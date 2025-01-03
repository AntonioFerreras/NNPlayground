import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import ImageClassifier 
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from data_augment import CIFAR10DataAugmentation
from torchvision.utils import save_image
import time
from torch.profiler import profile, record_function, ProfilerActivity

from torch.utils.tensorboard import SummaryWriter
import argparse

# torch.set_float32_matmul_precision('high')

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")

def unnormalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)  # Reverse the normalization
    return tensor

def unnormalize_batch(batch, mean, std):
    for i in range(batch.size(0)):
        batch[i] = unnormalize(batch[i], mean, std)
    return batch

class DummyContextManager:
    def __enter__(self):
        pass
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train a model with optional profiling and customizable epochs.")
    parser.add_argument("--profile", action="store_true", help="Enable PyTorch Profiler.")
    parser.add_argument("--epochs", type=int, default=40, help="Number of epochs to train the model.")
    parser.add_argument("--compile", action="store_true", help="Enable PyTorch compilation for the model.")
    args = parser.parse_args()

    # Hyperparameters
    batch_size = 500
    initial_lr = 0.01
    max_lr = 0.01
    grad_clip = 0.1
    weight_decay = 0.0001
    epochs = args.epochs

    # Initialize data augmentation
    data_augmentation = CIFAR10DataAugmentation()

    # Load datasets using CIFAR10DataAugmentation
    train_dataset = datasets.CIFAR10(root="../data", train=True, transform=data_augmentation.get_train_transforms(), download=True)
    test_dataset = datasets.CIFAR10(root="../data", train=False, transform=data_augmentation.get_test_transforms(), download=True)


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=2, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=2, persistent_workers=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"CUDA is available: {torch.cuda.is_available()}")

    model = ImageClassifier(num_classes=10).to(device)
    if args.compile:
        model = torch.compile(model)



    criterion = nn.CrossEntropyLoss(label_smoothing=0.2)
    optimizer = optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=len(train_loader), epochs=epochs)

    count_parameters(model)
    first_epoch = True
    total_time = 0.0

    # TensorBoard Writer
    writer = None
    if args.profile:
        writer = SummaryWriter(log_dir="./logs3")

    # scaler = torch.amp.GradScaler()

    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        running_loss = torch.tensor(0.0, device=device)
        correct_train = torch.tensor(0, device=device, dtype=torch.int64)
        total_train = torch.tensor(0, device=device, dtype=torch.int64)

        do_profile = args.profile and epoch >= 1
        prof = None
        if do_profile:
            # Profiler setup
            prof = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs3'),
                record_shapes=True,
                profile_memory=True,
                with_stack=False
            )
            prof.start()

        for batch_idx, (images, labels) in enumerate(train_loader):
            with (record_function("data transfer") if do_profile else DummyContextManager()):
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                with (record_function("model forward") if do_profile else DummyContextManager()):
                    outputs = model(images)
                with (record_function("loss computation") if do_profile else DummyContextManager()):
                    loss = criterion(outputs, labels)

            # outputs = model(images)
            # loss = criterion(outputs.to(dtype=torch.float32), labels)

            with (record_function("backward pass") if do_profile else DummyContextManager()):
                # scaler.scale(loss).backward()
                # loss.to(dtype=torch.bfloat16).backward()
                loss.backward()

            # scaler.unscale_(optimizer)

            if writer:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        writer.add_scalar(f"Gradients/{name}_norm", param.grad.norm().item(), epoch * len(train_loader) + batch_idx)
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            print(f"NaN or INF detected in gradients of {name} at batch {batch_idx} of epoch {epoch}")
                            break


            if grad_clip:
                with (record_function("gradient clipping") if do_profile else DummyContextManager()):
                    
                    nn.utils.clip_grad_value_(model.parameters(), grad_clip)

            with (record_function("optimizer step") if do_profile else DummyContextManager()):
                # scaler.step(optimizer)
                optimizer.step()

            # scaler.update()
                
            with (record_function("metrics calculation") if do_profile else DummyContextManager()):
                running_loss += loss.detach() * images.size(0)
                _, predicted = outputs.max(1)
                total_train += labels.size(0)
                correct_train += predicted.eq(labels).sum().detach()

            with (record_function("scheduler step") if do_profile else DummyContextManager()):
                scheduler.step()

            if do_profile:
                prof.step()


        if do_profile:
            prof.stop()

        # Transfer metrics to CPU only once
        running_loss_cpu = running_loss.item()
        correct_train_cpu = correct_train.item()
        total_train_cpu = total_train.item()

        # Compute final metrics
        avg_loss = running_loss_cpu / len(train_loader.dataset)
        train_accuracy = 100.0 * correct_train_cpu / total_train_cpu


        # Log training metrics
        if writer:
            writer.add_scalar("Loss/Train", avg_loss, epoch)
            writer.add_scalar("Accuracy/Train", train_accuracy, epoch)

        # Validation phase (test set)
        model.eval()
        correct_test = torch.tensor(0, device=device, dtype=torch.int64)
        total_test = torch.tensor(0, device=device, dtype=torch.int64)
        with torch.no_grad():
            for images, labels in test_loader: 
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                _, predicted = outputs.max(1)
                total_test += labels.size(0)
                correct_test += predicted.eq(labels).sum().detach()

        test_accuracy = 100.0 * correct_test.item() / total_test.item()
        # test_accuracy = 0.0
        # Log validation metrics
        if writer:
            writer.add_scalar("Accuracy/Test", test_accuracy, epoch)

        epoch_time = time.time() - start_time
        total_time += epoch_time
        print(f"Epoch {epoch+1}/{epochs}, "
              f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}, "
              f"Train Loss: {avg_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
              f"Test Acc: {test_accuracy:.2f}%, "
              f"Epoch Time: {epoch_time:.2f}s")

        # Save model every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), "model_parameters.pth")
            print(f"Model checkpoint saved at epoch {epoch+1}")

    avg_epoch_time = total_time / epochs
    print(f"Average Epoch Time: {avg_epoch_time:.2f}s")
    print(f"Total Training Time: {total_time:.2f}s")
    if writer:
        writer.close()
    
    del train_loader
    del test_loader