import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import ImageClassifier 
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from data_augment import train_transforms, test_transforms
from data_augment import CIFAR10DataAugmentation
from torchvision.utils import save_image
import time



def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")

def unnormalize(tensor, mean, std):
    """
    Reverse the normalization process.
    Args:
        tensor: Normalized tensor.
        mean: Tuple of means for each channel.
        std: Tuple of standard deviations for each channel.
    Returns:
        Unnormalized tensor.
    """
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)  # Reverse the normalization
    return tensor

def unnormalize_batch(batch, mean, std):
    """
    Reverse the normalization process for a batch of images.
    Args:
        batch: Normalized batch of images.
        mean: Tuple of means for each channel.
        std: Tuple of standard deviations for each channel.
    Returns:
        Unnormalized batch of images.
    """
    for i in range(batch.size(0)):
        batch[i] = unnormalize(batch[i], mean, std)
    return batch

if __name__ == '__main__':
    # Hyperparameters
    batch_size = 400
    epochs = 40
    initial_lr = 0.01 # doesnt do anything with 1 cycle policy
    max_lr = 0.01
    grad_clip = 0.1
    weight_decay = 0.0001

    # Initialize data augmentation
    data_augmentation = CIFAR10DataAugmentation()

    # Load datasets using CIFAR10DataAugmentation
    train_dataset = datasets.CIFAR10(root="../data", train=True, transform=data_augmentation.get_train_transforms(), download=True)
    test_dataset = datasets.CIFAR10(root="../data", train=False, transform=data_augmentation.get_test_transforms(), download=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"CUDA is available: {torch.cuda.is_available()}")

    model = ImageClassifier(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=len(train_loader), epochs=epochs)

    count_parameters(model)
    first_epoch = True
    total_time = 0.0

    for epoch in range(epochs):
        start_time = time.time()
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # Save first 24 images of the first batch of the first epoch and apply unnormalization
            if first_epoch and batch_idx == 0:
                unnormalized_images = unnormalize_batch(images[:24].cpu(), mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
                save_image(unnormalized_images, 'first_24_images.png', nrow=6)
                first_epoch = False

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            if grad_clip: 
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)

            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()

            scheduler.step()


        avg_loss = running_loss / len(train_loader.dataset)
        train_accuracy = 100.0 * correct_train / total_train
        train_error = 1.0 - (train_accuracy / 100.0)

        # Step the scheduler with training error

        # Validation phase (test set)
        model.eval()
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                total_test += labels.size(0)
                correct_test += predicted.eq(labels).sum().item()

        test_accuracy = 100.0 * correct_test / total_test

        epoch_time = time.time() - start_time
        total_time += epoch_time
        print(f"Epoch {epoch+1}/{epochs}, "
              f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}, "
              f"Train Loss: {avg_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
              f"Test Acc: {test_accuracy:.2f}%, "
              f"Epoch Time: {epoch_time:.2f}s")

        # Save model every 10 epochs (overwrite)
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), "model_parameters.pth")
            print(f"Model checkpoint saved at epoch {epoch+1}")

    avg_epoch_time = total_time / epochs
    print(f"Average Epoch Time: {avg_epoch_time:.2f}s")
    print(f"Total Training Time: {total_time:.2f}s")