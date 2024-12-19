import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class ResidualPair(nn.Module):
    """
    A pair of 3x3 convolutional layers with BN and ReLU, forming a residual unit.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int): Stride for the first convolution in the pair.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualPair, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, 
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, 
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # If there's a dimension change in channels or spatial size, use a projection
        self.projection = None
        if (in_channels != out_channels) or (stride != 1):
            self.projection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.projection is not None:
            identity = self.projection(identity)

        out += identity
        out = self.relu(out)
        return out

class ConvBlock(nn.Module):
    """
    A block of an even number of convolutional layers grouped into pairs.
    Each pair forms a residual connection. The first pair uses stride=2 and may use projection.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        num_layers (int): Total number of layers in the block (must be even).
    """
    def __init__(self, in_channels, out_channels, num_layers):
        super(ConvBlock, self).__init__()
        assert num_layers % 2 == 0, "Number of layers in a block must be even."

        layers = []
        # The first pair uses stride=2 for downsampling
        layers.append(ResidualPair(in_channels, out_channels, stride=2))
        
        # Subsequent pairs use stride=1
        for _ in range((num_layers // 2) - 1):
            layers.append(ResidualPair(out_channels, out_channels, stride=1))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ImageClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(ImageClassifier, self).__init__()

        # Initial 7x7 conv, stride=2, output=64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Residual blocks
        self.block1 = ConvBlock(in_channels=64, out_channels=128, num_layers=4)
        self.block2 = ConvBlock(in_channels=128, out_channels=256, num_layers=6)
        self.block3 = ConvBlock(in_channels=256, out_channels=512, num_layers=4)
        # self.block4 = ConvBlock(in_channels=256, out_channels=512, num_layers=2)

        # Adaptive global pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.bnfc = nn.BatchNorm1d(512)

        # Fully connected layer for classification
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # Initial conv
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # Blocks
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        # x = self.block4(x)

        # Global average pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # FC
        x = self.bnfc(x)
        x = self.fc(x)  # [N,10]
        return F.relu(x)

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")

if __name__ == '__main__':
    # Hyperparameters
    batch_size = 400
    epochs = 30
    initial_lr = 0.01
    grad_clip = 0.1
    weight_decay = 0.0001
    momentum = 0.9

    # Data transforms
    CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR10_STD = (0.2470, 0.2435, 0.2616)
    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)
    ])

    # Load datasets
    train_dataset = datasets.ImageFolder(root="../offline_augmented_data", transform=image_transform)
    test_dataset = datasets.CIFAR10(root="../data", train=False, transform=image_transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"CUDA is available: {torch.cuda.is_available()}")

    model = ImageClassifier(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=weight_decay)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5, threshold=0.025)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(train_loader), epochs=epochs)

    count_parameters(model)

    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

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

        print(f"Epoch {epoch+1}/{epochs}, "
              f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}, "
              f"Train Loss: {avg_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
              f"Test Acc: {test_accuracy:.2f}%")

        # scheduler.step(test_accuracy)

        
        # Save model every 10 epochs (overwrite)
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), "model_parameters.pth")
            print(f"Model checkpoint saved at epoch {epoch+1}")