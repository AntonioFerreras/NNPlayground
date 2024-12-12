import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Your CNN model definition (same as before)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv12 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn12 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv22 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn22 = nn.BatchNorm2d(128)
        self.conv23 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn23 = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv31 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn31 = nn.BatchNorm2d(256)
        self.conv32 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn32 = nn.BatchNorm2d(256)

        # Pooling layer
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling

        # BN 
        self.bnfc1 = nn.BatchNorm1d(256)
        self.bnfc2 = nn.BatchNorm1d(128)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

        # Fully connected layers
        self.fc1 = nn.Linear(256, 128)  
        self.fc2 = nn.Linear(128, 10)  

    def forward(self, x):
        # Convolution + BatchNorm + ReLU + Pooling
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn12(F.relu(self.conv12(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.bn22(F.relu(self.conv22(x)))
        x = self.bn23(F.relu(self.conv23(x)))
        x = self.bn3(F.relu(self.conv3(x)))
        x = self.bn31(F.relu(self.conv31(x)))
        x = self.bn32(F.relu(self.conv32(x)))

        x = self.pool(x)                      # Global average pooling

        # Flatten the tensor
        x = x.view(x.size(0), -1)       
        
        # Fully connected layers + ReLU + Dropout
        x = self.bnfc1(x)                     # BatchNorm before FC
        x = F.relu(self.fc1(x))               # Fully connected layer 1
        x = self.bnfc2(x)
        x = F.relu(self.fc2(x))               # Fully connected layer 2
        
        return x

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")

# Hyperparameters
batch_size = 32
epochs = 70
learning_rate = 0.001

# Use transforms appropriate for training and testing
# Since images are now PNGs in directories, they will be loaded as PIL images.
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # ensure single-channel
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # ensure single-channel
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


# Load the offline augmented dataset
train_dataset = datasets.ImageFolder(root="offline_augmented_data", transform=train_transform)
test_dataset = datasets.FashionMNIST(root="../data", train=False, transform=test_transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"CUDA is available: {torch.cuda.is_available()}")

    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=8, factor=0.5, threshold=0.025)

    count_parameters(model)

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Compute average loss for the epoch
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        # Testing loop
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_accuracy = 100 * correct / total
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Test Accuracy: {test_accuracy:.3f}%, Learning Rate: {current_lr:.6f}")

        # Step the scheduler with the test accuracy
        scheduler.step(test_accuracy)
