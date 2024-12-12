import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define the MLP model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 256)  # Input layer (28x28 pixels)
        self.fc2 = nn.Linear(256, 128)      # Hidden layer
        self.fc3 = nn.Linear(128, 128)       # Output layer (10 classes for digits 0-9)
        self.fc4 = nn.Linear(128, 10)       # Output layer (10 classes for digits 0-9)
        self.relu = nn.ReLU()               # Activation function
        self.dropout = nn.Dropout(0.2)     # Dropout for regularization

    def forward(self, x):
        x = x.view(-1, 28 * 28)            # Flatten the image
        x = self.relu(self.fc1(x))         # Input to hidden layer
        x = self.dropout(x)                # Dropout layer
        x = self.relu(self.fc2(x))         # Hidden to another hidden layer
        x = self.dropout(x)                # Dropout layer
        x = self.relu(self.fc3(x))         # Hidden to another hidden layer
        x = self.fc4(x)                    # Hidden to output layer
        return x

# Print total and trainable parameter counts
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")

# Hyperparameters
batch_size = 128
epochs = 20
learning_rate = 0.001

# Data preparation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.FashionMNIST(root="./data", train=True, transform=transform, download=True)
test_dataset = datasets.FashionMNIST(root="./data", train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Model, loss, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"CUDA is available: {torch.cuda.is_available()}")
model = MLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.1, threshold=0.4)
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
    print(f"Test Accuracy: {test_accuracy:.2f}%")

    # Step the scheduler with the test accuracy
    scheduler.step(test_accuracy)
