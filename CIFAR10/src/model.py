import torch
import torch.nn as nn
import torch.nn.functional as F

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