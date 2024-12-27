import torch
import torch.nn as nn
import torch.nn.functional as F

# low precision variants of modules, keeps weights and grad in FP32, but performs forward pass in FP16
class Linear(torch.nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        compute_type = input.dtype
        w = self.weight.to(compute_type)
        b = self.bias.to(compute_type) if self.bias is not None else None
        return torch.nn.functional.linear(input, w, b)

class Conv2d(torch.nn.Conv2d):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        compute_type = input.dtype
        w = self.weight.to(compute_type)
        b = self.bias.to(compute_type) if self.bias is not None else None
        return torch.nn.functional.conv2d(input, w, b, self.stride,
                                           self.padding, self.dilation, self.groups)

class BatchNorm2d(torch.nn.BatchNorm2d):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        compute_type = input.dtype
        w = self.weight.to(compute_type)
        b = self.bias.to(compute_type)
        m = self.running_mean.to(compute_type)
        v = self.running_var.to(compute_type)
        return torch.nn.functional.batch_norm(input, m, v, w, b, self.training,
                                              self.momentum, self.eps)

class BatchNorm1d(torch.nn.BatchNorm1d):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        compute_type = input.dtype
        w = self.weight.to(compute_type)
        b = self.bias.to(compute_type)
        m = self.running_mean.to(compute_type)
        v = self.running_var.to(compute_type)
        return torch.nn.functional.batch_norm(input, m, v, w, b, self.training,
                                              self.momentum, self.eps)

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
        x = self.relu(x)
        return x