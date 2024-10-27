import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any

def add_docstring(docstring: str):
    """Decorator to add a docstring to a function or class."""
    def decorator(func: Any):
        func.__doc__ = docstring
        return func
    return decorator

@add_docstring("Squeeze-and-Excitation Layer to adaptively recalibrate channel-wise feature responses.")
class SELayer(nn.Module):
    def __init__(self, channel: int, reduction: int = 8):
        """
        Args:
            channel (int): Number of input channels.
            reduction (int): Reduction ratio for the fully connected layers.
        """
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through SELayer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, channels, height, width).

        Returns:
            torch.Tensor: Output tensor after recalibrating channels.
        """
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

@add_docstring("Alternative weighting layer for feature recalibration.")
class WLayer(nn.Module):
    def __init__(self, channel: int, reduction: int = 8):
        """
        Args:
            channel (int): Number of input channels.
            reduction (int): Reduction ratio for fully connected layers.
        """
        super(WLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through WLayer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, channels, height, width).

        Returns:
            torch.Tensor: Output tensor after applying recalibration.
        """
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

@add_docstring("ResNeXt bottleneck layer with group convolutions for feature extraction.")
class ResNeXtBottleneck(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, 
                 cardinality: int = 16, base_width: int = 4, widen_factor: int = 1):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int): Stride for convolution. Defaults to 1.
            cardinality (int): Number of convolution groups.
            base_width (int): Base number of channels per group.
            widen_factor (int): Factor to adjust the input dimensionality before convolution.
        """
        super(ResNeXtBottleneck, self).__init__()
        width_ratio = out_channels / (widen_factor * 64.)
        D = cardinality * int(base_width * width_ratio)
        self.conv_reduce = nn.Conv2d(in_channels, D, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_reduce = nn.BatchNorm2d(D)
        self.conv_conv = nn.Conv2d(D, D, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn = nn.BatchNorm2d(D)
        self.conv_expand = nn.Conv2d(D, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_expand = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module('shortcut_conv',
                                     nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False))
            self.shortcut.add_module('shortcut_bn', nn.BatchNorm2d(out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ResNeXtBottleneck layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after ResNeXt block.
        """
        bottleneck = self.conv_reduce(x)
        bottleneck = F.relu(self.bn_reduce(bottleneck), inplace=True)
        bottleneck = self.conv_conv(bottleneck)
        bottleneck = F.relu(self.bn(bottleneck), inplace=True)
        bottleneck = self.conv_expand(bottleneck)
        bottleneck = self.bn_expand(bottleneck)
        residual = self.shortcut(x)
        return F.relu(residual + bottleneck, inplace=True)

@add_docstring("Residual Block with optional channel adjustment.")
class ResidualBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, stride: int = 1):
        """
        Args:
            input_channels (int): Number of input channels.
            output_channels (int): Number of output channels.
            stride (int): Stride for convolution. Defaults to 1.
        """
        super(ResidualBlock, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.stride = stride

        self.bn1 = nn.BatchNorm2d(input_channels)
        self.conv1 = nn.Conv2d(input_channels, output_channels // 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(output_channels // 4)
        self.conv2 = nn.Conv2d(output_channels // 4, output_channels // 4, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(output_channels // 4)
        self.conv3 = nn.Conv2d(output_channels // 4, output_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if input_channels != output_channels or stride != 1:
            self.conv4 = nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=stride, padding=0, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ResidualBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after residual block.
        """
        residual = x.clone()
        out = self.bn1(x)
        out1 = self.relu(out)
        out = self.conv1(out1)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        if (self.input_channels != self.output_channels) or (self.stride != 1):
            residual = self.conv4(out1)
        out += residual
        return out

