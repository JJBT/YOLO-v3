from torch import nn
import torch


class ConvBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 1,
            padding: int = 1,
            bias: bool = False
    ):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.leaky_relu(self.bn(self.conv(x)))


class DarknetBlock(nn.Module):
    def __init__(self, hid_channels: int, out_channels: int):
        super().__init__()
        self.block1 = ConvBlock(out_channels, hid_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.block2 = ConvBlock(hid_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block2(self.block1(x))


def first_darknet_layer() -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(32),
        nn.LeakyReLU(0.1)
    )


def darknet_layer(in_channels: int, out_channels: int, num_blocks: int) -> nn.Module:
    layers = [ConvBlock(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)]
    layers.extend([DarknetBlock(in_channels, out_channels) for _ in range(num_blocks)])
    return nn.Sequential(*layers)
