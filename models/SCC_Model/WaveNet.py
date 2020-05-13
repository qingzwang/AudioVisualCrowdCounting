import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


class WaveNet(nn.Module):
    def __init__(self, output_channels=128, layers=10, blocks=3, kernel_size=2):
        super(WaveNet, self).__init__()
        blocks_layers = []
        input_layer = True
        for b in range(blocks):
            for l in range(layers):
                dilation = 2**l
                if input_layer:
                    layer = GLU(1, output_channels, kernel_size, dilation=dilation)
                    input_layer = False
                else:
                    layer = ResidualBlocks(
                        in_channels=output_channels, out_channels=output_channels, kernel_size=kernel_size,
                        dilation=dilation
                    )
                blocks_layers.append(layer)

        self.model = nn.Sequential(*blocks_layers)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        y = self.model(x)
        return y


class GLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(GLU, self).__init__()
        self.conv1d_a = nn.Conv1d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
            padding=padding, dilation=dilation, groups=groups, bias=bias
        )
        self.conv1d_b = nn.Conv1d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
            padding=padding, dilation=dilation, groups=groups, bias=bias
        )
        self.gate = nn.Sigmoid()
        self.dilation = dilation

    def forward(self, x):
        x = F.pad(x, [self.dilation, 0])
        return torch.mul(self.conv1d_a(x), self.gate(self.conv1d_b(x)))


class ResidualBlocks(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(ResidualBlocks, self).__init__()
        self.GLU = GLU(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding,
            dilation=dilation, groups=groups, bias=bias
        )
        self.conv1_1 = nn.Conv1d(
            in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1,
            padding=0, bias=False
        )

    def forward(self, x):
        skip = self.conv1_1(self.GLU(x))
        return torch.add(x, skip)

