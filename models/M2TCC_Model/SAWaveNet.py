#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
from misc.utils import initialize_weights
import pdb
from ..SCC_Model.vggish import VGGish


class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn=False, **kwargs):
        super(BasicConv, self).__init__()
        self.use_bn = use_bn
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not self.use_bn, **kwargs)
        self.bn = nn.InstanceNorm2d(out_channels, affine=True) if self.use_bn else None

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        return F.relu(x, inplace=True)


class BasicDeconv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, use_bn=False):
        super(BasicDeconv, self).__init__()
        self.use_bn = use_bn
        self.tconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, bias=not self.use_bn)
        self.bn = nn.InstanceNorm2d(out_channels, affine=True) if self.use_bn else None

    def forward(self, x):
        # pdb.set_trace()
        x = self.tconv(x)
        if self.use_bn:
            x = self.bn(x)
        return F.relu(x, inplace=True)


class SAModule_Head(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn):
        super(SAModule_Head, self).__init__()
        branch_out = out_channels // 4
        self.branch1x1 = BasicConv(in_channels, branch_out, use_bn=use_bn,
                                   kernel_size=1)
        self.branch3x3 = BasicConv(in_channels, branch_out, use_bn=use_bn,
                                   kernel_size=3, padding=1)
        self.branch5x5 = BasicConv(in_channels, branch_out, use_bn=use_bn,
                                   kernel_size=5, padding=2)
        self.branch7x7 = BasicConv(in_channels, branch_out, use_bn=use_bn,
                                   kernel_size=7, padding=3)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch7x7 = self.branch7x7(x)
        out = torch.cat([branch1x1, branch3x3, branch5x5, branch7x7], 1)
        return out


class SAModule(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn):
        super(SAModule, self).__init__()
        branch_out = out_channels // 4
        self.branch1x1 = BasicConv(in_channels, branch_out, use_bn=use_bn,
                                   kernel_size=1)
        self.branch3x3 = nn.Sequential(
            BasicConv(in_channels, 2 * branch_out, use_bn=use_bn,
                      kernel_size=1),
            BasicConv(2 * branch_out, branch_out, use_bn=use_bn,
                      kernel_size=3, padding=1),
        )
        self.branch5x5 = nn.Sequential(
            BasicConv(in_channels, 2 * branch_out, use_bn=use_bn,
                      kernel_size=1),
            BasicConv(2 * branch_out, branch_out, use_bn=use_bn,
                      kernel_size=5, padding=2),
        )
        self.branch7x7 = nn.Sequential(
            BasicConv(in_channels, 2 * branch_out, use_bn=use_bn,
                      kernel_size=1),
            BasicConv(2 * branch_out, branch_out, use_bn=use_bn,
                      kernel_size=7, padding=3),
        )

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch7x7 = self.branch7x7(x)
        out = torch.cat([branch1x1, branch3x3, branch5x5, branch7x7], 1)
        return out


class SANet(nn.Module):
    def __init__(self, gray_input=False, use_bn=True):
        super(SANet, self).__init__()
        if gray_input:
            in_channels = 1
        else:
            in_channels = 3

        self.encoder = nn.Sequential(
            SAModule_Head(in_channels, 64, use_bn),
            nn.MaxPool2d(2, 2),
            SAModule(64, 128, use_bn),
            nn.MaxPool2d(2, 2),
            SAModule(128, 128, use_bn),
            nn.MaxPool2d(2, 2),
            SAModule(128, 128, use_bn),
        )

        # self.decoder = nn.Sequential(
        #     BasicConv(128, 64, use_bn=use_bn, kernel_size=9, padding=4),
        #     BasicDeconv(64, 64, 2, stride=2, use_bn=use_bn),
        #     BasicConv(64, 32, use_bn=use_bn, kernel_size=7, padding=3),
        #     BasicDeconv(32, 32, 2, stride=2, use_bn=use_bn),
        #     BasicConv(32, 16, use_bn=use_bn, kernel_size=5, padding=2),
        #     BasicDeconv(16, 16, 2, stride=2, use_bn=use_bn),
        #     BasicConv(16, 16, use_bn=use_bn, kernel_size=3, padding=1),
        #     BasicConv(16, 1, use_bn=False, kernel_size=1),
        # )
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=9, padding=4, bias=False),
            nn.InstanceNorm2d(64, affine=False),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(64, affine=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=7, padding=3, bias=False),
            nn.InstanceNorm2d(32, affine=False),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(32, affine=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=5, padding=2, bias=False),
            nn.InstanceNorm2d(16, affine=False),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(16, affine=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(16, affine=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1, padding=0, bias=True)
        )

        self.fully_connect_layers = nn.Sequential(
            nn.Linear(in_features=512, out_features=64 * 2),
            nn.Linear(in_features=512, out_features=64 * 2),
            nn.Linear(in_features=512, out_features=32 * 2),
            nn.Linear(in_features=512, out_features=32 * 2),
            nn.Linear(in_features=512, out_features=16 * 2),
            nn.Linear(in_features=512, out_features=16 * 2),
            nn.Linear(in_features=512, out_features=16 * 2)
        )

        self._initialize_weights()

        self.audio_vgg = VGGish()
        self.audio_vgg.load_state_dict(
            torch.load('/mnt/home/dongsheng/hudi/counting/C-3-Framework-python3.x/models/SCC_Model/pytorch_vggish.pth'))

    def forward(self, x):
        y = x[1]
        x = x[0]
        y = self.audio_vgg.features(y)
        by, cy, wy, hy = y.shape
        y = y.view(by, cy, -1).mean(-1)

        x = self.encoder(x)
        fl = 0
        for l in self.decoder:
            # print(l, x.shape)
            if isinstance(l, nn.InstanceNorm2d):
                x = l(x)
                b, c, h, w = x.shape
                gamma_beta = self.fully_connect_layers[fl](y)  # b x 1024
                x = torch.mul(x, gamma_beta[:, 0:c].unsqueeze(2).unsqueeze(2).repeat([1, 1, h, w])) + \
                    gamma_beta[:, c:].unsqueeze(2).unsqueeze(2).repeat([1, 1, h, w])
                fl += 1
            else:
                x = l(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d) and torch.typename(m.weight) != 'NoneType':
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.InstanceNorm2d) and torch.typename(m.weight) != 'NoneType':
                nn.init.constant_(m.weight, 1)