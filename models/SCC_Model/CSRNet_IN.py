import torch.nn as nn
import torch
from torchvision import models
import torch.nn.functional as F


class CSRNet(nn.Module):
    def __init__(self, load_weights=True):
        super(CSRNet, self).__init__()
        self.seen = 0
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat = [512, 512, 512, 256, 128, 64]
        self.frontend = make_layers(self.frontend_feat, instance_norm=False, batch_norm=False, affine=True)
        self.backend = make_layers(self.backend_feat, in_channels=512, dilation=True, instance_norm=False, batch_norm=True, affine=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        if load_weights:
            mod = models.vgg16(pretrained=True)
            # mod = models.vgg16_bn(pretrained=True)
            self._initialize_weights()
            print(len(self.frontend))
            self.frontend.load_state_dict(mod.features[0:len(self.frontend)].state_dict())
        else:
            self._initialize_weights()

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        x = F.upsample(x, scale_factor=8)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) and torch.typename(m.weight) != 'NoneType':
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.InstanceNorm2d) and torch.typename(m.weight) != 'NoneType':
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, in_channels=3, batch_norm=False, instance_norm=True, dilation=False, affine=False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            if batch_norm and instance_norm:
                layers += [conv2d, nn.BatchNorm2d(v, affine=affine), nn.InstanceNorm2d(v, affine=affine), nn.ReLU(inplace=True)]
            elif batch_norm and not instance_norm:
                layers += [conv2d, nn.BatchNorm2d(v, affine=affine), nn.ReLU(inplace=True)]
            elif not batch_norm and instance_norm:
                layers += [conv2d, nn.InstanceNorm2d(v, affine=affine), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)
