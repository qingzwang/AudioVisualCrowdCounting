import torch.nn as nn
import torch
from torchvision import models
import torch.nn.functional as F
from .WaveNet import WaveNet
from .vggish import VGGish


class CSRNet(nn.Module):
    def __init__(self, load_weights=True):
        super(CSRNet, self).__init__()
        self.seen = 0
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat = [512, 512, 512, 256, 128, 64]
        self.frontend = make_layers(self.frontend_feat, instance_norm=False)
        self.backend = make_layers(self.backend_feat, in_channels=512, dilation=True, batch_norm=True, instance_norm=False, affine=False)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        # self.wavenet = WaveNet()
        # self.fully_connected = make_fully_connected_layer(cfg=self.backend_feat)
        self.audio_vgg = VGGish()

        self.fully_connected = make_fully_connected_layer(cfg=self.backend_feat, in_channels=512)

        if load_weights:
            mod = models.vgg16(pretrained=True)
            self._initialize_weights()
            self.frontend.load_state_dict(mod.features[0:23].state_dict())
            self.audio_vgg.load_state_dict(torch.load(
                '/mnt/scratch/qingzhong/dataset/counting/C-3-Framework-python3.x/models/SCC_Model/pytorch_vggish.pth'))
        else:
            self._initialize_weights()
            self.audio_vgg.load_state_dict(torch.load(
                '/mnt/scratch/qingzhong/dataset/counting/C-3-Framework-python3.x/models/SCC_Model/pytorch_vggish.pth'))

    def forward(self, x):
        y = x[1]
        x = x[0]
        y = self.audio_vgg.features(y)
        by, cy, wy, hy = y.shape
        y = y.view(by, cy, -1).mean(-1)
        x = self.frontend(x)
        i = 0
        for l in self.backend:
            if isinstance(l, nn.BatchNorm2d):
                x = l(x)  # b x C x W x H
                b, c, h, w = x.shape
                # x = self._batch_norm(x)
                gamma_beta = self.fully_connected[i](y)  # b x 1024
                x = torch.mul(x, gamma_beta[:, 0:c].unsqueeze(2).unsqueeze(2).repeat([1, 1, h, w])) + \
                    gamma_beta[:, c:].unsqueeze(2).unsqueeze(2).repeat([1, 1, h, w])
                i += 1
            else:
                x = l(x)
        # x = self.backend(x)
        x = self.output_layer(x)
        x = F.upsample(x, scale_factor=8)
        return x

    def _batch_norm(self, x, epsilon=1e-10):
        b, c, w, h = x.shape
        y = x.transpose(1, 0).reshape(c, -1)
        x_mean = y.mean(-1).unsqueeze(0).unsqueeze(2).unsqueeze(2).repeat([b, 1, w, h])
        x_var = y.var(-1).unsqueeze(0).unsqueeze(2).unsqueeze(2).repeat([b, 1, w, h])
        return (x - x_mean) / torch.sqrt(x_var + epsilon)

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


class CSRNetConcat(nn.Module):
    def __init__(self, load_weights=True):
        super(CSRNetConcat, self).__init__()
        self.seen = 0
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat = [512, 512, 512, 256, 128, 64]
        self.frontend = make_layers(self.frontend_feat, instance_norm=False)
        self.backend = make_layers(
            self.backend_feat, in_channels=1024, dilation=True, batch_norm=False, instance_norm=False, affine=False
        )
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        # self.wavenet = WaveNet()
        # self.fully_connected = make_fully_connected_layer(cfg=self.backend_feat)
        self.audio_vgg = VGGish()
        self.audio_vgg.load_state_dict(
            torch.load('/mnt/home/dongsheng/hudi/counting/C-3-Framework-python3.x/models/SCC_Model/pytorch_vggish.pth')
        )

        if load_weights:
            mod = models.vgg16(pretrained=True)
            self._initialize_weights()
            self.frontend.load_state_dict(mod.features[0:23].state_dict())
        else:
            self._initialize_weights()

    def forward(self, x):
        y = x[1]
        x = x[0]
        y = self.audio_vgg.features(y)
        by, cy, wy, hy = y.shape
        y = y.view(by, cy, -1).mean(-1)  # by x cy
        x = self.frontend(x)
        bx, cx, wx, hx = x.shape
        y = y.unsqueeze(2).unsqueeze(2).repeat([1, 1, wx, hx])
        x = torch.cat([x, y], 1)  # b x 1024 x wx x hx
        x = self.backend(x)
        x = self.output_layer(x)
        x = F.upsample(x, scale_factor=8)
        return x

    def _batch_norm(self, x, epsilon=1e-10):
        b, c, w, h = x.shape
        y = x.transpose(1, 0).reshape(c, -1)
        x_mean = y.mean(-1).unsqueeze(0).unsqueeze(2).unsqueeze(2).repeat([b, 1, w, h])
        x_var = y.var(-1).unsqueeze(0).unsqueeze(2).unsqueeze(2).repeat([b, 1, w, h])
        return (x - x_mean) / torch.sqrt(x_var + epsilon)

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


class CSRNetGuided(nn.Module):
    def __init__(self, load_weights=True):
        super(CSRNetGuided, self).__init__()
        self.seen = 0
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat = [512, 512, 512, 256, 128, 64]
        self.frontend = make_layers(self.frontend_feat, instance_norm=False)
        # self.backend = make_layers(self.backend_feat, in_channels=512, dilation=True, batch_norm=True, instance_norm=False, affine=False)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        # self.wavenet = WaveNet()
        # self.fully_connected = make_fully_connected_layer(cfg=self.backend_feat)
        self.audio_vgg = VGGish()
        self.audio_vgg.load_state_dict(torch.load('/mnt/home/dongsheng/hudi/counting/C-3-Framework-python3.x/models/SCC_Model/pytorch_vggish.pth'))
        self.fc_feat = [50, 50, 50, 50, 50, 50]
        self.fully_connected = make_fully_connected_layer(
            cfg=self.fc_feat, in_channels=512
        )
        WB = []
        WC = []
        batch_norm = []
        new_backend_feat = [512, 512, 512, 512, 256, 128, 64]
        for i in range(len(new_backend_feat) - 1):
            c1 = new_backend_feat[i] * 3
            c2 = self.fc_feat[i] * 2
            c3 = new_backend_feat[i+1] * 3
            WB.append(LinearTrans(c_in=c2, c_out=c3))
            WC.append(LinearTrans(c_in=c2, c_out=c1))
            batch_norm.append(nn.BatchNorm2d(new_backend_feat[i+1]))
        self.WB = nn.Sequential(*WB)
        self.WC = nn.Sequential(*WC)
        self.batch_norm = nn.Sequential(*batch_norm)

        if load_weights:
            mod = models.vgg16(pretrained=True)
            self._initialize_weights()
            self.frontend.load_state_dict(mod.features[0:23].state_dict())
        else:
            self._initialize_weights()

    def forward(self, x):
        y = x[1]
        x = x[0]
        y = self.audio_vgg.features(y)
        by, cy, wy, hy = y.shape
        y = y.view(by, cy, -1).mean(-1)
        x = self.frontend(x)  # b x 512 x w x h

        for i in range(len(self.backend_feat)):
            w_a = self.fully_connected[i](y)  # b x 100
            w_a = torch.diag_embed(w_a)  # b x 100 x 100
            w_b = self.WB[i]
            w_c = self.WC[i]
            w = w_b(w_a)  # b x c_out*3 x 100
            w = w_c(w.transpose(1, 2)).transpose(1, 2)
            batch, width, height = w.shape
            # print(w.shape)
            w = w.contiguous().view(batch, int(width / 3), int(height / 3), 3, 3)
            output = []
            for j in range(batch):
                o = nn.functional.conv2d(x[j:(j+1), :, :, :], w[j, :, :, :, :], dilation=2, padding=2)
                output.append(o)
            x = torch.cat(output, 0)
            x = self.batch_norm[i](x)
            x = nn.functional.relu(x, inplace=True)

        # x = self.backend(x)
        x = self.output_layer(x)
        x = F.upsample(x, scale_factor=8)
        return x

    def _batch_norm(self, x, epsilon=1e-10):
        b, c, w, h = x.shape
        y = x.transpose(1, 0).reshape(c, -1)
        x_mean = y.mean(-1).unsqueeze(0).unsqueeze(2).unsqueeze(2).repeat([b, 1, w, h])
        x_var = y.var(-1).unsqueeze(0).unsqueeze(2).unsqueeze(2).repeat([b, 1, w, h])
        return (x - x_mean) / torch.sqrt(x_var + epsilon)

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


class LinearTrans(nn.Module):
    def __init__(self, c_in, c_out):
        super(LinearTrans, self).__init__()
        self._c_in = c_in
        self._c_out = c_out
        self.w = nn.Parameter(torch.randn(c_out, c_in) * 0.01)

    def forward(self, x):
        return torch.matmul(self.w, x)


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


def make_fully_connected_layer(cfg, in_channels=128):
    layers = []
    for v in cfg:
        layers += [nn.Linear(in_channels, 2 * v)]

    return nn.Sequential(*layers)
