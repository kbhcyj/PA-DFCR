import torch.nn as nn
import math
from . import utils
from collections import OrderedDict

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    # "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, cfg ,stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        m = OrderedDict()
        m['conv1'] = conv3x3(inplanes, cfg, stride)
        m['bn1'] = nn.BatchNorm2d(cfg)
        m['relu1'] = nn.ReLU(inplace=True)
        m['conv2'] = conv3x3(cfg, planes)
        m['bn2'] = nn.BatchNorm2d(planes)
        self.group1 = nn.Sequential(m)

        self.relu= nn.Sequential(nn.ReLU(inplace=True))
        self.downsample = downsample

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        out = self.group1(x) + residual

        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, cfg, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        m  = OrderedDict()
        m['conv1'] = nn.Conv2d(inplanes, cfg, kernel_size=1, bias=False)
        m['bn1'] = nn.BatchNorm2d(cfg)
        m['relu1'] = nn.ReLU(inplace=True)
        m['conv2'] = nn.Conv2d(cfg, cfg, kernel_size=3, stride=stride, padding=1, bias=False)
        m['bn2'] = nn.BatchNorm2d(cfg)
        m['relu2'] = nn.ReLU(inplace=True)
        m['conv3'] = nn.Conv2d(cfg, planes * 4, kernel_size=1, bias=False)
        m['bn3'] = nn.BatchNorm2d(planes * 4)
        self.group1 = nn.Sequential(m)

        self.relu= nn.Sequential(nn.ReLU(inplace=True))
        self.downsample = downsample

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        out = self.group1(x) + residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, cfg=None, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()

        if cfg == None:
            cfg = [64, 128, 256, 512]

        m = OrderedDict()
        m['conv1'] = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        m['bn1'] = nn.BatchNorm2d(64)
        m['relu1'] = nn.ReLU(inplace=True)
        m['maxpool'] = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.group1= nn.Sequential(m)

        self.layer1 = self._make_layer(block, 64, layers[0], cfg[0])
        self.layer2 = self._make_layer(block, 128, layers[1], cfg[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], cfg[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], cfg[3], stride=2)

        self.avgpool = nn.Sequential(nn.AvgPool2d(7))

        self.group2 = nn.Sequential(
            OrderedDict([
                ('fc', nn.Linear(512 * block.expansion, num_classes))
            ])
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, cfg, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, cfg, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,cfg))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.group1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.group2(x)

        return x



def resnet34(pretrained=False, model_root=None, **kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        utils.load_state_dict(model, model_urls['resnet34'], model_root)
    return model



def resnet101(pretrained=False, model_root=None, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        utils.load_state_dict(model, model_urls['resnet101'], model_root)
    return model

