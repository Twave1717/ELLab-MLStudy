import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, act_layer=nn.ReLU):
        super().__init__()

        self.bn = nn.BatchNorm2d(in_channels)
        self.act = act_layer(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)

    def forward(self, x):
        return self.conv(self.act(self.bn(x)))
    
class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, act_layer = nn.ReLU):
        super().__init__()

        inter_planes = out_planes * 4
        self.conv1 = ConvLayer(in_planes, inter_planes, kernel_size=1, padding=0, act_layer=act_layer)
        self.conv2 = ConvLayer(inter_planes, out_planes, kernel_size=3, padding=1, act_layer=act_layer)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        return torch.cat([x, out], dim=1)
    
class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, act_layer = nn.ReLU):
        super().__init__()

        self. conv = ConvLayer(in_planes, out_planes, kernel_size=1, padding=0, act_layer=act_layer)

    def forward(self, x):
        out = self.conv(x)
        
        return F.avg_pool2d(out, kernel_size=2, stride=2)

class DenseNet(nn.Module):
    def __init__(self, depth, num_classes, growth_rate = 12, reduction = 0.5, act_layer = nn.ReLU):
        super().__init__()

        n = int((depth - 4) / 6)

        in_planes = 2 * growth_rate

        self.features = nn.Sequential()
        self.features.add_module('init_conv', nn.Conv2d(3, in_planes, kernel_size=3, stride=1, padding=1, bias = False))

        for i in range(3):
            for j in range(n):
                block_name = f'dense_block_{i+1}_layer_{j+1}'
                self.features.add_module(block_name, BottleneckBlock(in_planes, growth_rate, act_layer))
                in_planes += growth_rate

            if i < 2:
                out_planes = int(in_planes * reduction)
                trans_name = f'transition_{i+1}'
                self.features.add_module(trans_name, TransitionBlock(in_planes, out_planes, act_layer))
                in_planes = out_planes

        self.final_bn = nn.BatchNorm2d(in_planes)
        self.final_act = act_layer(inplace=True)
        self.classifier = nn.Linear(in_planes, num_classes)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        features = self.features(x)
        out = self.final_act(self.final_bn(features))

        out = F.adaptive_avg_pool2d(out, (1,1))
        out = torch.flatten(out, 1)

        return self.classifier(out)
