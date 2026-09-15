import torch
import torch.nn as nn
import torch.nn.functional as F


class RotNetBlock(nn.Module):
    def __init__(self, in_channels, out1, out2, out3, kernel_size):
        super().__init__()

        padding = (kernel_size - 1) // 2

        self.conv1 = nn.Conv2d(in_channels, out1, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out1)

        self.conv2 = nn.Conv2d(out1, out2, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out2)

        self.conv3 = nn.Conv2d(out2, out3, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out3)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return x


class RotNet(nn.Module):
    def __init__(self, block_num):
        super().__init__()

        self.blocks = nn.ModuleList()
        self.blocks.append(RotNetBlock(3, 192, 160, 96, 5))
        self.blocks.append(RotNetBlock(96, 192, 192, 192, 5))
        for _ in range(2, block_num):
            self.blocks.append(RotNetBlock(192, 192, 192, 192, 3))

        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        self.num_features = 192

    def forward(self, x):
        x = self.blocks[0](x)
        x = self.max_pool(x)

        x = self.blocks[1](x)
        x = self.avg_pool(x)

        for i in range(2, len(self.blocks)):
            x = self.blocks[i](x)

        x = self.global_avg_pool(x)
        x = self.flatten(x)
        return x
    
    def forward_n(self, x, n):
        x = self.blocks[0](x)
        x = self.max_pool(x)

        if n == 1:
            return x

        x = self.blocks[1](x)
        x = self.avg_pool(x)

        if n == 2:
            return x

        for i in range(2, len(self.blocks)):
            x = self.blocks[i](x)
            if n == i + 1:
                return x

        return x