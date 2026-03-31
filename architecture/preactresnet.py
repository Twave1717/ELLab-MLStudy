from torch import nn

class ZeroPadShortcut(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        x = x[:, :, ::self.stride, ::self.stride]
        x = nn.functional.pad(x, (0, 0, 0, 0, 0, self.out_channels - self.in_channels))
        return x


class PreActResBlock3x3(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.plain_sequence = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
        )
        if stride == 1 and in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = ZeroPadShortcut(in_channels, out_channels, stride)

    def forward(self, input):
        x = self.shortcut(input)
        fx = self.plain_sequence(input)
        return x + fx


class PreActResNet(nn.Module):
    def __init__(self, num_layers, num_classes, feature_base=16):
        super().__init__()
        num_stage_layers = int((num_layers-2)/6)
        
        self.conv1 = nn.Conv2d(3, feature_base, 3, padding=1)
        self.stage1 = nn.Sequential(
            PreActResBlock3x3(feature_base, feature_base),
            *[PreActResBlock3x3(feature_base, feature_base) for _ in range(num_stage_layers-1)]
        )
        self.stage2 = nn.Sequential(
            PreActResBlock3x3(feature_base, feature_base*2, stride=2),
            *[PreActResBlock3x3(feature_base*2, feature_base*2) for _ in range(num_stage_layers-1)]
        )
        self.stage3 = nn.Sequential(
            PreActResBlock3x3(feature_base*2, feature_base*4, stride=2),
            *[PreActResBlock3x3(feature_base*4, feature_base*4) for _ in range(num_stage_layers-1)]
        )
        self.final_bn = nn.BatchNorm2d(feature_base*4)
        self.final_relu = nn.ReLU()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(feature_base*4, num_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.final_bn(x)
        x = self.final_relu(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        logits = self.fc(x)
        return logits

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)