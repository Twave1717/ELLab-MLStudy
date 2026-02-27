from torch import nn

class ResBlock3x3(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.plain_sequence = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.shortcut(input)
        fx = self.plain_sequence(input)
        return self.relu(x + fx)


class Resnet(nn.Module):
    def __init__(self, num_layers, num_classes, feature_base=16):
        super().__init__()
        num_stage_layers = int((num_layers-2)/6)
        
        self.conv1 = nn.Conv2d(3, feature_base, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(feature_base)
        self.relu = nn.ReLU()
        self.stage1 = nn.Sequential(*[ResBlock3x3(feature_base, feature_base) for _ in range(num_stage_layers)])
        self.conv2 = ResBlock3x3(feature_base, feature_base*2, stride=2)
        self.stage2 = nn.Sequential(*[ResBlock3x3(feature_base*2, feature_base*2) for _ in range(num_stage_layers-1)])
        self.conv3 = ResBlock3x3(feature_base*2, feature_base*4, stride=2)
        self.stage3 = nn.Sequential(*[ResBlock3x3(feature_base*4, feature_base*4) for _ in range(num_stage_layers-1)])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(feature_base*4, num_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.stage1(x)
        x = self.conv2(x)
        x = self.stage2(x)
        x = self.conv3(x)
        x = self.stage3(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        logits = self.fc(x)
        return logits

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)