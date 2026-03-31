import torch
from torch import nn

class DenseLayer_BC(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()

        self.bottleneck = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1, bias=False)
        )

        self.conv = nn.Sequential(
            nn.BatchNorm2d(4 * growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1)
        )

    def forward(self, x):
        out = self.bottleneck(x)
        out = self.conv(out)
        return torch.cat([x, out], dim=1)

class DenseBlock_BC(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate):
        super().__init__()

        layers = []
        for i in range(num_layers):
            layers.append(DenseLayer_BC(in_channels + i * growth_rate, growth_rate))

        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)

class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.transition = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2), # Avg Pooling 먼저 하는 것이 연산량 1/4로 줄어듦 (선형 연산이라 교환 가능)
            nn.Conv2d(in_channels, out_channels, kernel_size=1) 
        )
    
    def forward(self, x):
        return self.transition(x)

class DenseNet(nn.Module):
    def __init__(self, num_layers, num_classes, growth_rate=24, reduction=0.5):
        super().__init__()
        num_stage_layers = (num_layers - 4) // 6
        in_channels = 2 * growth_rate
        
        # Blcok 1
        self.conv1 = nn.Conv2d(3, in_channels, kernel_size=3, padding=1, bias=False)
        self.stage1 = DenseBlock_BC(num_stage_layers, in_channels, growth_rate)
        
        in_channels += num_stage_layers * growth_rate
        out_channels = int(in_channels * reduction)
        self.trans1 = TransitionLayer(in_channels, out_channels)
        in_channels = out_channels

        # Block 2
        self.stage2 = DenseBlock_BC(num_stage_layers, in_channels, growth_rate)
        
        in_channels += num_stage_layers * growth_rate
        out_channels = int(in_channels * reduction)
        self.trans2 = TransitionLayer(in_channels, out_channels)
        in_channels = out_channels

        # Block 3
        self.stage3 = DenseBlock_BC(num_stage_layers, in_channels, growth_rate)
        in_channels += num_stage_layers * growth_rate

        # Classifier
        self.final_bn = nn.BatchNorm2d(in_channels)
        self.final_relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_channels, num_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.stage1(x)
        x = self.trans1(x)
        x = self.stage2(x)
        x = self.trans2(x)
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