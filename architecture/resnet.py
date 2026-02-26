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
    def __init__(self, n):
        super().__init__()
        self.model = self._build_model(n)
        self._initialize_weights()

    def _build_model(self, n):
        n = int((n-2)/6)
        print(n)
        base = 16
        model = nn.Sequential(
            nn.Conv2d(3, base, 3, padding=1, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(),
            *[ResBlock3x3(base, base) for _ in range(n)],
            ResBlock3x3(base, base*2, stride=2),
            *[ResBlock3x3(base*2, base*2) for _ in range(n-1)],
            ResBlock3x3(base*2, base*4, stride=2),
            *[ResBlock3x3(base*4, base*4) for _ in range(n-1)],
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(base*4, 10)
        )
        return model

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        logits = self.model(x)
        return logits