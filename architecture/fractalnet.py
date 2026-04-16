import torch
from torch import nn

def conv3x3(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)


class FractalBlockDropPath(nn.Module):
    def __init__(self, in_channels, out_channels, columns, drop_prob=0.15):
        super().__init__()
        self.columns = columns
        self.drop_prob = drop_prob
        
        if columns == 1:
            self.f = nn.Sequential(
                nn.BatchNorm2d(in_channels),        # PreAct 구조 선택
                nn.ReLU(inplace=True),
                conv3x3(in_channels, out_channels),
                # nn.Dropout2d(0.15)
            )
        else:
            self.long1 = FractalBlockDropPath(in_channels, out_channels, columns - 1, drop_prob)   # 재귀 구조
            self.long2 = FractalBlockDropPath(out_channels, out_channels, columns - 1, drop_prob)
            self.short = nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                conv3x3(in_channels, out_channels),
                # nn.Dropout2d(0.15)
            )

    def _fractal_join(self, out_long, out_short):
        if not self.training or self.drop_prob == 0:
            return (out_long + out_short) * 0.5     # simple avg (inferencing)

        keep_prob = 1.0 - self.drop_prob
        mask = torch.bernoulli(out_long.new_tensor([keep_prob, keep_prob]))  # mini-batch shared mask
        if mask.sum() == 0:     # 모든 path가 drop된 경우 long path(idx=0) 살림
            mask[0] = 1.0

        long_mask = mask[0]
        short_mask = mask[1]

        out = out_long * long_mask + out_short * short_mask     # drop (broad casting) [B, C, H, W]
        alive = long_mask + short_mask                          
        return out / alive                                      # broad casting [B, C, H, W]

    def forward(self, x, route_token):
        if self.columns == 1:
            return self.f(x)

        # route_token in [0, root_columns - 1] -> global column 선택
        # route_token == root_columns -> local drop-path

        if route_token < self.columns:                  # global drop path
            if route_token == self.columns - 1:
                return self.short(x)

            out_long = self.long1(x, route_token)
            out_long = self.long2(out_long, route_token)
            return out_long

        out_long = self.long1(x, route_token)
        out_long = self.long2(out_long, route_token)
        out_short = self.short(x)
        return self._fractal_join(out_long, out_short)  # short drop path
        

class FractalNetDropPath(nn.Module):
    def __init__(self, num_layers, num_classes, blocks=5, columns=4, drop_prob=0.15, global_prob=0.5):
        super().__init__()
        self.drop_prob = drop_prob
        self.global_prob = global_prob
        if num_layers == 20:
            self.columns = 3
        elif num_layers == 40:
            self.columns = 4
        else:
            self.columns = columns

        self.block1 = FractalBlockDropPath(3, 64, self.columns)
        self.pool1 = nn.MaxPool2d(2)
        self.block2 = FractalBlockDropPath(64, 128, self.columns)
        self.pool2 = nn.MaxPool2d(2)
        self.block3 = FractalBlockDropPath(128, 256, self.columns)
        self.pool3 = nn.MaxPool2d(2)
        self.block4 = FractalBlockDropPath(256, 512, self.columns)
        self.pool4 = nn.MaxPool2d(2)
        self.block5 = FractalBlockDropPath(512, 512, self.columns)
        self.pool5 = nn.MaxPool2d(2)
        
        # 32 -> 16 -> 8 -> 4 -> 2 -> 1 로 feature_map 줄어들어서 별도 pooling 필요 없음
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, num_classes)

        self._initialize_weights()

    def _sample_route_token(self, device):
        if torch.rand((), device=device).item() < self.global_prob:      # local vs global
            return torch.randint(0, self.columns, (), device=device).item()     # global drop column choice
        return self.columns

    
    def forward(self, x):
        route_token = self.columns
        if self.training and self.drop_prob > 0:
            route_token = self._sample_route_token(x.device)

        x = self.block1(x, route_token)
        x = self.pool1(x)
        x = self.block2(x, route_token)
        x = self.pool2(x)
        x = self.block3(x, route_token)
        x = self.pool3(x)
        x = self.block4(x, route_token)
        x = self.pool4(x)
        x = self.block5(x, route_token)
        x = self.pool5(x)
        x = self.flatten(x)
        out = self.fc(x)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)




######## Simple FractalNet
class FractalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, columns):
        super().__init__()
        self.columns = columns
        
        if columns == 1:
            self.f = nn.Sequential(
                nn.BatchNorm2d(in_channels),        # PreAct 구조 선택
                nn.ReLU(inplace=True),
                conv3x3(in_channels, out_channels)
            )
        else:
            self.long1 = FractalBlock(in_channels, out_channels, columns - 1)   # 재귀 구조
            self.long2 = FractalBlock(out_channels, out_channels, columns - 1)
            self.short = nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                conv3x3(in_channels, out_channels)
            )

    def forward(self, x):
        if self.columns == 1:
            return self.f(x)
        
        out_long = self.long1(x)
        out_long = self.long2(out_long)
        out_short = self.short(x)
        return (out_long + out_short) * 0.5     # simple avg
        

class FractalNet(nn.Module):
    def __init__(self, num_layers, num_classes, blocks=5, columns=4):
        super().__init__()
        if num_layers == 20:
            self.columns = 3
        elif num_layers == 40:
            self.columns = 4
        else:
            self.columns = columns

        self.block1 = FractalBlock(3, 64, self.columns)
        self.pool1 = nn.MaxPool2d(2)
        self.block2 = FractalBlock(64, 128, self.columns)
        self.pool2 = nn.MaxPool2d(2)
        self.block3 = FractalBlock(128, 256, self.columns)
        self.pool3 = nn.MaxPool2d(2)
        self.block4 = FractalBlock(256, 512, self.columns)
        self.pool4 = nn.MaxPool2d(2)
        self.block5 = FractalBlock(512, 512, self.columns)
        self.pool5 = nn.MaxPool2d(2)
        
        # 32 -> 16 -> 8 -> 4 -> 2 -> 1 로 feature_map 줄어들어서 별도 pooling 필요 없음
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, num_classes)

        self._initialize_weights()

    
    def forward(self, x):
        x = self.block1(x)
        x = self.pool1(x)
        x = self.block2(x)
        x = self.pool2(x)
        x = self.block3(x)
        x = self.pool3(x)
        x = self.block4(x)
        x = self.pool4(x)
        x = self.block5(x)
        x = self.pool5(x)
        x = self.flatten(x)
        out = self.fc(x)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
