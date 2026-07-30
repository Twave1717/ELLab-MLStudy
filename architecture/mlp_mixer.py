import torch
import torch.nn.functional as F
from torch import nn


class MLPBlock(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.layers(x)


class MixerBlock(nn.Module):
    def __init__(self, num_patches, hidden_dim, tokens_mlp_dim, channels_mlp_dim, dropout=0.0):
        super().__init__()
        self.token_norm = nn.LayerNorm(hidden_dim)
        self.token_mlp = MLPBlock(num_patches, tokens_mlp_dim, dropout)
        self.channel_norm = nn.LayerNorm(hidden_dim)
        self.channel_mlp = MLPBlock(hidden_dim, channels_mlp_dim, dropout)

    def forward(self, x):
        y = self.token_norm(x)
        y = y.transpose(1, 2)   # [B, S, C] -> [B, C, S]
        y = self.token_mlp(y)   # [B, C, S] * [S, Ds] * GeLU * [Ds, S] = [B, C, S]
        y = y.transpose(1, 2)   # [B, C, S] -> [B, S, C]
        x = x + y

        y = self.channel_norm(x)    
        y = x + self.channel_mlp(y) # [B, S, C] * [C, Dc] * GeLU * [Dc, C] = [B, S, C]
        return y


class MLPMixer(nn.Module):
    def __init__(
        self,
        num_layers=12,
        image_size=224,
        patch_size=16,
        hidden_dim=768,
        tokens_mlp_dim=384,
        channels_mlp_dim=3072,
        dropout=0.0,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.tokens_mlp_dim = tokens_mlp_dim if image_size==224 else tokens_mlp_dim*4
        self.channels_mlp_dim = channels_mlp_dim

        num_patches = (image_size // patch_size) ** 2
        self.num_patches = num_patches
        self.stem = nn.Conv2d(3, hidden_dim, kernel_size=patch_size, stride=patch_size)         # conv로 linear projection 구현
        self.blocks = nn.Sequential(
            *[
                MixerBlock(num_patches, hidden_dim, self.tokens_mlp_dim, channels_mlp_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)

        self._initialize_weights()
        self.num_features = hidden_dim

    def forward(self, x):
        if x.shape[-2:] != (self.image_size, self.image_size):
            x = F.interpolate(x, size=(self.image_size, self.image_size))

        x = self.stem(x)                       # [B, 3, H, H] -> [B, C, H/P, H/P]
        x = x.flatten(2).transpose(1, 2)       # [B, C, sqrt(S), sqrt(S)] -> [B, S, C]
        x = self.blocks(x)
        x = self.norm(x)
        x = x.mean(dim=1)                      # [B, S, C] -> [B, C]
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
