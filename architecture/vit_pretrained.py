import torch.nn.functional as F
from torch import nn
from torchvision.models import ViT_B_16_Weights, vit_b_16


class VisionTransformer(nn.Module):
    def __init__(self, num_layers, num_classes):
        super().__init__()
        self.image_size = 224       # 논문 내용
        self.model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)

        in_features = self.model.heads.head.in_features
        self.model.heads.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = F.interpolate(
            x,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        out = self.model(x)
        return out
