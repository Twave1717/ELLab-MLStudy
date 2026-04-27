import torch.nn.functional as F
from torch import nn
from transformers import ViTForImageClassification


class VisionTransformer(nn.Module):
    def __init__(self, num_layers, num_classes):
        super().__init__()
        self.image_size = 224
        self.model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224-in21k",
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )

    def forward(self, x):
        x = F.interpolate(
            x,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        out = self.model(pixel_values=x).logits
        return out
