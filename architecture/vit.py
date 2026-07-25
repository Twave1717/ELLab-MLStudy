import torch.nn.functional as F
from torch import nn
from transformers import ViTConfig, ViTModel

PRETRAINED_VIT = "google/vit-base-patch16-224-in21k"


class VisionTransformer(nn.Module):
    def __init__(self, num_layers, pretrained=False):
        super().__init__()
        if pretrained:
            if num_layers != 12:
                raise ValueError("Pretrained ViT requires vit-12")
            model = ViTModel.from_pretrained(PRETRAINED_VIT)
        else:
            config = ViTConfig(num_hidden_layers=num_layers)
            model = ViTModel(config)

        self.model = model
        self.image_size = model.config.image_size
        self.num_features = model.config.hidden_size

    def forward(self, images):
        if images.shape[-2:] != (self.image_size, self.image_size):
            images = F.interpolate(
                images,
                size=(self.image_size, self.image_size),
                mode="bicubic",
            )
        return self.model(pixel_values=images).pooler_output
