import torch
import torch.nn.functional as F
from torch import nn

## 현재 image feture만 학습하도록 설정해놓음
class CLIP(nn.Module):
    def __init__(self, clip_model, tokenizer, classnames, template):
        super().__init__()

        classnames = [names[0].replace("_", " ") for names in classnames]
        tokens = tokenizer(
            [template.format(name) for name in classnames],
            padding=True,
            return_tensors="pt",
        )
        self.model = clip_model
        # freeze text encoder & projection layer
        self.model.text_model.requires_grad_(False)
        self.model.text_projection.requires_grad_(False)
        with torch.no_grad():
            text_outputs = self.model.text_model(
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"],
            )
            text_features = self.model.text_projection(text_outputs.pooler_output)
            text_features = F.normalize(text_features, dim=-1)
        self.register_buffer("text_features", text_features, persistent=False)

    @property
    def encoder(self):
        return self.model.vision_model

    def forward(self, images):
        vision_outputs = self.model.vision_model(pixel_values=images)
        image_features = self.model.visual_projection(vision_outputs.pooler_output)
        image_features = F.normalize(image_features, dim=-1)
        return self.model.logit_scale.exp() * image_features @ self.text_features.t()

    def training_step(self, batch, device):
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)
        return F.cross_entropy(self(images), labels)

    def validation_step(self, batch, device):
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)
        logits = self(images)
        loss = F.cross_entropy(logits, labels)
        correct = (logits.argmax(dim=1) == labels).sum().item()
        return loss.item(), correct, labels.size(0)
