import torch.nn.functional as F
from torch import nn


class CLIP(nn.Module):
    def __init__(self, clip_model, tokenizer, classnames):
        super().__init__()

        classnames = [names[0].replace("_", " ") for names in classnames]
        tokens = tokenizer(
            classnames,
            padding=True,
            return_tensors="pt",
        )
        self.model = clip_model
        # freeze text encoder & projection layer
        self.model.text_model.requires_grad_(False)
        self.model.text_projection.requires_grad_(False)
        self.register_buffer("input_ids", tokens["input_ids"], persistent=False)
        self.register_buffer("attention_mask", tokens["attention_mask"], persistent=False)

    @property
    def encoder(self):
        return self.model.vision_model

    def forward(self, images):
        return self.model(
            pixel_values=images,
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
        ).logits_per_image

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
