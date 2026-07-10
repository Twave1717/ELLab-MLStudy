import torch
from torch import nn
import torch.nn.functional as F


def rot_batch(images):
    r0 = images
    r1 = torch.rot90(images, k=1, dims=(2, 3))
    r2 = torch.rot90(images, k=2, dims=(2, 3))
    r3 = torch.rot90(images, k=3, dims=(2, 3))

    rot_images = torch.cat([r0, r1, r2, r3], dim=0)

    batch_size = images.size(0)
    rot_labels = torch.cat([
        torch.zeros(batch_size, dtype=torch.long),
        torch.ones(batch_size, dtype=torch.long),
        torch.full((batch_size,), 2, dtype=torch.long),
        torch.full((batch_size,), 3, dtype=torch.long),
    ], dim=0).to(images.device)

    return rot_images, rot_labels


class RotNetMethod(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(encoder.num_features, 4)

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x

    def training_step(self, batch, device):
        images, _ = batch
        images = images.to(device)

        rot_images, rot_labels = rot_batch(images)
        outputs = self(rot_images)

        loss = F.cross_entropy(outputs, rot_labels)
        return loss

    def validation_step(self, batch, device):
        images, _ = batch
        images = images.to(device)

        rot_images, rot_labels = rot_batch(images)
        outputs = self(rot_images)

        loss = F.cross_entropy(outputs, rot_labels)
        pred = outputs.argmax(dim=1)
        correct = (pred == rot_labels).sum().item()
        batch_size = rot_labels.size(0)

        return loss.item(), correct, batch_size