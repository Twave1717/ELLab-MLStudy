import torch
from torch import nn
import torch.nn.functional as F


class NonLinearClassifier(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()

        self.fc1 = nn.Linear(in_dim, 200)
        self.bn1 = nn.BatchNorm1d(200)

        self.fc2 = nn.Linear(200, 200)
        self.bn2 = nn.BatchNorm1d(200)

        self.fc3 = nn.Linear(200, num_classes)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x


class RotNetNonLinearEval(nn.Module):
    def __init__(self, encoder, num_classes):
        super().__init__()

        self.encoder = encoder
        self.block_idx = 2
        self.encoder.requires_grad_(False)

        in_dim = 192 * 8 * 8
        self.classifier = NonLinearClassifier(in_dim, num_classes)

    def forward(self, x):
        with torch.no_grad():
            feat = self.encoder.forward_n(x, self.block_idx)
        return self.classifier(feat)

    def training_step(self, batch, device):
        x, y = batch
        x, y = x.to(device), y.to(device)

        output = self(x)
        loss = F.cross_entropy(output, y)
        return loss

    def validation_step(self, batch, device):
        x, y = batch
        x, y = x.to(device), y.to(device)

        output = self(x)
        loss = F.cross_entropy(output, y)

        pred = output.argmax(dim=1)
        correct = (pred == y).sum().item()

        return loss.item(), correct, x.size(0)