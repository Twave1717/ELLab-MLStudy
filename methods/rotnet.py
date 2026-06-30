import copy

import torch
from torch import nn
import torch.nn.functional as F


class BYOL(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder


    def forward(self, x1, x2):
        x1 = self.online_encoder(x1)
        x1 = self.online_projector(x1)
        online = self.predictor(x1)

        with torch.no_grad():
            x2 = self.target_encoder(x2)
            target = self.target_projector(x2)

        return online, target

    def training_step(self, batch, device):
        (x1, x2), _ = batch
        x1, x2 = x1.to(device), x2.to(device)

        ## train online network
        loss = self.get_loss_(x1, x2)
        return loss
    
    def after_optimizer_step(self):
        ## update target network with EMA
        with torch.no_grad():
            for o, t in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
                t.mul_(0.99).add_(o, alpha=0.01)
            for o, t in zip(self.online_projector.parameters(), self.target_projector.parameters()):
                t.mul_(0.99).add_(o, alpha=0.01)

    def validation_step(self, batch, device):
        (x1, x2), _ = batch
        x1, x2 = x1.to(device), x2.to(device)
        loss = self.get_loss_(x1, x2)
        batch_size = x1.shape[0]
        return loss.item(), 0, batch_size

    def freeze(self):
        self.target_encoder.requires_grad_(False)
        self.target_projector.requires_grad_(False)