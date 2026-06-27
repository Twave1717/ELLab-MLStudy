import copy

import torch
from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_size, hidden_size=4096, out_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, out_size)
        )
    
    def forward(self, x):
        return self.mlp(x)


class BYOL(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.online_encoder = encoder
        self.target_encoder = copy.deepcopy(encoder)
        self.online_projector = MLP(encoder.num_features)
        self.target_projector = copy.deepcopy(self.online_projector)
        self.predictor = MLP(256)

        self.freeze()

    def forward(self, x1, x2):
        x1 = self.online_encoder(x1)
        x1 = self.online_projector(x1)
        online = self.predictor(x1)

        with torch.no_grad():
            x2 = self.target_encoder(x2)
            target = self.target_projector(x2)

        return online, target
    
    def get_loss_(self, x1, x2):
        online1, target2 = self.forward(x1, x2)
        online2, target1 = self.forward(x2, x1)
        online1 = F.normalize(online1, dim=1)
        online2 = F.normalize(online2, dim=1)
        target1 = F.normalize(target1, dim=1)
        target2 = F.normalize(target2, dim=1)
        loss =  2 - 2 * (online1 * target2).sum(dim=1).mean()
        loss +=  2 - 2 * (online2 * target1).sum(dim=1).mean()
        return loss

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