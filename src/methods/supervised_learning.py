from torch import nn
import torch.nn.functional as F

class SupervisedLearning(nn.Module):
    def __init__(self, encoder, num_classes):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(encoder.num_features, num_classes)

    def forward(self, x):
        features = self.encoder(x)
        logits = self.classifier(features)
        return logits
    
    def training_step(self, batch, device):
        x, y = batch
        x, y = x.to(device), y.to(device)
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)
        return loss

    def validation_step(self, batch, device):
        x, y = batch
        x, y = x.to(device), y.to(device)
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)
        correct = (logits.argmax(1) == y).float().sum().item()
        batch_size = y.shape[0]
        return loss.item(), correct, batch_size

        