import torch
from torch import nn
import torch.nn.functional as f

class MLP(nn.Module):
    def __init__(self, size, hidden_size=4096):#히든 사이즈? 
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, size)
        )

    def forward(self, x):
        return self.mlp(x)

class simCLR(nn.Module):
    def __init__(self, encoder, t=0.1):
        super().__init__()
        self.encoder = encoder
        self.proj = MLP(encoder.num_features)
        self.temperature = t

    def forward(self, x1, x2):
        h1 = self.encoder(x1)
        h2 = self.encoder(x2)

        return h1, h2

    def get_loss(self, h1, h2):
        z1 = self.proj(h1)
        z2 = self.proj(h2)
        batch_size = z1.size(0)
        loss = 0
        
        def ntx_ent_loss(z1, z2, t, batch_size, pos_idx):
            def sim(x, y):
                x_size = x.norm()
                y_size = y.norm()
                return (x@y)/(x_size*y_size*t)
            
            pos_sim = 0
            neg_sim = 0
            for j in range(batch_size):
                if pos_idx == j:
                    pos_sim += torch.exp(sim(z1[pos_idx], z2[j]))
                else:
                    neg_sim += torch.exp(sim(z1[pos_idx], z2[j]))
            loss = -torch.log(pos_sim/neg_sim)
        
            return loss
        
        for i in range(batch_size):
            loss += ntx_ent_loss(z1, z2, self.temperature, batch_size, i)
        loss /= (2*batch_size)

        return loss

    def training_step(self, batch, device):
        (x1, x2), _ = batch #x1,x2는 배치 사이즈 만큼의 사진 데이터
        x1, x2 = x1.to(device), x2.to(device)
        h1, h2 = self.forward(x1, x2)
        loss = self.get_loss(h1, h2)
        return loss

    def validation_step(self, batch, device):
        (x1, x2), _ = batch
        x1, x2 = x1.to(device), x2.to(device)
        h1, h2 = self.forward(x1, x2)
        loss = self.get_loss(h1, h2)
        batch_size = h1.size(0)
        return loss.item(), 0, batch_size