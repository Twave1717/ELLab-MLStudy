import copy
import torch
from torch import nn

class MoCo(nn.Module):

    def __init__(self, encoder, dim=128, K=4096, m=0.999, T=0.07):
        super().__init__()
        self.K = K
        self.m = m
        self.T = T
        self.encoder_q = encoder
        self.encoder_k = copy.deepcopy(encoder)

        self.projector_q = nn.Linear(encoder.num_features, dim)
        self.projector_k = copy.deepcopy(self.projector_q)

        self.freeze_key()

        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def freeze_key(self):
        for param in self.encoder_k.parameters():
            param.requires_grad = False
        for param in self.projector_k.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update_key(self):
        m=self.m
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = m*param_k.data + (1-m)*param_q.data
        for param_q, param_k in zip(self.projector_q.parameters(), self.projector_k.parameters()):
            param_k.data = m*param_k.data + (1-m)*param_q.data
            
    @torch.no_grad()
    def update_queue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        self.queue[:, ptr:ptr+batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr
    
    def forward(self, im_q, im_k):
        q = self.encoder_q(im_q)
        q = self.projector_q(q)
        q = nn.functional.normalize(q, dim=1)
        
        with torch.no_grad():
            self.update_key()
            k = self.encoder_k(im_k)
            k = self.projector_k(k)
            k = nn.functional.normalize(k, dim=1) 

        l_pos = (q * k).sum(dim=1, keepdim=True)
        l_neg = q @ self.queue.clone().detach()
        logits = torch.cat([l_pos, l_neg], dim=1)/self.T
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        
        self.update_queue(k)
        return logits, labels
        
        
    def training_step(self, batch, device):
        (im_q, im_k), _ = batch
        im_q = im_q.to(device)
        im_k = im_k.to(device)

        logits, labels = self.forward(im_q, im_k)
        return nn.functional.cross_entropy(logits, labels)

    @torch.no_grad()
    def validation_step(self, batch, device):
        (im_q, im_k), _ = batch
        im_q = im_q.to(device)
        im_k = im_k.to(device)

        logits, labels = self.forward(im_q, im_k)
        loss = nn.functional.cross_entropy(logits, labels)
        batch_size = im_q.shape[0]

        return loss.item(), 0, batch_size
