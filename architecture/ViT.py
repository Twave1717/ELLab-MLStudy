import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class Preprocess(nn.Module):
    def __init__(self, img_size = 224, patch_size = 16, in_channels = 3, dim = 768):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.patch_dim = in_channels * patch_size * patch_size

        self.proj = nn.Linear(self.patch_dim, dim)

    def split_patches(self, x):
        B, C, H, W = x.shape
        P = self.patch_size

        x = F.unfold(x, kernel_size= P, stride= P)
        x = x.transpose(1, 2)

        return x
    
    def forward(self, x):
        x = self.split_patches(x)
        x = self.proj(x)
        return x

class ViT(nn.Module):
    def __init__(self, img_size = 224, patch_size = 16, in_channels = 3, num_classes = 10, dim = 768, depth = 12, num_heads = 12, mlp_ratio = 4.0):
        super().__init__()
        self.patch_embed = Preprocess(img_size, patch_size, in_channels, dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1,1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches +1, dim))

        layer = nn.TransformerEncoderLayer(
            d_model= dim,
            nhead= num_heads,
            dim_feedforward= int(dim * mlp_ratio),
            dropout=0.0,
            activation="gelu",
            layer_norm_eps=1e-6,
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(layer, num_layers=depth, norm= nn.LayerNorm(dim, eps=1e-6)) #제가 다시 확인해보니 LazyNorm이 아니라 LayerNorm이 맞는 것 같습니다
        self.head = nn.Linear(dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std = 0.02)
        nn.init.trunc_normal_(self.cls_token, std = 0.02)

    def forward(self, x):
        x = self.patch_embed(x)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim = 1)

        x = x + self.pos_embed
        x = self.encoder(x)

        return self.head(x[:, 0])
    
def vit_base_patch16(num_classes = 10, img_size = 224):
    return ViT(img_size=img_size, patch_size=16, num_classes=num_classes, dim=768, depth=12, num_heads=12)

@torch.no_grad()
def load_timm_pretrained(model, timm_name = "vit_base_patch16_224.augreg_in21k", num_classes = 10):
    src = timm.create_model(timm_name, pretrained=True, num_classes = num_classes)
    sd = src.state_dict()
    new_sd = {}

    new_sd["patch_embed.proj.weight"] = sd["patch_embed.proj.weight"].flatten(1)
    new_sd["patch_embed.proj.bias"] = sd["patch_embed.proj.bias"]
 
    new_sd["cls_token"] = sd["cls_token"]
    new_sd["pos_embed"] = sd["pos_embed"]
    new_sd["encoder.norm.weight"] = sd["norm.weight"]
    new_sd["encoder.norm.bias"] = sd["norm.bias"]
    new_sd["head.weight"] = sd["head.weight"]
    new_sd["head.bias"] = sd["head.bias"]
 
    depth = len(model.encoder.layers)
    for i in range(depth):
        t = f"blocks.{i}"
        p = f"encoder.layers.{i}"

        new_sd[f"{p}.self_attn.in_proj_weight"] = sd[f"{t}.attn.qkv.weight"]
        new_sd[f"{p}.self_attn.in_proj_bias"] = sd[f"{t}.attn.qkv.bias"]
        new_sd[f"{p}.self_attn.out_proj.weight"] = sd[f"{t}.attn.proj.weight"]
        new_sd[f"{p}.self_attn.out_proj.bias"] = sd[f"{t}.attn.proj.bias"]
        # MLP
        new_sd[f"{p}.linear1.weight"] = sd[f"{t}.mlp.fc1.weight"]
        new_sd[f"{p}.linear1.bias"] = sd[f"{t}.mlp.fc1.bias"]
        new_sd[f"{p}.linear2.weight"] = sd[f"{t}.mlp.fc2.weight"]
        new_sd[f"{p}.linear2.bias"] = sd[f"{t}.mlp.fc2.bias"]
        # LayerNorm
        new_sd[f"{p}.norm1.weight"] = sd[f"{t}.norm1.weight"]
        new_sd[f"{p}.norm1.bias"] = sd[f"{t}.norm1.bias"]
        new_sd[f"{p}.norm2.weight"] = sd[f"{t}.norm2.weight"]
        new_sd[f"{p}.norm2.bias"] = sd[f"{t}.norm2.bias"]
 
    model.load_state_dict(new_sd, strict=True)
    print("timm pretrained loaded (strict=True OK)")
    return model