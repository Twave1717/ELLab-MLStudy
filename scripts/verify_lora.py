"""Verify configurable LoRA and LoRA-Pro."""

import math
import sys
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.peft import (  # noqa: E402
    LoRALinear,
    LoRAProOptimizer,
    apply_lora_to_clip,
    lora_modules,
    mark_only_lora_as_trainable,
)


class Block(nn.Module):
    def __init__(self, width=8):
        super().__init__()
        self.self_attn = nn.Module()
        for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
            setattr(self.self_attn, name, nn.Linear(width, width, bias=False))
        self.mlp = nn.Module()
        self.mlp.fc1 = nn.Linear(width, width, bias=False)
        self.mlp.fc2 = nn.Linear(width, width, bias=False)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList(Block() for _ in range(4))


class Clip(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_model = Encoder()
        self.text_model = Encoder()


def check(condition, message):
    if not condition:
        raise AssertionError(message)
    print(f"PASS  {message}")


model = Clip()
replaced = apply_lora_to_clip(model)
parameters = mark_only_lora_as_trainable(model)
check(replaced == 24, "default QKV on both encoders")
check(len(lora_modules(model)) == 24, "default adapter count")
check(len(parameters) == 48, "only LoRA parameters trainable")
check(all(module.lora_a.out_features == 2 for module in lora_modules(model).values()), "default rank 2")

model = Clip()
replaced = apply_lora_to_clip(
    model, targets=("fc1",), blocks="odd", modality="vision", rank=1
)
check(replaced == 2, "FC1 on odd vision blocks")
check(len(lora_modules(model.text_model)) == 0, "text encoder unchanged")
check(all(module.lora_a.out_features == 1 for module in lora_modules(model).values()), "custom rank 1")

torch.manual_seed(0)
adapter = LoRALinear(nn.Linear(8, 8, bias=False), rank=1)
optimizer = LoRAProOptimizer({"adapter": adapter}, lr=1e-3)
inputs = torch.randn(4, 8)
targets = torch.randn(4, 8)
for _ in range(3):
    optimizer.zero_grad()
    loss = ((adapter(inputs) - targets) ** 2).mean()
    loss.backward()
    optimizer.step()
check(math.isfinite(loss.item()), "LoRA-Pro finite step")
check(adapter.lora_b.weight.abs().sum().item() > 0, "LoRA-Pro updates B")

with patch.object(
    sys,
    "argv",
    [
        "train_2sfs.py",
        "--peft", "lora",
        "--lora_targets", "q", "k", "v",
        "--lora_blocks", "odd",
        "--lora_modality", "vision",
        "--lora_rank", "1",
        "--stage1_optimizer", "lora_pro",
    ],
):
    import train_2sfs

    args = train_2sfs.parse_args()

check(args.lora_blocks == "odd", "CLI block selector")
check(args.lora_modality == "vision", "CLI modality selector")
check(args.lora_rank == 1, "CLI rank selector")
check(args.stage1_optimizer == "lora_pro", "CLI LoRA-Pro selector")

with patch.object(sys, "argv", ["train_2sfs.py"]):
    defaults = train_2sfs.parse_args()

check(defaults.peft == "ln", "existing PEFT default")
check(defaults.lora_rank == 2, "existing LoRA rank default")

adapter = LoRALinear(nn.Linear(8, 2, bias=False), rank=1)
parameters = mark_only_lora_as_trainable(adapter)
gate = train_2sfs.AbsIdentityGate(parameters, online_images=2)
size = sum(parameter.numel() for parameter in parameters)
gate.first_moment = torch.ones(size)
gate.second_square = torch.ones(size)
optimizer = LoRAProOptimizer({"adapter": adapter}, lr=1e-3)
writer = type("Writer", (), {"add_scalar": lambda *args: None})()
train_2sfs.train_stage(
    adapter,
    parameters,
    [(torch.randn(4, 8), torch.tensor([0, 1, 0, 1]))],
    1,
    1e-3,
    "cpu",
    "test",
    writer,
    gradient_gate=gate,
    optimizer=optimizer,
)
check(adapter.lora_b.weight.abs().sum().item() > 0, "LoRA-Pro with Dynamic Gate")

print("16/16 passed")
