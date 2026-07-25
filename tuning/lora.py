import math

from torch import nn


RANK = 2
ALPHA = 1
DROPOUT = 0.25


class LoRALinear(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.scaling = ALPHA / math.sqrt(RANK)
        self.dropout = nn.Dropout(DROPOUT)
        self.lora_a = nn.Linear(
            base.in_features,
            RANK,
            bias=False,
        )
        self.lora_b = nn.Linear(
            RANK,
            base.out_features,
            bias=False,
        )
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, inputs):
        return self.base(inputs) + self.scaling * self.lora_b(
            self.lora_a(self.dropout(inputs))
        )


def apply_lora(vision_model):
    replaced = 0

    for layer in vision_model.encoder.layers:
        attention = layer.self_attn
        for name in ("q_proj", "k_proj", "v_proj"):
            projection = getattr(attention, name)
            setattr(
                attention,
                name,
                LoRALinear(projection),
            )
            replaced += 1

    return replaced


def mark_only_lora_as_trainable(model):
    model.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.lora_a.requires_grad_(True)
            module.lora_b.requires_grad_(True)


def lora_state_dict(model):
    return {
        name: value
        for name, value in model.state_dict().items()
        if "lora_a." in name or "lora_b." in name
    }
