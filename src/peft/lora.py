import math

from torch import nn


RANK = 2
ALPHA = 1
DROPOUT = 0.25
TARGETS = ("q", "k", "v", "o", "fc1", "fc2")

_ATTENTION_TARGETS = {
    "q": "q_proj",
    "k": "k_proj",
    "v": "v_proj",
    "o": "out_proj",
}


class LoRALinear(nn.Module):
    def __init__(self, base, rank=RANK):
        super().__init__()
        self.base = base
        self.scaling = ALPHA / math.sqrt(rank)
        self.dropout = nn.Dropout(DROPOUT)
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        self.to(base.weight)
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, inputs):
        return self.base(inputs) + self.scaling * self.lora_b(
            self.lora_a(self.dropout(inputs))
        )


def apply_lora(encoder, targets=("q", "k", "v"), blocks="all", rank=RANK):
    layers = encoder.encoder.layers
    indices = {
        "all": range(len(layers)),
        "odd": range(1, len(layers), 2),
        "even": range(0, len(layers), 2),
    }[blocks]
    replaced = 0

    for index in indices:
        layer = layers[index]
        for target in targets:
            parent = layer.self_attn if target in _ATTENTION_TARGETS else layer.mlp
            name = _ATTENTION_TARGETS.get(target, target)
            setattr(parent, name, LoRALinear(getattr(parent, name), rank))
            replaced += 1

    return replaced


def apply_lora_to_clip(
    model,
    targets=("q", "k", "v"),
    blocks="all",
    modality="both",
    rank=RANK,
):
    encoders = []
    if modality in {"both", "vision"}:
        encoders.append(model.vision_model)
    if modality in {"both", "text"}:
        encoders.append(model.text_model)
    return sum(apply_lora(encoder, targets, blocks, rank) for encoder in encoders)


def mark_only_lora_as_trainable(model):
    model.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.lora_a.requires_grad_(True)
            module.lora_b.requires_grad_(True)
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def lora_modules(model):
    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, LoRALinear)
    }


def lora_state_dict(model):
    return {
        name: value
        for name, value in model.state_dict().items()
        if "lora_a." in name or "lora_b." in name
    }
