import math

from torch import nn


RANK = 2
ALPHA = 1
DROPOUT = 0.25

TARGET_CHOICES = ("q", "k", "v", "o", "fc1", "fc2")
DEFAULT_TARGETS = ("q", "k", "v")

# OpenAI CLIP 계열 표기 -> Hugging Face CLIP 이름
TARGET_ALIASES = {
    "q_proj": "q",
    "k_proj": "k",
    "v_proj": "v",
    "out_proj": "o",
    "c_fc": "fc1",
    "c_proj": "fc2",
}

_ATTENTION_ATTRIBUTES = {
    "q": "q_proj",
    "k": "k_proj",
    "v": "v_proj",
    "o": "out_proj",
}


class LoRALinear(nn.Module):
    def __init__(self, base, rank=RANK, alpha=ALPHA, dropout=DROPOUT):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive: {rank}")
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / math.sqrt(rank)
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        self.to(base.weight)
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, inputs):
        return self.base(inputs) + self.scaling * self.lora_b(
            self.lora_a(self.dropout(inputs))
        )


def normalize_targets(targets):
    normalized = []
    for target in targets:
        name = TARGET_ALIASES.get(target, target)
        if name not in TARGET_CHOICES:
            raise ValueError(
                f"Unknown LoRA target: {target} (choose from {', '.join(TARGET_CHOICES)})"
            )
        if name not in normalized:
            normalized.append(name)
    if not normalized:
        raise ValueError("At least one LoRA target is required")
    return tuple(normalized)


def parse_blocks(blocks, num_blocks):
    """`all`, `odd`, `even` 또는 쉼표로 구분한 0-based index."""
    if blocks is None or blocks == "all":
        return tuple(range(num_blocks))
    if blocks == "odd":
        return tuple(index for index in range(num_blocks) if index % 2)
    if blocks == "even":
        return tuple(index for index in range(num_blocks) if not index % 2)

    indices = []
    for part in str(blocks).replace(" ", "").split(","):
        if not part:
            continue
        index = int(part)
        if not 0 <= index < num_blocks:
            raise ValueError(f"Block index out of range: {index} (0-{num_blocks - 1})")
        if index not in indices:
            indices.append(index)
    if not indices:
        raise ValueError(f"No block selected: {blocks}")
    return tuple(sorted(indices))


def apply_lora(
    encoder,
    targets=DEFAULT_TARGETS,
    rank=RANK,
    alpha=ALPHA,
    dropout=DROPOUT,
    blocks=None,
    rank_map=None
):
    """선택한 block의 projection을 `LoRALinear`로 교체한다.

    `rank_map`은 `(block_index, target)` -> rank인 선택적 dict이다. 없는 key는
    `rank`를 쓰고, rank가 0 이하면 그 module은 건너뛴다.
    """
    targets = normalize_targets(targets)
    layers = encoder.encoder.layers
    replaced = 0

    for index in parse_blocks(blocks, len(layers)):
        layer = layers[index]
        for target in targets:
            if target in _ATTENTION_ATTRIBUTES:
                parent, attribute = layer.self_attn, _ATTENTION_ATTRIBUTES[target]
            else:
                parent, attribute = layer.mlp, target

            projection = getattr(parent, attribute)
            if isinstance(projection, LoRALinear):
                raise ValueError(f"LoRA already applied to block {index} {target}")

            module_rank = rank if rank_map is None else int(rank_map.get((index, target), rank))
            if module_rank <= 0:
                continue

            setattr(parent, attribute, LoRALinear(projection, module_rank, alpha, dropout))
            replaced += 1

    return replaced


def lora_linear_modules(model):
    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, LoRALinear)
    }


def count_lora_parameters(model):
    return sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if "lora_a." in name or "lora_b." in name
    )


def mark_only_lora_as_trainable(model):
    model.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.lora_a.requires_grad_(True)
            module.lora_b.requires_grad_(True)
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def lora_state_dict(model):
    return {
        name: value
        for name, value in model.state_dict().items()
        if "lora_a." in name or "lora_b." in name
    }
