"""LoRA 회귀 검사: 기본값 동등성, 파라미터 회계, target/block 선택, LoRA-Pro, run 이름.

    python scripts/verify_lora.py

CLIP weight를 받지 않고 CLIPConfig로 ViT-B/16 구조만 만들어 CPU에서 검사한다.
"""

import math
import sys
import types
from pathlib import Path

import torch
from torch import nn
from transformers import CLIPConfig, CLIPModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from peft import (  # noqa: E402
    ALPHA,
    DROPOUT,
    RANK,
    LoRALinear,
    LoRAProOptimizer,
    apply_lora,
    count_lora_parameters,
    lora_linear_modules,
    mark_only_layernorm_as_trainable,
    mark_only_lora_as_trainable,
    normalize_targets,
    optimizer_state_bytes,
    parse_blocks,
)

CHECKS, FAILURES = [], []


def check(label, actual, expected):
    CHECKS.append(label)
    ok = actual == expected
    if not ok:
        FAILURES.append(label)
    print(f"{'PASS' if ok else 'FAIL'}  {label}: {actual}" + ("" if ok else f" != {expected}"))


def build_clip():
    config = CLIPConfig(
        text_config=dict(hidden_size=512, intermediate_size=2048,
                         num_hidden_layers=12, num_attention_heads=8),
        vision_config=dict(hidden_size=768, intermediate_size=3072,
                           num_hidden_layers=12, num_attention_heads=12,
                           image_size=224, patch_size=16),
        projection_dim=512,
    )
    torch.manual_seed(0)
    return CLIPModel(config)


def lora_weights(model):
    return {name: value.clone() for name, value in model.state_dict().items()
            if "lora_a." in name or "lora_b." in name}


# --- 1. 기본값에서 main(63057a4)과 초기 상태가 같은지 ---

class OriginalLoRALinear(nn.Module):
    """main의 peft/lora.py 구현."""

    def __init__(self, base):
        super().__init__()
        self.base = base
        self.scaling = ALPHA / math.sqrt(RANK)
        self.dropout = nn.Dropout(DROPOUT)
        self.lora_a = nn.Linear(base.in_features, RANK, bias=False)
        self.lora_b = nn.Linear(RANK, base.out_features, bias=False)
        self.to(base.weight)
        nn.init.zeros_(self.lora_b.weight)


def original_apply_lora(encoder):
    replaced = 0
    for layer in encoder.encoder.layers:
        for name in ("q_proj", "k_proj", "v_proj"):
            setattr(layer.self_attn, name, OriginalLoRALinear(getattr(layer.self_attn, name)))
            replaced += 1
    return replaced


old = build_clip()
torch.manual_seed(2026)
old_replaced = original_apply_lora(old.vision_model) + original_apply_lora(old.text_model)
old_weights = lora_weights(old)

model = build_clip()
torch.manual_seed(2026)
new_replaced = apply_lora(model.vision_model) + apply_lora(model.text_model)
new_weights = lora_weights(model)

check("main replaced count", old_replaced, 72)
check("new replaced count", new_replaced, 72)
check("state_dict keys identical", new_weights.keys() == old_weights.keys(), True)
check("initial weights bit-identical",
      all(torch.equal(new_weights[key], old_weights[key]) for key in old_weights), True)
check("scaling = alpha/sqrt(rank)",
      model.vision_model.encoder.layers[0].self_attn.q_proj.scaling, 1 / math.sqrt(2))

# --- 2. 파라미터 회계 ---

parameters = mark_only_lora_as_trainable(model)
check("default trainable tensors", len(parameters), 144)
check("default trainable params", sum(p.numel() for p in parameters), 184_320)
check("count_lora_parameters", count_lora_parameters(model), 184_320)
check("lora_linear_modules", len(lora_linear_modules(model)), 72)
check("lora_pro state bytes", optimizer_state_bytes(lora_linear_modules(model)), 245_366_784)

ln_parameters = mark_only_layernorm_as_trainable(build_clip())
check("layernorm tensors", len(ln_parameters), 102)
check("layernorm params", sum(p.numel() for p in ln_parameters), 65_536)

# --- 3. target / block / rank 선택 ---

cases = [
    (dict(targets=("q", "k", "v"), rank=1, blocks="odd"), 36, 46_080),
    (dict(targets=("q", "k", "v"), rank=1, blocks="even"), 36, 46_080),
    (dict(targets=("q", "k", "v"), rank=1, blocks="0,2,4,6"), 24, 30_720),  # half-LN과 동일 예산
    (dict(targets=("fc1",), rank=1, blocks="odd"), 12, 38_400),
    (dict(targets=("c_fc",), rank=1, blocks="odd"), 12, 38_400),           # 별칭
    (dict(targets=("q", "k", "v", "o"), rank=2), 96, 245_760),
    (dict(targets=("fc1", "fc2"), rank=2, blocks="0,2,4"), 12, 76_800),
]
for kwargs, expected_replaced, expected_params in cases:
    case = build_clip()
    replaced = apply_lora(case.vision_model, **kwargs) + apply_lora(case.text_model, **kwargs)
    label = ",".join(f"{key}={value}" for key, value in kwargs.items())
    check(f"replaced [{label}]", replaced, expected_replaced)
    check(f"params [{label}]", count_lora_parameters(case), expected_params)

odd = build_clip()
apply_lora(odd.vision_model, blocks="odd")
check("blocks=odd -> 0-based odd indices",
      [index for index, layer in enumerate(odd.vision_model.encoder.layers)
       if isinstance(layer.self_attn.q_proj, LoRALinear)], [1, 3, 5, 7, 9, 11])
check("parse_blocks even", parse_blocks("even", 12), (0, 2, 4, 6, 8, 10))
check("target aliases", normalize_targets(["c_fc", "out_proj", "c_proj"]), ("fc1", "o", "fc2"))

vision_only = build_clip()
apply_lora(vision_only.vision_model)
check("modality=vision only", count_lora_parameters(vision_only), 110_592)

# rank_map은 없는 key에서 rank로 fallback: (0,q)만 rank 4, 나머지 5개는 rank 2
partial = build_clip()
apply_lora(partial.vision_model, rank=2, blocks="0,1", rank_map={(0, "q"): 4})
check("rank_map falls back on missing key",
      count_lora_parameters(partial), 2 * 4 * 768 + 5 * 2 * 2 * 768)

for bad in (["qq"], []):
    try:
        normalize_targets(bad)
        check(f"reject targets {bad}", "no error", "ValueError")
    except ValueError:
        check(f"reject targets {bad}", "ValueError", "ValueError")
try:
    apply_lora(build_clip().vision_model, blocks="99")
    check("reject out-of-range block", "no error", "ValueError")
except ValueError:
    check("reject out-of-range block", "ValueError", "ValueError")

# --- 4. LoRA-Pro ---

torch.manual_seed(0)
adapter = LoRALinear(nn.Linear(16, 24, bias=False), rank=2, alpha=1, dropout=0.0)
optimizer = LoRAProOptimizer({"probe": adapter}, lr=1e-3)
check("B starts at zero", float(adapter.lora_b.weight.detach().abs().sum()), 0.0)

inputs, targets = torch.randn(8, 16), torch.randn(8, 24)
losses = []
for _ in range(20):
    optimizer.zero_grad()
    loss = ((adapter(inputs) - targets) ** 2).mean()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

check("no NaN", all(math.isfinite(value) for value in losses), True)
check("B leaves degenerate branch", float(adapter.lora_b.weight.detach().abs().sum()) > 0, True)
check("loss decreased", losses[-1] < losses[0], True)
check("optimizer_steps counted", optimizer.diagnostics()["optimizer_steps"], 20)
check("no ascent step", optimizer.diagnostics()["ascent_fraction"], 0.0)

# --- 5. run 디렉터리 이름 ---

def load_train_module():
    """train_2sfs는 dataset/architecture 의존성이 무거워 stub 후 import한다."""
    stubs = {
        "architecture": dict(CLIP_MODEL="", load_clip=None),
        "datasets": dict(get_dataloader=None),
        "datasets.vision": {},
        "datasets.vision.utils": dict(GLOBAL_SEED=2026),
        "methods": dict(TwoStageCLIP=object),
        "torch.utils.tensorboard": dict(SummaryWriter=object),
    }
    for name, attributes in stubs.items():
        module = sys.modules.setdefault(name, types.ModuleType(name))
        for key, value in attributes.items():
            if not hasattr(module, key):
                setattr(module, key, value)
    import train_2sfs

    return train_2sfs


train = load_train_module()
defaults = vars(train.build_parser().parse_args([]))


def run_name(argv):
    args = train.build_parser().parse_args(argv)
    base = f"{args.dataset}-{args.peft}-{args.shots}shot-ratio{args.stage_one_ratio}"
    return base + train.run_suffix(args, defaults)


BASE = "cifar10-ln-1shot-ratio0.6"
LORA = "cifar10-lora-1shot-ratio0.6"
name_cases = [
    ([], BASE),
    (["--peft", "lora"], LORA),
    (["--peft", "ln", "--lora_rank", "4"], BASE),                       # ln에서 미사용
    (["--peft", "ln", "--stage1_eta_min", "1e-7"], f"{BASE}-stage1_eta_min1e-07"),
    (["--batch_size", "64"], f"{BASE}-batch_size64"),
    (["--peft", "lora", "--lora_rank", "1", "--lora_blocks", "odd"],
     f"{LORA}-lora_rank1-lora_blocksodd"),
    (["--peft", "lora", "--lora_alpha", "2", "--lora_dropout", "0.1"],
     f"{LORA}-lora_alpha2.0-lora_dropout0.1"),
    (["--peft", "lora", "--stage1_optimizer", "lora_pro", "--lora_pro_lr", "5e-6"],
     f"{LORA}-stage1_optimizerlora_pro-lora_pro_lr5e-06"),
    (["--peft", "lora", "--lora_targets", "q", "k", "v", "o"], f"{LORA}-qkvo"),
    (["--peft", "lora", "--lr", "1e-4", "--weight_decay", "0"],
     f"{LORA}-lr0.0001-weight_decay0.0"),
]
for argv, expected in name_cases:
    check(f"run name [{' '.join(argv) or 'defaults'}]", run_name(argv), expected)

check("data_root는 hash로 기록",
      run_name(["--data_root", "alternate_data"]) != BASE
      and "alternate_data" not in run_name(["--data_root", "alternate_data"]), True)
check("서로 다른 data_root는 서로 다른 이름",
      run_name(["--data_root", "a"]) != run_name(["--data_root", "b"]), True)

print(f"\n{len(CHECKS) - len(FAILURES)}/{len(CHECKS)} passed")
print("FAILURES:", FAILURES or "none")
raise SystemExit(1 if FAILURES else 0)
