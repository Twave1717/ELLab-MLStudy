import argparse
from hashlib import sha1
from random import Random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from architecture import CLIP_MODEL, load_clip
from datasets import get_dataloader
from datasets.vision.utils import GLOBAL_SEED
from methods import TwoStageCLIP
from peft import (
    DEFAULT_TARGETS,
    RANK,
    TARGET_ALIASES,
    TARGET_CHOICES,
    LoRAProOptimizer,
    apply_lora,
    count_lora_parameters,
    lora_linear_modules,
    mark_only_layernorm_as_trainable,
    mark_only_lora_as_trainable,
    normalize_targets,
    optimizer_state_bytes,
)


def train_stage(logits_fn, optimizer, loader, steps, device, name, writer,
                eta_min=1e-6, on_epoch_end=None):
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps, eta_min=eta_min)
    scaler = torch.amp.GradScaler()
    cur_step = 0

    while cur_step < steps:
        for batch_index, (images, labels) in enumerate(loader, 1):
            optimizer.zero_grad()
            images, labels = images.to(device), labels.to(device)
            with torch.amp.autocast(device):
                loss = F.cross_entropy(logits_fn(images), labels)

            scale = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < scale:
                continue

            scheduler.step()
            cur_step += 1
            print(f"{name} [{cur_step}/{steps}] Loss: {loss.item():.4f}")
            writer.add_scalar(f"Loss/{name}", loss.item(), cur_step)
            if cur_step == steps:
                break
        if on_epoch_end and batch_index == len(loader):
            on_epoch_end(cur_step)


def evaluate(method, loader, classifier, device, split):
    total_loss, total_correct, total_size = 0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = method.classifier_logits(images, classifier)
            batch_size = labels.size(0)
            total_loss += F.cross_entropy(logits, labels).item() * batch_size
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_size += batch_size

    loss = total_loss / total_size
    accuracy = total_correct / total_size
    print(f"{split.title()} - Accuracy: {accuracy * 100:.1f}%, Avg loss: {loss:.6f}")
    return accuracy


def build_breakpoint_loader(method, loader, classifier, device, split):
    incorrect_by_class = {label: [] for label in range(len(loader.dataset.classes))}
    total_correct = total_size = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            predictions = method.classifier_logits(images, classifier).argmax(dim=1)
            total_correct += (predictions == labels).sum().item()
            for index in (predictions != labels).nonzero().flatten().tolist():
                incorrect_by_class[labels[index].item()].append(total_size + index)
            total_size += labels.size(0)

    rng = Random(GLOBAL_SEED)
    indices = []
    for incorrect_indices in incorrect_by_class.values():
        indices.extend(rng.sample(incorrect_indices, min(50, len(incorrect_indices))))

    dataset = Subset(loader.dataset, indices)
    dataset.classes = loader.dataset.classes
    dataset.template = loader.dataset.template
    print(f"{split.title()} full - Accuracy: {total_correct / total_size * 100:.1f}%, Breakpoint samples: {len(dataset)}")
    return DataLoader(dataset, batch_size=loader.batch_size)


def configure_peft(args, method):
    """Attach the adapter and return the Stage 1 trainable parameters."""
    if args.peft != "lora":
        parameters = mark_only_layernorm_as_trainable(method.model)
        total = sum(parameter.numel() for parameter in parameters)
        print(f"LayerNorm: {len(parameters)} tensors, {total:,} trainable parameters")
        return parameters

    targets = normalize_targets(args.lora_targets)
    encoders = []
    if args.lora_modality in ("both", "vision"):
        encoders.append(method.model.vision_model)
    if args.lora_modality in ("both", "text"):
        encoders.append(method.model.text_model)

    replaced = sum(
        apply_lora(
            encoder,
            targets=targets,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            blocks=args.lora_blocks,
        )
        for encoder in encoders
    )
    parameters = mark_only_lora_as_trainable(method.model)
    print(
        f"LoRA: {replaced} layers replaced, targets={'+'.join(targets)}, "
        f"rank={args.lora_rank}, alpha={args.lora_alpha}, blocks={args.lora_blocks}, "
        f"modality={args.lora_modality}, "
        f"{count_lora_parameters(method.model):,} trainable parameters"
    )
    return parameters


def build_stage_one_optimizer(args, method, parameters):
    """Return the Stage 1 optimizer and its cosine schedule floor."""
    if args.stage1_optimizer == "lora_pro":
        modules = lora_linear_modules(method.model)
        optimizer = LoRAProOptimizer(modules, lr=args.lora_pro_lr)
        eta_min = (
            args.stage1_eta_min
            if args.stage1_eta_min is not None
            else args.lora_pro_lr / 100
        )
        megabytes = optimizer_state_bytes(modules) / 1024 ** 2
        print(
            f"LoRA-Pro: {len(modules)} adapters, lr={args.lora_pro_lr}, "
            f"eta_min={eta_min}, optimizer state ~{megabytes:.0f} MiB"
        )
        return optimizer, eta_min

    optimizer = torch.optim.AdamW(
        parameters, lr=args.lr, weight_decay=args.weight_decay
    )
    eta_min = args.stage1_eta_min if args.stage1_eta_min is not None else 1e-6
    return optimizer, eta_min


def train_2sfs(args, method, train_loader, validation_loader, test_loader, device, writer):
    total_steps = args.shots * args.steps_per_shot
    stage_one_steps = int(total_steps * args.stage_one_ratio)
    method.to(device)

    parameters = configure_peft(args, method)

    on_epoch_end = None
    if args.setting == "base2new":
        validation_base_loader, validation_new_loader = validation_loader
        method.eval()
        with torch.no_grad():
            base_classifier = method.encode_text()
            new_classifier = method.encode_classnames(validation_new_loader.dataset.classes)
        breakpoint_base_loader = build_breakpoint_loader(method, validation_base_loader, base_classifier, device, "stage1 [0] base")
        breakpoint_new_loader = build_breakpoint_loader(method, validation_new_loader, new_classifier, device, "stage1 [0] new")
        previous_step = previous_base_accuracy = previous_new_accuracy = 0
        writer.add_scalar("Accuracy/stage1_base", 0, 0)
        writer.add_scalar("Accuracy/stage1_new", 0, 0)

        def track_breakpoint(cur_step):
            nonlocal previous_step, previous_base_accuracy, previous_new_accuracy
            method.eval()
            with torch.no_grad():
                base_classifier = method.encode_text()
                new_classifier = method.encode_classnames(breakpoint_new_loader.dataset.classes)
            base_accuracy = evaluate(method, breakpoint_base_loader, base_classifier, device, f"stage1 [{cur_step}] base")
            new_accuracy = evaluate(method, breakpoint_new_loader, new_classifier, device, f"stage1 [{cur_step}] new")
            writer.add_scalar("Accuracy/stage1_base", base_accuracy, cur_step)
            writer.add_scalar("Accuracy/stage1_new", new_accuracy, cur_step)
            if cur_step > previous_step:
                step_gap = cur_step - previous_step
                base_rate = (base_accuracy - previous_base_accuracy) / step_gap
                new_rate = (new_accuracy - previous_new_accuracy) / step_gap
                writer.add_scalar("Rate/stage1_base", base_rate, cur_step)
                writer.add_scalar("Rate/stage1_new", new_rate, cur_step)
                writer.add_scalar("Rate/stage1_gap", base_rate - new_rate, cur_step)
            previous_step, previous_base_accuracy, previous_new_accuracy = cur_step, base_accuracy, new_accuracy
            method.train()

        on_epoch_end = track_breakpoint

    method.train()
    stage_one_optimizer, stage_one_eta_min = build_stage_one_optimizer(
        args, method, parameters
    )
    train_stage(
        method.stage_one_logits,
        stage_one_optimizer,
        train_loader,
        stage_one_steps,
        device,
        "stage1",
        writer,
        stage_one_eta_min,
        on_epoch_end
    )
    if isinstance(stage_one_optimizer, LoRAProOptimizer):
        for key, value in stage_one_optimizer.diagnostics().items():
            print(f"lora_pro/{key}: {value}")
            writer.add_scalar(f"LoRAPro/{key}", value, stage_one_steps)

    method.initialize_classifier()
    method.eval()
    train_stage(
        method.stage_two_logits,
        torch.optim.AdamW(
            [method.classifier], lr=args.lr, weight_decay=args.weight_decay
        ),
        train_loader,
        total_steps - stage_one_steps,
        device,
        "stage2",
        writer
    )

    method.eval()
    if args.setting == "base2new":
        validation_base_loader, validation_new_loader = validation_loader
        evaluate(method, validation_base_loader, method.classifier, device, "validation base")
        with torch.no_grad():
            new_classifier = method.encode_classnames(validation_new_loader.dataset.classes)
        evaluate(method, validation_new_loader, new_classifier, device, "validation new")
        test_base_loader, test_new_loader = test_loader
        evaluate(method, test_base_loader, method.classifier, device, "test base")
        with torch.no_grad():
            new_classifier = method.encode_classnames(test_new_loader.dataset.classes)
        evaluate(method, test_new_loader, new_classifier, device, "test new")
    else:
        evaluate(method, validation_loader, method.classifier, device, "validation")
        evaluate(method, test_loader, method.classifier, device, "test")


# run 이름 앞부분에 이미 들어가는 설정
BASE_KEYS = ("dataset", "shots", "peft", "stage_one_ratio", "setting")
# `--peft ln`에서는 쓰이지 않는 설정. `stage1_eta_min`은 두 경우 모두 쓰이므로 제외하지 않는다.
LORA_ONLY_KEYS = ("lora_rank", "lora_alpha", "lora_dropout", "lora_blocks",
                  "lora_modality", "lora_pro_lr", "stage1_optimizer")


def run_suffix(args, defaults):
    """기본값과 다른 설정을 모두 접미사로 남긴다."""
    tags = []
    if args.peft == "lora":
        targets = normalize_targets(args.lora_targets)
        if targets != DEFAULT_TARGETS:
            tags.append("".join(targets))
    for key, default in defaults.items():
        if key in BASE_KEYS or key in ("lora_targets", "data_root"):
            continue
        if args.peft != "lora" and key in LORA_ONLY_KEYS:
            continue
        value = getattr(args, key)
        if value != default:
            tags.append(f"{key}{value}")
    if args.data_root != defaults["data_root"]:
        tags.append("data" + sha1(args.data_root.encode()).hexdigest()[:8])
    return "-" + "-".join(tags) if tags else ""


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--shots", type=int, choices=[1, 2, 4, 8, 16], default=1)
    parser.add_argument("--peft", choices=["ln", "lora"], default="ln")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--steps_per_shot", type=int, default=300)
    parser.add_argument("--stage_one_ratio", type=float, default=0.6)
    parser.add_argument("--setting", choices=["standard", "base2new"], default="standard")
    parser.add_argument("--data_root", default="data")

    lora = parser.add_argument_group("LoRA")
    lora.add_argument(
        "--lora_targets",
        nargs="+",
        default=list(DEFAULT_TARGETS),
        metavar="TARGET",
        help=f"any of {TARGET_CHOICES}; aliases: {TARGET_ALIASES}",
    )
    lora.add_argument("--lora_rank", type=int, default=RANK)
    lora.add_argument("--lora_alpha", type=float, default=1.0)
    lora.add_argument("--lora_dropout", type=float, default=0.25)
    lora.add_argument(
        "--lora_blocks",
        default="all",
        help="all, odd, even, or 0-based indices such as 0,2,4",
    )
    lora.add_argument(
        "--lora_modality", choices=["both", "vision", "text"], default="both"
    )

    optimizer = parser.add_argument_group("Stage 1 optimizer")
    optimizer.add_argument(
        "--stage1_optimizer", choices=["adamw", "lora_pro"], default="adamw"
    )
    optimizer.add_argument("--lora_pro_lr", type=float, default=2e-6)
    optimizer.add_argument(
        "--stage1_eta_min",
        type=float,
        default=None,
        help="cosine floor; defaults to 1e-6 for adamw and lr/100 for lora_pro",
    )

    return parser


def parse_args():
    parser = build_parser()
    args = parser.parse_args()
    if args.stage1_optimizer == "lora_pro" and args.peft != "lora":
        parser.error("--stage1_optimizer lora_pro requires --peft lora")
    normalize_targets(args.lora_targets)
    return args


def main():
    args = parse_args()
    torch.manual_seed(GLOBAL_SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, validation_loader, test_loader, _ = get_dataloader(
        args.batch_size,
        args.dataset,
        "2sfs",
        root=args.data_root,
        shots=args.shots,
        setting=args.setting
    )
    model, tokenizer = load_clip(CLIP_MODEL)
    method = TwoStageCLIP(
        model,
        tokenizer,
        train_loader.dataset.classes,
        train_loader.dataset.template
    )
    run_name = f"{args.dataset}-{args.peft}-{args.shots}shot-ratio{args.stage_one_ratio}"
    if args.setting == "base2new":
        run_name += "-base2new"
    run_name += run_suffix(args, vars(build_parser().parse_args([])))
    with SummaryWriter(f"runs/2sfs/{run_name}") as writer:
        train_2sfs(
            args,
            method,
            train_loader,
            validation_loader,
            test_loader,
            device,
            writer
        )


if __name__ == "__main__":
    main()
