import argparse
import hashlib
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from datasets.official_2sfs import (
    OFFICIAL_2SFS_DATASETS,
    OFFICIAL_SPLIT_SEEDS,
    build_official_2sfs_loaders,
)
from datasets.vision.utils import GLOBAL_SEED
from src.architecture import CLIP_MODEL, load_clip
from src.methods import TwoStageCLIP
from src.peft import (
    AbsIdentityGate,
    LoRAProOptimizer,
    lora,
    mark_only_half_layernorm_as_trainable,
    mark_only_layernorm_as_trainable,
)


def train_stage(
    logits_fn, parameters, loader, steps, lr, device, name, writer,
    gradient_gate=None, optimizer=None, eta_min=1e-6, early_stop=False,
):
    if optimizer is None:
        optimizer = torch.optim.AdamW(parameters, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, steps, eta_min=eta_min
    )
    scaler = torch.amp.GradScaler()
    cur_step = 0
    beta = 0.98
    loss_ema = None
    loss_best = float('inf')
    loss_bad = 0

    while cur_step < steps:
        for images, labels in loader:
            optimizer.zero_grad()
            images, labels = images.to(device), labels.to(device)
            with torch.amp.autocast(device):
                losses = F.cross_entropy(
                    logits_fn(images), labels,
                    reduction="none" if gradient_gate else "mean",
                )
                loss = losses.mean() if gradient_gate else losses

            previous = q = None
            if gradient_gate:
                previous, q = gradient_gate.prepare(losses, cur_step + 1)

            scale = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < scale:
                continue
            if gradient_gate:
                gradient_gate.apply(previous, q)
                writer.add_scalar(f"Q/{name}", q.mean().item(), cur_step + 1)

            scheduler.step()
            cur_step += 1

            print(f"{name} [{cur_step}/{steps}] Loss: {loss.item():.4f}")
            writer.add_scalar(f"Loss/{name}", loss.item(), cur_step)

            if early_stop:
                loss_ema = loss.item() if loss_ema is None else beta * loss_ema + (1-beta) * loss.item()
                if cur_step > steps * 0.2:
                    if loss_ema < loss_best - 1e-4:
                        loss_best = loss_ema
                        loss_bad = 0
                    else:
                        loss_bad += 1
                    loss_ok = loss_bad >= 60  # LOSS_PATIENCE
                    if loss_ok:
                        return cur_step

            if cur_step == steps:
                break
    return cur_step


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

    if total_size == 0:
        raise ValueError(f"Empty evaluation split: {split}")
    metrics = {
        "loss": total_loss / total_size,
        "correct": total_correct,
        "total": total_size,
        "accuracy": total_correct / total_size,
    }
    print(
        f"{split.title()} - Accuracy: {metrics['accuracy'] * 100:.1f}%, "
        f"Avg loss: {metrics['loss']:.6f}"
    )
    return metrics


def harmonic_mean(base_accuracy, novel_accuracy):
    denominator = base_accuracy + novel_accuracy
    if denominator == 0:
        return 0.0
    return 2 * base_accuracy * novel_accuracy / denominator


def seed_training(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def reset_train_stream(loader, seed):
    if loader.generator is None:
        raise RuntimeError("The official 2SFS train loader needs a generator")
    loader.generator.manual_seed(seed)
    sampler_generator = getattr(loader.sampler, "generator", None)
    if sampler_generator is None:
        raise RuntimeError("The official 2SFS train sampler needs a generator")
    sampler_generator.manual_seed(seed)


def config_fingerprint(args):
    excluded = {"data_root", "results_dir"}
    config = {
        key: value for key, value in vars(args).items() if key not in excluded
    }
    config["training_seed"] = GLOBAL_SEED
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:10]


def train_2sfs(args, method, train_loader, test_loader, device, writer):
    total_steps = args.shots * args.steps_per_shot
    stage_one_steps = int(total_steps * args.stage_one_ratio)
    method.to(device)

    if args.peft in ("lora", "ln_lora"):
        lora.apply_lora_to_clip(
            method.model,
            targets=args.lora_targets,
            blocks=args.lora_blocks,
            modality=args.lora_modality,
            rank=args.lora_rank,
        )
        lora.mark_only_lora_as_trainable(method.model)
        if args.peft == "ln_lora":
            for module in method.model.modules():
                if isinstance(module, torch.nn.LayerNorm):
                    module.requires_grad_(True)
    elif args.peft == "ln_half":
        mark_only_half_layernorm_as_trainable(method.model)
    else:
        mark_only_layernorm_as_trainable(method.model)

    parameters = [parameter for parameter in method.model.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError(f"No trainable parameters found after applying PEFT mode: {args.peft}")

    gradient_gate = None
    if args.gradient_gate == "abs_identity":
        gradient_gate = AbsIdentityGate(parameters, seed=GLOBAL_SEED)
        gradient_gate.initialize(method.stage_one_logits, train_loader.dataset, device)

    seed_training(GLOBAL_SEED)
    reset_train_stream(train_loader, GLOBAL_SEED)
    method.train()
    stage_one_optimizer = None
    stage_one_eta_min = 1e-6
    if args.peft == "lora" and args.stage1_optimizer == "lora_pro":
        stage_one_optimizer = LoRAProOptimizer(
            lora.lora_modules(method.model), args.lora_pro_lr
        )
        stage_one_eta_min = args.lora_pro_lr / 100
    stage_one_steps_run = train_stage(
        method.stage_one_logits,
        parameters,
        train_loader,
        stage_one_steps,
        args.lr,
        device,
        "stage1",
        writer,
        gradient_gate,
        optimizer=stage_one_optimizer,
        eta_min=stage_one_eta_min,
        early_stop=args.ema_early_stop,
    )

    method.initialize_classifier()
    method.eval()
    seed_training(GLOBAL_SEED)
    reset_train_stream(train_loader, GLOBAL_SEED)
    stage_two_steps_run = train_stage(
        method.stage_two_logits,
        [method.classifier],
        train_loader,
        total_steps - stage_one_steps_run,
        args.lr,
        device,
        "stage2",
        writer
    )
    training_metrics = {
        "stage1_steps": stage_one_steps_run,
        "stage2_steps": stage_two_steps_run,
        "total_steps": stage_one_steps_run + stage_two_steps_run,
    }

    method.eval()
    if args.setting == "base2new":
        test_base_loader, test_novel_loader = test_loader
        base_metrics = evaluate(
            method, test_base_loader, method.classifier, device, "test base"
        )
        with torch.no_grad():
            novel_classifier = method.encode_classnames(
                test_novel_loader.dataset.classes
            )
        novel_metrics = evaluate(
            method,
            test_novel_loader,
            novel_classifier,
            device,
            "test novel",
        )
        hm = harmonic_mean(
            base_metrics["accuracy"], novel_metrics["accuracy"]
        )
        print(f"Test - Harmonic mean: {hm * 100:.1f}%")
        writer.add_scalar("Accuracy/test_base", base_metrics["accuracy"], 0)
        writer.add_scalar("Accuracy/test_novel", novel_metrics["accuracy"], 0)
        writer.add_scalar("Accuracy/test_harmonic_mean", hm, 0)
        return {
            "training": training_metrics,
            "test": {
                "base": base_metrics,
                "novel": novel_metrics,
                "harmonic_mean": hm,
            }
        }
    else:
        test_metrics = evaluate(
            method, test_loader, method.classifier, device, "test"
        )
        writer.add_scalar("Accuracy/test", test_metrics["accuracy"], 0)
        return {"training": training_metrics, "test": {"all": test_metrics}}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", choices=OFFICIAL_2SFS_DATASETS, default="dtd"
    )
    parser.add_argument("--shots", type=int, choices=[1, 2, 4, 8, 16], default=1)
    parser.add_argument(
        "--split_seed", type=int, choices=OFFICIAL_SPLIT_SEEDS, default=1
    )
    parser.add_argument(
        "--peft", choices=["ln", "lora", "ln_lora", "ln_half"], default="ln"
    )
    parser.add_argument("--gradient_gate", choices=["none", "abs_identity"], default="none")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--steps_per_shot", type=int, default=300)
    parser.add_argument("--stage_one_ratio", type=float, default=0.6)
    parser.add_argument("--setting", choices=["standard", "base2new"], default="standard")
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--test_batch_size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--results_dir", default="results/2sfs")
    parser.add_argument("--ema_early_stop", action="store_true")
    parser.add_argument("--lora_targets", nargs="+", choices=lora.TARGETS, default=["q", "k", "v"])
    parser.add_argument("--lora_blocks", choices=["all", "odd", "even"], default="all")
    parser.add_argument("--lora_modality", choices=["both", "vision", "text"], default="both")
    parser.add_argument("--lora_rank", type=int, default=lora.RANK)
    parser.add_argument("--stage1_optimizer", choices=["adamw", "lora_pro"], default="adamw")
    parser.add_argument("--lora_pro_lr", type=float, default=2e-6)
    args = parser.parse_args()
    if args.stage1_optimizer == "lora_pro" and args.peft != "lora":
        parser.error("--stage1_optimizer lora_pro requires --peft lora")
    if args.workers < 0:
        parser.error("--workers cannot be negative")
    return args


def main():
    args = parse_args()
    seed_training(GLOBAL_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, _, test_loader, _ = build_official_2sfs_loaders(
        batch_size=args.batch_size,
        dataset_name=args.dataset,
        root=args.data_root,
        shots=args.shots,
        setting=args.setting,
        split_seed=args.split_seed,
        training_seed=GLOBAL_SEED,
        test_batch_size=args.test_batch_size,
        num_workers=args.workers,
    )
    model, tokenizer = load_clip(CLIP_MODEL)
    method = TwoStageCLIP(
        model,
        tokenizer,
        train_loader.dataset.classes,
        train_loader.dataset.template
    )
    run_name = (
        f"{args.dataset}-{args.peft}-{args.shots}shot"
        f"-split{args.split_seed}-seed{GLOBAL_SEED}"
        f"-ratio{args.stage_one_ratio}"
    )
    if args.gradient_gate != "none":
        run_name += f"-{args.gradient_gate}"
    if args.setting == "base2new":
        run_name += "-base2new"
    if args.peft in ("lora", "ln_lora") and (
        args.lora_targets != ["q", "k", "v"]
        or args.lora_blocks != "all"
        or args.lora_modality != "both"
        or args.lora_rank != lora.RANK
        or args.stage1_optimizer != "adamw"
    ):
        targets = "".join(args.lora_targets)
        run_name += (
            f"-{targets}-{args.lora_blocks}-{args.lora_modality}"
            f"-r{args.lora_rank}-{args.stage1_optimizer}"
        )
        if args.stage1_optimizer == "lora_pro":
            run_name += f"-lr{args.lora_pro_lr:g}"
    run_name += f"-cfg{config_fingerprint(args)}"
    run_dir = Path("runs/2sfs") / run_name
    with SummaryWriter(str(run_dir)) as writer:
        metrics = train_2sfs(
            args,
            method,
            train_loader,
            test_loader,
            device,
            writer
        )
    result = {
        "schema_version": "2sfs-result-v1",
        "run_name": run_name,
        "model": CLIP_MODEL,
        "dataset": args.dataset,
        "setting": args.setting,
        "shots": args.shots,
        "split_seed": args.split_seed,
        "training_seed": GLOBAL_SEED,
        "protocol": train_loader.dataset.protocol,
        "config": vars(args),
        "metrics": metrics,
    }
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    result_path = results_dir / f"{run_name}.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"Saved results to {result_path}")


if __name__ == "__main__":
    main()
