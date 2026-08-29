import argparse
from random import Random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from architecture import CLIP_MODEL, load_clip
from datasets import get_dataloader
from datasets.vision.utils import GLOBAL_SEED
from methods import TwoStageCLIP
from peft import AbsIdentityGate, apply_lora, mark_only_layernorm_as_trainable, mark_only_lora_as_trainable


def train_stage(
    logits_fn, parameters, loader, steps, lr, device, name, writer,
    on_epoch_end=None, gradient_gate=None,
):
    optimizer = torch.optim.AdamW(parameters, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps, eta_min=1e-6)
    scaler = torch.amp.GradScaler()
    cur_step = 0

    while cur_step < steps:
        for batch_index, (images, labels) in enumerate(loader, 1):
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


def train_2sfs(args, method, train_loader, validation_loader, test_loader, device, writer):
    total_steps = args.shots * args.steps_per_shot
    stage_one_steps = int(total_steps * args.stage_one_ratio)
    method.to(device)

    if args.peft == "lora":
        apply_lora(method.model.vision_model)
        apply_lora(method.model.text_model)
        mark_only_lora_as_trainable(method.model)
    else:
        mark_only_layernorm_as_trainable(method.model)

    parameters = [parameter for parameter in method.model.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError(f"No trainable parameters found after applying PEFT mode: {args.peft}")

    gradient_gate = None
    if args.gradient_gate == "abs_identity":
        gradient_gate = AbsIdentityGate(parameters)
        gradient_gate.initialize(method.stage_one_logits, train_loader.dataset, device)

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
    train_stage(
        method.stage_one_logits,
        parameters,
        train_loader,
        stage_one_steps,
        args.lr,
        device,
        "stage1",
        writer,
        on_epoch_end,
        gradient_gate
    )

    method.initialize_classifier()
    method.eval()
    train_stage(
        method.stage_two_logits,
        [method.classifier],
        train_loader,
        total_steps - stage_one_steps,
        args.lr,
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--shots", type=int, choices=[1, 2, 4, 8, 16], default=1)
    parser.add_argument("--peft", choices=["ln", "lora"], default="ln")
    parser.add_argument("--gradient_gate", choices=["none", "abs_identity"], default="none")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--steps_per_shot", type=int, default=300)
    parser.add_argument("--stage_one_ratio", type=float, default=0.6)
    parser.add_argument("--setting", choices=["standard", "base2new"], default="standard")
    parser.add_argument("--data_root", default="data")
    return parser.parse_args()


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
    if args.gradient_gate != "none":
        run_name += f"-{args.gradient_gate}"
    if args.setting == "base2new":
        run_name += "-base2new"
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
