import argparse
import os
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

import architecture
import methods
from datasets import get_dataloader
from datasets.vision.utils import GLOBAL_SEED
from tuning import apply_lora, lora_state_dict, mark_only_lora_as_trainable


def train(
    epochs,
    train_loader,
    validation_loader,
    test_loader,
    device,
    method,
    optimizer,
    scheduler,
    writer,
    grad_clip=None,
):
    global_step = 0

    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}\n-------------------------------")
        method.train()

        for batch in train_loader:
            optimizer.zero_grad()
            loss = method.training_step(batch, device)
            loss.backward()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(method.parameters(), grad_clip)

            optimizer.step()
            if hasattr(method, "after_optimizer_step"):
                method.after_optimizer_step()

            if global_step % 100 == 0:
                print(f"Step [{global_step}/{len(train_loader) * epochs}] Loss: {loss.item():.4f}")
                writer.add_scalar("Loss/train", loss.item(), global_step)
            global_step += 1

        scheduler.step()
        test(epoch, validation_loader, device, method, writer, "validation")

    test(epochs, test_loader, device, method, writer, "test")


def test(epoch, loader, device, method, writer, split):
    method.eval()
    total_loss, total_correct, total_size = 0, 0, 0

    with torch.no_grad():
        for batch in loader:
            loss, correct, batch_size = method.validation_step(batch, device)
            total_loss += loss * batch_size
            total_correct += correct
            total_size += batch_size

    loss = total_loss / total_size
    accuracy = total_correct / total_size
    print(f"{split.title()} - Accuracy: {accuracy * 100:.1f}%, Avg loss: {loss:.6f}")
    writer.add_scalar(f"Loss/{split}", loss, epoch)
    writer.add_scalar(f"Accuracy/{split}", accuracy * 100, epoch)


def parse_args():
    parser = argparse.ArgumentParser()

    # experiment
    parser.add_argument(
        '--method',
        choices=["supervised", "byol", "simclr", "rotnet", "moco", "clip"],
        default="supervised",
    )
    parser.add_argument('--model', choices=architecture.MODEL_CHOICES)
    parser.add_argument('--dataset', default="cifar10")
    parser.add_argument('--tuning', choices=["none", "lora"], default="none")
    parser.add_argument('--pretrained', action='store_true')

    # training
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--scheduler', choices=["multistep", "cosine"])
    parser.add_argument('--grad_clip', type=float)

    # paths
    parser.add_argument('--data_root', default="data")
    parser.add_argument('--save_path', default="checkpoint")
    return parser.parse_args()


def build_method(args, num_classes, dataset):
    if args.method == "clip":
        model, tokenizer = architecture.load_clip(args.model)
        return methods.CLIP(model, tokenizer, dataset.classes, dataset.template)

    encoder = architecture.build_encoder(args.model, args.pretrained)
    if args.method == "supervised":
        return methods.SupervisedLearning(encoder, num_classes)
    if args.method == "byol":
        return methods.BYOL(encoder)
    if args.method == "simclr":
        return methods.simCLR(encoder)
    if args.method == "rotnet":
        return methods.RotNetMethod(encoder)
    if args.method == "moco":
        return methods.MoCo(encoder)


def configure_tuning(args, method):
    if args.tuning == "lora":
        replaced = apply_lora(method.encoder)
        mark_only_lora_as_trainable(method)
        print(f"Applied LoRA to {replaced} Linear layers")


def build_optimizer(args, method):
    lr = args.lr if args.lr is not None else (2e-4 if args.method == "clip" else 0.1)
    weight_decay = args.weight_decay if args.weight_decay is not None else (
        1e-2 if args.method == "clip" else 1e-4
    )
    parameters = [parameter for parameter in method.parameters() if parameter.requires_grad]

    if args.method == "clip":
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    return torch.optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=weight_decay)


def build_scheduler(args, optimizer):
    name = args.scheduler or ("cosine" if args.method == "clip" else "multistep")
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    return torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        [args.epochs // 2, args.epochs * 3 // 4],
        gamma=0.1,
    )


def save_checkpoint(args, method):
    model_name = args.model.rsplit("/", 1)[-1]
    run_name = f"{model_name}-{args.method}-{args.epochs}-{datetime.now():%m%d_%H%M}"
    save_dir = os.path.join(args.save_path, run_name)
    os.makedirs(save_dir, exist_ok=True)

    if args.tuning == "lora":
        path = os.path.join(save_dir, "lora.pth")
        torch.save({"state_dict": lora_state_dict(method), "model": args.model}, path)
        print(f"Saved LoRA state to {path}")
        return

    encoder_path = os.path.join(save_dir, "encoder.pth")
    method_path = os.path.join(save_dir, "method.pth")
    torch.save(method.encoder.state_dict(), encoder_path)
    torch.save(method.state_dict(), method_path)
    print(f"Saved encoder state to {encoder_path}")
    print(f"Saved method state to {method_path}")


def main():
    args = parse_args()
    torch.manual_seed(GLOBAL_SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.model = architecture.CLIP_MODEL if args.method == "clip" else args.model or "resnet-20"

    train_loader, validation_loader, test_loader, num_classes = get_dataloader(
        args.batch_size,
        args.dataset,
        args.method,
        root=args.data_root,
        pretrained=args.pretrained,
        model_name=args.model,
    )
    method = build_method(args, num_classes, train_loader.dataset)
    configure_tuning(args, method)
    method.to(device)
    optimizer = build_optimizer(args, method)
    scheduler = build_scheduler(args, optimizer)

    with SummaryWriter() as writer:
        train(
            args.epochs,
            train_loader,
            validation_loader,
            test_loader,
            device,
            method,
            optimizer,
            scheduler,
            writer,
            args.grad_clip,
        )

    save_checkpoint(args, method)
    print("Done!")


if __name__ == "__main__":
    main()
