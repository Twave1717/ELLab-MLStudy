import argparse

import torch
from torch.utils.tensorboard import SummaryWriter

from datasets import get_dataloader
from datasets.vision.utils import GLOBAL_SEED
from src import architecture
from src.architecture.rotnet import RotNet
from src.eval import RotNetNonLinearEval
from src.methods.clip import CLIP
from train import train


def evaluate_clip(dataset, device, batch_size, data_root):
    _, _, test_loader, _ = get_dataloader(
        batch_size,
        dataset,
        "clip",
        root=data_root,
    )
    clip_model, tokenizer = architecture.load_clip(architecture.CLIP_MODEL)
    model = CLIP(
        clip_model,
        tokenizer,
        test_loader.dataset.classes,
        test_loader.dataset.template,
    ).to(device)
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            labels = labels.to(device)
            correct += (model(images.to(device)).argmax(1) == labels).sum().item()
            total += labels.size(0)
    print(f"Zero-shot accuracy: {correct / total * 100:.1f}%")


def evaluate_rotnet(checkpoint, device, batch_size, data_root):
    train_loader, validation_loader, test_loader, _ = get_dataloader(
        batch_size,
        "cifar10",
        root=data_root,
    )
    encoder = RotNet(4)
    encoder.load_state_dict(torch.load(checkpoint, weights_only=True))
    model = RotNetNonLinearEval(encoder, 10).to(device)
    optimizer = torch.optim.SGD(model.classifier.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [100, 150], gamma=0.1)
    with SummaryWriter() as writer:
        train(
            200,
            train_loader,
            validation_loader,
            test_loader,
            device,
            model,
            optimizer,
            scheduler,
            writer,
        )


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="method", required=True)

    clip_parser = subparsers.add_parser("clip")
    clip_parser.add_argument("dataset")
    clip_parser.add_argument("--batch_size", type=int, default=128)
    clip_parser.add_argument("--data_root", default="data")

    rotnet_parser = subparsers.add_parser("rotnet")
    rotnet_parser.add_argument("checkpoint")
    rotnet_parser.add_argument("--batch_size", type=int, default=128)
    rotnet_parser.add_argument("--data_root", default="data")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(GLOBAL_SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.method == "clip":
        evaluate_clip(args.dataset, device, args.batch_size, args.data_root)
    else:
        evaluate_rotnet(args.checkpoint, device, args.batch_size, args.data_root)


if __name__ == "__main__":
    main()
