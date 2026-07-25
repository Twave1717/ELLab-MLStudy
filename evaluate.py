import sys

import torch
from torch.utils.tensorboard import SummaryWriter

import architecture
from architecture.rotnet import RotNet
from datasets import get_dataloader
from datasets.vision.utils import GLOBAL_SEED
from eval import RotNetNonLinearEval
from methods.clip import CLIP
from train import train


def evaluate_clip(dataset, device):
    _, _, test_loader, _ = get_dataloader(128, dataset, "clip")
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


def evaluate_rotnet(checkpoint, device):
    train_loader, validation_loader, test_loader, _ = get_dataloader(128, "cifar10")
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


def main():
    torch.manual_seed(GLOBAL_SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if sys.argv[1] == "clip":
        evaluate_clip(sys.argv[2], device)
    else:
        evaluate_rotnet(sys.argv[2], device)


if __name__ == "__main__":
    main()
