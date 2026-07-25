import sys
import torch
from torch.utils.tensorboard import SummaryWriter
from architecture.rotnet import RotNet
from datasets import get_dataloader
from eval import RotNetNonLinearEval
from train import train

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, test_loader, _ = get_dataloader(128, "cifar10")
    encoder = RotNet(4)
    encoder.load_state_dict(torch.load(sys.argv[1], weights_only=True))
    model = RotNetNonLinearEval(encoder, 10).to(device)
    optimizer = torch.optim.SGD(model.classifier.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [100, 150], gamma=0.1)
    with SummaryWriter() as writer:
        train(200, train_loader, test_loader, device, model, optimizer, scheduler, writer)

if __name__ == "__main__":
    main()
