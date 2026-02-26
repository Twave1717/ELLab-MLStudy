import architecture

import os
import argparse
from datetime import datetime

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, RandomHorizontalFlip, RandomCrop, Normalize
from torch.utils.tensorboard import SummaryWriter


def get_dataloader(batch_size: int):
    training_data = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=Compose([
            RandomHorizontalFlip(),
            RandomCrop(32, padding=4),
            ToTensor(),
            Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    )
    test_data = datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=Compose([
            ToTensor(),
            Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    )
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    return train_dataloader, test_dataloader


def train(steps, train_dataloader, test_dataloader, device, model, loss_fn, optimizer, scheduler, tensorboard_writer):
    model.train()
    running_loss = 0
    running_step = 0

    while running_step < steps:
        for X, y in train_dataloader:
            if running_step > steps:
                break
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            running_loss += loss.item()
            running_step += 1

            if running_step % 100 == 0:
                avg_loss = running_loss / 100
                print(f"Step [{running_step}/{steps}] Loss: {avg_loss:.4f}")
                tensorboard_writer.add_scalar('Loss/train', avg_loss, running_step)
                running_loss = 0.0
            
            if running_step % 1000 == 0:
                test(running_step, test_dataloader, device, model, loss_fn, tensorboard_writer)
                model.train()


def test(steps, dataloader, device, model, loss_fn, tensorboard_writer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"{steps} Steps Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    tensorboard_writer.add_scalar("Loss/test", test_loss, steps)
    tensorboard_writer.add_scalar("Accuracy/test", correct * 100, steps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--steps', type=int, default=64000)
    parser.add_argument('--model', type=str, default="resnet-20")
    parser.add_argument('--save_path', type=str, default="checkpoint")
    args = parser.parse_args()

    train_dataloader, test_dataloader = get_dataloader(args.batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using {device} device")
    
    model_name, n = args.model.split('-')
    if model_name == 'resnet':
        model = architecture.Resnet(int(n)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[32000,48000], gamma=0.1)
    tensorboard_writer = SummaryWriter()

    train(
        args.steps,
        train_dataloader,
        test_dataloader,
        device,
        model,
        loss_fn,
        optimizer,
        scheduler,
        tensorboard_writer
    )
    print("Done!")

    if args.save_path:
        os.makedirs(args.save_path, exist_ok=True)
        filename = f"{args.model}-{args.steps}-{datetime.now().strftime("%m%d_%H%M")}.pth"
        torch.save(model.state_dict(), f"{args.save_path}/{filename}")
        print(f"Saved PyTorch Model State to {args.save_path}/{filename}")


if __name__ == "__main__":
    main()