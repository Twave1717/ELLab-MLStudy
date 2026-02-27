import architecture

import os
import argparse
from datetime import datetime

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, RandomHorizontalFlip, RandomResizedCrop, Normalize
from torch.utils.tensorboard import SummaryWriter


def get_dataloader(batch_size, dataset_name):
    dataset_class = getattr(datasets, dataset_name)
    training_data = dataset_class(
        root="data",
        train=True,
        download=True,
        transform=Compose([
            RandomHorizontalFlip(),
            RandomResizedCrop(32),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    )
    test_data = dataset_class(
        root="data",
        train=False,
        download=True,
        transform=Compose([
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    )
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    num_classes = len(training_data.classes)

    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    return train_dataloader, test_dataloader, num_classes


def train(epochs, train_dataloader, test_dataloader, device, model, loss_fn, optimizer, scheduler, tensorboard_writer):
    global_step = -1
    for epoch in range(1, epochs+1):
        print(f"Epoch {epoch}\n-------------------------------")
        model.train()
        for X, y in train_dataloader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            
            global_step += 1
            if global_step % 100 == 0:
                print(f"Step [{global_step}/{len(train_dataloader)*epochs}] Loss: {loss.item():.4f}")
                tensorboard_writer.add_scalar('Loss/train', loss.item(), global_step)

        scheduler.step()
        test(epoch, test_dataloader, device, model, loss_fn, tensorboard_writer)


def test(epoch, dataloader, device, model, loss_fn, tensorboard_writer):
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
    print(f"{epoch} Epochs Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    tensorboard_writer.add_scalar("Loss/test", test_loss, epoch)
    tensorboard_writer.add_scalar("Accuracy/test", correct * 100, epoch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--model', type=str, default="resnet-20")
    parser.add_argument('--dataset_name', type=str, default="CIFAR10")
    parser.add_argument('--save_path', type=str, default="checkpoint")
    args = parser.parse_args()

    train_dataloader, test_dataloader, num_classes = get_dataloader(args.batch_size, args.dataset_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using {device} device")
    
    model_name, n = args.model.split('-')
    if model_name == 'resnet':
        model = architecture.Resnet(int(n), num_classes).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.epochs // 2, args.epochs * 3 // 4], gamma=0.1)
    tensorboard_writer = SummaryWriter()

    train(
        args.epochs,
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
        filename = f"{args.model}-{args.epochs}-{datetime.now().strftime('%m%d_%H%M')}.pth"
        torch.save(model.state_dict(), f"{args.save_path}/{filename}")
        print(f"Saved PyTorch Model State to {args.save_path}/{filename}")


if __name__ == "__main__":
    main()