import architecture

import os
import argparse
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

from datasets import get_dataloader
from methods.supervised_learning import SupervisedLearning

def train(epochs, train_dataloader, test_dataloader, device, method, optimizer, scheduler, tensorboard_writer, grad_clip=None):
    global_step = 0

    for epoch in range(1, epochs+1):
        print(f"Epoch {epoch}\n-------------------------------")
        method.train()

        for batch in train_dataloader:
            optimizer.zero_grad()

            loss = method.training_step(batch, device)
            loss.backward()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(method.parameters(), grad_clip)

            optimizer.step()
            
            if global_step % 100 == 0:
                print(f"Step [{global_step}/{len(train_dataloader)*epochs}] Loss: {loss.item():.4f}")
                tensorboard_writer.add_scalar('Loss/train', loss.item(), global_step)

            global_step += 1
            
        scheduler.step()
        test(epoch, test_dataloader, device, method, tensorboard_writer)


def test(epoch, dataloader, device, method, tensorboard_writer):
    method.eval()
    total_loss, total_correct, total_size = 0, 0, 0
    with torch.no_grad():
        for batch in dataloader:
            loss, correct, batch_size = method.validation_step(batch, device)
            total_loss += loss * batch_size
            total_correct += correct
            total_size += batch_size
    test_loss = total_loss / total_size
    accuracy = total_correct / total_size
    print(f"{epoch} Epochs Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    tensorboard_writer.add_scalar("Loss/test", test_loss, epoch)
    tensorboard_writer.add_scalar("Accuracy/test", accuracy * 100, epoch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--model', type=str, default="resnet-20")
    parser.add_argument('--method', type=str, default="supervised")
    parser.add_argument('--dataset', type=str, default="cifar10")
    parser.add_argument('--save_path', type=str, default="checkpoint")
    parser.add_argument('--scheduler', type=str, default="multistep", choices=["multistep", "cosine"])
    parser.add_argument('--grad_clip', type=float, default=None)
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--pretrained', action='store_true')
    args = parser.parse_args()

    train_dataloader, test_dataloader, num_classes = get_dataloader(args.batch_size, args.dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using {device} device")


    ## choose architecture
    model_name, n = args.model.split('-')
    lr = args.lr
    if model_name == 'resnet':
        model = architecture.ResNet(int(n), num_classes).to(device)
    elif model_name == 'preactresnet':
        model = architecture.PreActResNet(int(n), num_classes).to(device)
    elif model_name == "densenet":
        model = architecture.DenseNet(int(n), num_classes).to(device)
    elif model_name == "fractalnet":
        model = architecture.FractalNet(int(n), num_classes).to(device)
    elif model_name == "fractalnet_droppath":
        model = architecture.FractalNetDropPath(int(n), num_classes).to(device)
    elif model_name == "mlp_mixer":
        image_size = 448 if args.pretrained else 224
        model = architecture.MLPMixer(num_classes, num_layers=int(n), image_size=image_size)
        if args.pretrained:
            model.load_mixer_b16_google_imagenet21k()
            print("pre-trained mlp mixer B/16 is loaded")
        model = model.to(device)
    elif model_name == "vit_pretrained":
        model = architecture.VisionTransformer(int(n), num_classes).to(device)


    ## choose learning method
    method_name = args.method
    if method_name == 'supervised':
        method = SupervisedLearning(encoder=model, num_classes=num_classes).to(device)


    ## get optimizer & scheduler
    optimizer = torch.optim.SGD(method.parameters(), lr=lr, momentum=0.9, weight_decay=args.weight_decay)
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == "multistep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.epochs // 2, args.epochs * 3 // 4], gamma=0.1)


    ## start training
    tensorboard_writer = SummaryWriter(log_dir=args.log_dir)
    if args.log_dir:
        print(f"TensorBoard log dir: {args.log_dir}")

    try:
        train(
            args.epochs,
            train_dataloader,
            test_dataloader,
            device,
            method,
            optimizer,
            scheduler,
            tensorboard_writer,
            grad_clip=args.grad_clip
        )
    finally:
        tensorboard_writer.close()
        print("Done!")


    ## finish training
    if args.save_path:
        run_name = f"{args.model}-{args.method}-{args.epochs}-{datetime.now().strftime('%m%d_%H%M')}"
        save_dir = os.path.join(args.save_path, run_name)
        os.makedirs(save_dir, exist_ok=True)

        encoder_path = os.path.join(save_dir, "encoder.pth")
        method_path = os.path.join(save_dir, "method.pth")

        torch.save(method.encoder.state_dict(), encoder_path)
        torch.save(method.state_dict(), method_path)
        print(f"Saved encoder state to {encoder_path}")
        print(f"Saved method state to {method_path}")


if __name__ == "__main__":
    main()
