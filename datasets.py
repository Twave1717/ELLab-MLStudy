import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose, RandomHorizontalFlip, RandomResizedCrop, Normalize

def load_dataset(name):
    if name == "cifar10":
        dataset = torchvision.datasets.CIFAR10
    else:
        print("dataset name is wrong")
        return None
    return dataset

def get_dataloader(batch_size, dataset_name):
    dataset_class = load_dataset(dataset_name)
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
        print(f"Shape of X [B, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break
    
    return train_dataloader, test_dataloader, num_classes