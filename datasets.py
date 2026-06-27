import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose, RandomHorizontalFlip, RandomResizedCrop, Normalize

def make_two_view_transform(transform):
    def two_view(x):
        return transform(x), transform(x)
    return two_view

def load_dataset(name, train_transform, test_transform):
    if name == "cifar10":
        dataset = torchvision.datasets.CIFAR10
        training_data = dataset(
            root="data",
            train=True,
            download=True,
            transform=train_transform
        )
        test_data = dataset(
            root="data",
            train=False,
            download=True,
            transform=test_transform
        )
    elif name == "imagenette":
        dataset = torchvision.datasets.Imagenette
        training_data = dataset(
            root="data",
            split="train",
            size="160px",
            download=True,
            transform=train_transform
        )
        test_data = dataset(
            root="data",
            split="val",
            size="160px",
            download=True,
            transform=test_transform
        )
    elif name == "imagenet-ilsvrc2012":
        # dataset 파일 수동 다운로드 필요 (https://image-net.org/download.php)
        dataset = torchvision.datasets.ImageNet
    else:
        print("dataset name is wrong")
        return None
    return training_data, test_data

def get_dataloader(batch_size, dataset_name, method='supervised'):
    
    crop_size = 32 if dataset_name == "cifar10" else 160

    train_transform = Compose([
        RandomHorizontalFlip(),
        RandomResizedCrop(crop_size),
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    test_transform = Compose([
        RandomResizedCrop(crop_size),
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    if method != 'supervised':
        ## supervised가 아닐 경우, ((x1, x2), y) 형태로 반환
        train_transform = make_two_view_transform(train_transform)
        test_transform = make_two_view_transform(test_transform)

    training_data, test_data = load_dataset(dataset_name, train_transform, test_transform)
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    num_classes = len(training_data.classes)

    for X, y in test_dataloader:
        if method == 'supervised':
            print(f"Shape of X [B, C, H, W]: {X.shape}")
        else:
            print(f"Shape of X1 [B, C, H, W]: {X[0].shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break
    
    return train_dataloader, test_dataloader, num_classes
