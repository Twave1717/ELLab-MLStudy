from .oxford_pets import OxfordPets
from .eurosat import EuroSAT
from .ucf101 import UCF101
from .sun397 import SUN397
from .caltech101 import Caltech101
from .dtd import DescribableTextures
from .fgvc import FGVCAircraft
from .food101 import Food101
from .oxford_flowers import OxfordFlowers
from .stanford_cars import StanfordCars
from .imagenet import ImageNet

from .utils import build_data_loader
import torchvision.transforms as transforms


dataset_list = {
    "oxford_pets": OxfordPets,
    "eurosat": EuroSAT,
    "ucf101": UCF101,
    "sun397": SUN397,
    "caltech101": Caltech101,
    "dtd": DescribableTextures,
    "fgvc": FGVCAircraft,
    "food101": Food101,
    "oxford_flowers": OxfordFlowers,
    "stanford_cars": StanfordCars,
    "imagenet": ImageNet,
}


def build_dataset(dataset, root_path, shots, setting, seed):
    return dataset_list[dataset](root_path, shots, setting, seed)


def build_dataloaders(args, dataset, preprocess):
    val_loader = build_data_loader(
        data_source=dataset.val, 
        batch_size=args.test_batch_size, 
        is_train=False, 
        tfm=preprocess, 
        shuffle=False,  
        num_workers=args.workers
    )
    test_loader = build_data_loader(
        data_source=dataset.test, 
        batch_size=args.test_batch_size, 
        is_train=False, tfm=preprocess, 
        shuffle=False,  
        num_workers=args.workers
    )
    if dataset.test_new is not None:
        test_new_loader = build_data_loader(
            data_source=dataset.test_new, 
            batch_size=args.test_batch_size, 
            is_train=False, tfm=preprocess, 
            shuffle=False,  
            num_workers=args.workers
        )
        test_loader = (test_loader, test_new_loader)
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.08, 1), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])
    train_loader = build_data_loader(
        data_source=dataset.train_x,
        data_root=args.root_path, 
        batch_size=args.batch_size, 
        tfm=train_transform, 
        is_train=True, 
        shuffle=True, 
        num_workers=args.workers
    )

    return train_loader, val_loader, test_loader