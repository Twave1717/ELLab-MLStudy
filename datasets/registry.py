from .caltech101 import build_caltech101
from .cifar10 import build_cifar10
from .dtd import build_dtd
from .eurosat import build_eurosat
from .fgvc import build_fgvc
from .food101 import build_food101
from .imagenet import build_imagenet
from .imagenette import build_imagenette
from .oxford_flowers import build_oxford_flowers
from .oxford_pets import build_oxford_pets
from .stanford_cars import build_stanford_cars
from .sun397 import build_sun397
from .ucf101 import build_ucf101


class DatasetConfig:
    def __init__(self, builder, crop_size):
        self.builder = builder
        self.crop_size = crop_size


def get_dataset_config(name):
    DATASET_REGISTRY = {
        "caltech101": DatasetConfig(builder=build_caltech101, crop_size=224),
        "cifar10": DatasetConfig(builder=build_cifar10, crop_size=32),
        "dtd": DatasetConfig(builder=build_dtd, crop_size=224),
        "eurosat": DatasetConfig(builder=build_eurosat, crop_size=224),
        "fgvc": DatasetConfig(builder=build_fgvc, crop_size=224),
        "food101": DatasetConfig(builder=build_food101, crop_size=224),
        "imagenet": DatasetConfig(builder=build_imagenet, crop_size=224),
        "imagenette": DatasetConfig(builder=build_imagenette, crop_size=160),
        "imagenet-ilsvrc2012": DatasetConfig(builder=build_imagenet, crop_size=224),
        "oxford_flowers": DatasetConfig(builder=build_oxford_flowers, crop_size=224),
        "oxford_pets": DatasetConfig(builder=build_oxford_pets, crop_size=224),
        "stanford_cars": DatasetConfig(builder=build_stanford_cars, crop_size=224),
        "sun397": DatasetConfig(builder=build_sun397, crop_size=224),
        "ucf101": DatasetConfig(builder=build_ucf101, crop_size=224),
    }
    return DATASET_REGISTRY[name]
