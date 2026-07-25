from .vision.caltech101 import DATASET as CALTECH101
from .vision.cifar10 import DATASET as CIFAR10
from .vision.dtd import DATASET as DTD
from .vision.eurosat import DATASET as EUROSAT
from .vision.fgvc import DATASET as FGVC
from .vision.food101 import DATASET as FOOD101
from .vision.imagenet import DATASET as IMAGENET
from .vision.imagenette import DATASET as IMAGENETTE
from .vision.oxford_flowers import DATASET as OXFORD_FLOWERS
from .vision.oxford_pets import DATASET as OXFORD_PETS
from .vision.stanford_cars import DATASET as STANFORD_CARS
from .vision.sun397 import DATASET as SUN397
from .vision.ucf101 import DATASET as UCF101


DATASETS = {
    "caltech101": CALTECH101,
    "cifar10": CIFAR10,
    "dtd": DTD,
    "eurosat": EUROSAT,
    "fgvc": FGVC,
    "food101": FOOD101,
    "imagenet": IMAGENET,
    "imagenette": IMAGENETTE,
    "imagenet-ilsvrc2012": IMAGENET,
    "oxford_flowers": OXFORD_FLOWERS,
    "oxford_pets": OXFORD_PETS,
    "stanford_cars": STANFORD_CARS,
    "sun397": SUN397,
    "ucf101": UCF101,
}
