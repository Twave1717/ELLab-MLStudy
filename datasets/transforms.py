from torchvision.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

DEFAULT_MEAN = (0.5, 0.5, 0.5)
DEFAULT_STD = (0.5, 0.5, 0.5)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
CLIP_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class TwoViewTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, image):
        return self.transform(image), self.transform(image)


def build_transforms(
    crop_size,
    clip=False,
    imagenet=False,
    two_views=False,
):
    if clip:
        crop_size = CLIP_SIZE
        mean, std = CLIP_MEAN, CLIP_STD
        interpolation = InterpolationMode.BICUBIC
    elif imagenet:
        mean, std = IMAGENET_MEAN, IMAGENET_STD
        interpolation = InterpolationMode.BILINEAR
    else:
        mean, std = DEFAULT_MEAN, DEFAULT_STD
        interpolation = InterpolationMode.BILINEAR

    train_transform = Compose([
        RandomResizedCrop(
            crop_size,
            interpolation=interpolation,
        ),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(mean, std),
    ])
    test_transform = Compose([
        Resize(crop_size, interpolation=interpolation),
        CenterCrop(crop_size),
        ToTensor(),
        Normalize(mean, std),
    ])

    if two_views:
        return TwoViewTransform(train_transform), TwoViewTransform(test_transform)
    return train_transform, test_transform
