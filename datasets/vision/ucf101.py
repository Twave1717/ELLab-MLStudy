import os

from torch.utils.data import Dataset
from torchvision.datasets import UCF101
from torchvision.transforms.functional import to_pil_image
from .utils import DatasetSpec


dataset_dir = "ucf101"


class UCF101FrameDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform
        self.classes = dataset.classes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        video, _, label = self.dataset[index]
        frame = to_pil_image(video[0].permute(2, 0, 1))
        return self.transform(frame), label


def build_ucf101(root, train_transform, test_transform):
    dataset_root = os.path.join(root, dataset_dir)
    video_root = os.path.join(dataset_root, "UCF-101")
    annotation_root = os.path.join(dataset_root, "ucfTrainTestlist")

    # Dataset 파일 수동 다운로드 필요: https://www.crcv.ucf.edu/data/UCF101.php
    train_dataset = UCF101(
        video_root,
        annotation_root,
        frames_per_clip=1,
        step_between_clips=10**9,
        fold=1,
        train=True,
    )
    test_dataset = UCF101(
        video_root,
        annotation_root,
        frames_per_clip=1,
        step_between_clips=10**9,
        fold=1,
        train=False,
    )
    return (
        UCF101FrameDataset(train_dataset, train_transform),
        UCF101FrameDataset(test_dataset, test_transform),
    )


DATASET = DatasetSpec(build_ucf101)
