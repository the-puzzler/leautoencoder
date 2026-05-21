from pathlib import Path

from datasets import load_dataset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class HFDatasetWrapper(Dataset):
    def __init__(self, hf_dataset, transform):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        sample = self.hf_dataset[idx]
        image = sample["image"]
        if self.transform is not None:
            image = self.transform(image)
        return image, 0


def load_data(
    batch_size: int = 64,
    test_batch_size: int | None = None,
    data_dir: str | Path = "data",
    num_workers: int = 2,
    pin_memory: bool = True,
    dataset_name: str = "cifar10",
) -> tuple[DataLoader, DataLoader]:
    test_batch_size = test_batch_size or batch_size
    root = Path(data_dir)
    dataset_name = dataset_name.lower()

    if dataset_name == "cifar10":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        train_dataset = datasets.CIFAR10(
            root=root,
            train=True,
            download=True,
            transform=transform,
        )
        test_dataset = datasets.CIFAR10(
            root=root,
            train=False,
            download=True,
            transform=transform,
        )
    elif dataset_name == "celeba":
        transform = transforms.Compose(
            [
                transforms.CenterCrop(128),
                transforms.ToTensor(),
            ]
        )
        train_dataset = HFDatasetWrapper(
            load_dataset("flwrlabs/celeba", split="train", cache_dir=root / "hf"),
            transform=transform,
        )
        test_dataset = HFDatasetWrapper(
            load_dataset("flwrlabs/celeba", split="test", cache_dir=root / "hf"),
            transform=transform,
        )
    else:
        raise ValueError(f"unsupported dataset_name: {dataset_name}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, test_loader
