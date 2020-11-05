from typing import Tuple, Union

import pathlib

import torch
import torchvision
import yacs.config

from torch.utils.data import Dataset

from pytorch_image_classification import create_transform


class SubsetDataset(Dataset):
    def __init__(self, subset_dataset, transform=None):
        self.subset_dataset = subset_dataset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset_dataset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset_dataset)


def create_dataset(config: yacs.config.CfgNode,
                   is_train: bool) -> Union[Tuple[Dataset, Dataset], Dataset]:
    if config.dataset.name in [
            'CIFAR10',
            'CIFAR100',
            'MNIST',
            'FashionMNIST',
            'KMNIST',
    ]:  # getattr(name, attribute) get the named attribute from the object, which is equivalent to create
        # an alias of the callable dataset for efficient configuration
        module = getattr(torchvision.datasets, config.dataset.name)
        if is_train:
            # use test set for validation
            if config.train.use_test_as_val:
                # The returned value will be a composed transform
                train_transform = create_transform(config, is_train=True)
                val_transform = create_transform(config, is_train=False)
                # efficient configuration
                train_dataset = module(config.dataset.dataset_dir,
                                       train=is_train,
                                       transform=train_transform,
                                       download=True)
                test_dataset = module(config.dataset.dataset_dir,
                                      train=False,
                                      transform=val_transform,
                                      download=True)
                return train_dataset, test_dataset
            else:
                # split training set into training and validation
                dataset = module(config.dataset.dataset_dir,
                                 train=is_train,
                                 transform=None,
                                 download=True)
                val_ratio = config.train.val_ratio
                assert val_ratio < 1
                val_num = int(len(dataset) * val_ratio)
                train_num = len(dataset) - val_num
                lengths = [train_num, val_num]
                train_subset, val_subset = torch.utils.data.dataset.random_split(
                    dataset, lengths)
                # After splitting, transform will be executed.
                train_transform = create_transform(config, is_train=True)
                val_transform = create_transform(config, is_train=False)
                train_dataset = SubsetDataset(train_subset, train_transform)
                val_dataset = SubsetDataset(val_subset, val_transform)
                return train_dataset, val_dataset
        else:
            transform = create_transform(config, is_train=False)
            dataset = module(config.dataset.dataset_dir,
                             train=is_train,
                             transform=transform,
                             download=True)
            return dataset
    elif config.dataset.name == 'ImageNet':
        dataset_dir = pathlib.Path(config.dataset.dataset_dir).expanduser()
        train_transform = create_transform(config, is_train=True)
        val_transform = create_transform(config, is_train=False)
        train_dataset = torchvision.datasets.ImageFolder(
            dataset_dir / 'train', transform=train_transform)
        val_dataset = torchvision.datasets.ImageFolder(dataset_dir / 'val',
                                                       transform=val_transform)
        return train_dataset, val_dataset
    else:
        raise ValueError()
