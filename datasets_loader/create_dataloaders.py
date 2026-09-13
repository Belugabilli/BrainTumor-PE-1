"""
create_dataloaders.py

Creates train and validation DataLoaders for
the BRISC2025 Classification Dataset.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from datasets_loader.classification_dataset import (
    BrainTumorClassificationDataset,
)

from datasets_loader.transforms import (
    get_train_transforms,
    get_validation_transforms,
)


def create_dataloaders(config):
    """
    Create train and validation dataloaders.

    Returns
    -------
    train_loader : DataLoader
    val_loader : DataLoader
    """

    train_dataset = BrainTumorClassificationDataset(
        root_dir=config.get(
            "dataset",
            "train_dir",
        ),
        transform=get_train_transforms(),
    )

    validation_dataset = BrainTumorClassificationDataset(
        root_dir=config.get(
            "dataset",
            "train_dir",
        ),
        transform=get_validation_transforms(),
    )

    validation_split = config.get(
        "training",
        "validation_split",
    )

    total_size = len(train_dataset)

    validation_size = int(
        total_size * validation_split
    )

    train_size = total_size - validation_size

    generator = torch.Generator().manual_seed(
        config["seed"]
    )

    train_indices, validation_indices = random_split(
        range(total_size),
        [train_size, validation_size],
        generator=generator,
    )

    train_subset = torch.utils.data.Subset(
        train_dataset,
        train_indices.indices,
    )

    validation_subset = torch.utils.data.Subset(
        validation_dataset,
        validation_indices.indices,
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=config.get(
            "training",
            "batch_size",
        ),
        shuffle=True,
        num_workers=config.get(
            "dataset",
            "num_workers",
        ),
        pin_memory=False,
        persistent_workers=False,
    )

    validation_loader = DataLoader(
        validation_subset,
        batch_size=config.get(
            "training",
            "batch_size",
        ),
        shuffle=False,
        num_workers=config.get(
            "dataset",
            "num_workers",
        ),
        pin_memory=False,
        persistent_workers=False,
    )

    return (
        train_loader,
        validation_loader,
    )