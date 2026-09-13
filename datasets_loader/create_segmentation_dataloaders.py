import torch
from torch.utils.data import DataLoader, Dataset, random_split

from datasets_loader.segmentation_dataset import (
    BRISCSegmentationDataset
)


class SegmentationSubset(Dataset):

    def __init__(
        self,
        dataset,
        indices
    ):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]


def create_segmentation_dataloaders(
    root_dir,
    image_size=512,
    batch_size=8,
    num_workers=0,
    seed=42,
):

    train_root = f"{root_dir}/train"
    test_root = f"{root_dir}/test"

    # -------------------------------------------------
    # Official BRISC2025 TRAIN set
    # -------------------------------------------------

    train_dataset_plain = BRISCSegmentationDataset(
        root_dir=train_root,
        image_size=image_size,
        augment=False,
    )

    # Separate copy with training augmentation enabled
    train_dataset_augmented = BRISCSegmentationDataset(
        root_dir=train_root,
        image_size=image_size,
        augment=True,
    )

    # -------------------------------------------------
    # Official BRISC2025 TEST set
    # -------------------------------------------------

    test_dataset = BRISCSegmentationDataset(
        root_dir=test_root,
        image_size=image_size,
        augment=False,
    )

    # -------------------------------------------------
    # Train / Validation split
    #
    # IMPORTANT:
    # Split ONLY the official training set.
    # Official test set remains completely untouched.
    # -------------------------------------------------

    total_train = len(train_dataset_plain)

    validation_size = int(0.15 * total_train)
    training_size = total_train - validation_size

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_subset_indices, val_subset_indices = random_split(
        range(total_train),
        [training_size, validation_size],
        generator=generator,
    )

    # random_split returns Subset objects.
    train_indices = train_subset_indices.indices
    val_indices = val_subset_indices.indices

    # -------------------------------------------------
    # Training dataset
    # Augmentation ON
    # -------------------------------------------------

    train_dataset = SegmentationSubset(
        train_dataset_augmented,
        train_indices,
    )

    # -------------------------------------------------
    # Validation dataset
    # Augmentation OFF
    # -------------------------------------------------

    val_dataset = SegmentationSubset(
        train_dataset_plain,
        val_indices,
    )

    # -------------------------------------------------
    # Test dataset
    # Official BRISC2025 test set
    # Augmentation OFF
    # -------------------------------------------------

    # -------------------------------------------------
    # DataLoaders
    # -------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    # -------------------------------------------------
    # Dataset information
    # -------------------------------------------------

    print()
    print("Segmentation Dataset")
    print("--------------------")
    print("Official Train :", total_train)
    print("Training       :", len(train_dataset))
    print("Validation     :", len(val_dataset))
    print("Official Test  :", len(test_dataset))

    print()
    print("Data Leakage Check")
    print("------------------")
    print("Training uses official train set only.")
    print("Validation uses official train set only.")
    print("Testing uses official test set only.")
    print()

    return (
        train_loader,
        val_loader,
        test_loader,
    )