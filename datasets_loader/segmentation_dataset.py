from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A


class BRISCSegmentationDataset(Dataset):

    def __init__(
        self,
        root_dir,
        image_size=512,
        augment=False,
    ):
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.augment = augment

        self.images_dir = self.root_dir / "images"
        self.masks_dir = self.root_dir / "masks"

        if not self.images_dir.exists():
            raise FileNotFoundError(
                f"Images directory not found: {self.images_dir}"
            )

        if not self.masks_dir.exists():
            raise FileNotFoundError(
                f"Masks directory not found: {self.masks_dir}"
            )

        image_paths = sorted(
            [
                p for p in self.images_dir.iterdir()
                if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
            ]
        )

        self.samples = []

        for image_path in image_paths:

            mask_path = (
                self.masks_dir /
                f"{image_path.stem}.png"
            )

            if mask_path.exists():
                self.samples.append(
                    (image_path, mask_path)
                )

        if len(self.samples) == 0:
            raise RuntimeError(
                "No matching image-mask pairs found."
            )

        # Paper: augmentation is applied only to training.
        if self.augment:
            self.transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),

                    A.VerticalFlip(p=0.5),

                    A.RandomRotate90(p=0.5),

                    A.RandomBrightnessContrast(
                        brightness_limit=0.2,
                        contrast_limit=0.0,
                        p=0.5,
                    ),

                    A.ShiftScaleRotate(
                        shift_limit=0.05,
                        scale_limit=0.10,
                        rotate_limit=15,
                        border_mode=cv2.BORDER_CONSTANT,
                        p=0.5,
                    ),

                    A.GaussNoise(
                        p=0.2,
                    ),
                ]
            )

        else:
            self.transform = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        image_path, mask_path = self.samples[index]

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR
        )

        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            raise RuntimeError(
                f"Could not read image: {image_path}"
            )

        if mask is None:
            raise RuntimeError(
                f"Could not read mask: {mask_path}"
            )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        # Paper:
        # MRI -> 512x512 using area interpolation
        image = cv2.resize(
            image,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_AREA
        )

        # Mask -> 512x512 using nearest-neighbour
        mask = cv2.resize(
            mask,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_NEAREST
        )

        if self.transform is not None:

            transformed = self.transform(
                image=image,
                mask=mask,
            )

            image = transformed["image"]
            mask = transformed["mask"]

        # Image normalization: [0,255] -> [0,1]
        image = image.astype(
            np.float32
        ) / 255.0

        # Binary mask
        mask = (
            mask > 0
        ).astype(np.float32)

        # HWC -> CHW
        image = np.transpose(
            image,
            (2, 0, 1)
        )

        image = torch.tensor(
            image,
            dtype=torch.float32
        )

        mask = torch.tensor(
            mask,
            dtype=torch.float32
        ).unsqueeze(0)

        return image, mask