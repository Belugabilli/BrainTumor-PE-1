"""
classification_dataset.py

Dataset loader for the BRISC2025 Brain Tumor Classification dataset.

Classes:
    0 -> glioma
    1 -> meningioma
    2 -> no_tumor
    3 -> pituitary

Author:
Paper Reproduction Project
"""

from pathlib import Path
from typing import List, Tuple

from PIL import Image
import numpy as np

from torch.utils.data import Dataset


CLASS_TO_INDEX = {
    "glioma": 0,
    "meningioma": 1,
    "no_tumor": 2,
    "pituitary": 3,
}


class BrainTumorClassificationDataset(Dataset):
    """
    PyTorch Dataset for BRISC2025 Classification Task.
    """

    def __init__(
        self,
        root_dir: str,
        transform=None,
    ) -> None:

        self.root_dir = Path(root_dir)
        self.transform = transform

        self.image_paths: List[Path] = []
        self.labels: List[int] = []

        self._load_dataset()

    def _load_dataset(self) -> None:

        for class_name, label in CLASS_TO_INDEX.items():

            class_folder = self.root_dir / class_name

            if not class_folder.exists():
                continue

            for image_path in sorted(class_folder.glob("*")):

                if image_path.suffix.lower() in [
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".bmp",
                    ".tif",
                    ".tiff",
                ]:

                    self.image_paths.append(image_path)
                    self.labels.append(label)

    def __len__(self) -> int:

        return len(self.image_paths)

    def __getitem__(
        self,
        index: int,
    ) -> Tuple:

        image = np.array(
            Image.open(
                self.image_paths[index]
            ).convert("RGB")
        )

        label = self.labels[index]

        if self.transform:

            image = self.transform(image=image)["image"]

        return image, label