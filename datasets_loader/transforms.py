"""
Image Transformations
"""

import albumentations as A

from albumentations.pytorch import ToTensorV2


IMAGENET_MEAN = (
    0.485,
    0.456,
    0.406,
)

IMAGENET_STD = (
    0.229,
    0.224,
    0.225,
)


def get_train_transforms():

    return A.Compose(

        [

            A.Resize(
                224,
                224,
            ),

            A.HorizontalFlip(
                p=0.5,
            ),

            A.VerticalFlip(
                p=0.5,
            ),

            A.Rotate(
                limit=20,
                p=0.5,
            ),

            A.RandomBrightnessContrast(
                p=0.5,
            ),

            A.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),

            ToTensorV2(),

        ]

    )


def get_validation_transforms():

    return A.Compose(

        [

            A.Resize(
                224,
                224,
            ),

            A.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),

            ToTensorV2(),

        ]

    )