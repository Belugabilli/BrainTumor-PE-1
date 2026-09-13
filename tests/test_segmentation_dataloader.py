from datasets_loader.create_segmentation_dataloaders import (
    create_segmentation_dataloaders
)


def main():

    (
        train_loader,
        val_loader,
        test_loader,
    ) = create_segmentation_dataloaders(
        root_dir=(
            "datasets/BRISC2025/"
            "segmentation_task"
        ),
        image_size=512,
        batch_size=8,
        num_workers=0,
        seed=42,
    )

    images, masks = next(iter(train_loader))

    print("Batch Test")
    print("----------")
    print("Images :", images.shape)
    print("Masks  :", masks.shape)

    print(
        "Image min/max :",
        images.min().item(),
        images.max().item()
    )

    print(
        "Mask values   :",
        masks.unique().tolist()
    )


if __name__ == "__main__":
    main()