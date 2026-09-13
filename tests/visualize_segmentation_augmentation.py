import matplotlib.pyplot as plt

from datasets_loader.create_segmentation_dataloaders import (
    create_segmentation_dataloaders
)


def main():

    train_loader, _, _ = create_segmentation_dataloaders(
        root_dir="datasets/BRISC2025/segmentation_task",
        image_size=512,
        batch_size=1,
        num_workers=0,
        seed=42,
    )

    image, mask = next(iter(train_loader))

    image = image[0].permute(1, 2, 0).numpy()
    mask = mask[0, 0].numpy()

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.imshow(mask, alpha=0.5)
    plt.title("Augmented Image + Mask")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap="gray")
    plt.title("Augmented Mask")
    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        "results/segmentation_augmentation.png",
        dpi=300,
    )

    plt.show()

    print("Saved:")
    print("results/segmentation_augmentation.png")


if __name__ == "__main__":
    main()