import matplotlib.pyplot as plt

from datasets_loader.segmentation_dataset import (
    BRISCSegmentationDataset
)


def main():

    dataset = BRISCSegmentationDataset(
        root_dir="datasets/BRISC2025/segmentation_task/train",
        image_size=224,
    )

    image, mask = dataset[0]

    image = image.permute(1, 2, 0).numpy()
    mask = mask.squeeze(0).numpy()

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("MRI Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(image)
    plt.imshow(mask, alpha=0.5)
    plt.title("MRI + Ground Truth Mask")
    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        "results/segmentation_sample.png",
        dpi=300
    )

    plt.show()

    print()
    print("Saved:")
    print("results/segmentation_sample.png")


if __name__ == "__main__":
    main()