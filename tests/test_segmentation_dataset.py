from datasets_loader.segmentation_dataset import (
    BRISCSegmentationDataset
)


def main():

    train_dir = (
        "datasets/BRISC2025/"
        "segmentation_task/train"
    )

    dataset = BRISCSegmentationDataset(
        root_dir=train_dir,
        image_size=512,
    )

    print()
    print("Segmentation Dataset Test")
    print("--------------------------")
    print("Total Samples :", len(dataset))

    image, mask = dataset[0]

    print("Image Shape   :", image.shape)
    print("Mask Shape    :", mask.shape)

    print(
        "Image Range   :",
        image.min().item(),
        "to",
        image.max().item()
    )

    print(
        "Mask Values   :",
        mask.unique().tolist()
    )


if __name__ == "__main__":
    main()