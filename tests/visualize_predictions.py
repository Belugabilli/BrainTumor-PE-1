import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from datasets_loader.create_segmentation_dataloaders import (
    create_segmentation_dataloaders,
)

from models.unetplusplus import (
    UNetPlusPlusEfficientNetB1,
)


def main():

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print()
    print("Segmentation Prediction Visualization")
    print("======================================")
    print("Device :", device)

    # -------------------------------------------------
    # Test Dataset
    # -------------------------------------------------

    _, _, test_loader = (
        create_segmentation_dataloaders(
            root_dir=(
                "datasets/BRISC2025/"
                "segmentation_task"
            ),
            image_size=512,
            batch_size=1,
            num_workers=0,
            seed=42,
        )
    )

    # -------------------------------------------------
    # Model
    # -------------------------------------------------

    model = UNetPlusPlusEfficientNetB1(
        in_channels=3,
        classes=1,
        pretrained=False,
    ).to(device)

    checkpoint = torch.load(
        "weights/segmentation/best_model.pth",
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print(
        "Loaded Best Model - Epoch",
        checkpoint["epoch"],
    )

    # -------------------------------------------------
    # Output directory
    # -------------------------------------------------

    output_dir = "results/predictions"

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    # -------------------------------------------------
    # Generate predictions
    # -------------------------------------------------

    num_samples = 5

    with torch.no_grad():

        for index, (image, mask) in enumerate(
            test_loader
        ):

            if index >= num_samples:
                break

            image = image.to(device)

            logits = model(image)

            probability = torch.sigmoid(
                logits
            )

            prediction = (
                probability >= 0.5
            ).float()

            # Convert to numpy
            image_np = (
                image[0]
                .cpu()
                .numpy()
                .transpose(1, 2, 0)
            )

            mask_np = (
                mask[0, 0]
                .cpu()
                .numpy()
            )

            prediction_np = (
                prediction[0, 0]
                .cpu()
                .numpy()
            )

            # -------------------------------------------------
            # Plot
            # -------------------------------------------------

            figure, axes = plt.subplots(
                1,
                3,
                figsize=(15, 5),
            )

            axes[0].imshow(
                image_np
            )

            axes[0].set_title(
                "Original MRI"
            )

            axes[0].axis("off")

            axes[1].imshow(
                mask_np,
                cmap="gray",
            )

            axes[1].set_title(
                "Ground Truth"
            )

            axes[1].axis("off")

            axes[2].imshow(
                prediction_np,
                cmap="gray",
            )

            axes[2].set_title(
                "Prediction"
            )

            axes[2].axis("off")

            figure.suptitle(
                f"Segmentation Sample {index + 1}"
            )

            figure.tight_layout()

            output_path = os.path.join(
                output_dir,
                f"sample_{index + 1}.png",
            )

            figure.savefig(
                output_path,
                dpi=150,
                bbox_inches="tight",
            )

            plt.close(figure)

            print(
                "Saved:",
                output_path,
            )

    print()
    print(
        "Prediction visualization complete."
    )


if __name__ == "__main__":
    main()