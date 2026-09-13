import numpy as np
import torch
from scipy.ndimage import (
    binary_erosion,
    distance_transform_edt,
)
from tqdm import tqdm

from datasets_loader.create_segmentation_dataloaders import (
    create_segmentation_dataloaders,
)

from models.unetplusplus import (
    UNetPlusPlusEfficientNetB1,
)


def surface_distances(mask_a, mask_b):
    """
    Calculate distances between the surfaces of two binary masks.
    """

    mask_a = mask_a.astype(bool)
    mask_b = mask_b.astype(bool)

    if not mask_a.any() or not mask_b.any():
        return None

    surface_a = mask_a ^ binary_erosion(
        mask_a
    )

    surface_b = mask_b ^ binary_erosion(
        mask_b
    )

    distance_b = distance_transform_edt(
        ~surface_b
    )

    distance_a = distance_transform_edt(
        ~surface_a
    )

    distances_a_to_b = distance_b[
        surface_a
    ]

    distances_b_to_a = distance_a[
        surface_b
    ]

    return np.concatenate(
        [
            distances_a_to_b,
            distances_b_to_a,
        ]
    )


def calculate_hd95(
    prediction,
    target,
):
    """
    Calculate 95th-percentile Hausdorff distance.
    """

    prediction = prediction.astype(bool)
    target = target.astype(bool)

    # Both empty
    if not prediction.any() and not target.any():
        return 0.0

    # One empty, one non-empty
    if not prediction.any() or not target.any():
        return float("inf")

    distances = surface_distances(
        prediction,
        target,
    )

    if distances is None:
        return float("inf")

    return float(
        np.percentile(
            distances,
            95,
        )
    )


def calculate_dice(
    prediction,
    target,
    smooth=1e-6,
):

    prediction = prediction.astype(bool)
    target = target.astype(bool)

    intersection = np.logical_and(
        prediction,
        target,
    ).sum()

    return (
        2.0 * intersection + smooth
    ) / (
        prediction.sum()
        + target.sum()
        + smooth
    )


def calculate_iou(
    prediction,
    target,
    smooth=1e-6,
):

    prediction = prediction.astype(bool)
    target = target.astype(bool)

    intersection = np.logical_and(
        prediction,
        target,
    ).sum()

    union = np.logical_or(
        prediction,
        target,
    ).sum()

    return (
        intersection + smooth
    ) / (
        union + smooth
    )


def main():

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print()
    print("Segmentation Metrics")
    print("====================")
    print("Device :", device)

    # -------------------------------------------------
    # Test Data
    # -------------------------------------------------

    _, _, test_loader = (
        create_segmentation_dataloaders(
            root_dir=(
                "datasets/BRISC2025/"
                "segmentation_task"
            ),
            image_size=512,
            batch_size=8,
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

    print()
    print(
        "Loaded Best Model - Epoch",
        checkpoint["epoch"],
    )

    # -------------------------------------------------
    # Metrics
    # -------------------------------------------------

    all_dice = []
    all_iou = []
    all_hd95 = []

    with torch.no_grad():

        progress = tqdm(
            test_loader,
            desc="Calculating Metrics",
        )

        for images, masks in progress:

            images = images.to(device)

            logits = model(images)

            probabilities = torch.sigmoid(
                logits
            )

            predictions = (
                probabilities >= 0.5
            ).cpu().numpy()

            targets = (
                masks
                .cpu()
                .numpy()
            )

            for prediction, target in zip(
                predictions,
                targets,
            ):

                prediction = prediction[0]
                target = target[0]

                dice = calculate_dice(
                    prediction,
                    target,
                )

                iou = calculate_iou(
                    prediction,
                    target,
                )

                hd95 = calculate_hd95(
                    prediction,
                    target,
                )

                all_dice.append(dice)
                all_iou.append(iou)
                all_hd95.append(hd95)

    # -------------------------------------------------
    # Remove infinite HD95 cases
    # -------------------------------------------------

    finite_hd95 = [
        value
        for value in all_hd95
        if np.isfinite(value)
    ]

    # -------------------------------------------------
    # Final Results
    # -------------------------------------------------

    mean_dice = np.mean(
        all_dice
    )

    mean_iou = np.mean(
        all_iou
    )

    mean_hd95 = np.mean(
        finite_hd95
    )

    print()
    print("=" * 60)
    print("FINAL SEGMENTATION METRICS")
    print("=" * 60)

    print(
        f"Dice : {mean_dice:.4f}"
    )

    print(
        f"IoU  : {mean_iou:.4f}"
    )

    print(
        f"HD95 : {mean_hd95:.4f} pixels"
    )

    print(
        f"Test Samples : {len(all_dice)}"
    )

    print(
        f"Finite HD95  : {len(finite_hd95)}"
    )

    print("=" * 60)

    # -------------------------------------------------
    # Save
    # -------------------------------------------------

    with open(
        "results/segmentation_metrics.txt",
        "w",
    ) as file:

        file.write(
            "Final Segmentation Metrics\n"
        )

        file.write(
            "==========================\n\n"
        )

        file.write(
            f"Best Epoch : "
            f"{checkpoint['epoch']}\n"
        )

        file.write(
            f"Validation Dice : "
            f"{checkpoint['val_dice']:.4f}\n\n"
        )

        file.write(
            f"Test Dice : "
            f"{mean_dice:.4f}\n"
        )

        file.write(
            f"Test IoU : "
            f"{mean_iou:.4f}\n"
        )

        file.write(
            f"Test HD95 : "
            f"{mean_hd95:.4f} pixels\n"
        )

        file.write(
            f"Test Samples : "
            f"{len(all_dice)}\n"
        )

    print()
    print("Saved:")
    print(
        "results/segmentation_metrics.txt"
    )


if __name__ == "__main__":
    main()