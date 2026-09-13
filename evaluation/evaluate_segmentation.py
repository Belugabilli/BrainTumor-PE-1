import os
import csv
import torch
from tqdm import tqdm

from datasets_loader.create_segmentation_dataloaders import (
    create_segmentation_dataloaders
)

from models.unetplusplus import (
    UNetPlusPlusEfficientNetB1
)

from losses.segmentation_loss import (
    CombinedSegmentationLoss
)


def calculate_dice(logits, targets, threshold=0.5, smooth=1e-6):

    probabilities = torch.sigmoid(logits)

    predictions = (
        probabilities >= threshold
    ).float()

    predictions = predictions.view(
        predictions.size(0), -1
    )

    targets = targets.view(
        targets.size(0), -1
    )

    intersection = (
        predictions * targets
    ).sum(dim=1)

    dice = (
        2 * intersection + smooth
    ) / (
        predictions.sum(dim=1)
        + targets.sum(dim=1)
        + smooth
    )

    return dice


def calculate_iou(logits, targets, threshold=0.5, smooth=1e-6):

    probabilities = torch.sigmoid(logits)

    predictions = (
        probabilities >= threshold
    ).float()

    predictions = predictions.view(
        predictions.size(0), -1
    )

    targets = targets.view(
        targets.size(0), -1
    )

    intersection = (
        predictions * targets
    ).sum(dim=1)

    union = (
        predictions
        + targets
        - predictions * targets
    ).sum(dim=1)

    iou = (
        intersection + smooth
    ) / (
        union + smooth
    )

    return iou


def main():

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print()
    print("Segmentation Evaluation")
    print("=======================")
    print("Device :", device)

    # -------------------------------------------------
    # Data
    # -------------------------------------------------

    _, _, test_loader = create_segmentation_dataloaders(

        root_dir=(
            "datasets/BRISC2025/"
            "segmentation_task"
        ),

        image_size=512,

        batch_size=8,

        num_workers=0,

        seed=42,
    )

    # -------------------------------------------------
    # Model
    # -------------------------------------------------

    model = UNetPlusPlusEfficientNetB1(
        in_channels=3,
        classes=1,
        pretrained=False,
    )

    model = model.to(device)

    # -------------------------------------------------
    # Load BEST checkpoint
    # -------------------------------------------------

    checkpoint_path = (
        "weights/segmentation/"
        "best_model.pth"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print()
    print(
        "Loaded checkpoint :",
        checkpoint_path
    )

    print(
        "Best Epoch        :",
        checkpoint["epoch"]
    )

    print(
        "Validation Dice   :",
        checkpoint["val_dice"]
    )

    # -------------------------------------------------
    # Evaluation
    # -------------------------------------------------

    model.eval()

    all_dice = []
    all_iou = []

    total_loss = 0.0

    criterion = CombinedSegmentationLoss(
        dice_weight=0.7,
        focal_weight=0.3,
    )

    with torch.no_grad():

        progress = tqdm(
            test_loader,
            desc="Testing"
        )

        for images, masks in progress:

            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)

            loss = criterion(
                logits,
                masks
            )

            dice = calculate_dice(
                logits,
                masks
            )

            iou = calculate_iou(
                logits,
                masks
            )

            total_loss += loss.item()

            all_dice.extend(
                dice.cpu().tolist()
            )

            all_iou.extend(
                iou.cpu().tolist()
            )

    # -------------------------------------------------
    # Final metrics
    # -------------------------------------------------

    test_loss = (
        total_loss
        / len(test_loader)
    )

    test_dice = (
        sum(all_dice)
        / len(all_dice)
    )

    test_iou = (
        sum(all_iou)
        / len(all_iou)
    )

    print()
    print("=" * 60)
    print("Segmentation Test Results")
    print("=" * 60)

    print(
        f"Test Loss : {test_loss:.4f}"
    )

    print(
        f"Test Dice : {test_dice:.4f}"
    )

    print(
        f"Test IoU  : {test_iou:.4f}"
    )

    print("=" * 60)

    # -------------------------------------------------
    # Save results
    # -------------------------------------------------

    os.makedirs(
        "results",
        exist_ok=True
    )

    result_path = (
        "results/"
        "segmentation_evaluation.txt"
    )

    with open(
        result_path,
        "w"
    ) as file:

        file.write(
            "Segmentation Evaluation\n"
        )

        file.write(
            "=======================\n\n"
        )

        file.write(
            f"Checkpoint Epoch : "
            f"{checkpoint['epoch']}\n"
        )

        file.write(
            f"Validation Dice : "
            f"{checkpoint['val_dice']:.4f}\n\n"
        )

        file.write(
            f"Test Loss : "
            f"{test_loss:.4f}\n"
        )

        file.write(
            f"Test Dice : "
            f"{test_dice:.4f}\n"
        )

        file.write(
            f"Test IoU : "
            f"{test_iou:.4f}\n"
        )

    print()
    print("Saved:")
    print(result_path)


if __name__ == "__main__":
    main()