import torch

from losses.segmentation_loss import (
    DiceLoss,
    FocalLoss,
    CombinedSegmentationLoss,
)


def main():

    print()
    print("Segmentation Loss Test")
    print("======================")

    batch_size = 4
    height = 512
    width = 512

    logits = torch.randn(
        batch_size,
        1,
        height,
        width
    )

    targets = torch.randint(
        0,
        2,
        (
            batch_size,
            1,
            height,
            width
        )
    ).float()

    dice = DiceLoss()
    focal = FocalLoss()
    combined = CombinedSegmentationLoss()

    dice_value = dice(
        logits,
        targets
    )

    focal_value = focal(
        logits,
        targets
    )

    combined_value = combined(
        logits,
        targets
    )

    print()
    print("Logits Shape  :", logits.shape)
    print("Target Shape  :", targets.shape)

    print()
    print("Dice Loss     :", dice_value.item())
    print("Focal Loss    :", focal_value.item())
    print("Combined Loss :", combined_value.item())

    assert torch.isfinite(dice_value)
    assert torch.isfinite(focal_value)
    assert torch.isfinite(combined_value)

    print()
    print("PASS: All losses are finite.")


if __name__ == "__main__":
    main()