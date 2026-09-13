import torch

from datasets_loader.create_segmentation_dataloaders import (
    create_segmentation_dataloaders
)

from models.unetplusplus import (
    UNetPlusPlusEfficientNetB1
)

from losses.segmentation_loss import (
    CombinedSegmentationLoss
)

from engine.segmentation_optimizer import (
    create_segmentation_optimizer,
    create_segmentation_scheduler
)

from engine.segmentation_trainer import (
    SegmentationTrainer
)


def main():

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print()
    print("Segmentation Trainer Test")
    print("==========================")
    print("Device :", device)

    (
        train_loader,
        val_loader,
        _
    ) = create_segmentation_dataloaders(
        root_dir=(
            "datasets/BRISC2025/"
            "segmentation_task"
        ),
        image_size=512,
        batch_size=1,
        num_workers=0,
        seed=42,
    )

    model = UNetPlusPlusEfficientNetB1(
        pretrained=True
    ).to(device)

    criterion = CombinedSegmentationLoss()

    optimizer = create_segmentation_optimizer(
        model,
        learning_rate=1e-4,
        weight_decay=1e-5,
    )

    scheduler = create_segmentation_scheduler(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-6,
    )

    # Take only ONE batch for testing.
    images, masks = next(
        iter(train_loader)
    )

    images = images.to(device)
    masks = masks.to(device)

    print()
    print("Input Images :", images.shape)
    print("Input Masks  :", masks.shape)

    optimizer.zero_grad(
        set_to_none=True
    )

    logits = model(images)

    loss = criterion(
        logits,
        masks
    )

    loss.backward()

    optimizer.step()

    print()
    print("Logits :", logits.shape)
    print("Loss   :", loss.item())

    assert logits.shape == masks.shape
    assert torch.isfinite(loss)

    print()
    print("PASS: Forward + loss + backward + optimizer step.")


if __name__ == "__main__":
    main()