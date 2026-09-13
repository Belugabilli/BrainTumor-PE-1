import torch

from models.unetplusplus import (
    UNetPlusPlusEfficientNetB1
)

from engine.segmentation_optimizer import (
    create_segmentation_optimizer,
    create_segmentation_scheduler,
)


def main():

    print()
    print("Segmentation Optimizer Test")
    print("===========================")

    model = UNetPlusPlusEfficientNetB1(
        pretrained=True
    )

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

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    optimized_parameters = sum(
        p.numel()
        for group in optimizer.param_groups
        for p in group["params"]
    )

    print()
    print(
        "Trainable Parameters :",
        trainable_parameters
    )

    print(
        "Optimizer Parameters :",
        optimized_parameters
    )

    print(
        "Learning Rate        :",
        optimizer.param_groups[0]["lr"]
    )

    print(
        "Weight Decay         :",
        optimizer.param_groups[0]["weight_decay"]
    )

    print(
        "Scheduler             :",
        scheduler.__class__.__name__
    )

    assert (
        trainable_parameters
        == optimized_parameters
    )

    # Test scheduler
    scheduler.step()

    print(
        "LR after scheduler    :",
        optimizer.param_groups[0]["lr"]
    )

    print()
    print("PASS: Optimizer and scheduler working.")


if __name__ == "__main__":
    main()