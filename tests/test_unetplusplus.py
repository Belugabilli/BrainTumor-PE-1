import torch

from models.unetplusplus import (
    UNetPlusPlusEfficientNetB1
)


def main():

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print()
    print("U-Net++ Model Test")
    print("------------------")
    print("Device :", device)

    model = UNetPlusPlusEfficientNetB1(
        in_channels=3,
        classes=1,
        pretrained=True,
    )

    model = model.to(device)

    # One 512x512 segmentation image
    x = torch.randn(
        1,
        3,
        512,
        512,
        device=device,
    )

    with torch.no_grad():

        output = model(x)

    print("Input Shape  :", x.shape)
    print("Output Shape :", output.shape)

    print(
        "Parameters   :",
        sum(
            p.numel()
            for p in model.parameters()
        )
    )

    print(
        "Trainable    :",
        sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )
    )


if __name__ == "__main__":
    main()