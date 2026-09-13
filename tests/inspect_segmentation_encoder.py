from models.unetplusplus import (
    UNetPlusPlusEfficientNetB1
)


def main():

    model = UNetPlusPlusEfficientNetB1(
        pretrained=True
    )

    print()
    print("EfficientNet-B1 Encoder")
    print("=======================")

    print()

    for name, module in model.model.encoder.named_children():
        print(
            f"{name:25s} "
            f"{module.__class__.__name__}"
        )

    print()
    print("Detailed Encoder Modules")
    print("========================")

    for name, module in model.model.encoder.named_modules():

        if "block" in name.lower():
            print(name)


if __name__ == "__main__":
    main()