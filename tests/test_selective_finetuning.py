from models.unetplusplus import UNetPlusPlusEfficientNetB1


def main():

    model = UNetPlusPlusEfficientNetB1(
        pretrained=True
    )

    encoder = model.model.encoder

    print()
    print("Selective Fine-Tuning Test")
    print("==========================")

    print()
    print("Encoder Blocks:")

    for i, block in enumerate(encoder._blocks):

        trainable = any(
            param.requires_grad
            for param in block.parameters()
        )

        status = "TRAINABLE" if trainable else "FROZEN"

        print(
            f"Block {i:02d} : {status}"
        )

    print()
    print("Decoder trainable:",
          any(
              p.requires_grad
              for p in model.model.decoder.parameters()
          ))

    print()
    print("Segmentation Head trainable:",
          any(
              p.requires_grad
              for p in model.model.segmentation_head.parameters()
          ))

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print()
    print("Total Parameters     :", total)
    print("Trainable Parameters :", trainable)


if __name__ == "__main__":
    main()