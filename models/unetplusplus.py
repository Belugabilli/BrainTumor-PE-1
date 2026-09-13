import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class UNetPlusPlusEfficientNetB1(nn.Module):

    def __init__(
        self,
        in_channels=3,
        classes=1,
        pretrained=True,
    ):
        super().__init__()

        encoder_weights = "imagenet" if pretrained else None

        self.model = smp.UnetPlusPlus(
            encoder_name="efficientnet-b1",
            encoder_depth=5,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None,
        )

# Freeze the complete model first
        for param in self.model.parameters():
            param.requires_grad = False


        # EfficientNet-B1 encoder
        encoder = self.model.encoder


        # Unfreeze EfficientNet-B1 blocks 15–21
        for block_idx in range(15, 22):
            for param in encoder._blocks[block_idx].parameters():
                param.requires_grad = True


        # Keep decoder trainable
        for param in self.model.decoder.parameters():
            param.requires_grad = True


        # Keep segmentation head trainable
        for param in self.model.segmentation_head.parameters():
            param.requires_grad = True

    def forward(self, x):

        return self.model(x)