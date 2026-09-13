"""
EfficientNet-B1 Classifier

Research Paper Reproduction
"""

from __future__ import annotations

import torch
import torch.nn as nn
import timm


class EfficientNetB1Classifier(nn.Module):
    """
    EfficientNet-B1 classifier for BRISC2025.

    Classes:
        0 -> Glioma
        1 -> Meningioma
        2 -> No Tumor
        3 -> Pituitary
    """

    def __init__(
        self,
        num_classes: int = 4,
        pretrained: bool = True,
        dropout: float = 0.3,
    ) -> None:

        super().__init__()

        self.backbone = timm.create_model(
            "efficientnet_b1",
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )

        in_features = self.backbone.num_features

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(
                in_features,
                num_classes,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        features = self.backbone(x)

        output = self.classifier(features)

        return output

    def freeze_backbone(self):

        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def unfreeze_backbone(self):

        for parameter in self.backbone.parameters():
            parameter.requires_grad = True