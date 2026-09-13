"""
PE1 Brain Tumor Classifier

Modified architecture based on the BRISC2025 reproduction baseline.
Backbone: EfficientNet-B3
"""

import timm
import torch.nn as nn


class PE1Classifier(nn.Module):

    def __init__(
        self,
        num_classes=4,
        pretrained=True,
        dropout=0.30,
        hidden_dim=512,
    ):
        super().__init__()

        self.backbone = timm.create_model(
            "efficientnet_b3",
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )

        feature_dim = self.backbone.num_features  # 1536

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(feature_dim),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)
