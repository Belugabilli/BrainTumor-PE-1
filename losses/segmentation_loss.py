import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        logits : [B, 1, H, W]
        targets: [B, 1, H, W]
        """

        probs = torch.sigmoid(logits)

        probs = probs.contiguous().view(probs.size(0), -1)
        targets = targets.contiguous().view(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)

        dice = (
            2.0 * intersection + self.smooth
        ) / (
            probs.sum(dim=1)
            + targets.sum(dim=1)
            + self.smooth
        )

        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        """
        Binary focal loss.
        """

        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none"
        )

        probs = torch.sigmoid(logits)

        p_t = (
            probs * targets
            + (1.0 - probs) * (1.0 - targets)
        )

        alpha_t = (
            self.alpha * targets
            + (1.0 - self.alpha) * (1.0 - targets)
        )

        focal_weight = alpha_t * (
            1.0 - p_t
        ).pow(self.gamma)

        return (focal_weight * bce).mean()


class CombinedSegmentationLoss(nn.Module):
    def __init__(
        self,
        dice_weight=0.7,
        focal_weight=0.3,
        smooth=1.0,
        alpha=0.25,
        gamma=2.0,
    ):
        super().__init__()

        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

        self.dice = DiceLoss(
            smooth=smooth
        )

        self.focal = FocalLoss(
            alpha=alpha,
            gamma=gamma
        )

    def forward(self, logits, targets):

        dice_loss = self.dice(
            logits,
            targets
        )

        focal_loss = self.focal(
            logits,
            targets
        )

        total_loss = (
            self.dice_weight * dice_loss
            + self.focal_weight * focal_loss
        )

        return total_loss