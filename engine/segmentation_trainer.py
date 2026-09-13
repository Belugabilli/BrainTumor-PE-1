import csv
import os
import time

import torch
from tqdm import tqdm


class SegmentationTrainer:

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        criterion,
        device,
        epochs=60,
        patience=10,
        save_dir="weights/segmentation",
        log_dir="logs/segmentation",
    ):

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion

        self.device = device
        self.epochs = epochs
        self.patience = patience

        self.save_dir = save_dir
        self.log_dir = log_dir

        os.makedirs(
            self.save_dir,
            exist_ok=True
        )

        os.makedirs(
            self.log_dir,
            exist_ok=True
        )

        self.best_dice = -1.0
        self.epochs_without_improvement = 0

        self.metrics_file = os.path.join(
            self.log_dir,
            "metrics.csv"
        )

        self._create_metrics_file()

    def _create_metrics_file(self):

        with open(
            self.metrics_file,
            "w",
            newline=""
        ) as file:

            writer = csv.writer(file)

            writer.writerow([
                "epoch",
                "train_loss",
                "val_loss",
                "train_dice",
                "val_dice",
                "train_iou",
                "val_iou",
                "learning_rate",
                "epoch_time",
            ])

    @staticmethod
    def calculate_dice(
        logits,
        targets,
        threshold=0.5,
        smooth=1e-6,
    ):

        probabilities = torch.sigmoid(logits)

        predictions = (
            probabilities >= threshold
        ).float()

        predictions = predictions.view(
            predictions.size(0),
            -1
        )

        targets = targets.view(
            targets.size(0),
            -1
        )

        intersection = (
            predictions * targets
        ).sum(dim=1)

        dice = (
            2.0 * intersection + smooth
        ) / (
            predictions.sum(dim=1)
            + targets.sum(dim=1)
            + smooth
        )

        return dice.mean().item()

    @staticmethod
    def calculate_iou(
        logits,
        targets,
        threshold=0.5,
        smooth=1e-6,
    ):

        probabilities = torch.sigmoid(logits)

        predictions = (
            probabilities >= threshold
        ).float()

        predictions = predictions.view(
            predictions.size(0),
            -1
        )

        targets = targets.view(
            targets.size(0),
            -1
        )

        intersection = (
            predictions * targets
        ).sum(dim=1)

        union = (
            predictions
            + targets
            - predictions * targets
        ).sum(dim=1)

        iou = (
            intersection + smooth
        ) / (
            union + smooth
        )

        return iou.mean().item()

    def train_one_epoch(self, epoch):

        self.model.train()

        total_loss = 0.0
        total_dice = 0.0
        total_iou = 0.0

        num_batches = len(
            self.train_loader
        )

        progress = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch} [Train]"
        )

        for batch_idx, (
            images,
            masks
        ) in enumerate(progress):

            images = images.to(
                self.device
            )

            masks = masks.to(
                self.device
            )

            self.optimizer.zero_grad(
                set_to_none=True
            )

            logits = self.model(images)

            loss = self.criterion(
                logits,
                masks
            )

            loss.backward()

            self.optimizer.step()

            # Warm-restart scheduler
            # uses fractional epoch progress
            scheduler_position = (
                (epoch - 1)
                + (
                    batch_idx
                    / max(num_batches, 1)
                )
            )

            self.scheduler.step(
                scheduler_position
            )

            dice = self.calculate_dice(
                logits.detach(),
                masks
            )

            iou = self.calculate_iou(
                logits.detach(),
                masks
            )

            total_loss += loss.item()
            total_dice += dice
            total_iou += iou

            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                dice=f"{dice:.4f}",
                iou=f"{iou:.4f}",
            )

        return (
            total_loss / num_batches,
            total_dice / num_batches,
            total_iou / num_batches,
        )

    @torch.no_grad()
    def validate(self):

        self.model.eval()

        total_loss = 0.0
        total_dice = 0.0
        total_iou = 0.0

        num_batches = len(
            self.val_loader
        )

        progress = tqdm(
            self.val_loader,
            desc="Validation"
        )

        for images, masks in progress:

            images = images.to(
                self.device
            )

            masks = masks.to(
                self.device
            )

            logits = self.model(
                images
            )

            loss = self.criterion(
                logits,
                masks
            )

            dice = self.calculate_dice(
                logits,
                masks
            )

            iou = self.calculate_iou(
                logits,
                masks
            )

            total_loss += loss.item()
            total_dice += dice
            total_iou += iou

        return (
            total_loss / num_batches,
            total_dice / num_batches,
            total_iou / num_batches,
        )

    def save_checkpoint(
        self,
        filename,
        epoch,
        val_dice,
    ):

        path = os.path.join(
            self.save_dir,
            filename
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict":
                    self.model.state_dict(),
                "optimizer_state_dict":
                    self.optimizer.state_dict(),
                "scheduler_state_dict":
                    self.scheduler.state_dict(),
                "val_dice": val_dice,
            },
            path,
        )

    def fit(self):

        print()
        print("=" * 60)
        print("Segmentation Training")
        print("=" * 60)

        print(
            "Device          :",
            self.device
        )

        print(
            "Epochs          :",
            self.epochs
        )

        print(
            "Train Batches   :",
            len(self.train_loader)
        )

        print(
            "Validation Batches :",
            len(self.val_loader)
        )

        print("=" * 60)

        for epoch in range(
            1,
            self.epochs + 1
        ):

            start_time = time.time()

            train_loss, train_dice, train_iou = (
                self.train_one_epoch(epoch)
            )

            val_loss, val_dice, val_iou = (
                self.validate()
            )

            epoch_time = (
                time.time() - start_time
            )

            learning_rate = (
                self.optimizer
                .param_groups[0]["lr"]
            )

            print()
            print("-" * 60)
            print(
                f"Epoch [{epoch}/{self.epochs}]"
            )

            print(
                f"Train Loss : {train_loss:.4f}"
            )

            print(
                f"Val Loss   : {val_loss:.4f}"
            )

            print(
                f"Train Dice : {train_dice:.4f}"
            )

            print(
                f"Val Dice   : {val_dice:.4f}"
            )

            print(
                f"Train IoU  : {train_iou:.4f}"
            )

            print(
                f"Val IoU    : {val_iou:.4f}"
            )

            print(
                f"LR         : {learning_rate:.8f}"
            )

            print(
                f"Time       : {epoch_time:.2f} sec"
            )

            # Save metrics
            with open(
                self.metrics_file,
                "a",
                newline=""
            ) as file:

                writer = csv.writer(file)

                writer.writerow([
                    epoch,
                    train_loss,
                    val_loss,
                    train_dice,
                    val_dice,
                    train_iou,
                    val_iou,
                    learning_rate,
                    epoch_time,
                ])

            # Latest checkpoint
            self.save_checkpoint(
                "latest_model.pth",
                epoch,
                val_dice,
            )

            # Best checkpoint
            if val_dice > self.best_dice:

                self.best_dice = val_dice

                self.epochs_without_improvement = 0

                self.save_checkpoint(
                    "best_model.pth",
                    epoch,
                    val_dice,
                )

                print(
                    "Best model updated."
                )

            else:

                self.epochs_without_improvement += 1

                print(
                    "No improvement:",
                    self.epochs_without_improvement,
                    "/",
                    self.patience
                )

            if (
                self.epochs_without_improvement
                >= self.patience
            ):

                print()
                print(
                    "Early stopping triggered."
                )

                break

        print()
        print("=" * 60)
        print("Training Complete")
        print("=" * 60)

        print(
            "Best Validation Dice :",
            f"{self.best_dice:.4f}"
        )