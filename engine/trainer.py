"""
engine/trainer.py

Classification Trainer
Research Paper Reproduction

Paper:
Brain Tumor Segmentation and Classification in MRI Images
Using EfficientNet and U-Net++

Author:
BrainTumor-Reproduction
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.metrics import calculate_metrics


class Trainer:
    """
    Classification Trainer.

    Responsible for

    - Training
    - Validation
    - Scheduler
    - Metrics
    - Checkpointing
    - Logging
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        scheduler: ReduceLROnPlateau,
        device: torch.device,
        logger,
        config,
    ) -> None:

        self.model = model.to(device)

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.device = device

        self.logger = logger

        self.config = config

        self.best_accuracy = -1.0

        self.best_record = None

        self.start_epoch = 1

        self.history = []

        checkpoint_dir = Path(
            config.get(
                "checkpoint",
                "save_dir",
            )
        )

        checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.best_model_path = (
            checkpoint_dir /
            "best_model.pth"
        )

        self.pe1_best_model_path = (
            checkpoint_dir /
            "pe1_best.pth"
        )

        self.latest_model_path = (
            checkpoint_dir /
            "latest_model.pth"
        )

        self.pe1_latest_model_path = (
            checkpoint_dir /
            "pe1_latest.pth"
        )


        log_directory = Path(
            config.get(
                "logging",
                "log_dir",
            )
        )

        log_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.csv_path = (
            log_directory /
            "metrics.csv"
        )

        self.summary_path = (
            log_directory /
            "training_summary.json"
        )

        with open(
            self.csv_path,
            "w",
            newline="",
        ) as file:

            writer = csv.writer(file)

            writer.writerow(

                [

                    "epoch",

                    "train_loss",

                    "val_loss",

                    "accuracy",

                    "precision",

                    "recall",

                    "f1",

                    "learning_rate",

                    "epoch_time",

                ]

            )

        self.logger.info(
            "=" * 60
        )

        self.logger.info(
            "Classification Trainer Initialized"
        )

        self.logger.info(
            f"Device : {device}"
        )

        self.logger.info(
            f"Train Images : {len(train_loader.dataset)}"
        )

        self.logger.info(
            f"Validation Images : {len(val_loader.dataset)}"
        )

        self.logger.info(
            "=" * 60
        )

    def train_one_epoch(
        self,
        epoch: int,
    ) -> float:

        """
        Train the model for one epoch.
        """

        self.model.train()

        running_loss = 0.0

        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch} [Train]",
            leave=False,
        )

        for images, labels in progress_bar:

            images = images.to(self.device)

            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(images)

            loss = self.criterion(
                outputs,
                labels,
            )

            if not torch.isfinite(loss):
                raise ValueError(
                    f"Non-finite training loss detected at epoch {epoch}."
                )

            if (
                epoch == self.start_epoch
                and progress_bar.n == 0
            ):
                expected_classes = self.config.get(
                    "model",
                    "num_classes",
                )
                expected_shape = (
                    images.shape[0],
                    expected_classes,
                )
                if tuple(outputs.shape) != expected_shape:
                    raise ValueError(
                        "Unexpected model output shape: "
                        f"{tuple(outputs.shape)} != {expected_shape}"
                    )

                self.logger.info(
                    f"First training output shape : {list(outputs.shape)}"
                )

            loss.backward()

            self.optimizer.step()

            running_loss += loss.item()

            average_loss = running_loss / (
                progress_bar.n + 1
            )

            progress_bar.set_postfix(
                {
                    "loss": f"{average_loss:.4f}"
                }
            )

        epoch_loss = (
            running_loss /
            len(self.train_loader)
        )

        return epoch_loss

    def validate(
        self,
    ):

        """
        Validate the model on the validation dataset.
        """

        self.model.eval()

        running_loss = 0.0

        all_predictions = []

        all_labels = []

        with torch.no_grad():

            progress_bar = tqdm(
                self.val_loader,
                desc="Validation",
                leave=False,
            )

            for images, labels in progress_bar:

                images = images.to(self.device)

                labels = labels.to(self.device)

                outputs = self.model(images)

                loss = self.criterion(
                    outputs,
                    labels,
                )

                if not torch.isfinite(loss):
                    raise ValueError(
                        "Non-finite validation loss detected."
                    )

                running_loss += loss.item()

                predictions = torch.argmax(
                    outputs,
                    dim=1,
                )

                all_predictions.extend(
                    predictions.cpu().numpy().tolist()
                )

                all_labels.extend(
                    labels.cpu().numpy().tolist()
                )

                average_loss = (
                    running_loss /
                    (progress_bar.n + 1)
                )

                progress_bar.set_postfix(
                    {
                        "loss": f"{average_loss:.4f}"
                    }
                )

        validation_loss = (
            running_loss /
            len(self.val_loader)
        )

        metrics = calculate_metrics(
            all_labels,
            all_predictions,
        )

        return (
            validation_loss,
            metrics,
        )

    def fit(self):
        """
        Complete training loop.
        """

        epochs = self.config.get(
            "training",
            "epochs",
        )

        self.logger.info("Starting Training...")

        training_start_time = time.time()

        for epoch in range(
            self.start_epoch,
            epochs + 1,
        ):

            start_time = time.time()

            train_loss = self.train_one_epoch(epoch)

            val_loss, metrics = self.validate()

            self.scheduler.step(val_loss)

            current_lr = self.optimizer.param_groups[0]["lr"]

            epoch_time = time.time() - start_time

            self.logger.info(
                "-" * 60
            )

            self.logger.info(
                f"Epoch [{epoch}/{epochs}]"
            )

            self.logger.info(
                f"Train Loss : {train_loss:.4f}"
            )

            self.logger.info(
                f"Validation Loss : {val_loss:.4f}"
            )

            self.logger.info(
                f"Accuracy : {metrics['accuracy']:.4f}"
            )

            self.logger.info(
                f"Precision : {metrics['precision']:.4f}"
            )

            self.logger.info(
                f"Recall : {metrics['recall']:.4f}"
            )

            self.logger.info(
                f"F1 Score : {metrics['f1']:.4f}"
            )

            self.logger.info(
                f"Learning Rate : {current_lr:.8f}"
            )

            self.logger.info(
                f"Epoch Time : {epoch_time:.2f} sec"
            )

            epoch_record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "learning_rate": current_lr,
                "epoch_time": epoch_time,
            }

            self.history.append(epoch_record)

            with open(
                self.csv_path,
                "a",
                newline="",
            ) as file:

                writer = csv.writer(file)

                writer.writerow(
                    [
                        epoch,
                        train_loss,
                        val_loss,
                        metrics["accuracy"],
                        metrics["precision"],
                        metrics["recall"],
                        metrics["f1"],
                        current_lr,
                        epoch_time,
                    ]
                )

            latest_checkpoint = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "accuracy": metrics["accuracy"],
                "val_loss": val_loss,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "epoch_time": epoch_time,
                "model_name": self.config.get(
                    "model",
                    "name",
                ),
                "backbone": self.config.get(
                    "model",
                    "backbone",
                ),
            }

            torch.save(
                latest_checkpoint,
                self.latest_model_path,
            )
            torch.save(
                latest_checkpoint,
                self.pe1_latest_model_path,
            )

            if metrics["accuracy"] > self.best_accuracy:

                self.best_accuracy = metrics["accuracy"]

                self.best_record = epoch_record

                torch.save(
                    latest_checkpoint,
                    self.best_model_path,
                )
                torch.save(
                    latest_checkpoint,
                    self.pe1_best_model_path,
                )

                self.logger.info(
                    "Best model updated."
                )


        total_training_time = time.time() - training_start_time

        summary = {
            "epochs": epochs,
            "total_training_time_seconds": total_training_time,
            "sum_epoch_time_seconds": sum(
                record["epoch_time"]
                for record in self.history
            ),
            "best_validation_epoch": (
                self.best_record["epoch"]
                if self.best_record
                else None
            ),
            "best_validation_accuracy": (
                self.best_record["accuracy"]
                if self.best_record
                else None
            ),
            "best_validation_loss_at_best_accuracy": (
                self.best_record["val_loss"]
                if self.best_record
                else None
            ),
            "best_validation_f1_at_best_accuracy": (
                self.best_record["f1"]
                if self.best_record
                else None
            ),
            "history_csv": str(self.csv_path),
            "best_checkpoint": str(self.best_model_path),
            "latest_checkpoint": str(self.latest_model_path),
        }

        with open(
            self.summary_path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                summary,
                file,
                indent=2,
            )

        self.logger.info(
            f"Total Training Time : {total_training_time:.2f} sec"
        )

        self.logger.info(
            "=" * 60
        )

        self.logger.info(
            "Training Finished Successfully."
        )

        self.logger.info(
            "=" * 60
        )

        return summary
