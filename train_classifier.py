"""
Main Training Script

Brain Tumor Classification
Project Exhibition 1 (PE1) Experiment
"""

import argparse

import torch
import torch.nn as nn

from datasets_loader.create_dataloaders import create_dataloaders
from engine.trainer import Trainer
from models.pe1_classifier import PE1Classifier

from utils.config import Config
from utils.logger import get_logger
from utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the Project Exhibition 1 (PE1) EfficientNet-B3 classifier."
    )

    parser.add_argument(
        "--config",
        default="configs/classification.yaml",
        help="Path to the classification YAML config.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Temporarily override training.epochs without editing the YAML file.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Temporarily override checkpoint.save_dir.",
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Temporarily override logging.log_dir.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Temporarily override dataset.num_workers.",
    )
    parser.add_argument(
        "--require-mps",
        action="store_true",
        help="Fail fast unless Apple MPS is selected.",
    )
    return parser.parse_args()


def apply_overrides(config, args):
    if args.epochs is not None:
        config.config["training"]["epochs"] = args.epochs

    if args.checkpoint_dir is not None:
        config.config["checkpoint"]["save_dir"] = args.checkpoint_dir

    if args.log_dir is not None:
        config.config["logging"]["log_dir"] = args.log_dir

    if args.num_workers is not None:
        config.config["dataset"]["num_workers"] = args.num_workers


def get_device(require_mps=False):

    if torch.backends.mps.is_available():

        return torch.device("mps")

    elif torch.cuda.is_available():

        if require_mps:
            raise RuntimeError(
                "MPS was required for this run, but CUDA would be selected."
            )

        return torch.device("cuda")

    if require_mps:
        raise RuntimeError(
            "MPS was required for this run, but torch.backends.mps is not available."
        )

    return torch.device("cpu")


def count_parameters(model):
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return total, trainable


def main():

    args = parse_args()

    config = Config(
        args.config
    )

    apply_overrides(config, args)

    set_seed(
        config["seed"]
    )

    logger = get_logger(
        config.get(
            "logging",
            "log_dir",
        )
    )

    device = get_device(
        require_mps=args.require_mps,
    )

    logger.info(
        f"Using Device : {device}"
    )

    train_loader, val_loader = create_dataloaders(
        config
    )

    model = PE1Classifier(
        num_classes=config.get(
            "model",
            "num_classes",
        ),
        pretrained=config.get(
            "model",
            "pretrained",
        ),
        dropout=config.get(
            "model",
            "dropout",
        ),
        hidden_dim=config.get(
            "model",
            "hidden_dim",
        ),
    )

    total_parameters, trainable_parameters = count_parameters(model)

    logger.info(
        f"Model : {config.get('model', 'name')} ({config.get('model', 'backbone')})"
    )
    logger.info(
        f"Total Parameters : {total_parameters}"
    )
    logger.info(
        f"Trainable Parameters : {trainable_parameters}"
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(

        model.parameters(),

        lr=config.get(
            "training",
            "learning_rate",
        ),

        weight_decay=config.get(
            "training",
            "weight_decay",
        ),

    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

        optimizer,

        mode="min",

        factor=config.get(
            "scheduler",
            "factor",
        ),

        patience=config.get(
            "scheduler",
            "patience",
        ),

    )

    trainer = Trainer(

        model=model,

        train_loader=train_loader,

        val_loader=val_loader,

        criterion=criterion,

        optimizer=optimizer,

        scheduler=scheduler,

        device=device,

        logger=logger,

        config=config,

    )

    trainer.fit()


if __name__ == "__main__":

    main()
