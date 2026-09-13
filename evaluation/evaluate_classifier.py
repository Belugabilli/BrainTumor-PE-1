"""
Evaluate the Project Exhibition 1 (PE1) classifier on the untouched BRISC2025 test set.
"""


from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

from datasets_loader.classification_dataset import (
    CLASS_TO_INDEX,
    BrainTumorClassificationDataset,
)
from datasets_loader.transforms import get_validation_transforms
from models.classifier import EfficientNetB1Classifier
from models.pe1_classifier import PE1Classifier
from utils.config import Config


CLASS_NAMES = [
    class_name
    for class_name, _ in sorted(
        CLASS_TO_INDEX.items(),
        key=lambda item: item[1],
    )
]

BASELINE = {
    "model": "Original EfficientNet-B1 reproduction",
    "accuracy": 0.9920,
    "macro_precision": 0.9930,
    "macro_recall": 0.9932,
    "macro_f1": 0.9931,
    "weighted_precision": 0.9921,
    "weighted_recall": 0.9920,
    "weighted_f1": 0.9920,
    "per_class": {
        "glioma": {
            "precision": 0.9883,
            "recall": 0.9961,
            "f1": 0.9922,
            "support": 254,
        },
        "meningioma": {
            "precision": 0.9838,
            "recall": 0.9902,
            "f1": 0.9870,
            "support": 306,
        },
        "pituitary": {
            "precision": 1.0000,
            "recall": 1.0000,
            "f1": 1.0000,
            "support": 140,
        },
        "no_tumor": {
            "precision": 1.0000,
            "recall": 0.9867,
            "f1": 0.9933,
            "support": 300,
        },
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Project Exhibition 1 (PE1) on the official BRISC2025 classification test set."
    )

    parser.add_argument(
        "--config",
        default="configs/classification.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        default="weights/classification/best_model.pth",
    )
    parser.add_argument(
        "--history-csv",
        default="logs/classification/metrics.csv",
    )
    parser.add_argument(
        "--training-summary",
        default="logs/classification/training_summary.json",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
    )
    parser.add_argument(
        "--require-mps",
        action="store_true",
    )
    return parser.parse_args()


def get_device(require_mps=False):
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        if require_mps:
            raise RuntimeError("MPS was required, but CUDA would be selected.")
        return torch.device("cuda")

    if require_mps:
        raise RuntimeError("MPS was required, but no MPS device is available.")

    return torch.device("cpu")


def synchronize(device):
    if device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def count_parameters(model):
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return total, trainable


def clean_class_metrics(class_metrics):
    return {
        "precision": float(class_metrics["precision"]),
        "recall": float(class_metrics["recall"]),
        "f1-score": float(class_metrics["f1-score"]),
        "support": int(class_metrics["support"]),
    }


def load_model(config, checkpoint_path, device):
    model = PE1Classifier(
        num_classes=config.get("model", "num_classes"),
        pretrained=False,
        dropout=config.get("model", "dropout"),
        hidden_dim=config.get("model", "hidden_dim"),
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    model.to(device)
    model.eval()

    return model, checkpoint


def evaluate(model, config, device):
    dataset = BrainTumorClassificationDataset(
        root_dir=config.get("dataset", "test_dir"),
        transform=get_validation_transforms(),
    )

    loader = DataLoader(
        dataset,
        batch_size=config.get("training", "batch_size"),
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    inference_time_seconds = 0.0
    all_labels = []
    all_predictions = []
    rows = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(device)
            labels = labels.to(device)

            synchronize(device)
            start_time = time.perf_counter()
            outputs = model(images)
            synchronize(device)
            inference_time_seconds += time.perf_counter() - start_time

            expected_shape = (
                images.shape[0],
                config.get("model", "num_classes"),
            )
            if tuple(outputs.shape) != expected_shape:
                raise ValueError(
                    "Unexpected model output shape during evaluation: "
                    f"{tuple(outputs.shape)} != {expected_shape}"
                )

            loss = criterion(outputs, labels)

            if not torch.isfinite(loss):
                raise ValueError("Non-finite test loss detected.")

            total_loss += loss.item()

            predictions = torch.argmax(outputs, dim=1)

            label_values = labels.cpu().numpy().tolist()
            prediction_values = predictions.cpu().numpy().tolist()

            all_labels.extend(label_values)
            all_predictions.extend(prediction_values)

            for index_in_batch, (label, prediction) in enumerate(
                zip(label_values, prediction_values)
            ):
                sample_index = batch_idx * loader.batch_size + index_in_batch
                rows.append(
                    {
                        "index": sample_index,
                        "image_path": str(dataset.image_paths[sample_index]),
                        "ground_truth": CLASS_NAMES[label],
                        "prediction": CLASS_NAMES[prediction],
                        "correct": label == prediction,
                    }
                )

    avg_loss = total_loss / len(loader)

    report_dict = classification_report(
        all_labels,
        all_predictions,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    report_text = classification_report(
        all_labels,
        all_predictions,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )

    matrix = confusion_matrix(
        all_labels,
        all_predictions,
        labels=list(range(len(CLASS_NAMES))),
    )

    metrics = {
        "test_loss": avg_loss,
        "accuracy": float(accuracy_score(all_labels, all_predictions)),
        "macro_precision": float(precision_score(
            all_labels,
            all_predictions,
            average="macro",
            zero_division=0,
        )),
        "macro_recall": float(recall_score(
            all_labels,
            all_predictions,
            average="macro",
            zero_division=0,
        )),
        "macro_f1": float(f1_score(
            all_labels,
            all_predictions,
            average="macro",
            zero_division=0,
        )),
        "weighted_precision": float(precision_score(
            all_labels,
            all_predictions,
            average="weighted",
            zero_division=0,
        )),
        "weighted_recall": float(recall_score(
            all_labels,
            all_predictions,
            average="weighted",
            zero_division=0,
        )),
        "weighted_f1": float(f1_score(
            all_labels,
            all_predictions,
            average="weighted",
            zero_division=0,
        )),
        "per_class": {
            class_name: clean_class_metrics(report_dict[class_name])
            for class_name in CLASS_NAMES
        },
        "support": {
            class_name: int(report_dict[class_name]["support"])
            for class_name in CLASS_NAMES
        },
        "confusion_matrix": matrix.tolist(),
        "inference_time_seconds": inference_time_seconds,
        "inference_time_per_image_seconds": (
            inference_time_seconds / len(dataset)
        ),
        "num_test_images": len(dataset),
    }

    return metrics, report_text, matrix, rows


def save_predictions(rows, output_path):
    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "index",
                "image_path",
                "ground_truth",
                "prediction",
                "correct",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def save_confusion_matrix(matrix, output_path):
    fig, ax = plt.subplots(figsize=(7, 7))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=range(len(CLASS_NAMES)),
        yticks=range(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ylabel="True label",
        xlabel="Predicted label",
        title="PE1 BRISC2025 Test Confusion Matrix",
    )
    plt.setp(
        ax.get_xticklabels(),
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )

    threshold = matrix.max() / 2.0
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            ax.text(
                col_idx,
                row_idx,
                format(matrix[row_idx, col_idx], "d"),
                ha="center",
                va="center",
                color="white" if matrix[row_idx, col_idx] > threshold else "black",
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def copy_and_plot_history(history_csv, results_dir):
    history_source = Path(history_csv)
    if not history_source.exists():
        return None

    history_output = results_dir / "pe1_training_history.csv"
    shutil.copyfile(history_source, history_output)

    history = pd.read_csv(history_output)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(history["epoch"], history["train_loss"], label="Train loss")
    axes[0].plot(history["epoch"], history["val_loss"], label="Validation loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("PE1 Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(history["epoch"], history["accuracy"], label="Accuracy")
    axes[1].plot(history["epoch"], history["f1"], label="Weighted F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_title("PE1 Validation Metrics")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(results_dir / "pe1_training_history.png", dpi=300)
    plt.close(fig)

    return history_output


def read_training_summary(summary_path, history_csv):
    summary_file = Path(summary_path)
    if summary_file.exists():
        with open(summary_file, "r", encoding="utf-8") as file:
            return json.load(file)

    history_file = Path(history_csv)
    if not history_file.exists():
        return {}

    history = pd.read_csv(history_file)
    best_accuracy_row = history.loc[history["accuracy"].idxmax()]
    min_loss_row = history.loc[history["val_loss"].idxmin()]

    return {
        "epochs": int(history["epoch"].max()),
        "total_training_time_seconds": float(history["epoch_time"].sum()),
        "sum_epoch_time_seconds": float(history["epoch_time"].sum()),
        "best_validation_epoch": int(best_accuracy_row["epoch"]),
        "best_validation_accuracy": float(best_accuracy_row["accuracy"]),
        "best_validation_loss_at_best_accuracy": float(best_accuracy_row["val_loss"]),
        "best_validation_f1_at_best_accuracy": float(best_accuracy_row["f1"]),
        "min_validation_loss_epoch": int(min_loss_row["epoch"]),
        "min_validation_loss": float(min_loss_row["val_loss"]),
    }


def enrich_training_summary(training_summary, history_csv):
    history_file = Path(history_csv)
    if not history_file.exists():
        return training_summary

    history = pd.read_csv(history_file)
    if history.empty:
        return training_summary

    best_accuracy_row = history.loc[history["accuracy"].idxmax()]
    min_loss_row = history.loc[history["val_loss"].idxmin()]

    training_summary.update(
        {
            "epochs": int(history["epoch"].max()),
            "sum_epoch_time_seconds": float(history["epoch_time"].sum()),
            "best_validation_epoch": int(best_accuracy_row["epoch"]),
            "best_validation_accuracy": float(best_accuracy_row["accuracy"]),
            "best_validation_loss_at_best_accuracy": float(
                best_accuracy_row["val_loss"]
            ),
            "best_validation_f1_at_best_accuracy": float(best_accuracy_row["f1"]),
            "min_validation_loss_epoch": int(min_loss_row["epoch"]),
            "min_validation_loss": float(min_loss_row["val_loss"]),
        }
    )
    return training_summary


def format_value(value):
    if value is None:
        return "unavailable"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def difference(pe1_value, baseline_value):
    if pe1_value is None or baseline_value is None:
        return None
    return pe1_value - baseline_value


def write_comparison(results_dir, metrics, complexity, training_summary):
    rows = [
        (
            "Accuracy",
            BASELINE["accuracy"],
            metrics["accuracy"],
        ),
        (
            "Macro Precision",
            BASELINE["macro_precision"],
            metrics["macro_precision"],
        ),
        (
            "Macro Recall",
            BASELINE["macro_recall"],
            metrics["macro_recall"],
        ),
        (
            "Macro F1",
            BASELINE["macro_f1"],
            metrics["macro_f1"],
        ),
        (
            "Weighted F1",
            BASELINE["weighted_f1"],
            metrics["weighted_f1"],
        ),
        (
            "Parameters",
            complexity["baseline_total_parameters"],
            complexity["pe1_total_parameters"],
        ),
        (
            "Training time seconds",
            None,
            training_summary.get("total_training_time_seconds"),
        ),
        (
            "Inference time per image seconds",
            None,
            metrics["inference_time_per_image_seconds"],
        ),
    ]

    csv_path = results_dir / "baseline_vs_pe1_comparison.csv"
    csv_legacy = results_dir / "comparison.csv"
    for p in (csv_path, csv_legacy):
        with open(
            p,
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.writer(file)
            writer.writerow(["Metric", "Original B1", "PE1", "Difference"])
            for metric, baseline_value, pe1_value in rows:
                writer.writerow(
                    [
                        metric,
                        format_value(baseline_value),
                        format_value(pe1_value),
                        format_value(difference(pe1_value, baseline_value)),
                    ]
                )

    md_path = results_dir / "baseline_vs_pe1_comparison.md"
    md_legacy = results_dir / "comparison.md"
    for p in (md_path, md_legacy):
        with open(p, "w", encoding="utf-8") as file:
            file.write("# PE1 vs Original EfficientNet-B1\n\n")
            file.write("| Metric | Original B1 | PE1 | Difference |\n")
            file.write("| --- | ---: | ---: | ---: |\n")
            for metric, baseline_value, pe1_value in rows:
                file.write(
                    "| "
                    f"{metric} | "
                    f"{format_value(baseline_value)} | "
                    f"{format_value(pe1_value)} | "
                    f"{format_value(difference(pe1_value, baseline_value))} |\n"
                )

    return csv_path, md_path


def write_analysis(results_dir, metrics, complexity):
    accuracy_difference = metrics["accuracy"] - BASELINE["accuracy"]
    macro_f1_difference = metrics["macro_f1"] - BASELINE["macro_f1"]
    parameter_difference = (
        complexity["pe1_total_parameters"]
        - complexity["baseline_total_parameters"]
    )

    class_lines = []
    for class_name in CLASS_NAMES:
        baseline_class = BASELINE["per_class"][class_name]
        pe1_class = metrics["per_class"][class_name]
        class_lines.append(
            "- "
            f"{class_name}: PE1 F1 {pe1_class['f1-score']:.4f} vs "
            f"baseline F1 {baseline_class['f1']:.4f} "
            f"({pe1_class['f1-score'] - baseline_class['f1']:+.4f})"
        )

    if accuracy_difference > 0:
        accuracy_sentence = "PE1 improved accuracy over the original baseline."
    elif accuracy_difference < 0:
        accuracy_sentence = "PE1 did not improve accuracy; it finished below the original baseline."
    else:
        accuracy_sentence = "PE1 matched the original baseline accuracy."

    if macro_f1_difference > 0:
        macro_sentence = "PE1 improved macro F1 over the original baseline."
    elif macro_f1_difference < 0:
        macro_sentence = "PE1 did not improve macro F1; it finished below the original baseline."
    else:
        macro_sentence = "PE1 matched the original baseline macro F1."

    if accuracy_difference > 0 and macro_f1_difference > 0:
        verdict = "On these measured aggregate metrics, PE1 beats the baseline."
    elif accuracy_difference < 0 or macro_f1_difference < 0:
        verdict = "On these measured aggregate metrics, PE1 does not beat the baseline."
    else:
        verdict = "On these measured aggregate metrics, PE1 is effectively tied with the baseline."

    analysis_path = results_dir / "pe1_analysis.md"
    analysis_legacy = results_dir / "analysis.md"
    for p in (analysis_path, analysis_legacy):
        with open(p, "w", encoding="utf-8") as file:
            file.write("# PE1 Scientific Interpretation\n\n")
            file.write(f"{accuracy_sentence} ")
            file.write(
                f"Accuracy difference: {accuracy_difference:+.6f}.\n\n"
            )
            file.write(f"{macro_sentence} ")
            file.write(
                f"Macro F1 difference: {macro_f1_difference:+.6f}.\n\n"
            )
            file.write("## Per-Class F1\n\n")
            file.write("\n".join(class_lines))
            file.write("\n\n")
            file.write("## Complexity\n\n")
            file.write(
                f"PE1 has {complexity['pe1_total_parameters']} parameters versus "
                f"{complexity['baseline_total_parameters']} for the original B1 "
                f"architecture, an increase of {parameter_difference} parameters. "
                "This confirms that the B3 backbone increases model size.\n\n"
            )
            file.write("## Verdict\n\n")
            file.write(verdict)
            file.write(
                " No statistical significance or clinical significance is claimed; "
                "this is a controlled machine-learning experiment on the provided "
                "BRISC2025 split.\n"
            )

    return analysis_path



def main():
    args = parse_args()
    config = Config(args.config)
    device = get_device(require_mps=args.require_mps)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    model, checkpoint = load_model(
        config=config,
        checkpoint_path=args.checkpoint,
        device=device,
    )

    baseline_model = EfficientNetB1Classifier(
        num_classes=config.get("model", "num_classes"),
        pretrained=False,
        dropout=config.get("model", "dropout"),
    )

    pe1_total_parameters, pe1_trainable_parameters = count_parameters(model)
    baseline_total_parameters, baseline_trainable_parameters = count_parameters(
        baseline_model
    )

    metrics, report_text, matrix, rows = evaluate(
        model=model,
        config=config,
        device=device,
    )

    checkpoint_path = Path(args.checkpoint)
    complexity = {
        "pe1_total_parameters": pe1_total_parameters,
        "pe1_trainable_parameters": pe1_trainable_parameters,
        "baseline_total_parameters": baseline_total_parameters,
        "baseline_trainable_parameters": baseline_trainable_parameters,
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
        "checkpoint_size_mb": checkpoint_path.stat().st_size / (1024 * 1024),
    }

    training_summary = read_training_summary(
        args.training_summary,
        args.history_csv,
    )
    training_summary = enrich_training_summary(
        training_summary,
        args.history_csv,
    )

    save_predictions(
        rows,
        results_dir / "pe1_predictions.csv",
    )

    with open(
        results_dir / "pe1_classification_report.txt",
        "w",
        encoding="utf-8",
    ) as file:
        file.write(report_text)

    save_confusion_matrix(
        matrix,
        results_dir / "pe1_confusion_matrix.png",
    )

    copy_and_plot_history(
        args.history_csv,
        results_dir,
    )

    test_metrics = {
        **metrics,
        "device": str(device),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "model": {
            "name": config.get("model", "name"),
            "backbone": config.get("model", "backbone"),
            "hidden_dim": config.get("model", "hidden_dim"),
            "dropout": config.get("model", "dropout"),
            "num_classes": config.get("model", "num_classes"),
        },
        "complexity": complexity,
    }

    with open(
        results_dir / "pe1_test_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            test_metrics,
            file,
            indent=2,
        )

    write_comparison(
        results_dir=results_dir,
        metrics=metrics,
        complexity=complexity,
        training_summary=training_summary,
    )
    write_analysis(
        results_dir=results_dir,
        metrics=metrics,
        complexity=complexity,
    )

    summary = {
        "model": test_metrics["model"],
        "training": training_summary,
        "validation": {
            "best_validation_epoch": training_summary.get(
                "best_validation_epoch"
            ),
            "best_validation_accuracy": training_summary.get(
                "best_validation_accuracy"
            ),
            "best_validation_loss_at_best_accuracy": training_summary.get(
                "best_validation_loss_at_best_accuracy"
            ),
            "best_validation_f1_at_best_accuracy": training_summary.get(
                "best_validation_f1_at_best_accuracy"
            ),
            "min_validation_loss_epoch": training_summary.get(
                "min_validation_loss_epoch"
            ),
            "min_validation_loss": training_summary.get(
                "min_validation_loss"
            ),
        },
        "test": {
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "test_loss": metrics["test_loss"],
        },
        "complexity": complexity,
        "baseline": BASELINE,
        "comparison": {
            "accuracy_difference": metrics["accuracy"] - BASELINE["accuracy"],
            "macro_f1_difference": metrics["macro_f1"] - BASELINE["macro_f1"],
            "weighted_f1_difference": (
                metrics["weighted_f1"] - BASELINE["weighted_f1"]
            ),
            "parameter_difference": (
                complexity["pe1_total_parameters"]
                - complexity["baseline_total_parameters"]
            ),
        },
    }

    with open(
        results_dir / "pe1_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    print()
    print("Evaluation Complete")
    print(f"Device: {device}")
    print(f"Test Loss: {metrics['test_loss']:.6f}")
    print(f"Accuracy: {metrics['accuracy']:.6f}")
    print(f"Macro F1: {metrics['macro_f1']:.6f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.6f}")
    print(f"Inference seconds/image: {metrics['inference_time_per_image_seconds']:.6f}")
    print()
    print("Saved PE1 results under:")
    print(results_dir)


if __name__ == "__main__":
    main()
