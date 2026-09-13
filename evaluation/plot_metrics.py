"""
Plot Training Curves
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():

    csv_file = Path("logs/classification/metrics.csv")

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    df = pd.read_csv(csv_file)

    # ---------------- Loss Curve ---------------- #

    plt.figure(figsize=(8, 5))

    plt.plot(
        df["epoch"],
        df["train_loss"],
        linewidth=2,
        label="Train Loss",
    )

    plt.plot(
        df["epoch"],
        df["val_loss"],
        linewidth=2,
        label="Validation Loss",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    loss_curve_path = results_dir / "pe1_loss_curve.png"

    plt.savefig(loss_curve_path, dpi=300)

    plt.close()

    # ---------------- Accuracy Curve ---------------- #

    plt.figure(figsize=(8, 5))

    plt.plot(
        df["epoch"],
        df["accuracy"],
        linewidth=2,
    )

    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy")
    plt.grid(True)

    plt.tight_layout()

    accuracy_curve_path = results_dir / "pe1_accuracy_curve.png"

    plt.savefig(accuracy_curve_path, dpi=300)

    plt.close()

    print()
    print("Saved:")
    print(loss_curve_path)
    print(accuracy_curve_path)


if __name__ == "__main__":
    main()
