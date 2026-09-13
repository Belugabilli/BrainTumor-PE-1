"""
Generate PE1 Confusion Matrix from pe1_predictions.csv
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from datasets_loader.classification_dataset import CLASS_TO_INDEX


CLASS_NAMES = [
    class_name
    for class_name, _ in sorted(
        CLASS_TO_INDEX.items(),
        key=lambda item: item[1],
    )
]


def main():

    results_dir = Path("results")

    predictions = pd.read_csv(
        results_dir / "pe1_predictions.csv"
    )

    cm = confusion_matrix(
        predictions["ground_truth"],
        predictions["prediction"],
        labels=CLASS_NAMES,
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASS_NAMES,
    )

    fig, ax = plt.subplots(figsize=(7, 7))

    disp.plot(
        cmap="Blues",
        ax=ax,
        colorbar=False,
    )

    plt.title(
        "Brain Tumor Classification\nConfusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        results_dir / "pe1_confusion_matrix.png",
        dpi=300,
    )

    plt.close()

    print()

    print("Saved:")

    print("results/pe1_confusion_matrix.png")


if __name__ == "__main__":
    main()
