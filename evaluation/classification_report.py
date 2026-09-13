"""
Generate PE1 Classification Report
"""

from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report

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

    report = classification_report(
        predictions["ground_truth"],
        predictions["prediction"],
        labels=CLASS_NAMES,
        target_names=CLASS_NAMES,
        digits=4,
    )

    output_file = (
        results_dir /
        "pe1_classification_report.txt"
    )

    with open(output_file, "w") as f:
        f.write(report)

    print()
    print(report)
    print()
    print("Saved:")
    print(output_file)


if __name__ == "__main__":
    main()
