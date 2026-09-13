"""
Classification Metrics
"""

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def calculate_metrics(
    labels,
    predictions,
):

    accuracy = accuracy_score(
        labels,
        predictions,
    )

    precision = precision_score(
        labels,
        predictions,
        average="weighted",
        zero_division=0,
    )

    recall = recall_score(
        labels,
        predictions,
        average="weighted",
        zero_division=0,
    )

    f1 = f1_score(
        labels,
        predictions,
        average="weighted",
        zero_division=0,
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }