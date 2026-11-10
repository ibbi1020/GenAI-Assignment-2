"""Criterion classification metrics. Labels are never collapsed to yes/no."""

from __future__ import annotations

from collections import Counter


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def confusion_matrix(y_true: list[str], y_pred: list[str], labels: list[str]) -> list[list[int]]:
    index = {label: position for position, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for gold, pred in zip(y_true, y_pred, strict=True):
        if gold not in index:
            continue
        row = index[gold]
        column = index.get(pred)
        if column is None:
            continue
        matrix[row][column] += 1
    return matrix


def per_class_scores(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict[str, dict[str, float]]:
    scores = {}
    for label in labels:
        tp = sum(gold == label and pred == label for gold, pred in zip(y_true, y_pred, strict=True))
        fp = sum(gold != label and pred == label for gold, pred in zip(y_true, y_pred, strict=True))
        fn = sum(gold == label and pred != label for gold, pred in zip(y_true, y_pred, strict=True))
        support = sum(gold == label for gold in y_true)
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        predicted = sum(pred == label for pred in y_pred)
        scores[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "predicted": predicted,
        }
    return scores


def macro_f1(class_scores: dict[str, dict[str, float]]) -> float:
    """Unweighted mean of F1 over labels that appear in the gold or the predictions.

    A label with no support and no predictions is left out so an absent class
    does not force the mean to zero.
    """
    return _safe_div(
        sum(row["f1"] for row in class_scores.values() if _class_is_active(row)),
        _active_count(class_scores),
    )


def _class_is_active(row: dict[str, float]) -> bool:
    return row["support"] > 0 or row.get("predicted", 0) > 0


def _active_count(class_scores: dict[str, dict[str, float]]) -> int:
    return sum(1 for row in class_scores.values() if _class_is_active(row))


def classification_summary(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict[str, object]:
    if len(y_true) != len(y_pred):
        raise ValueError("Gold and predicted label lists have different lengths.")
    present = [label for label in labels if label in set(y_true) or label in set(y_pred)]
    used = present or list(labels)
    scores = per_class_scores(y_true, y_pred, used)
    correct = sum(gold == pred for gold, pred in zip(y_true, y_pred, strict=True))
    invalid = sum(pred not in labels for pred in y_pred)
    return {
        "n": len(y_true),
        "accuracy": _safe_div(correct, len(y_true)),
        "macro_f1": macro_f1(scores) if scores else 0.0,
        "per_class": scores,
        "confusion_matrix": {
            "labels": used,
            "matrix": confusion_matrix(y_true, y_pred, used),
        },
        "invalid_predictions": invalid,
        "label_support": dict(Counter(y_true)),
    }
