"""Evidence-sentence precision, recall, and F1.

Empty sets are explicit. Both empty scores 1. A non-empty prediction with an
empty gold set scores 0 for precision and 0 for recall. An empty prediction
with a non-empty gold set scores 0 for both. Neither case is treated as proof
that the patient meets or fails the criterion.
"""

from __future__ import annotations


def example_precision_recall(gold: set[int], predicted: set[int]) -> tuple[float, float]:
    if not predicted and not gold:
        return 1.0, 1.0
    if not predicted and gold:
        return 0.0, 0.0
    if predicted and not gold:
        return 0.0, 0.0
    overlap = len(gold & predicted)
    return overlap / len(predicted), overlap / len(gold)


def example_f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def evidence_summary(pairs: list[tuple[set[int], set[int]]]) -> dict[str, float | int]:
    precisions = []
    recalls = []
    f1s = []
    true_positive = 0
    false_positive = 0
    false_negative = 0
    for gold, predicted in pairs:
        precision, recall = example_precision_recall(gold, predicted)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(example_f1(precision, recall))
        true_positive += len(gold & predicted)
        false_positive += len(predicted - gold)
        false_negative += len(gold - predicted)
    if true_positive + false_positive + false_negative == 0:
        micro_precision = 1.0
        micro_recall = 1.0
    else:
        micro_precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        micro_recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    micro_f1 = example_f1(micro_precision, micro_recall)
    return {
        "n": len(pairs),
        "example_macro_precision": _mean(precisions),
        "example_macro_recall": _mean(recalls),
        "example_macro_f1": _mean(f1s),
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "empty_empty_policy": "both empty scores precision = recall = f1 = 1",
        "empty_gold_nonempty_prediction_policy": "precision = 0 and recall = 0",
    }
