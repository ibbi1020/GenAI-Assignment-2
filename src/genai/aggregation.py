"""Trial-level line. This is a rule, not a labeled outcome in the dataset."""

from __future__ import annotations

from genai.constants import AGGREGATION_RULE


def aggregate(criteria: list[dict[str, str]]) -> dict[str, object]:
    hard: list[str] = []
    missing: list[str] = []
    for index, row in enumerate(criteria):
        label = row["eligibility"]
        kind = row["criterion_type"]
        if label == "not applicable":
            continue
        if (kind == "inclusion" and label == "not included") or (kind == "exclusion" and label == "excluded"):
            hard.append(f"{index}:{kind}:{label}")
        elif label == "not enough information":
            missing.append(f"{index}:{kind}:{label}")
    if hard:
        outcome = "ineligible"
    elif missing:
        outcome = "needs more information"
    else:
        outcome = "potentially eligible"
    return {
        "outcome": outcome,
        "is_ground_truth": False,
        "rule": AGGREGATION_RULE,
        "hard_fail_count": len(hard),
        "missing_information_count": len(missing),
    }
