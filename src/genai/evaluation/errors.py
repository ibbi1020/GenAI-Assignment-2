"""Pick at least eight distinct failure rows from a finished test prediction file."""

from __future__ import annotations


def select_errors(rows: list[dict], minimum: int = 8, truncation_ids: set[object] | None = None) -> list[dict]:
    """Return distinct annotated failures. Categories are assigned in order."""
    used: set[object] = set()
    chosen: list[dict] = []

    def take(predicate, category: str) -> None:
        if len(chosen) >= minimum and category != "truncation_drops_gold":
            return
        for row in rows:
            if row.get("annotation_id") in used:
                continue
            if predicate(row):
                used.add(row.get("annotation_id"))
                chosen.append({**row, "error_category": category})
                return

    take(lambda row: row["pred_label"] is None, "unparsed_output")
    take(lambda row: row["pred_label"] not in (None, row["gold_label"]), "wrong_label")
    take(
        lambda row: row["pred_label"] is not None
        and (
            (row["criterion_type"] == "inclusion" and row["pred_label"] in {"excluded", "not excluded"})
            or (row["criterion_type"] == "exclusion" and row["pred_label"] in {"included", "not included"})
        ),
        "cross_type_label",
    )
    take(lambda row: row["gold_label"] == "not enough information" and row["pred_label"] != row["gold_label"], "missing_information")
    take(lambda row: bool(row.get("invalid_sentence_ids")), "unsupported_sentence")
    take(
        lambda row: set(row["gold_sentences"]) - set(row["pred_sentences"]) != set() and row["pred_label"] != row["gold_label"],
        "missed_gold_sentence",
    )
    take(
        lambda row: not row["gold_sentences"] and bool(row["pred_sentences"]),
        "empty_gold_but_model_cited",
    )
    take(
        lambda row: row["gold_label"] == "not applicable" and row["pred_label"] != "not applicable",
        "not_applicable_confusion",
    )
    truncation_ids = truncation_ids or set()
    take(
        lambda row: row.get("annotation_id") in truncation_ids,
        "truncation_drops_gold",
    )
    for row in rows:
        if len(chosen) >= minimum:
            break
        if row.get("annotation_id") in used:
            continue
        if row["pred_label"] != row["gold_label"] or set(row["gold_sentences"]) != set(row["pred_sentences"]):
            used.add(row.get("annotation_id"))
            chosen.append({**row, "error_category": "additional_mismatch"})
    return chosen
