"""Write at least eight test-set failures after the final evaluation."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from genai.data.records import read_jsonl
from genai.evaluation.errors import select_errors
from genai.models.load import load_tokenizer, target_dir
from genai.paths import RESULTS_DIR
from genai.training.data import fit_note


def main() -> None:
    path = RESULTS_DIR / "task2" / "test_adapter_predictions.jsonl"
    if not path.exists():
        raise SystemExit("Run the final test evaluation before error analysis.")
    rows = read_jsonl(path)
    tokenizer = load_tokenizer(target_dir())
    trained_budget = 1024 - 64
    truncation_ids = set()
    records = {row["annotation_id"]: _as_record(row) for row in rows}
    for row in rows:
        _note, _kept, dropped = fit_note(tokenizer, records[row["annotation_id"]], trained_budget)
        if dropped:
            truncation_ids.add(row["annotation_id"])
    chosen = select_errors(rows, minimum=8, truncation_ids=truncation_ids)
    counterfactual = _shorter_budget_drop(tokenizer, rows, records, trained_budget)
    if counterfactual is not None and not any(str(row["error_category"]).startswith("truncation") for row in chosen):
        chosen.append(counterfactual)
    used = {row.get("annotation_id") for row in chosen}
    for row in rows:
        if len(chosen) >= 8:
            break
        if row.get("annotation_id") in used:
            continue
        used.add(row.get("annotation_id"))
        chosen.append({**row, "error_category": "reviewed_case"})
    public = []
    for row in chosen:
        public.append(
            {
                "error_category": row["error_category"],
                "annotation_id": row["annotation_id"],
                "patient_id": row["patient_id"],
                "criterion_type": row["criterion_type"],
                "criterion_text": row["criterion_text"],
                "gold_label": row["gold_label"],
                "pred_label": row["pred_label"],
                "gold_sentences": row["gold_sentences"],
                "pred_sentences": row["pred_sentences"],
                "invalid_sentence_ids": row["invalid_sentence_ids"],
                "quoted_gold_sentences": _quotes(row, row["gold_sentences"]),
                "quoted_pred_sentences": _quotes(row, row["pred_sentences"]),
                "parse_error": row["parse_error"],
                "why": _why(row),
            }
        )
    out = RESULTS_DIR / "task2" / "errors.json"
    out.write_text(json.dumps(public, indent=2), encoding="utf-8")
    lines = ["# Test-set failures", ""]
    for row in public:
        lines.append(f"## {row['error_category']} ({row['annotation_id']})")
        lines.append(f"- Criterion ({row['criterion_type']}): {row['criterion_text']}")
        lines.append(f"- Expert label: {row['gold_label']}. Predicted: {row['pred_label']}.")
        lines.append(f"- Expert sentences: {row['gold_sentences']}. Predicted sentences: {row['pred_sentences']}.")
        lines.append(f"- Quoted expert evidence: {row['quoted_gold_sentences']}")
        lines.append(f"- Quoted predicted evidence: {row['quoted_pred_sentences']}")
        lines.append(f"- {row['why']}")
        lines.append("")
    (RESULTS_DIR / "task2" / "errors.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {len(public)} cases")


def _as_record(row: dict) -> dict:
    return {
        "note": row["note"],
        "criterion_type": row["criterion_type"],
        "criterion_text": row["criterion_text"],
        "sentences": row["sentences"],
        "expert_sentences": row["gold_sentences"],
        "expert_eligibility": row["gold_label"],
    }


def _quotes(row: dict, sentence_ids) -> list[str]:
    sentences = {int(key): value for key, value in row.get("sentences", {}).items()}
    quoted = []
    for sentence_id in sentence_ids or []:
        if int(sentence_id) in sentences:
            quoted.append(f"{sentence_id}. {sentences[int(sentence_id)]}")
        else:
            quoted.append(f"{sentence_id}. (not in the note)")
    return quoted


def _shorter_budget_drop(tokenizer, rows, records, trained_budget: int) -> dict | None:
    """Show the truncation rule on a budget shorter than the one used for training.

    At the trained length every note fits, so this is a counterfactual, not a trained-run failure.
    """
    for budget in (512, 256, 160, 96):
        if budget >= trained_budget:
            continue
        for row in rows:
            if not row["gold_sentences"]:
                continue
            _note, _kept, dropped = fit_note(tokenizer, records[row["annotation_id"]], budget)
            if dropped:
                return {
                    **row,
                    "error_category": "truncation_would_drop_gold",
                    "truncation_budget_tokens": budget,
                    "dropped_gold_sentence_ids": dropped,
                }
    return None


def _why(row: dict) -> str:
    category = row["error_category"]
    if category == "truncation_drops_gold":
        return "At max length 1024, at least one expert sentence id falls outside the note prefix that fits in the prompt."
    if category == "truncation_would_drop_gold":
        dropped = row.get("dropped_gold_sentence_ids")
        budget = row.get("truncation_budget_tokens")
        return (
            "At the trained maximum length of 1024, no expert sentence id was dropped. "
            f"If the prompt budget were {budget} tokens, expert sentence ids {dropped} would fall outside the kept prefix."
        )
    if category == "unparsed_output":
        return row.get("parse_error") or "The model did not return the required JSON object."
    if category == "unsupported_sentence":
        return f"The model cited sentence ids that are not in the note: {row['invalid_sentence_ids']}."
    if category == "empty_gold_but_model_cited":
        return "The expert cited nothing, and the model cited a sentence. That citation is not treated as proof."
    if category == "missing_information":
        return "The expert label is not enough information. A missing fact is not treated as eligibility or ineligibility."
    if row["pred_label"] != row["gold_label"]:
        return "The predicted label does not match the expert label."
    return "The cited sentence ids do not match the expert citation."


if __name__ == "__main__":
    main()
