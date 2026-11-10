import pytest

from genai.aggregation import aggregate
from genai.constants import FORBIDDEN_FIELDS
from genai.explain import evidence_explanation, missing_information
from genai.metrics.classification import classification_summary
from genai.metrics.evidence import evidence_summary, example_precision_recall
from genai.parsing import OutputParseError, check_sentence_ids, parse_prediction
from genai.prompts import build_messages, prompt_from_record


def test_prompt_rejects_gpt4_fields():
    with pytest.raises(ValueError, match="gpt4_explanation"):
        build_messages("0. Note.", "inclusion", "Age over 18", gpt4_explanation="leak")
    record = {
        "note": "0. Note.",
        "criterion_type": "exclusion",
        "criterion_text": "Prior stroke",
        "gpt4_eligibility": "excluded",
    }
    with pytest.raises(ValueError, match="gpt4_eligibility"):
        prompt_from_record(record)


def test_prompt_lists_the_label_set_for_the_criterion_type():
    inclusion = build_messages("0. Note.", "inclusion", "Age over 18")
    exclusion = build_messages("0. Note.", "exclusion", "Prior stroke")
    assert "not included" in inclusion[1]["content"]
    assert "excluded" not in inclusion[1]["content"]
    assert "not excluded" in exclusion[1]["content"]
    assert "included" not in exclusion[1]["content"].split("Allowed eligibility labels: ")[1]
    joined = inclusion[0]["content"] + inclusion[1]["content"]
    for field in FORBIDDEN_FIELDS:
        assert field not in joined
    assert "empty list" in inclusion[0]["content"]
    assert "not proof" in inclusion[0]["content"]
    assert "not enough information" in inclusion[1]["content"]


def test_answer_weights_balance_real_notes_without_copying():
    from genai.training.data import answer_weights

    records = []
    for label, count in (("not excluded", 4), ("excluded", 1), ("not enough information", 2)):
        for index in range(count):
            records.append({"criterion_type": "exclusion", "expert_eligibility": label, "annotation_id": f"{label}-{index}"})
    weights = answer_weights(records)
    totals = {}
    for record in records:
        key = (record["criterion_type"], record["expert_eligibility"])
        totals[key] = totals.get(key, 0.0) + weights[key]
    assert len(totals) == 3
    assert len({round(value, 6) for value in totals.values()}) == 1
    mean = sum(weights[(row["criterion_type"], row["expert_eligibility"])] for row in records) / len(records)
    assert mean == pytest.approx(1.0)
    assert weights[("exclusion", "excluded")] == pytest.approx(4 * weights[("exclusion", "not excluded")])


def test_parse_prediction_rejects_wrong_label_and_bad_ids():
    parsed = parse_prediction('{"eligibility": "Included", "sentence_ids": ["0", 1]}', "inclusion")
    assert parsed == {"eligibility": "included", "sentence_ids": [0, 1]}
    with pytest.raises(OutputParseError, match="not allowed"):
        parse_prediction('{"eligibility": "excluded", "sentence_ids": []}', "inclusion")
    with pytest.raises(OutputParseError, match="not in the numbered note"):
        check_sentence_ids([0, 4], {0: "only"})


def test_evidence_empty_sets():
    assert example_precision_recall(set(), set()) == (1.0, 1.0)
    assert example_precision_recall(set(), {1}) == (0.0, 0.0)
    assert example_precision_recall({1}, set()) == (0.0, 0.0)
    assert example_precision_recall({1, 2}, {2, 3}) == (0.5, 0.5)
    summary = evidence_summary([(set(), set()), ({1}, {1}), (set(), {2})])
    assert summary["example_macro_f1"] == pytest.approx((1.0 + 1.0 + 0.0) / 3)
    assert summary["micro_precision"] == pytest.approx(1 / 2)


def test_classification_keeps_separate_labels():
    gold = ["included", "not included", "not enough information"]
    pred = ["included", "included", "not enough information"]
    summary = classification_summary(gold, pred, ["included", "not included", "not enough information", "not applicable"])
    assert summary["accuracy"] == pytest.approx(2 / 3)
    assert summary["per_class"]["not included"]["recall"] == 0.0
    assert "not applicable" not in summary["per_class"]


def test_aggregation_rule():
    assert aggregate([{"criterion_type": "inclusion", "eligibility": "included"}])["outcome"] == "potentially eligible"
    assert aggregate([{"criterion_type": "inclusion", "eligibility": "not applicable"}])["outcome"] == "potentially eligible"
    assert aggregate([{"criterion_type": "inclusion", "eligibility": "not included"}])["outcome"] == "ineligible"
    assert aggregate([{"criterion_type": "exclusion", "eligibility": "excluded"}])["outcome"] == "ineligible"
    mixed = aggregate(
        [
            {"criterion_type": "inclusion", "eligibility": "not included"},
            {"criterion_type": "exclusion", "eligibility": "not enough information"},
        ]
    )
    assert mixed["outcome"] == "ineligible"
    assert mixed["is_ground_truth"] is False
    missing = aggregate([{"criterion_type": "exclusion", "eligibility": "not enough information"}])
    assert missing["outcome"] == "needs more information"


def test_explanation_does_not_treat_empty_citations_as_proof():
    text = evidence_explanation("not enough information", [], {0: "Age 58."})
    assert "not proof" in text
    quoted = evidence_explanation("included", [0], {0: "Age 58."})
    assert "Age 58." in quoted
    assert missing_information("included", []) is True
    assert missing_information("not enough information", [0]) is True
    assert missing_information("included", [0]) is False
