"""Fixed choices from the assignment plan. Do not retune these on the test set."""

from __future__ import annotations

DATASET_ID = "ncbi/TrialGPT-Criterion-Annotations"
TARGET_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DRAFT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
SPLIT_SEED = 42
TRAIN_PATIENTS = 37
VAL_PATIENTS = 8

INCLUSION_LABELS = (
    "included",
    "not included",
    "not enough information",
    "not applicable",
)
EXCLUSION_LABELS = (
    "excluded",
    "not excluded",
    "not enough information",
    "not applicable",
)
ALL_LABELS = (
    "included",
    "not included",
    "excluded",
    "not excluded",
    "not enough information",
    "not applicable",
)

# Model-generated columns. Never used as inputs, features, or training targets.
FORBIDDEN_FIELDS = (
    "gpt4_explanation",
    "explanation_correctness",
    "gpt4_sentences",
    "gpt4_eligibility",
)

DISCLAIMER = (
    "This is an educational screening prototype. It is not a clinical "
    "enrollment decision and must not be used to enroll or exclude a patient from a trial."
)

AGGREGATION_RULE = (
    "This overall line is an aggregation, not a ground-truth label. "
    "The trial is ineligible if any inclusion criterion is 'not included' or any "
    "exclusion criterion is 'excluded'. Otherwise it needs more information if any "
    "scored criterion is 'not enough information'. Otherwise it is potentially eligible. "
    "Criteria labeled 'not applicable' do not count. An empty evidence list is not "
    "proof of eligibility or ineligibility."
)

# Graded KV-cache length pairs: (prompt tokens, new tokens).
KV_LENGTH_PAIRS = ((32, 16), (128, 32), (512, 64))
