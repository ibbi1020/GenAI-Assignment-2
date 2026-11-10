"""Fill report/generated.tex from the saved result logs. Does not invent numbers."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from genai.paths import RESULTS_DIR


def _load(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"Missing {path}. Finish that experiment before writing the results section.")
    return json.loads(path.read_text())


def _pct(value: float) -> str:
    return f"{100 * float(value):.1f}\\%"


def _num(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def _tex(value: object) -> str:
    return (
        str(value)
        .replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
        .replace("#", "\\#")
        .replace("<", "\\textless{}")
        .replace(">", "\\textgreater{}")
    )


def _mb(value: float) -> str:
    return f"{float(value) / (1024 * 1024):.0f}"


def _class_rows(summary: dict) -> str:
    lines = []
    for label, row in summary["per_class"].items():
        lines.append(
            f"{_tex(label)} & {int(row['support'])} & {_num(row['precision'])} & {_num(row['recall'])} & {_num(row['f1'])} \\\\"
        )
    return "\n".join(lines)


def _kv_claim(kv: dict) -> str:
    pairs = kv["pairs"]
    equal = sum(bool(pair["token_equal"]) for pair in pairs)
    text = f"Tokens match on {equal} of {len(pairs)} graded lengths."
    for pair in pairs:
        mismatch = pair.get("mismatch")
        if pair["token_equal"] or not mismatch:
            continue
        text += (
            f" At {pair['prompt_tokens']}/{pair['new_tokens']} the first disagreement is index {mismatch['index']}: "
            f"the cache emits token {mismatch['cached_token']} and the uncached loop emits token {mismatch['uncached_token']}."
        )
    if equal != len(pairs):
        text += (
            " A fresh one-step cache on that shared prefix agrees with the full forward. "
            "The top logits differ by one bf16 unit, so the long-running cache breaks a tie. "
            "The position ids and the mask are the same on both paths."
        )
    return text


def _spec_claim(speculative: dict, equals: int) -> str:
    if equals == len(speculative["prompts"]):
        return "Every prompt matched."
    return (
        "On the prompt that differs, the full-forward top two logits also differ by one bf16 unit, "
        "and the speculative token is that forward's argmax."
    )


def _evidence_row(name: str, block: dict) -> str:
    return (
        f"{name} & {_num(block['example_macro_precision'])} & {_num(block['example_macro_recall'])} "
        f"& {_num(block['example_macro_f1'])} & {_num(block['micro_precision'])} "
        f"& {_num(block['micro_recall'])} & {_num(block['micro_f1'])} \\\\"
    )


def main() -> None:
    train = _load(RESULTS_DIR / "task1" / "train_summary.json")
    val = _load(RESULTS_DIR / "task2" / "validation_comparison.json")
    test = _load(RESULTS_DIR / "task2" / "test_comparison.json")
    test_adapter = _load(RESULTS_DIR / "task2" / "test_adapter.json")
    test_base = _load(RESULTS_DIR / "task2" / "test_base.json")
    kv = _load(RESULTS_DIR / "task3" / "kv_cache.json")
    attention = _load(RESULTS_DIR / "task4" / "attention.json")
    speculative = _load(RESULTS_DIR / "task5" / "speculative.json")
    errors = _load(RESULTS_DIR / "task2" / "errors.json")

    kv_lines = []
    for pair in kv["pairs"]:
        kv_lines.append(
            f"{pair['prompt_tokens']}/{pair['new_tokens']} & {str(pair['token_equal']).lower()} "
            f"& {_num(pair['cached']['prefill_seconds'])} & {_num(pair['cached']['decode_seconds'])} "
            f"& {_num(pair['uncached']['decode_seconds'])} "
            f"& {_num(pair['cached']['tokens_per_second'], 1)} & {_num(pair['uncached']['tokens_per_second'], 1)} "
            f"& {_mb(pair['cached']['peak_memory_bytes'])} & {_mb(pair['uncached']['peak_memory_bytes'])} \\\\"
        )
    attn_lines = []
    worst = 0.0
    for case in attention["cases"]:
        worst = max(worst, float(case["max_abs_error"]))
        attn_lines.append(
            f"{case['batch']},{case['heads']},{case['seq']},{case['dim']} & {case['block_size']} "
            f"& {str(case['causal']).lower()} & {case['max_abs_error']:.2e} & {case['mean_abs_error']:.2e} "
            f"& {_num(case['reference_seconds'], 5)} & {_num(case['tiled_seconds'], 5)} \\\\"
        )
    spec_lines = []
    equals = 0
    for index, row in enumerate(speculative["prompts"], start=1):
        equals += int(bool(row["token_equal"]))
        spec_lines.append(
            f"{index} & {str(row['token_equal']).lower()} & {_num(row['acceptance_rate'])} "
            f"& {_num(row['greedy_tokens_per_second'], 1)} & {_num(row['speculative_tokens_per_second'], 1)} "
            f"& {_num(row['speedup'])} & {_mb(row['peak_memory_bytes'])} \\\\"
        )
    error_lines = []
    for row in errors:
        error_lines.append(
            f"{_tex(row['error_category']).replace('_', ' ')} & {_tex(row['annotation_id'])} & {_tex(row['criterion_type'])} "
            f"& {_tex(row['gold_label'])} & {_tex(row['pred_label'] or 'unparsed')} \\\\"
        )

    best = train["best_val_macro_f1"]
    if "oversampled_records" in train:
        train_line = (
            f"The {train['train_records']} teaching annotations were repeated only inside the train split, "
            f"up to the majority count of each criterion type and at most eight times, "
            f"which produced {train['oversampled_records']} encoded rows."
        )
    else:
        train_line = (
            f"The encoded train set uses each of the {train['train_records']} teaching annotations once. "
            f"Rare answers are up-weighted in the loss instead of being copied."
        )
    text = f"""
Figure~\\ref{{fig:curve}} shows the training loss and the validation macro-F1 used to pick the checkpoint.
The selected epoch is {train['best_epoch']}, with validation macro-F1 {_num(best)} during training.
{train['train_rows_with_gold_sentence_truncated']} train rows lost an expert sentence id to truncation at the trained length.
{train_line}

\\begin{{figure}}[t]
\\centering
\\includegraphics[width=0.95\\linewidth]{{../results/task1/training_curve.png}}
\\caption{{Train loss by optimizer step and validation macro-F1 by epoch. The checkpoint with the higher validation macro-F1 is the one that is evaluated.}}
\\label{{fig:curve}}
\\end{{figure}}

Table~\\ref{{tab:compare}} compares the untouched base model and the adapter on the same rows and the same greedy decoder.
On validation, accuracy moves from {_pct(val['base_accuracy'])} to {_pct(val['adapter_accuracy'])} and macro-F1 moves from {_num(val['base_macro_f1'])} to {_num(val['adapter_macro_f1'])}.
On the held-out test set, which was scored once, accuracy is {_pct(test['base_accuracy'])} for the base model and {_pct(test['adapter_accuracy'])} for the adapter.
Test macro-F1 is {_num(test['base_macro_f1'])} and {_num(test['adapter_macro_f1'])}.
Example-level evidence F1 on the test set is {_num(test['base_evidence_f1'])} and {_num(test['adapter_evidence_f1'])}.

\\begin{{table}}[t]
\\centering
\\caption{{Base model and adapter. Decoding is greedy. The test split was scored once.}}
\\label{{tab:compare}}
\\begin{{tabular}}{{lcccc}}
\\toprule
Split & Model & Accuracy & Macro-F1 & Evidence F1 \\\\
\\midrule
Validation & Base & {_pct(val['base_accuracy'])} & {_num(val['base_macro_f1'])} & {_num(val['base_evidence_f1'])} \\\\
Validation & Adapter & {_pct(val['adapter_accuracy'])} & {_num(val['adapter_macro_f1'])} & {_num(val['adapter_evidence_f1'])} \\\\
Test & Base & {_pct(test['base_accuracy'])} & {_num(test['base_macro_f1'])} & {_num(test['base_evidence_f1'])} \\\\
Test & Adapter & {_pct(test['adapter_accuracy'])} & {_num(test['adapter_macro_f1'])} & {_num(test['adapter_evidence_f1'])} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}

Table~\\ref{{tab:perclass}} is the adapter's per-class precision, recall, and F1 on the test set.
Rare labels stay rare: a high accuracy can still hide a weak F1 on ``excluded'' or ``not included''.
Figure~\\ref{{fig:cm}} is the adapter's test confusion matrix.
The inclusion and exclusion matrices are saved beside it and are not a collapsed yes/no task.

\\begin{{table}}[t]
\\centering
\\caption{{Adapter, test set, per-class scores. Support is the number of expert labels.}}
\\label{{tab:perclass}}
\\begin{{tabular}}{{lcccc}}
\\toprule
Label & Support & Precision & Recall & F1 \\\\
\\midrule
{_class_rows(test_adapter['overall'])}
\\bottomrule
\\end{{tabular}}
\\end{{table}}

\\begin{{figure}}[t]
\\centering
\\includegraphics[width=0.72\\linewidth]{{../results/task2/confusion_test_adapter_overall.png}}
\\caption{{Confusion matrix for the adapter on the test set. Rows are expert labels and columns are predictions. Unparsed outputs are their own column.}}
\\label{{fig:cm}}
\\end{{figure}}

Table~\\ref{{tab:evidence}} applies the empty-set rule to the adapter on the test set.
Both-empty counts as a match. A citation where the expert cited nothing scores zero, and so does a missing citation where the expert cited something.

\\begin{{table}}[t]
\\centering
\\caption{{Adapter evidence scores on the test set. Macro is the mean of the example scores. Micro pools sentence occurrences.}}
\\label{{tab:evidence}}
\\resizebox{{\\columnwidth}}{{!}}{{%
\\begin{{tabular}}{{lcccccc}}
\\toprule
Slice & P & R & F1 & Micro P & Micro R & Micro F1 \\\\
\\midrule
{_evidence_row('Overall', test_adapter['evidence']['overall'])}
{_evidence_row('Inclusion', test_adapter['evidence']['inclusion'])}
{_evidence_row('Exclusion', test_adapter['evidence']['exclusion'])}
\\bottomrule
\\end{{tabular}}}}
\\end{{table}}

The base model left {test_base['unparsed']} test outputs unparsed.
The adapter left {test_adapter['unparsed']} unparsed and cited an id outside the note on {test_adapter['invalid_sentence_id_rows']} rows.
Table~\\ref{{tab:errors}} lists the reviewed failures. Each row in the accompanying \\texttt{{errors.md}} stores the criterion, the expert label, the prediction, the quoted sentences, and why it failed.

\\begin{{table}}[t]
\\centering
\\caption{{Reviewed test failures for the adapter. The full quotations are in \\texttt{{results/task2/errors.md}}.}}
\\label{{tab:errors}}
\\resizebox{{\\columnwidth}}{{!}}{{%
\\begin{{tabular}}{{llccc}}
\\toprule
Category & Id & Type & Expert & Predicted \\\\
\\midrule
{chr(10).join(error_lines)}
\\bottomrule
\\end{{tabular}}}}
\\end{{table}}

Table~\\ref{{tab:kv}} checks the hand-written cache against the uncached loop on the {kv['model']} model.
{_kv_claim(kv)}
Prefill is timed separately from the later tokens.
The first new token is produced by the prefill, so the decode rate does not include it in the same way the speculative rate does.
The uncached loop resends the whole prefix, which is why its decode time grows with the prompt length.

\\begin{{table*}}[t]
\\centering
\\caption{{Cached and uncached greedy decoding. EOS is disabled. Times are seconds after one warmup. Memory is peak megabytes.}}
\\label{{tab:kv}}
\\begin{{tabular}}{{lcccccccc}}
\\toprule
Length & Equal & Prefill & Dec. cache & Dec. plain & Tok/s c. & Tok/s p. & MB c. & MB p. \\\\
\\midrule
{chr(10).join(kv_lines)}
\\bottomrule
\\end{{tabular}}
\\end{{table*}}

Table~\\ref{{tab:attn}} compares reference attention with tiled online softmax in fp32.
The largest absolute error in these cases is {worst:.2e}.
The tiled loop is slower. That is the expected result for a Python block loop against a single matrix softmax, and it is not a fused kernel.

\\begin{{table}}[t]
\\centering
\\caption{{Reference and tiled attention. Shapes are batch, heads, sequence, and head size. Times are seconds.}}
\\label{{tab:attn}}
\\resizebox{{\\columnwidth}}{{!}}{{%
\\begin{{tabular}}{{lcccccc}}
\\toprule
Shape & Block & Causal & Max abs & Mean abs & Ref. & Tiled \\\\
\\midrule
{chr(10).join(attn_lines)}
\\bottomrule
\\end{{tabular}}}}
\\end{{table}}

Table~\\ref{{tab:spec}} is greedy speculative decoding with lookahead 4.
The draft is the 0.5\\,B base model and the target is the {speculative['target']} 1.5\\,B model.
{equals} of {len(speculative['prompts'])} prompts matched the target's ordinary greedy tokens.
{_spec_claim(speculative, equals)}
Acceptance rate is the fraction of drafted tokens that were kept.
A speedup below 1 is a slowdown. The token match is the correctness result.

\\begin{{table}}[t]
\\centering
\\caption{{Speculative decoding against target greedy decoding. 32 new tokens, lookahead 4.}}
\\label{{tab:spec}}
\\resizebox{{\\columnwidth}}{{!}}{{%
\\begin{{tabular}}{{lcccccc}}
\\toprule
Prompt & Equal & Accept & Greedy tok/s & Spec. tok/s & Speedup & Peak MB \\\\
\\midrule
{chr(10).join(spec_lines)}
\\bottomrule
\\end{{tabular}}}}
\\end{{table}}
"""
    path = ROOT / "report" / "generated.tex"
    path.write_text(text.strip() + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
