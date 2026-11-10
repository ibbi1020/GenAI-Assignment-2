"""Compare the base model and the adapter. Test requires --final."""

import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import gc
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from genai.data.records import read_jsonl, write_jsonl
from genai.evaluation.predict import predict_records, summarize_predictions, write_metrics
from genai.models.load import adapter_ready, load_causal_lm, load_tokenizer, target_dir
from genai.paths import ADAPTER_DIR, PROCESSED_DIR, RESULTS_DIR


def _plot(summary: dict, path) -> None:
    matrix = summary["confusion_matrix"]
    labels = matrix["labels"]
    values = matrix["matrix"]
    figure, axis = plt.subplots(figsize=(6.2, 5.2))
    image = axis.imshow(values, cmap="Blues")
    axis.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    axis.set_yticks(range(len(labels)), labels)
    for row_index, row in enumerate(values):
        for col_index, count in enumerate(row):
            axis.text(col_index, row_index, str(count), ha="center", va="center", color="#221e1a")
    figure.colorbar(image, ax=axis, fraction=0.046)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Expert label")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _run(name: str, records: list[dict], adapter: bool, max_new_tokens: int) -> dict:
    tokenizer = load_tokenizer(target_dir())
    model = load_causal_lm(target_dir(), ADAPTER_DIR if adapter else None)
    print(f"scoring {name}", flush=True)
    rows = predict_records(
        model,
        tokenizer,
        records,
        max_new_tokens=max_new_tokens,
        progress=lambda done, total: print(f"  {name} {done}/{total}", flush=True) if done % 10 == 0 or done == total else None,
    )
    summary = summarize_predictions(rows)
    summary["model"] = "adapter" if adapter else "base"
    summary["decoding"]["max_new_tokens"] = max_new_tokens
    del model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return summary, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["validation", "test"], required=True)
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()
    if args.split == "test" and not args.final:
        raise SystemExit("The test split is held out. Re-run with --final only for the final evaluation.")
    if not adapter_ready():
        raise SystemExit("No adapter at outputs/task1/adapter. Train first.")
    records = read_jsonl(PROCESSED_DIR / f"{args.split}.jsonl")
    out = RESULTS_DIR / "task2"
    base_summary, base_rows = _run("base", records, adapter=False, max_new_tokens=args.max_new_tokens)
    adapter_summary, adapter_rows = _run("adapter", records, adapter=True, max_new_tokens=args.max_new_tokens)
    write_metrics(out / f"{args.split}_base.json", base_summary)
    write_metrics(out / f"{args.split}_adapter.json", adapter_summary)
    write_jsonl(out / f"{args.split}_base_predictions.jsonl", base_rows)
    write_jsonl(out / f"{args.split}_adapter_predictions.jsonl", adapter_rows)
    for label, summary in ("base", base_summary), ("adapter", adapter_summary):
        _plot(summary["overall"], out / f"confusion_{args.split}_{label}_overall.png")
        _plot(summary["inclusion"], out / f"confusion_{args.split}_{label}_inclusion.png")
        _plot(summary["exclusion"], out / f"confusion_{args.split}_{label}_exclusion.png")
    comparison = {
        "split": args.split,
        "decoding": adapter_summary["decoding"],
        "base_accuracy": base_summary["overall"]["accuracy"],
        "adapter_accuracy": adapter_summary["overall"]["accuracy"],
        "base_macro_f1": base_summary["overall"]["macro_f1"],
        "adapter_macro_f1": adapter_summary["overall"]["macro_f1"],
        "base_evidence_f1": base_summary["evidence"]["overall"]["example_macro_f1"],
        "adapter_evidence_f1": adapter_summary["evidence"]["overall"]["example_macro_f1"],
    }
    (out / f"{args.split}_comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
