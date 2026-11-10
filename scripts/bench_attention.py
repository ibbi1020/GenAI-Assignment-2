"""Write the attention benchmark log. This does not load the language model."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from genai.benchmarks.attention_bench import run_attention_case
from genai.paths import RESULTS_DIR

CASES = [
    {"batch": 1, "heads": 1, "seq": 8, "dim": 16, "block_size": 4, "causal": True},
    {"batch": 2, "heads": 4, "seq": 17, "dim": 32, "block_size": 8, "causal": True},
    {"batch": 1, "heads": 2, "seq": 64, "dim": 64, "block_size": 16, "causal": True},
    {"batch": 1, "heads": 2, "seq": 64, "dim": 64, "block_size": 16, "causal": False},
    {"batch": 1, "heads": 4, "seq": 128, "dim": 64, "block_size": 32, "causal": True},
]


def main() -> None:
    rows = [run_attention_case(**case, repeats=5) for case in CASES]
    path = RESULTS_DIR / "task4" / "attention.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"cases": rows}, indent=2), encoding="utf-8")
    worst = max(row["max_abs_error"] for row in rows)
    print(f"wrote {path} worst max abs error {worst}")


if __name__ == "__main__":
    main()
