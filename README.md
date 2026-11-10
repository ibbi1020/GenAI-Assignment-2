# Criterion screening

Educational prototype that matches a patient note to individual inclusion and exclusion criteria. It is not a clinical enrollment tool.

The dataset is [ncbi/TrialGPT-Criterion-Annotations](https://huggingface.co/datasets/ncbi/TrialGPT-Criterion-Annotations). The split is patient-grouped: sort the patient ids, shuffle with `numpy.random.default_rng(42)`, then 37 train, 8 validation, and the rest test. Do not tune prompts, thresholds, or checkpoints on the test patients. The test script refuses to run unless you pass `--final`.

GPT-4 columns in the dataset are never model inputs or training targets. The dataset `training` flag is ignored.

## Setup

From this folder:

```bash
uv sync
uv run python scripts/prepare_data.py
uv run python scripts/download_models.py
```

`prepare_data.py` writes `data/splits/manifest.json`. `download_models.py` saves `Qwen/Qwen2.5-1.5B-Instruct` and `Qwen/Qwen2.5-0.5B-Instruct` under `models/`. Those folders are gitignored. NumPy is pinned to 2.2.6 so the patient shuffle stays fixed.

```bash
uv run pytest
```

## Train

Training reads only the train and validation jsonl files. The checkpoint with the best validation macro-F1 is saved to `outputs/task1/adapter`. If length 1024 does not fit in memory, the script retries 768 and then 512.

```bash
uv run python scripts/train.py
```

Curves and the run summary land in `results/task1/`.

## Evaluate

Validation, any time after training:

```bash
uv run python scripts/evaluate.py --split validation
```

Final test, once:

```bash
uv run python scripts/evaluate.py --split test --final
uv run python scripts/error_analysis.py
```

Both commands compare the untouched base model with the adapter under greedy decoding.

## Benchmarks

```bash
uv run python scripts/bench_attention.py
uv run python scripts/bench_kv.py
uv run python scripts/bench_speculative.py
```

Attention is a plain PyTorch reference and a tiled online-softmax implementation. It does not call a fused kernel, and it is not expected to be faster than one.

The KV benchmark calls the model forward itself, with and without a key/value cache, at prompt/new-token lengths 32/16, 128/32, and 512/64. EOS is off so both paths emit the same number of tokens.

Speculative decoding uses the 0.5B model as the draft and the 1.5B model as the target. The draft is not fine-tuned. A slowdown is recorded if that is what the timing shows.

## Application

Restart the API after training so it loads the adapter.

```bash
uv run uvicorn app.backend.main:app --port 8000
```

In another terminal:

```bash
cd app/frontend
npm install
npm run dev
```

Open http://127.0.0.1:5173.

The Screening page accepts a note and any number of criteria, shows each label with the quoted sentences, and downloads an HTML report. The Inference lab runs the cache comparison, speculative decoding, and tiled attention when you click the buttons.

Invalid notes, empty criteria, unknown types, and model text that is not the required JSON are shown as errors. Earlier criterion cards stay on the page.

## Demo shot list (5–7 minutes)

1. Show the notice that this is not an enrollment decision.
2. Load a train-split example, or paste a short unseen note and two criteria, one inclusion and one exclusion.
3. Run screening. Point at the label, the sentence ids, the quoted text, and the missing-information flag.
4. Show the overall line and read the aggregation rule. Say that it is not a ground-truth trial label.
5. Download the HTML report and open it.
6. Open the Inference lab. Run cached versus uncached greedy and point at token equality and the times.
7. Run speculative versus greedy and point at the acceptance rate, whether the tokens match, and the speedup or slowdown.
8. Run reference versus tiled attention and point at the max and mean error.
9. Submit an empty note and show the error, then show that the previous result is still there.

Record the video yourself. Do not push this repository until you decide to.

## Models and hardware

Target: Qwen2.5-1.5B-Instruct, about 1.54B parameters, 32,768 context, Apache 2.0. Draft: Qwen2.5-0.5B-Instruct, same tokenizer. LoRA rank 8 on `q_proj` and `v_proj`. This project was sized for an Apple M5 with 16 GB of unified memory. Docker is not the supported path, because a Linux container cannot use the Mac GPU.

## Report

The IEEE paper is `report/main.tex`, and the built file is `report/main.pdf`. From `report/`, rebuild it with `tectonic main.tex` or `pdflatex main.tex` after `python scripts/write_report_section.py`.
