"""LoRA fine-tuning on the train patients, with model choice on validation only."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from genai.device import best_device
from genai.evaluation.predict import score_records
from genai.paths import ADAPTER_DIR, RESULTS_DIR
from genai.training.data import answer_weights, encode_example, label_counts


class _Encoded(Dataset):
    def __init__(self, rows: list[dict[str, list[int]]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.rows[index]


def _collate(features: list[dict[str, list[int]]], pad_id: int) -> dict[str, torch.Tensor]:
    width = max(len(row["input_ids"]) for row in features)
    input_ids = []
    labels = []
    mask = []
    for row in features:
        pad = width - len(row["input_ids"])
        input_ids.append(row["input_ids"] + [pad_id] * pad)
        labels.append(row["labels"] + [-100] * pad)
        mask.append([1] * len(row["input_ids"]) + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(mask, dtype=torch.long),
        "weight": torch.tensor([float(row["weight"]) for row in features], dtype=torch.float32),
    }


def _save_curve(history: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    steps = [row["step"] for row in history if "loss" in row]
    losses = [row["loss"] for row in history if "loss" in row]
    epochs = [row["epoch"] for row in history if "val_macro_f1" in row]
    scores = [row["val_macro_f1"] for row in history if "val_macro_f1" in row]
    figure, axes = plt.subplots(1, 2, figsize=(8.2, 3.2))
    axes[0].plot(steps, losses, color="#0e6b62")
    axes[0].set_title("Train loss")
    axes[0].set_xlabel("Optimizer step")
    axes[1].plot(epochs, scores, marker="o", color="#8c3a2f")
    axes[1].set_title("Validation macro-F1")
    axes[1].set_xlabel("Epoch")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def train_lora(
    train_records: list[dict],
    val_records: list[dict],
    model_dir: Path,
    config: dict,
) -> dict[str, object]:
    device = best_device()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    weights = answer_weights(train_records)
    encoded = []
    dropped_examples = 0
    gold_dropped = 0
    for record in train_records:
        row = encode_example(tokenizer, record, int(config["max_length"]))
        if row is None:
            dropped_examples += 1
            continue
        key = (str(record["criterion_type"]), str(record["expert_eligibility"]))
        row["weight"] = weights[key]
        encoded.append(row)
    for record in train_records:
        _note, _kept, dropped_ids = _dropped_ids(tokenizer, record, int(config["max_length"]))
        if dropped_ids:
            gold_dropped += 1
    if not encoded:
        raise RuntimeError("Every training example was longer than max_length.")

    dtype = torch.bfloat16 if device.type in {"mps", "cuda"} else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=dtype, low_cpu_mem_usage=True)
    model = get_peft_model(
        model,
        LoraConfig(
            r=int(config["lora_r"]),
            lora_alpha=int(config["lora_alpha"]),
            lora_dropout=float(config["lora_dropout"]),
            target_modules=list(config["target_modules"]),
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.to(device)
    pad_id = int(tokenizer.pad_token_id)
    loader = DataLoader(
        _Encoded(encoded),
        batch_size=1,
        shuffle=True,
        collate_fn=lambda rows: _collate(rows, pad_id),
    )
    accum = int(config["grad_accum"])
    epochs = int(config["epochs"])
    steps_per_epoch = (len(loader) + accum - 1) // accum
    total_steps = max(1, steps_per_epoch * epochs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]))
    warmup = max(1, int(total_steps * float(config["warmup_ratio"])))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup, total_steps)
    history: list[dict] = []
    best_f1 = -1.0
    best_epoch = 0
    stale = 0
    step = 0
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    result_dir = RESULTS_DIR / "task1"
    result_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        model.config.use_cache = False
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        seen = 0
        for index, batch in enumerate(loader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            weight = batch.pop("weight")
            loss = model(**batch).loss * weight.mean() / accum
            loss.backward()
            running += float(loss.item()) * accum
            seen += 1
            if index % accum == 0 or index == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                history.append({"step": step, "epoch": epoch, "loss": running / seen})
                if step % 20 == 0:
                    print(f"epoch {epoch} step {step} loss {running / seen:.4f}", flush=True)
        model.eval()
        model.gradient_checkpointing_disable()
        model.config.use_cache = True
        if device.type == "mps":
            torch.mps.empty_cache()
        print(f"epoch {epoch}: scoring validation", flush=True)
        metrics = score_records(
            model,
            tokenizer,
            val_records,
            max_new_tokens=int(config["max_new_tokens"]),
            use_cache=True,
            progress=lambda done, total: print(f"  val {done}/{total}", flush=True) if done % 10 == 0 or done == total else None,
        )
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        macro = float(metrics["overall"]["macro_f1"])
        history.append({"epoch": epoch, "val_macro_f1": macro, "val_accuracy": metrics["overall"]["accuracy"]})
        epoch_dir = ADAPTER_DIR.parent / f"epoch-{epoch}"
        model.save_pretrained(epoch_dir)
        tokenizer.save_pretrained(epoch_dir)
        if macro > best_f1:
            best_f1 = macro
            best_epoch = epoch
            stale = 0
            model.save_pretrained(ADAPTER_DIR)
            tokenizer.save_pretrained(ADAPTER_DIR)
        else:
            stale += 1
        if stale >= int(config["patience"]):
            break

    _save_curve(history, result_dir / "training_curve.png")
    summary = {
        "device": device.type,
        "train_records": len(train_records),
        "repeated_training_rows": False,
        "encoded_records": len(encoded),
        "examples_dropped_for_length": dropped_examples,
        "train_rows_with_gold_sentence_truncated": gold_dropped,
        "label_counts": label_counts(train_records),
        "answer_weighting": "Each criterion type and answer has the same total weight. Each teaching note is used once.",
        "answer_weights": {f"{kind}|{label}": value for (kind, label), value in sorted(weights.items())},
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1,
        "history": history,
        "config": config,
        "adapter_dir": str(ADAPTER_DIR),
    }
    (result_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (result_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return summary


def _dropped_ids(tokenizer, record: dict, max_length: int) -> tuple[str, dict, list[int]]:
    from genai.training.data import fit_note

    return fit_note(tokenizer, record, max_length - 64)
