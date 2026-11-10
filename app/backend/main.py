"""Browser API for screening and live inference comparisons."""

from __future__ import annotations

import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from genai.aggregation import aggregate
from genai.benchmarks.attention_bench import run_attention_case
from genai.constants import DISCLAIMER, DRAFT_MODEL_ID, TARGET_MODEL_ID
from genai.data.records import read_jsonl
from genai.inference.kv_cache import greedy_decode
from genai.inference.speculative import speculative_decode
from genai.models.load import adapter_ready, draft_dir, load_causal_lm, load_tokenizer, target_dir
from genai.parsing import OutputParseError
from genai.paths import ADAPTER_DIR, PROCESSED_DIR
from genai.report_html import render_report
from genai.screening import screen_criterion

app = FastAPI(title="Trial screening prototype")
_lock = threading.Lock()
_state: dict = {"target": None, "tokenizer": None, "draft": None, "draft_tokenizer": None, "error": None, "loading": False, "adapter": False}


class CriterionRequest(BaseModel):
    note: str
    criterion_type: str
    criterion_text: str


class AggregateRequest(BaseModel):
    criteria: list[dict]


class ReportRequest(BaseModel):
    numbered_note: str
    criteria: list[dict]
    aggregate: dict


class AttentionRequest(BaseModel):
    batch: int = 1
    heads: int = 2
    seq: int = 32
    dim: int = 32
    block_size: int = 8
    causal: bool = True


class DecodeRequest(BaseModel):
    prompt: str = "Patient note: 0. A 58-year-old woman has chest pain."
    max_new_tokens: int = Field(default=16, ge=1, le=64)
    gamma: int = Field(default=4, ge=1, le=8)


def _load_target() -> None:
    with _lock:
        if _state["target"] is not None or _state["loading"]:
            return
        if not target_dir().exists():
            _state["error"] = "The target model is not downloaded. Run scripts/download_models.py."
            return
        _state["loading"] = True
    try:
        tokenizer = load_tokenizer(target_dir())
        adapter = ADAPTER_DIR if adapter_ready() else None
        model = load_causal_lm(target_dir(), adapter)
        with _lock:
            _state["tokenizer"] = tokenizer
            _state["target"] = model
            _state["adapter"] = adapter is not None
            _state["error"] = None
    except Exception as exc:
        with _lock:
            _state["error"] = f"The model failed to load: {exc}"
    finally:
        with _lock:
            _state["loading"] = False


def _ensure_target():
    _load_target()
    if _state["target"] is None:
        detail = _state["error"] or "The model is still loading. Wait a moment and try again."
        raise HTTPException(status_code=503, detail=detail)
    return _state["target"], _state["tokenizer"]


def _ensure_draft():
    with _lock:
        if _state["draft"] is None:
            if not draft_dir().exists():
                raise HTTPException(status_code=503, detail="The draft model is not downloaded.")
            _state["draft_tokenizer"] = load_tokenizer(draft_dir())
            _state["draft"] = load_causal_lm(draft_dir(), None)
        return _state["draft"], _state["draft_tokenizer"]


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "disclaimer": DISCLAIMER,
        "target_model": TARGET_MODEL_ID,
        "target_downloaded": target_dir().exists(),
        "draft_downloaded": draft_dir().exists(),
        "adapter_ready": adapter_ready(),
        "model_loaded": _state["target"] is not None,
        "loading": _state["loading"],
        "error": _state["error"],
        "device": "mps" if torch.backends.mps.is_available() else "cpu",
    }


@app.get("/api/examples")
def examples() -> dict:
    path = PROCESSED_DIR / "train.jsonl"
    if not path.exists():
        return {"examples": []}
    grouped: dict[tuple[str, str], dict] = {}
    for row in read_jsonl(path):
        key = (str(row["patient_id"]), str(row["trial_id"]))
        bucket = grouped.setdefault(
            key,
            {
                "patient_id": row["patient_id"],
                "trial_id": row["trial_id"],
                "trial_title": row["trial_title"],
                "note": row["note"],
                "criteria": [],
            },
        )
        if len(bucket["criteria"]) < 4:
            bucket["criteria"].append(
                {"criterion_type": row["criterion_type"], "criterion_text": row["criterion_text"]}
            )
    return {"examples": list(grouped.values())[:4]}


@app.post("/api/screen/criterion")
def screen(body: CriterionRequest) -> dict:
    model, tokenizer = _ensure_target()
    try:
        result = screen_criterion(model, tokenizer, body.note, body.criterion_type, body.criterion_text)
    except OutputParseError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    result["adapter"] = bool(_state.get("adapter"))
    return result


@app.post("/api/aggregate")
def aggregate_criteria(body: AggregateRequest) -> dict:
    scored = []
    for row in body.criteria:
        if row.get("ok") is False or "eligibility" not in row:
            continue
        scored.append({"eligibility": row["eligibility"], "criterion_type": row["criterion_type"]})
    if not scored:
        return {
            "outcome": "not scored",
            "is_ground_truth": False,
            "rule": "No criterion produced a label, so there is no overall line.",
            "unscored": len(body.criteria),
        }
    result = aggregate(scored)
    result["unscored"] = len(body.criteria) - len(scored)
    return result


@app.post("/api/report")
def report(body: ReportRequest):
    html = render_report(
        {
            "numbered_note": body.numbered_note,
            "criteria": body.criteria,
            "aggregate": body.aggregate,
            "provenance": {
                "target_model": TARGET_MODEL_ID,
                "adapter": str(ADAPTER_DIR) if adapter_ready() else "none",
                "decoding": "greedy, cache enabled, temperature 0",
                "draft_model": DRAFT_MODEL_ID,
            },
        }
    )
    return HTMLResponse(html)


@app.post("/api/bench/attention")
def bench_attention(body: AttentionRequest) -> dict:
    if body.seq > 256 or body.dim > 128 or body.batch > 2 or body.heads > 8:
        raise HTTPException(status_code=400, detail="Keep the live attention run at seq <= 256, dim <= 128, batch <= 2 and heads <= 8.")
    if body.block_size < 1:
        raise HTTPException(status_code=400, detail="Block size must be at least 1.")
    return run_attention_case(body.batch, body.heads, body.seq, body.dim, body.block_size, body.causal, repeats=3)


@app.post("/api/bench/kv")
def bench_kv(body: DecodeRequest) -> dict:
    model, tokenizer = _ensure_target()
    device = next(model.parameters()).device
    input_ids = tokenizer(body.prompt, return_tensors="pt").input_ids.to(device)
    if input_ids.shape[1] > 256:
        raise HTTPException(status_code=400, detail="Use a prompt of 256 tokens or fewer for the live comparison.")
    cached = greedy_decode(model, input_ids, body.max_new_tokens, use_cache=True, eos_id=tokenizer.eos_token_id)
    plain = greedy_decode(model, input_ids, body.max_new_tokens, use_cache=False, eos_id=tokenizer.eos_token_id)
    return {
        "token_equal": bool(torch.equal(cached.sequences, plain.sequences)),
        "new_tokens": cached.new_tokens,
        "cached": {
            "prefill_seconds": cached.prefill_seconds,
            "decode_seconds": cached.decode_seconds,
            "tokens_per_second": cached.tokens_per_second,
            "peak_memory_bytes": cached.peak_memory_bytes,
        },
        "uncached": {
            "prefill_seconds": plain.prefill_seconds,
            "decode_seconds": plain.decode_seconds,
            "tokens_per_second": plain.tokens_per_second,
            "peak_memory_bytes": plain.peak_memory_bytes,
        },
        "text": tokenizer.decode(cached.sequences[0, input_ids.shape[1] :], skip_special_tokens=True),
    }


@app.post("/api/bench/speculative")
def bench_speculative(body: DecodeRequest) -> dict:
    target, tokenizer = _ensure_target()
    draft, draft_tokenizer = _ensure_draft()
    if tokenizer.get_vocab() != draft_tokenizer.get_vocab():
        raise HTTPException(status_code=500, detail="The draft tokenizer does not match the target tokenizer.")
    device = next(target.parameters()).device
    input_ids = tokenizer(body.prompt, return_tensors="pt").input_ids.to(device)
    greedy = greedy_decode(target, input_ids, body.max_new_tokens, use_cache=True, eos_id=tokenizer.eos_token_id)
    speculated = speculative_decode(
        target,
        draft,
        input_ids,
        max_new_tokens=body.max_new_tokens,
        gamma=body.gamma,
        eos_id=tokenizer.eos_token_id,
    )
    greedy_seconds = greedy.prefill_seconds + greedy.decode_seconds
    return {
        "token_equal": bool(torch.equal(greedy.sequences, speculated.sequences)),
        "acceptance_rate": speculated.acceptance_rate,
        "draft_proposed": speculated.draft_proposed,
        "draft_accepted": speculated.draft_accepted,
        "bonus_tokens": speculated.bonus_tokens,
        "speculative_seconds": speculated.seconds,
        "greedy_seconds": greedy_seconds,
        "speedup": greedy_seconds / speculated.seconds if speculated.seconds else 0,
        "speculative_tokens_per_second": speculated.tokens_per_second,
        "peak_memory_bytes": speculated.peak_memory_bytes,
        "draft_model": DRAFT_MODEL_ID,
        "text": tokenizer.decode(speculated.sequences[0, input_ids.shape[1] :], skip_special_tokens=True),
    }


@app.on_event("startup")
def _startup() -> None:
    threading.Thread(target=_load_target, daemon=True).start()
