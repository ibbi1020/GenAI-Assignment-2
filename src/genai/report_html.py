"""HTML screening report. The file is the downloadable assessment, not a screenshot."""

from __future__ import annotations

import html
from datetime import datetime, timezone

from genai.constants import AGGREGATION_RULE, DISCLAIMER


def _esc(value: object) -> str:
    return html.escape(str(value))


def render_report(payload: dict) -> str:
    criteria = payload.get("criteria") or []
    rows = []
    for index, row in enumerate(criteria, start=1):
        evidence = row.get("evidence") or []
        if evidence:
            quotes = "".join(
                f"<li><strong>[{_esc(item['id'])}]</strong> {_esc(item['text'])}</li>" for item in evidence
            )
        else:
            quotes = "<li>No sentence was cited.</li>"
        flag = "Yes" if row.get("missing_information") else "No"
        rows.append(
            "<article>"
            f"<h2>Criterion {index}: {_esc(row.get('criterion_type'))}</h2>"
            f"<p>{_esc(row.get('criterion_text'))}</p>"
            f"<p><strong>Label:</strong> {_esc(row.get('eligibility') or row.get('error') or 'not scored')}</p>"
            f"<p><strong>Missing-information flag:</strong> {flag}</p>"
            f"<ul>{quotes}</ul>"
            f"<p>{_esc(row.get('explanation') or row.get('error') or '')}</p>"
            "</article>"
        )
    aggregate = payload.get("aggregate") or {}
    provenance = payload.get("provenance") or {}
    generated = payload.get("generated_at") or datetime.now(timezone.utc).isoformat()
    body = "\n".join(rows) or "<p>No criteria were scored.</p>"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Eligibility screening report</title>
  <style>
    body {{ font-family: "Iowan Old Style", Palatino, Georgia, serif; margin: 2rem; color: #221e1a; }}
    .notice {{ background: #f3e6cf; border: 1px solid #8a5a12; padding: 0.8rem 1rem; }}
    h1 {{ font-size: 1.6rem; }}
    article {{ border-top: 1px solid #d9d1c7; padding-top: 0.8rem; }}
  </style>
</head>
<body>
  <p class="notice">{_esc(DISCLAIMER)}</p>
  <h1>Eligibility screening report</h1>
  <p><strong>Generated:</strong> {_esc(generated)}</p>
  <p><strong>Model:</strong> {_esc(provenance.get("target_model", ""))}</p>
  <p><strong>Adapter:</strong> {_esc(provenance.get("adapter", "none"))}</p>
  <p><strong>Decoding:</strong> {_esc(provenance.get("decoding", "greedy"))}</p>
  <h2>Numbered patient note</h2>
  <pre>{_esc(payload.get("numbered_note", ""))}</pre>
  {body}
  <h2>Overall line</h2>
  <p><strong>{_esc(aggregate.get("outcome", "not scored"))}</strong></p>
  <p>{_esc(aggregate.get("rule", AGGREGATION_RULE))}</p>
</body>
</html>
"""
