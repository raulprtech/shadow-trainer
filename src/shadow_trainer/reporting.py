"""Self-contained, redacted HTML run reports."""

from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any

from .events import atomic_json

_SECRET = re.compile(r"(token|secret|password|credential|authorization)", re.I)


def redact(value: Any, key: str = "") -> Any:
    if _SECRET.search(key):
        return "[REDACTED]"
    if isinstance(value, dict):
        return {name: redact(item, name) for name, item in value.items()}
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, str):
        value = re.sub(r"(?i)(bearer\s+)[A-Za-z0-9._~+/-]+", r"\1[REDACTED]", value)
        value = re.sub(r"(?i)(token|secret|password)=([^&\s]+)", r"\1=[REDACTED]", value)
    return value


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _read_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def render_report(run_dir: Path) -> Path:
    run_dir = run_dir.resolve()
    summary = redact(_read_json(run_dir / "summary.json"))
    plan = redact(_read_json(run_dir / "plan.json"))
    environment = redact(_read_json(run_dir / "environment.json"))
    events = redact(_read_events(run_dir / "events.jsonl"))
    steps = [event for event in events if event.get("kind") == "train_step"]
    staging = [
        event for event in events
        if event.get("kind") == "staging" and event.get("event") == "cache_miss"
    ]
    losses = [float(step["loss"]) for step in steps if "loss" in step]
    peak_vram = max(
        (int(step.get("peak_gpu_reserved_bytes", 0)) for step in steps), default=0
    )
    transferred = sum(int(event.get("bytes", 0)) for event in staging)
    status = str(summary.get("status", "unknown"))
    status_class = "ok" if status == "success" else "bad"

    def dump(value: Any) -> str:
        return html.escape(json.dumps(value, indent=2, sort_keys=True))

    rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(step.get('global_step', '')))}</td>"
        f"<td>{html.escape(str(step.get('case_id', '')))}</td>"
        f"<td>{float(step.get('loss', 0)):.6f}</td>"
        f"<td>{int(step.get('peak_gpu_reserved_bytes', 0))/2**20:.1f}</td>"
        "</tr>"
        for step in steps[-100:]
    )
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Shadow Trainer — {html.escape(str(summary.get("job_id", "run")))}</title>
<style>
:root {{ color-scheme: dark; --bg:#07111f; --panel:#102238; --ink:#edf7ff;
--muted:#9eb4c7; --accent:#6ee7b7; --danger:#fb7185; }}
body {{ margin:0; font:15px/1.5 system-ui,sans-serif; color:var(--ink);
background:linear-gradient(145deg,#07111f,#0d2230); }}
main {{ max-width:1100px; margin:auto; padding:36px 22px 70px; }}
h1 {{ font-size:clamp(2rem,5vw,4rem); margin:.1em 0; letter-spacing:-.04em; }}
h2 {{ margin-top:32px; }}
.muted {{ color:var(--muted); }} .grid {{ display:grid;
grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:12px; }}
.card {{ background:rgba(16,34,56,.88); border:1px solid #24425c;
border-radius:16px; padding:18px; box-shadow:0 12px 40px #0004; }}
.metric {{ font-size:1.8rem; font-weight:750; color:var(--accent); }}
.ok {{ color:var(--accent); }} .bad {{ color:var(--danger); }}
table {{ width:100%; border-collapse:collapse; }} th,td {{ padding:8px;
border-bottom:1px solid #24425c; text-align:left; }}
pre {{ white-space:pre-wrap; overflow:auto; color:#cbe5f6; }}
details {{ margin:10px 0; }}
</style>
</head>
<body><main>
<p class="muted">AUDITABLE RESOURCE-AWARE TRAINING</p>
<h1>Shadow Trainer</h1>
<h2 class="{status_class}">{html.escape(status.upper())}</h2>
<p class="muted">Job {html.escape(str(summary.get("job_id", "unknown")))}</p>
<section class="grid">
<div class="card"><div class="metric">{len(steps)}</div><div>training steps</div></div>
<div class="card"><div class="metric">{(losses[-1] if losses else 0):.5f}</div><div>final loss</div></div>
<div class="card"><div class="metric">{peak_vram/2**20:.1f} MiB</div><div>peak reserved VRAM</div></div>
<div class="card"><div class="metric">{transferred/2**20:.1f} MiB</div><div>staged data</div></div>
<div class="card"><div class="metric">{html.escape(str(plan.get("selected_strategy", "—")))}</div><div>selected strategy</div></div>
</section>
<h2>Recent trajectory</h2>
<div class="card"><table><thead><tr><th>Step</th><th>Case</th><th>Loss</th>
<th>VRAM MiB</th></tr></thead><tbody>{rows}</tbody></table></div>
<h2>Audit evidence</h2>
<details class="card"><summary>Plan and admission</summary><pre>{dump(plan)}</pre></details>
<details class="card"><summary>Environment</summary><pre>{dump(environment)}</pre></details>
<details class="card"><summary>Summary</summary><pre>{dump(summary)}</pre></details>
<p class="muted">This report describes system behavior, not clinical performance.</p>
</main></body></html>
"""
    target = run_dir / "report.html"
    temporary = target.with_suffix(".html.tmp")
    temporary.write_text(document, encoding="utf-8")
    temporary.replace(target)
    atomic_json(
        run_dir / "report-summary.json",
        {
            "schema_version": "shadowtrainer.report/v1",
            "status": status,
            "steps": len(steps),
            "final_loss": losses[-1] if losses else None,
            "peak_gpu_reserved_bytes": peak_vram,
            "staged_bytes": transferred,
        },
    )
    return target
