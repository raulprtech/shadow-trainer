"""Build traceable scientific tables from audited Shadow Trainer runs."""

from __future__ import annotations

import csv
import hashlib
import html
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

from .errors import ConfigurationError, IntegrityError
from .events import atomic_json

SPEC_SCHEMA = "shadowtrainer.evidence-spec/v1"
BUNDLE_SCHEMA = "shadowtrainer.evidence-bundle/v1"
CLASSIFICATIONS = {"valid", "engineering_only", "diagnostic", "invalid", "pending"}
REQUIRED = ("job.json", "plan.json", "environment.json", "events.jsonl", "summary.json")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IntegrityError(f"cannot read evidence artifact {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise IntegrityError(f"evidence artifact {path.name} must contain an object")
    return value


def _read_events(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    number = 0
    try:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("event must be an object")
            rows.append(row)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise IntegrityError(f"invalid events.jsonl at line {number}: {exc}") from exc
    return rows


def _digest(run_dir: Path) -> str:
    digest = hashlib.sha256()
    for name in REQUIRED:
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update((run_dir / name).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _numbers(rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            values.append(float(value))
    return values


@dataclass(frozen=True)
class EvidenceRow:
    evidence_id: str
    label: str
    classification: str
    permitted_use: str
    status: str
    workload: str
    strategy: str
    gpu: str
    steps: int
    expected_steps: int | None
    duration_seconds: float | None
    peak_gpu_allocated_bytes: int | None
    peak_gpu_reserved_bytes: int | None
    cache_budget_bytes: int | None
    cache_occupancy_bytes: int | None
    bytes_transferred: int
    staging_seconds: float
    cache_hits: int
    cache_misses: int
    mean_loss: float | None
    final_loss: float | None
    all_steps_finite: bool
    bundle_sha256: str
    notes: str


def collect_run(record: dict[str, Any], root: Path) -> EvidenceRow:
    evidence_id = record.get("id")
    classification = record.get("classification")
    if not isinstance(evidence_id, str) or not evidence_id:
        raise ConfigurationError("each evidence record requires a non-empty id")
    if classification not in CLASSIFICATIONS:
        raise ConfigurationError(f"{evidence_id}: unsupported evidence classification")
    raw_dir = record.get("run_dir")
    if not isinstance(raw_dir, str) or not raw_dir:
        raise ConfigurationError(f"{evidence_id}: run_dir must be a path string")
    run_dir = Path(raw_dir).expanduser()
    run_dir = run_dir.resolve() if run_dir.is_absolute() else (root / run_dir).resolve()
    missing = [name for name in REQUIRED if not (run_dir / name).is_file()]
    if missing:
        raise IntegrityError(f"{evidence_id}: incomplete run bundle: {', '.join(missing)}")
    job = _read_json(run_dir / "job.json")
    plan = _read_json(run_dir / "plan.json")
    environment = _read_json(run_dir / "environment.json")
    summary = _read_json(run_dir / "summary.json")
    events = _read_events(run_dir / "events.jsonl")
    train = [row for row in events if row.get("kind") == "train_step"]
    staging = [row for row in events if row.get("kind") == "staging"]
    losses = _numbers(train, "loss")
    allocated = _numbers(train, "peak_gpu_allocated_bytes")
    reserved = _numbers(train, "peak_gpu_reserved_bytes")
    stage_times = _numbers(staging, "seconds")
    devices = environment.get("nvidia", {}).get("devices", [])
    gpu = devices[0].get("name", "unavailable") if devices else "unavailable"
    cache = summary.get("cache") if isinstance(summary.get("cache"), dict) else {}
    duration = summary.get("duration_seconds")
    duration = float(duration) if isinstance(duration, (int, float)) and math.isfinite(duration) else None
    return EvidenceRow(
        evidence_id=evidence_id,
        label=str(record.get("label", evidence_id)),
        classification=classification,
        permitted_use=str(record.get("permitted_use", "")),
        status=str(summary.get("status", "unknown")),
        workload=str(job.get("workload", {}).get("type", "unknown")),
        strategy=str(summary.get("selected_strategy", plan.get("selected_strategy", "unknown"))),
        gpu=str(gpu),
        steps=int(summary.get("global_step", len(train))),
        expected_steps=int(summary["expected_steps"]) if isinstance(summary.get("expected_steps"), int) else None,
        duration_seconds=duration,
        peak_gpu_allocated_bytes=int(max(allocated)) if allocated else None,
        peak_gpu_reserved_bytes=int(max(reserved)) if reserved else None,
        cache_budget_bytes=int(cache["budget_bytes"]) if isinstance(cache.get("budget_bytes"), int) else None,
        cache_occupancy_bytes=int(cache["occupancy_bytes"]) if isinstance(cache.get("occupancy_bytes"), int) else None,
        bytes_transferred=int(sum(row.get("bytes", 0) for row in staging if row.get("event") == "cache_miss")),
        staging_seconds=float(sum(stage_times)),
        cache_hits=sum(row.get("event") == "cache_hit" for row in staging),
        cache_misses=sum(row.get("event") == "cache_miss" for row in staging),
        mean_loss=fmean(losses) if losses else None,
        final_loss=losses[-1] if losses else None,
        all_steps_finite=bool(train) and all(row.get("finite") is True for row in train),
        bundle_sha256=_digest(run_dir),
        notes=str(record.get("notes", "")),
    )


def _display(value: Any) -> str:
    if value is None:
        return "—"
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def _markdown(rows: list[EvidenceRow]) -> str:
    lines = ["# Canonical Shadow Trainer evidence", "",
             "Generated from immutable run artifacts. Numeric publication figures use only `valid` records.", "",
             "| ID | Class | Workload | Strategy | Status | Steps | Time (s) | Peak VRAM (MiB) | Transfer (MiB) | Permitted use |",
             "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |"]
    for row in rows:
        vram = row.peak_gpu_reserved_bytes / 2**20 if row.peak_gpu_reserved_bytes is not None else None
        cells = [row.evidence_id, row.classification, row.workload, row.strategy, row.status,
                 str(row.steps), _display(row.duration_seconds), _display(vram),
                 _display(row.bytes_transferred / 2**20), row.permitted_use]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "## Integrity digests", ""]
    lines += [f"- `{row.evidence_id}`: `{row.bundle_sha256}`" for row in rows]
    return "\n".join(lines) + "\n"


def _bar_svg(rows: list[EvidenceRow], key: str, title: str, unit: str, divisor: float) -> str:
    selected = [row for row in rows if row.classification == "valid" and getattr(row, key) is not None]
    width, left, height = 900, 250, 100 + 64 * max(1, len(selected))
    values = [float(getattr(row, key)) / divisor for row in selected]
    maximum = max(values, default=1.0)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
             '<rect width="100%" height="100%" fill="#fbfcfe"/>',
             f'<text x="24" y="38" font-family="sans-serif" font-size="22" font-weight="700">{html.escape(title)}</text>']
    if not selected:
        parts.append('<text x="24" y="86" font-family="sans-serif">No valid numeric evidence.</text>')
    for index, (row, value) in enumerate(zip(selected, values, strict=True)):
        y = 72 + index * 64
        bar = max(2.0, (width - left - 110) * value / maximum)
        parts += [f'<text x="24" y="{y + 20}" font-family="sans-serif" font-size="15">{html.escape(row.label)}</text>',
                  f'<rect x="{left}" y="{y}" width="{bar:.2f}" height="28" rx="4" fill="#3157d5"/>',
                  f'<text x="{left + bar + 8:.2f}" y="{y + 20}" font-family="monospace" font-size="14">{value:.2f} {html.escape(unit)}</text>']
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def build_evidence_bundle(spec_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    spec_path = Path(spec_path).expanduser().resolve()
    spec = _read_json(spec_path)
    if spec.get("schema_version") != SPEC_SCHEMA:
        raise ConfigurationError(f"schema_version must be {SPEC_SCHEMA!r}")
    records = spec.get("records")
    if not isinstance(records, list) or not records:
        raise ConfigurationError("evidence spec requires a non-empty records list")
    rows = [collect_run(record, spec_path.parent) for record in records]
    ids = [row.evidence_id for row in rows]
    if len(ids) != len(set(ids)):
        raise ConfigurationError("evidence ids must be unique")
    output = Path(output_dir).expanduser().resolve()
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": BUNDLE_SCHEMA,
               "policy": "numeric figures include only classification=valid",
               "records": [asdict(row) for row in rows]}
    atomic_json(output / "evidence.json", payload)
    fields = list(asdict(rows[0]))
    with (output / "evidence.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    (output / "evidence.md").write_text(_markdown(rows), encoding="utf-8")
    (figures / "peak-vram.svg").write_text(_bar_svg(rows, "peak_gpu_reserved_bytes", "Peak reserved GPU memory", "MiB", 2**20), encoding="utf-8")
    (figures / "runtime.svg").write_text(_bar_svg(rows, "duration_seconds", "Observed runtime (independent runs; not a speedup comparison)", "s", 1.0), encoding="utf-8")
    return payload
