"""Guarded statistical summaries for exact paired experiment matrices."""

from __future__ import annotations

import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import median, quantiles
from typing import Any

from .errors import ConfigurationError, IntegrityError
from .events import atomic_json

MATRIX_SCHEMA = "shadowtrainer.pair-matrix/v1"
SUMMARY_SCHEMA = "shadowtrainer.pair-summary/v1"


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IntegrityError(f"cannot read pair matrix: {exc}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != MATRIX_SCHEMA:
        raise ConfigurationError(f"matrix schema must be {MATRIX_SCHEMA!r}")
    return value


def _finite(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return None


def _bootstrap_median(values: list[float], seed: int = 1401,
                      samples: int = 10000) -> tuple[float, float]:
    generator = random.Random(seed)
    size = len(values)
    draws = sorted(median([values[generator.randrange(size)] for _ in range(size)])
                   for _ in range(samples))
    return draws[int(0.025 * samples)], draws[min(samples - 1, int(0.975 * samples))]


def summarize_pair_matrix(matrix_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    matrix_path = Path(matrix_path).expanduser().resolve()
    matrix = _read(matrix_path)
    pairs = matrix.get("pairs")
    if not isinstance(pairs, list):
        raise IntegrityError("pair matrix requires a pairs list")
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    excluded = []
    for index, pair in enumerate(pairs):
        if not isinstance(pair, dict):
            raise IntegrityError(f"pair {index} is not an object")
        workload, condition = pair.get("workload"), pair.get("condition")
        repetition = pair.get("repetition")
        if not isinstance(workload, str) or condition not in {"cold", "warm"}:
            raise IntegrityError(f"pair {index} has invalid workload/condition")
        if not isinstance(repetition, int) or isinstance(repetition, bool) or repetition < 1:
            raise IntegrityError(f"pair {index} has invalid repetition")
        audit = pair.get("audit")
        sync = _finite(pair.get("sync_seconds"))
        prefetch = _finite(pair.get("prefetch_seconds"))
        exact = (isinstance(audit, dict) and audit.get("status") == "exact"
                 and audit.get("performance_comparison_eligible") is True)
        eligible = exact and sync is not None and prefetch is not None and sync > 0 and prefetch > 0
        if eligible:
            groups[(workload, condition)].append({"repetition": repetition,
                                                   "sync": sync, "prefetch": prefetch,
                                                   "ratio": sync / prefetch})
        else:
            excluded.append({"index": index, "workload": workload, "condition": condition,
                             "reason": "non_exact_or_invalid_timing"})
    protocol = matrix.get("protocol") if isinstance(matrix.get("protocol"), dict) else {}
    expected_workloads = protocol.get("workloads") if isinstance(protocol.get("workloads"), list) else []
    expected_conditions = protocol.get("conditions") if isinstance(protocol.get("conditions"), list) else []
    expected_repetitions = protocol.get("repetitions") if isinstance(protocol.get("repetitions"), int) else 0
    timing_field = protocol.get("timing_field")
    expected_groups = {(workload, condition) for workload in expected_workloads for condition in expected_conditions}
    expected_repetition_ids = set(range(1, expected_repetitions + 1))
    coverage_complete = (bool(expected_groups) and set(groups) == expected_groups
                         and all({row["repetition"] for row in groups[key]}
                                 == expected_repetition_ids
                                 and len(groups[key]) == expected_repetitions
                                 for key in expected_groups))
    matrix_complete = (matrix.get("status") == "success" and not excluded
                       and coverage_complete and timing_field == "execution_seconds")
    summaries = []
    for group_index, ((workload, condition), rows) in enumerate(sorted(groups.items())):
        ratios = [row["ratio"] for row in rows]
        syncs = [row["sync"] for row in rows]
        prefetches = [row["prefetch"] for row in rows]
        eligible = matrix_complete and len(rows) >= expected_repetitions
        quartiles = quantiles(ratios, n=4, method="inclusive") if len(rows) >= 2 else None
        q1, q3 = (quartiles[0], quartiles[2]) if quartiles else (None, None)
        low, high = _bootstrap_median(ratios, seed=1401 + group_index) if eligible else (None, None)
        summaries.append({
            "workload": workload, "condition": condition, "exact_pairs": len(rows),
            "performance_claim_eligible": eligible,
            "median_sync_seconds": median(syncs) if eligible else None,
            "median_prefetch_seconds": median(prefetches) if eligible else None,
            "median_ratio_sync_over_prefetch": median(ratios) if eligible else None,
            "ratio_q1": q1 if eligible else None, "ratio_q3": q3 if eligible else None,
            "ratio_bootstrap_95_low": low, "ratio_bootstrap_95_high": high,
        })
    complete = matrix_complete
    payload = {
        "schema_version": SUMMARY_SCHEMA,
        "source_matrix_status": matrix.get("status", "unknown"),
        "complete_matrix": complete,
        "policy": "timing summaries require execution_seconds, exact audit, and complete protocol coverage",
        "groups": summaries,
        "excluded_pairs": excluded,
        "all_performance_claims_eligible": (complete and not excluded and bool(summaries)
                                             and all(row["performance_claim_eligible"] for row in summaries)),
    }
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "pair-summary.json", payload)
    fields = ["workload", "condition", "exact_pairs", "performance_claim_eligible",
              "median_sync_seconds", "median_prefetch_seconds", "median_ratio_sync_over_prefetch",
              "ratio_q1", "ratio_q3", "ratio_bootstrap_95_low", "ratio_bootstrap_95_high"]
    with (output / "pair-summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in summaries)
    lines = ["# Exact paired timing summary", "", payload["policy"], "",
             "| Workload | Cache | Exact pairs | Eligible | Sync median (s) | Prefetch median (s) | Ratio | 95% bootstrap CI |",
             "| --- | --- | ---: | --- | ---: | ---: | ---: | --- |"]
    for row in summaries:
        value = lambda key: "—" if row[key] is None else f"{row[key]:.4f}"
        interval = "—" if row["ratio_bootstrap_95_low"] is None else (
            f'[{row["ratio_bootstrap_95_low"]:.4f}, {row["ratio_bootstrap_95_high"]:.4f}]')
        lines.append(f'| {row["workload"]} | {row["condition"]} | {row["exact_pairs"]} | '
                     f'{row["performance_claim_eligible"]} | {value("median_sync_seconds")} | '
                     f'{value("median_prefetch_seconds")} | {value("median_ratio_sync_over_prefetch")} | {interval} |')
    if excluded:
        lines += ["", f"Excluded non-exact or invalid pairs: {len(excluded)}."]
    (output / "pair-summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload
