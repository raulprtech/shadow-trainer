"""Export an allowlisted, offline presentation of the existing synthetic bridge.

Not a training runner, general run exporter, or certification of source truth.
"""
import argparse
import hashlib
import html
import json
from pathlib import Path

CHECKS = (
    "real_plan_admitted_sync", "two_confirmed_finite_cpu_steps",
    "receipt_artifact_hashes_and_pending_human_review",
    "duplicate_run_rejected_without_evidence_changes", "locked_test_rejected",
    "disk_rejected_before_execution", "tampered_summary_receipt_rejected",
)
LIMITS = [
    "Fixture sintético CPU de dos pasos; no nueva ejecución GPU.",
    "Los checks son resultados registrados por el harness, no repetidos por este exportador.",
    "Integridad de bytes no demuestra autenticidad ni corrección científica.",
    "No demuestra mejora clínica, aceleración universal ni equivalencia STU-Net.",
    "sync sigue predeterminado; Stage38 largo continúa pendiente.",
    "La revisión humana permanece pendiente; ninguna promoción automática.",
]


def reject_constant(value):
    raise ValueError("nonfinite_json")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_json_key")
        result[key] = value
    return result


def export(source, output):
    source, output = Path(source), Path(output)
    if source.is_symlink():
        raise ValueError("symlink_source")
    source = source.resolve(strict=True)
    if output.resolve().is_relative_to(source.parent):
        raise ValueError("output_must_be_separate_from_source_bundle")
    if source.stat().st_size > 65536:
        raise ValueError("source_too_large")
    raw = source.read_bytes()
    if len(raw) > 65536:
        raise ValueError("source_too_large")
    data = json.loads(raw, parse_constant=reject_constant, object_pairs_hook=unique_object)
    if not isinstance(data, dict):
        raise ValueError("invalid_object")
    required = {
        "schema_version": "shadowtrainer.bridge-smoke/v1",
        "classification": "synthetic_cpu_integration_only",
        "status": "success", "training_steps_limit": 2, "cuda_visible_devices": "",
    }
    for key, expected in required.items():
        if type(data.get(key)) is not type(expected) or data[key] != expected:
            raise ValueError("unsupported_bridge_record")
    checks = data.get("checks")
    if not isinstance(checks, list) or checks != list(CHECKS):
        raise ValueError("unexpected_checks")
    # Never export free-form input, file paths, IDs, environment, or patient data.
    public = {
        "schema_version": "shadowtrainer.public-bridge-demo/v1",
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "evidence_kind": "synthetic_cpu_integration_only",
        "source_reported_status": "success",
        "source_reported_checks": list(CHECKS),
        "source_reported_step_limit": 2,
        "default_strategy": "sync", "limits": LIMITS,
    }
    content = json.dumps(public, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    checks_html = "".join("<li>" + html.escape(c) + "</li>" for c in CHECKS)
    limits_html = "".join("<li>" + html.escape(c) + "</li>" for c in LIMITS)
    page = (
        '<!doctype html><html lang="es"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>Shadow Trainer — evidencia de integración</title>'
        '<style>body{font:18px system-ui;max-width:850px;margin:3rem auto;padding:1rem;'
        'line-height:1.6;background:#101827;color:#edf2fa}h1,h2{color:#82dacc}'
        'code{overflow-wrap:anywhere}li{margin:.5rem 0}</style>'
        '<h1>Shadow Trainer</h1><p>Entrenar con recursos limitados y conservar evidencia verificable.</p>'
        '<h2>Puente local demostrado</h2>'
        '<p>Clinical-Nigma → CLI aislada → Shadow Trainer → recibo de evidencia.</p>'
        '<p>El registro sintético CPU contiene dos pasos y siete checks satisfactorios. '
        'Esta presentación no vuelve a ejecutar las pruebas.</p>'
        '<ol>' + checks_html + '</ol><h2>Alcance y pendientes</h2><ul>' + limits_html +
        '</ul><p>Las validaciones físicas históricas se documentan por separado en '
        'CLAIMS_LEDGER; este paquete no las reproduce ni acredita.</p>'
        '<p>SHA-256 del resumen fuente: <code>' + public["source_sha256"] +
        '</code></p></html>'
    )
    # Exclusive output directory: never overwrite a previous presentation.
    output.mkdir(parents=False, exist_ok=False)
    artifacts = {}
    for name, body in (("evidence.json", content), ("index.html", page)):
        encoded = body.encode("utf-8")
        with (output / name).open("xb") as stream:
            stream.write(encoded)
        artifacts[name] = {"bytes": len(encoded), "sha256": hashlib.sha256(encoded).hexdigest()}
    # Written last: its absence marks an incomplete export. No original is changed.
    with (output / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump({"schema_version": "shadowtrainer.public-pack/v1", "artifacts": artifacts},
                  stream, indent=2, allow_nan=False)
    return public


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args.source, args.output)
    print("export_complete: evidence.json, index.html, manifest.json")
