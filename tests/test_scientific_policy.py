from shadow_trainer.scientific import EvidenceRow, _bar_svg


def _row(evidence_id, classification):
    return EvidenceRow(evidence_id, evidence_id, classification, "test", "success",
        "tiny3d", "sync", "GPU", 1, 1, 1.0, 1, 2, 10, 5, 5, 0.1,
        0, 1, 0.2, 0.2, True, "a" * 64, "")


def test_numeric_figure_excludes_non_valid_records():
    svg = _bar_svg([_row("valid-run", "valid"),
                    _row("diagnostic-run", "diagnostic")],
                   "duration_seconds", "Runtime", "s", 1.0)
    assert "valid-run" in svg
    assert "diagnostic-run" not in svg
