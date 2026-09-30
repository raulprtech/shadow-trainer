"""Epoch selection that remains fixed across invocations; no GPU/data imports."""
import json
from pathlib import Path


def frozen_epochs(state, session, proposed):
    """Validate before mutating state. Conflicting legacy evidence needs review."""
    def valid(value):
        if type(value) is not int or value not in (2, 4):
            raise ValueError("invalid_campaign_epochs")
        return value

    selected = state.get("epochs_selected")
    summaries = []
    for arm in ("A", "B"):
        path = Path(session) / f"arm_{arm}" / "summary.json"
        if path.exists():
            summary = json.loads(path.read_text())
            requested = valid(summary.get("epochs_requested"))
            if summary.get("configuration", {}).get("epochs", requested) != requested:
                raise ValueError("arm_epoch_configuration_conflict")
            summaries.append(requested)
    if selected is None:
        if summaries or any(name in state.get("phases", {}) for name in ("train_A", "train_B")):
            raise ValueError("legacy_epoch_plan_missing_requires_review")
        selected = valid(proposed)
    else:
        selected = valid(selected)
    if any(value != selected for value in summaries):
        raise ValueError("frozen_epoch_plan_conflicts_with_arm_requires_review")
    state["epochs_selected"] = selected
    return selected
