#!/usr/bin/env python3
"""Render deterministic reference-selected CT/B0/A comparison panels.

Reads only the completed, provenance-audited six-case evaluation. The chosen
slice and crop depend on the reference mask, never on either prediction.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

import nibabel as nib
import numpy as np

from complete_stunet_train64_heldout import CANDIDATE, GIB, MIB, SESSION, audit, snapshot


def check_resources() -> None:
    state = snapshot()
    if (state["disk_free_bytes"] < 20 * GIB + 512 * MIB
            or state["available_ram_bytes"] < 1536 * MIB
            or state["swap_used_bytes"] > 192 * MIB):
        raise RuntimeError("resource guard before figure rendering")


def chosen_slice(target: np.ndarray) -> int:
    # Count one z-slab at a time; avoid a full-volume boolean copy.
    score = np.zeros(target.shape[2], dtype=np.int64)
    renal = np.zeros_like(score)
    for start in range(0, target.shape[2], 16):
        stop = min(start + 16, target.shape[2])
        block = target[:, :, start:stop]
        score[start:stop] = np.count_nonzero(block == 2, axis=(0, 1))
        renal[start:stop] = np.count_nonzero(block > 0, axis=(0, 1))
    return int(np.argmax(score if np.any(score) else renal))


def reference_crop(target: np.ndarray) -> tuple[slice, slice]:
    region = target == 2
    if not np.any(region):
        region = target > 0
    coordinates = np.argwhere(region)
    if len(coordinates) == 0:
        return slice(0, target.shape[0]), slice(0, target.shape[1])
    ranges = []
    for axis in (0, 1):
        lo, hi = int(coordinates[:, axis].min()), int(coordinates[:, axis].max()) + 1
        width = min(target.shape[axis], max(256, hi - lo + 64))
        start = max(0, min((lo + hi - width) // 2, target.shape[axis] - width))
        ranges.append(slice(start, start + width))
    return ranges[0], ranges[1]


def render_case(session: Path, case_id: str, public_id: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    source = session / "evaluation_cache" / case_id
    image_nii = nib.load(source / "imaging.nii.gz")
    target_nii = nib.load(source / "segmentation.nii.gz")
    baseline_nii = nib.load(session / "evaluation/evaluation/B0/predictions" / f"{case_id}.nii.gz")
    candidate_nii = nib.load(session / "evaluation/evaluation/A/predictions" / f"{case_id}.nii.gz")
    if (image_nii.shape != target_nii.shape or image_nii.shape != baseline_nii.shape
            or image_nii.shape != candidate_nii.shape
            or any(not np.allclose(image_nii.affine, other.affine)
                   for other in (target_nii, baseline_nii, candidate_nii))):
        raise ValueError("image/reference/prediction geometry differs")
    target_volume = np.asarray(target_nii.dataobj, dtype=np.uint8)
    index = chosen_slice(target_volume)
    target = np.asarray(target_volume[:, :, index])
    del target_volume
    crop = reference_crop(target)
    ct = np.asarray(image_nii.dataobj[:, :, index], dtype=np.float32)[crop]
    truth = target[crop]
    baseline = np.asarray(baseline_nii.dataobj[:, :, index], dtype=np.uint8)[crop]
    candidate = np.asarray(candidate_nii.dataobj[:, :, index], dtype=np.uint8)[crop]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4), constrained_layout=True)
    for axis, mask, name in zip(axes,
                                (None, truth, baseline, candidate),
                                ("CT", "Referencia", "B0 · Stage20", "A · 64 casos")):
        axis.imshow(np.clip(ct, -200, 300).T, cmap="gray", origin="lower",
                    vmin=-200, vmax=300)
        if mask is not None:
            for class_id, color in ((1, "#00d084"), (2, "#ff3b30"), (3, "#ffd60a")):
                if np.any(mask == class_id):
                    axis.contour((mask == class_id).T, levels=[0.5],
                                 colors=[color], linewidths=0.9)
        axis.set_title(name)
        axis.axis("off")
    fig.suptitle(f"{public_id} · corte axial {index} · recorte por referencia · riñón/tumor/quiste")
    return fig, {"public_id": public_id, "case_id": case_id,
                 "slice_axis": 2, "slice_index": index,
                 "crop": [[crop[0].start, crop[0].stop], [crop[1].start, crop[1].stop]],
                 "selection": "maximum reference tumor area, else renal area"}


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, default=SESSION)
    parser.add_argument("--candidate", type=Path, default=CANDIDATE)
    args = parser.parse_args()
    session = args.session.resolve()
    check_resources()
    paired = audit(session, args.candidate.resolve())
    if paired["case_count"] != 6:
        raise ValueError("expected six audited pairs")
    output = session / "figures"
    if output.exists():
        raise FileExistsError("figure output already exists; do not overwrite")
    output.mkdir()
    metadata = []
    pdf = output / "stunet-b0-a-six-cases.pdf"
    with PdfPages(pdf) as pages:
        for number, case in enumerate(paired["paired_cases"], 1):
            check_resources()
            public_id = f"P{number}"
            fig, item = render_case(session, case["case_id"], public_id)
            path = output / f"{public_id}.png"
            fig.savefig(path, dpi=150)
            pages.savefig(fig)
            plt.close(fig)
            item["png"] = str(path)
            metadata.append(item)
    with tempfile.NamedTemporaryFile(mode="w", dir=output, suffix=".json", delete=False) as stream:
        json.dump({"schema_version": "shadowtrainer.stunet-figures/v1",
                   "source_audit": str(session / "paired-audit.json"),
                   "rules": "reference-selected slice/crop; same slice and crop for CT, truth, B0, A",
                   "figures": metadata, "pdf": str(pdf)}, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    os.replace(temporary, output / "manifest.json")
    print(json.dumps({"status": "success", "figures": len(metadata), "pdf": str(pdf)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
