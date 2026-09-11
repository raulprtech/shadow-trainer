import numpy as np
import nibabel as nib

from shadow_trainer.nifti_workload import NiftiPatch3DWorkload


def test_nifti_adapter_reads_only_a_patch_and_checkpoints(tmp_path):
    case = tmp_path / "case_001"
    case.mkdir()
    image = np.linspace(-1000, 500, 20**3, dtype=np.float32).reshape((20, 20, 20))
    label = np.zeros((20, 20, 20), dtype=np.uint8)
    label[8:12, 8:12, 8:12] = 1
    nib.save(nib.Nifti1Image(image, np.eye(4)), case / "imaging.nii.gz")
    nib.save(nib.Nifti1Image(label, np.eye(4)), case / "segmentation.nii.gz")
    workload = NiftiPatch3DWorkload(
        7,
        {
            "device": "cpu",
            "patch_size": 16,
            "classes": 4,
            "base_channels": 2,
            "amp": False,
        },
    )
    result = workload.train_case(case, "case_001")
    assert result["finite"]
    assert result["patch_size"] == 16
    state = workload.checkpoint_state()
    restored = NiftiPatch3DWorkload(
        7,
        {
            "device": "cpu",
            "patch_size": 16,
            "classes": 4,
            "base_channels": 2,
            "amp": False,
        },
    )
    restored.restore(state)
    assert restored.step == 1
