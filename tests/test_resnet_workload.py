from shadow_trainer.resnet_workload import ResNet2p5DWorkload


def test_resnet18_2p5d_trains_and_restores(tmp_path):
    case = tmp_path / "case"
    case.mkdir()
    (case / "input.bin").write_bytes(b"calibration")
    options = {"device": "cpu", "depth": 18, "input_size": 32,
               "base_width": 8, "classes": 3}
    workload = ResNet2p5DWorkload(11, options)
    row = workload.train_case(case, "case")
    assert row["finite"]
    assert row["resnet_depth"] == 18
    assert row["input_mode"] == "2.5d-proxy"
    restored = ResNet2p5DWorkload(11, options)
    restored.restore(workload.checkpoint_state())
    assert restored.parameter_count == workload.parameter_count
