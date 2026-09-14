import json

from shadow_trainer.pair_analysis import summarize_pair_matrix


def _pair(index, status="exact"):
    return {"workload": "tiny3d", "condition": "cold", "repetition": index,
            "audit": {"status": status, "performance_comparison_eligible": status == "exact"}, "sync_seconds": 2.0 + index / 10,
            "prefetch_seconds": 1.0 + index / 10}


def test_three_exact_pairs_enable_guarded_statistics(tmp_path):
    matrix = tmp_path / "matrix.json"
    matrix.write_text(json.dumps({"schema_version": "shadowtrainer.pair-matrix/v1",
                                  "status": "success", "protocol": {"workloads": ["tiny3d"], "conditions": ["cold"], "repetitions": 3, "timing_field": "execution_seconds"}, "pairs": [_pair(i) for i in range(1, 4)]}))
    result = summarize_pair_matrix(matrix, tmp_path / "out")
    assert result["all_performance_claims_eligible"] is True
    assert result["groups"][0]["exact_pairs"] == 3
    assert result["groups"][0]["ratio_bootstrap_95_low"] is not None


def test_diverged_pair_is_excluded_and_blocks_claim(tmp_path):
    matrix = tmp_path / "matrix.json"
    matrix.write_text(json.dumps({"schema_version": "shadowtrainer.pair-matrix/v1",
                                  "status": "failed", "protocol": {"workloads": ["tiny3d"], "conditions": ["cold"], "repetitions": 3},
                                  "pairs": [_pair(1), _pair(2), _pair(3, "diverged")]}))
    result = summarize_pair_matrix(matrix, tmp_path / "out")
    assert result["all_performance_claims_eligible"] is False
    assert result["groups"][0]["performance_claim_eligible"] is False
    assert len(result["excluded_pairs"]) == 1
    assert result["groups"][0]["median_ratio_sync_over_prefetch"] is None
