import json

from shadow_trainer.pair_analysis import summarize_pair_matrix


def test_missing_expected_group_blocks_all_timing_claims(tmp_path):
    pairs = [{"workload": "tiny3d", "condition": "cold", "repetition": index,
              "audit": {"status": "exact", "performance_comparison_eligible": True}, "sync_seconds": 2.0,
              "prefetch_seconds": 1.0} for index in range(1, 4)]
    matrix = tmp_path / "matrix.json"
    matrix.write_text(json.dumps({
        "schema_version": "shadowtrainer.pair-matrix/v1", "status": "success",
        "protocol": {"workloads": ["tiny3d", "resnet18"],
                     "conditions": ["cold"], "repetitions": 3, "timing_field": "execution_seconds"},
        "pairs": pairs,
    }))
    result = summarize_pair_matrix(matrix, tmp_path / "out")
    assert result["complete_matrix"] is False
    assert result["all_performance_claims_eligible"] is False
    assert result["groups"][0]["performance_claim_eligible"] is False
    assert result["groups"][0]["median_ratio_sync_over_prefetch"] is None

    duplicate_repetitions = [dict(row, repetition=1) for row in pairs]
    matrix.write_text(json.dumps({
        "schema_version": "shadowtrainer.pair-matrix/v1", "status": "success",
        "protocol": {"workloads": ["tiny3d"], "conditions": ["cold"],
                     "repetitions": 3, "timing_field": "execution_seconds"},
        "pairs": duplicate_repetitions,
    }))
    duplicate_result = summarize_pair_matrix(matrix, tmp_path / "duplicates")
    assert duplicate_result["complete_matrix"] is False
    assert duplicate_result["all_performance_claims_eligible"] is False

    matrix.write_text(json.dumps({
        "schema_version": "shadowtrainer.pair-matrix/v1", "status": "success",
        "protocol": {"workloads": ["tiny3d"], "conditions": ["cold"],
                     "repetitions": 3},
        "pairs": pairs,
    }))
    legacy_timing = summarize_pair_matrix(matrix, tmp_path / "legacy-timing")
    assert legacy_timing["complete_matrix"] is False
    assert legacy_timing["all_performance_claims_eligible"] is False
