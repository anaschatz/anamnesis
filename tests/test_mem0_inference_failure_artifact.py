from __future__ import annotations

import json
from pathlib import Path


def test_mem0_inference_v1_failure_is_not_reported_as_memory_quality() -> None:
    artifact = json.loads(Path("results/mem0_inference_v1_failure.json").read_text())
    assert artifact["attempt"] == 1
    assert artifact["observed_model_calls"] == 7
    assert artifact["observed_stop_completions"] == 7
    assert artifact["integrity_passed"] is False
    assert artifact["semantic_metrics_interpreted"] is False
    assert artifact["result_artifact_written"] is False
    assert artifact["provider_api_cost_usd"] == 0.0
    assert artifact["external_network_calls"] == 0
    assert artifact["context_observation"] == {
        "configured_context_length": 8192,
        "resident_context_length": 8192,
        "server_input_limit_observed": 4098,
        "explicit_truncation_warnings": [
            {
                "prompt_tokens_before_truncation": 8374,
                "prompt_tokens_after_truncation": 4098,
            },
            {
                "prompt_tokens_before_truncation": 8419,
                "prompt_tokens_after_truncation": 4098,
            },
            {
                "prompt_tokens_before_truncation": 8220,
                "prompt_tokens_after_truncation": 4098,
            },
        ],
    }
