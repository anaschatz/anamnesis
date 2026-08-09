from __future__ import annotations

import hashlib
import json
from pathlib import Path

PROTOCOL = Path("eval/mem0_inference_v2.protocol.json")


def test_mem0_inference_v2_protocol_is_fresh_and_frozen_before_calls() -> None:
    protocol = json.loads(PROTOCOL.read_text())
    assert protocol["schema_version"] == "mem0_inference_protocol.v2"
    assert protocol["preregistered_before_model_calls"] is True
    assert protocol["hypothesis_test_eligible"] is False
    assert len(protocol["events"]) == 7
    assert {event["id"] for event in protocol["events"]} == {
        f"mi2-e{index}" for index in range(1, 8)
    }
    v1 = json.loads(Path("eval/mem0_inference_v1.protocol.json").read_text())
    assert {event["text"] for event in protocol["events"]}.isdisjoint(
        event["text"] for event in v1["events"]
    )


def test_mem0_inference_v2_context_and_stopping_rule_are_explicit() -> None:
    protocol = json.loads(PROTOCOL.read_text())
    assert protocol["model"]["context_length"] == 32768
    assert protocol["model"]["expected_model_calls"] == 7
    assert protocol["context_fidelity"] == {
        "v1_largest_observed_prompt_tokens": 8419,
        "minimum_context_headroom_tokens": 8192,
        "required_operator_observation": (
            "no Ollama truncating input prompt warning during the one measured attempt"
        ),
        "on_observed_truncation": "integrity_failure",
    }
    assert protocol["stopping_rule"]["attempts"] == 1
    assert protocol["stopping_rule"]["prompt_tuning_on_v2_events"] is False


def test_mem0_inference_v2_protocol_bytes_are_pinned() -> None:
    assert (
        hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
        == "e0031a0e9044b02b816afacb2ff1ecf4fe96bd4b26d2b6239d72dc496c3f5f7d"
    )
