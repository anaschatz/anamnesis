from __future__ import annotations

import hashlib
import json
from pathlib import Path

PROTOCOL = Path("eval/mem0_inference_v1.protocol.json")


def _protocol() -> dict:
    return json.loads(PROTOCOL.read_text())


def test_protocol_is_frozen_before_calls_and_diagnostic_only() -> None:
    value = _protocol()
    assert value["schema_version"] == "mem0_inference_protocol.v1"
    assert value["preregistered_before_model_calls"] is True
    assert value["hypothesis_test_eligible"] is False
    assert value["stopping_rule"] == {
        "attempts": 1,
        "on_integrity_failure": "stop_without_interpreting_metrics",
        "on_valid_result": "publish_all_event_results_and_stop",
        "prompt_tuning_on_v1_events": False,
        "next_prompt_requires_new_events": True,
    }


def test_protocol_pins_exact_free_local_runtime() -> None:
    value = _protocol()
    model = value["model"]
    assert model["name"] == "qwen3.5:9b-q4_K_M"
    assert model["manifest_sha256"] == (
        "6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7"
    )
    assert model["model_blob_sha256"] == (
        "dec52a44569a2a25341c4e4d3fee25846eed4f6f0b936278e3a3c900bb99d37c"
    )
    assert model["base_url"] == "http://127.0.0.1:11434"
    assert model["seed"] == 101
    assert model["temperature"] == 0.0
    assert model["thinking"] is False
    assert model["json_mode"] is True
    assert model["expected_model_calls"] == 7
    assert model["timeout_seconds"] == 180
    assert model["retries"] == model["repairs"] == 0
    assert model["cache"] is False
    assert value["storage"]["provider_api_cost_usd"] == 0.0
    assert value["storage"]["telemetry"] is False


def test_events_have_exact_order_and_cover_declared_mechanisms() -> None:
    value = _protocol()
    events = value["events"]
    assert [event["id"] for event in events] == [f"mi1-e{i}" for i in range(1, 8)]
    assert [event["assertion"]["type"] for event in events] == [
        "contains_fact",
        "deduplicated_fact",
        "corrected_fact",
        "contains_fact",
        "cancelled_or_absent",
        "no_hard_obligation",
        "contains_fact",
    ]
    assert [event["scope"] for event in events] == ["a"] * 6 + ["b"]


def test_inputs_are_fresh_and_do_not_reuse_prior_smoke_entities() -> None:
    encoded = PROTOCOL.read_text().casefold()
    for forbidden in (
        "compatibility check",
        "orion dome",
        "cedar incubator",
        "atlas",
        "apollo",
        "theater sponsor",
    ):
        assert forbidden not in encoded
    assert "argo permit" in encoded
    assert "amber museum" in encoded


def test_protocol_bytes_have_stable_identity() -> None:
    assert hashlib.sha256(PROTOCOL.read_bytes()).hexdigest() == (
        "7bc9532c599397414ddf856fa3e74dbfdf6039af39b1d7dae3656454261be5d1"
    )
