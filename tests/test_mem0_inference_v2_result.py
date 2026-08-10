from __future__ import annotations

import hashlib
from pathlib import Path

from anamnesis.mem0_inference_diagnostic import Mem0InferenceResult

RESULT = Path("results/mem0_inference_v2.json")


def test_mem0_inference_v2_result_bytes_and_integrity_are_frozen() -> None:
    assert (
        hashlib.sha256(RESULT.read_bytes()).hexdigest()
        == "f3463898cddc713470342f991a97366937c733cb8f3d1bf1218108c26fca460b"
    )
    result = Mem0InferenceResult.model_validate_json(RESULT.read_text())
    assert result.source_commit == "1b40db587538295500b6fcc3308a09f6ea801c30"
    assert result.integrity_passed is True
    assert result.semantic_passed is False
    assert result.localhost_model_calls == 7
    assert result.all_calls_finished is True
    assert result.usage_complete is True
    assert result.scope_isolation_passed is True
    assert result.cleanup_passed is True
    assert result.prompt_tokens == 58_103
    assert result.completion_tokens == 373
    assert result.provider_api_cost_usd == 0.0
    assert result.external_network_calls == 0


def test_mem0_inference_v2_result_preserves_all_semantic_failures() -> None:
    result = Mem0InferenceResult.model_validate_json(RESULT.read_text())
    outcomes = {
        event.event_id: event.assertion_passed for event in result.event_results
    }
    assert outcomes == {
        "mi2-e1": True,
        "mi2-e2": True,
        "mi2-e3": False,
        "mi2-e4": False,
        "mi2-e5": False,
        "mi2-e6": True,
        "mi2-e7": True,
    }
    correction = result.event_results[2]
    assert correction.sdk_events == ("ADD",)
    assert correction.record_count == 2
    assert any("Spanish" in memory for memory in correction.memories)
    assert any("French" in memory for memory in correction.memories)
    cancellation = result.event_results[4]
    assert cancellation.sdk_events == ("ADD",)
    assert any("cancelled" in memory for memory in cancellation.memories)
    assert any("needs to renew" in memory for memory in cancellation.memories)


def test_mem0_inference_v2_result_has_no_local_paths_or_secrets() -> None:
    text = RESULT.read_text().lower()
    for forbidden in ("/users/", "/private/tmp", "api_key", "bearer "):
        assert forbidden not in text
