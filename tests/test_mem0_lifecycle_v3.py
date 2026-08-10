from __future__ import annotations

from pathlib import Path

import pytest

from anamnesis.mem0_lifecycle_v3 import (
    PROTOCOL_SHA256,
    LifecycleQueryResult,
    Mem0LifecycleResult,
    _raw_stale_present,
    load_protocol,
)


def test_raw_stale_opportunity_does_not_use_vacuous_truth() -> None:
    assert _raw_stale_present(("stale",), ("current", "stale")) is True
    assert _raw_stale_present(("stale",), ("current",)) is False
    assert _raw_stale_present((), ("current",)) is False


def test_mem0_lifecycle_v3_loader_is_byte_locked() -> None:
    value = load_protocol(Path("eval/mem0_lifecycle_v3.protocol.json"))
    assert value["model"]["expected_model_calls"] == 6
    assert value["model"]["context_length"] == 32768
    assert value["gate"]["filtered_query_exact"] == 4
    assert PROTOCOL_SHA256 == (
        "4e9d63ebd6d66b2c76175f94d55262ab74b0045d025297e2f813e07073aaef9a"
    )


def test_mem0_lifecycle_v3_loader_rejects_other_protocol() -> None:
    with pytest.raises(RuntimeError, match="bytes drifted"):
        load_protocol(Path("eval/mem0_inference_v2.protocol.json"))


def test_mem0_lifecycle_v3_source_has_no_provider_id_in_result_models() -> None:
    source = Path("src/anamnesis/mem0_lifecycle_v3.py").read_text()
    assert "provider_id" not in LifecycleQueryResult.model_fields
    assert "provider_id" not in Mem0LifecycleResult.model_fields
    assert "action_evidence_ids=()" in source
    assert "_serialize_llm_calls(calls)" in source
