from __future__ import annotations

import hashlib
from pathlib import Path

from anamnesis.lifecycle_writer_v4 import LifecycleWriterResult

RAW = Path("results/mem0_lifecycle_writer_v4.raw.json")


def _result() -> LifecycleWriterResult:
    return LifecycleWriterResult.model_validate_json(RAW.read_text())


def test_lifecycle_writer_v4_raw_result_is_byte_pinned() -> None:
    assert (
        hashlib.sha256(RAW.read_bytes()).hexdigest()
        == "e0bab6f607daecbcf2892a6b4827cb3d0bc3c029bc244f862f2eaeb6d64d532f"
    )
    result = _result()
    assert result.source_commit == "5aa8b081d38396da3d11aee6d158993e63cbbc90"
    assert result.integrity_passed is True
    assert result.semantic_passed is False
    assert result.passed is False


def test_lifecycle_writer_v4_failure_metrics_are_exact() -> None:
    result = _result()
    assert result.localhost_model_calls == 9
    assert (result.prompt_tokens, result.completion_tokens) == (3414, 562)
    assert result.wire_valid == 3
    assert result.directive_exact == 2
    assert result.filter_accepts == 2
    assert result.ignored == 0
    assert result.final_active_source_event_ids == {
        "a": ("mw4-e5", "mw4-e8"),
        "b": (),
    }
    assert all(call.done_reason == "stop" for call in result.llm_calls)


def test_lifecycle_writer_v4_failure_taxonomy_is_preserved() -> None:
    result = _result()
    invalid = [event for event in result.event_results if not event.wire_valid]
    assert [event.event_id for event in invalid] == [
        "mw4-e1",
        "mw4-e2",
        "mw4-e3",
        "mw4-e4",
        "mw4-e6",
        "mw4-e7",
    ]
    assert (
        sum(
            "Extra inputs are not permitted" in (event.error_detail or "")
            for event in invalid
        )
        == 4
    )
    assert sum("Field required" in (event.error_detail or "") for event in invalid) == 3
    reschedule = next(
        event for event in result.event_results if event.event_id == "mw4-e9"
    )
    assert (
        reschedule.key == "project_borealis.lab_permit_renewal.obligation_2046_august"
    )
    assert reschedule.supersedes_event_ids == ()
    assert reschedule.filter_accepted is False
    assert reschedule.error_code == "filter_rejected"


def test_lifecycle_writer_v4_result_has_no_paths_or_secrets() -> None:
    text = RAW.read_text().lower()
    for forbidden in ("/users/", "/private/tmp", "api_key", "bearer "):
        assert forbidden not in text
