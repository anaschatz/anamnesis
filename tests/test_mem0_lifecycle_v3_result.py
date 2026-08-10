from __future__ import annotations

import hashlib
import json
from pathlib import Path

from anamnesis.mem0_lifecycle_v3 import Mem0LifecycleResult

RAW = Path("results/mem0_lifecycle_v3.raw.json")
RECOMPUTED = Path("results/mem0_lifecycle_v3.json")


def test_mem0_lifecycle_v3_raw_result_is_byte_exact_and_valid() -> None:
    assert (
        hashlib.sha256(RAW.read_bytes()).hexdigest()
        == "2c31620cba59e532b65ea6c646be285539440a2642ed18ccb834afb534187db4"
    )
    result = Mem0LifecycleResult.model_validate_json(RAW.read_text())
    assert result.source_commit == "a80b2b1ffb546e7907869654060f7c4bfada305f"
    assert result.integrity_passed is True
    assert result.localhost_model_calls == 6
    assert result.scope_isolation_passed is True
    assert result.cleanup_passed is True
    assert result.filtered_query_exact == 4
    assert result.filtered_stale_hits == 0


def test_mem0_lifecycle_v3_gate_recomputes_without_vacuous_truth() -> None:
    raw = Mem0LifecycleResult.model_validate_json(RAW.read_text())
    stale_queries = [
        query.query_id
        for query in raw.query_results
        if query.required_raw_stale_source_event_ids
        and all(
            source in query.raw_source_event_ids
            for source in query.required_raw_stale_source_event_ids
        )
    ]
    exact = sum(query.filtered_exact for query in raw.query_results)
    filtered_stale = sum(query.filtered_stale_hits for query in raw.query_results)
    recomputed = json.loads(RECOMPUTED.read_text())
    assert stale_queries == ["ml3-q1", "ml3-q2"]
    assert recomputed["raw_stale_query_ids"] == stale_queries
    assert recomputed["raw_stale_recall_opportunities"] == 2
    assert recomputed["filtered_query_exact"] == exact == 4
    assert recomputed["filtered_stale_hits"] == filtered_stale == 0
    assert recomputed["recomputed_semantic_passed"] is True
    assert recomputed["model_rerun_performed"] is False


def test_mem0_lifecycle_v3_filter_outputs_are_exact() -> None:
    raw = Mem0LifecycleResult.model_validate_json(RAW.read_text())
    assert {
        query.query_id: query.filtered_source_event_ids for query in raw.query_results
    } == {
        "ml3-q1": ("ml3-e2",),
        "ml3-q2": (),
        "ml3-q3": ("ml3-e5",),
        "ml3-q4": ("ml3-e6",),
    }
    assert all(query.filtered_stale_hits == 0 for query in raw.query_results)


def test_mem0_lifecycle_v3_artifacts_have_no_paths_or_secrets() -> None:
    text = (RAW.read_text() + RECOMPUTED.read_text()).lower()
    for forbidden in ("/users/", "/private/tmp", "api_key", "bearer "):
        assert forbidden not in text
