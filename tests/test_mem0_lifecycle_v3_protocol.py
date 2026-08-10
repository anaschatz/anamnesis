from __future__ import annotations

import hashlib
import json
from pathlib import Path

PROTOCOL = Path("eval/mem0_lifecycle_v3.protocol.json")


def test_mem0_lifecycle_v3_protocol_is_fresh_and_preregistered() -> None:
    value = json.loads(PROTOCOL.read_text())
    assert value["schema_version"] == "mem0_lifecycle_protocol.v3"
    assert value["preregistered_before_implementation_and_model_calls"] is True
    assert value["hypothesis_test_eligible"] is False
    assert len(value["events"]) == 6
    assert len(value["queries"]) == 4
    previous = json.loads(Path("eval/mem0_inference_v2.protocol.json").read_text())
    assert {event["text"] for event in value["events"]}.isdisjoint(
        event["text"] for event in previous["events"]
    )


def test_mem0_lifecycle_v3_directives_are_causal_and_scope_local() -> None:
    value = json.loads(PROTOCOL.read_text())
    observed: dict[str, set[str]] = {"a": set(), "b": set()}
    event_scope: dict[str, str] = {}
    for event in value["events"]:
        scope = event["scope"]
        supersedes = set(event["directive"]["supersedes_event_ids"])
        assert supersedes <= observed[scope]
        assert all(event_scope[item] == scope for item in supersedes)
        observed[scope].add(event["id"])
        event_scope[event["id"]] = scope
    assert value["gate"]["raw_stale_recall_opportunities"] == 2
    assert value["gate"]["filtered_query_exact"] == 4
    assert value["gate"]["filtered_stale_hits"] == 0
    assert [query["top_k"] for query in value["queries"]] == [2, 2, 1, 1]


def test_mem0_lifecycle_v3_protocol_bytes_are_pinned() -> None:
    assert (
        hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
        == "4e9d63ebd6d66b2c76175f94d55262ab74b0045d025297e2f813e07073aaef9a"
    )
