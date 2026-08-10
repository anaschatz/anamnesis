from __future__ import annotations

import hashlib
import json
from pathlib import Path

PROTOCOL = Path("eval/mem0_lifecycle_writer_v4.protocol.json")


def test_lifecycle_writer_v4_protocol_bytes_are_pinned() -> None:
    assert (
        hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
        == "3152b3ae3599e2badb42cf94168ed99520800c514760142228d540c01118b6bb"
    )


def test_lifecycle_writer_v4_is_fresh_and_frozen_before_prompt() -> None:
    value = json.loads(PROTOCOL.read_text())
    assert value["schema_version"] == "mem0_lifecycle_writer_protocol.v4"
    assert value["hypothesis_test_eligible"] is False
    assert value["preregistered_before_writer_prompt_implementation_and_model_calls"]
    assert len(value["events"]) == value["model"]["expected_model_calls"] == 9
    previous = json.loads(Path("eval/mem0_lifecycle_v3.protocol.json").read_text())
    assert {event["text"] for event in value["events"]}.isdisjoint(
        event["text"] for event in previous["events"]
    )


def test_lifecycle_writer_v4_references_only_causally_prior_same_scope_events() -> None:
    value = json.loads(PROTOCOL.read_text())
    observed: dict[str, set[str]] = {scope: set() for scope in value["scopes"]}
    event_scope: dict[str, str] = {}
    for event in value["events"]:
        expected = event["expected"]
        references = set(expected["supersedes_event_ids"])
        source = expected.get("key_source_event_id")
        if source is not None:
            references.add(source)
        assert references <= observed[event["scope"]]
        assert all(event_scope[item] == event["scope"] for item in references)
        observed[event["scope"]].add(event["id"])
        event_scope[event["id"]] = event["scope"]


def test_lifecycle_writer_v4_gate_is_exact_and_single_attempt() -> None:
    value = json.loads(PROTOCOL.read_text())
    assert value["gate"] == {
        "integrity_model_calls": 9,
        "wire_valid": 9,
        "directive_exact": 9,
        "filter_accepts": 8,
        "ignored": 1,
        "final_active_source_event_ids": {
            "a": ["mw4-e2", "mw4-e5", "mw4-e9"],
            "b": ["mw4-e7"],
        },
        "action_evidence_ids": [],
    }
    assert value["stopping_rule"]["attempts"] == 1
    assert value["stopping_rule"]["prompt_tuning_on_v4_events"] is False
