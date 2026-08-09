import hashlib

from anamnesis.action_canonicalizer import canonicalize_immediate_decision
from anamnesis.openmemory_vllm_v6 import _correct
from anamnesis.openmemory_vllm_v8 import (
    ADAPTER_PATH,
    FIXTURE_PATH,
    SDK_PIN_PATH,
    V8Fixture,
    _load_inputs,
)
from anamnesis.schema import Decision, ProposedAction


def test_v8_frozen_inputs_are_real_sdk_and_fresh() -> None:
    pin, fixture, runtime, sdk_pin = _load_inputs()
    assert len(fixture.cases) == 6
    assert sum(case.helpful_opportunity for case in fixture.cases) == 3
    assert runtime.served_model == "anamnesis-openmemory-v8"
    assert sdk_pin.package_version == "1.3.0"
    assert sdk_pin.embedding_provider == "synthetic"
    assert hashlib.sha256(ADAPTER_PATH.read_bytes()).hexdigest() == (
        pin.openmemory_adapter_sha256
    )
    assert hashlib.sha256(SDK_PIN_PATH.read_bytes()).hexdigest() == (
        pin.openmemory_sdk_pin_sha256
    )
    fixture_text = FIXTURE_PATH.read_text()
    assert "omr2_" not in fixture_text
    assert "omsdk8_" in fixture_text


def test_v8_expected_helpful_actions_are_reachable_by_frozen_rules() -> None:
    fixture = V8Fixture.model_validate_json(FIXTURE_PATH.read_text())
    raw_payloads = (
        {
            "subject": "send harbor access renewal",
            "recipient": "Meridian Harbor Office",
            "room": "18 Lantern Quay",
        },
        {
            "subject": "track shipment",
            "item": "ceramic glaze samples",
            "shipment": "GLAZE-842",
        },
        {
            "subject": "upload mountain lichen field summary",
            "project": "Silver Ridge",
        },
    )
    for case, payload in zip(fixture.cases[:3], raw_payloads, strict=True):
        raw = Decision(
            actions=[
                ProposedAction(
                    kind="reminder",
                    action_key=case.event.id,
                    payload=payload,
                    summary="Execute action",
                    evidence_event_ids=[case.event.id],
                )
            ]
        )
        normalized = canonicalize_immediate_decision(
            event=case.event,
            retrospective_recall=tuple(item.content for item in case.memories[:1]),
            decision=raw,
        )
        assert _correct(case.expected, normalized.decision)


def test_v8_controls_preserve_event_authority() -> None:
    fixture = V8Fixture.model_validate_json(FIXTURE_PATH.read_text())
    no_action = Decision(actions=[])
    injection = fixture.cases[4]
    normalized = canonicalize_immediate_decision(
        event=injection.event,
        retrospective_recall=tuple(item.content for item in injection.memories),
        decision=no_action,
    )
    assert normalized.decision == no_action
    assert not normalized.changes
