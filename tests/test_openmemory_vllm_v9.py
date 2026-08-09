import hashlib

from anamnesis.action_canonicalizer_v2 import (
    ACTION_CANONICALIZER_VERSION,
    canonicalize_immediate_decision_v2,
)
from anamnesis.openmemory_vllm_v6 import _correct
from anamnesis.openmemory_vllm_v9 import (
    ADAPTER_PATH,
    FIXTURE_PATH,
    SDK_PIN_PATH,
    _load_inputs,
)
from anamnesis.schema import Decision, ProposedAction


def _decision(case, payload: dict[str, str]) -> Decision:
    return Decision(
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


def test_v9_frozen_inputs_bind_real_sdk_and_canonicalizer_v2() -> None:
    pin, fixture, runtime, sdk_pin = _load_inputs()
    assert ACTION_CANONICALIZER_VERSION == "immediate-action-canonicalizer.v2"
    assert len(fixture.cases) == 6
    assert sum(case.helpful_opportunity for case in fixture.cases) == 3
    assert runtime.served_model == "anamnesis-openmemory-v9"
    assert sdk_pin.package_version == "1.3.0"
    assert hashlib.sha256(ADAPTER_PATH.read_bytes()).hexdigest() == (
        pin.openmemory_adapter_sha256
    )
    assert hashlib.sha256(SDK_PIN_PATH.read_bytes()).hexdigest() == (
        pin.openmemory_sdk_pin_sha256
    )
    text = FIXTURE_PATH.read_text()
    assert "omsdk8_" not in text
    assert "omsdk9_" in text


def test_v9_three_canonicalizer_v2_residuals_are_reachable() -> None:
    _, fixture, _, _ = _load_inputs()
    cases = {case.id: case for case in fixture.cases}
    examples = (
        (
            cases["omsdk9_moss_project"],
            {
                "subject": "upload survey digest",
                "item": "alpine moss survey",
                "project": "Cloud Needle",
            },
        ),
        (
            cases["omsdk9_current_workshop_wins"],
            {
                "subject": "send lens inventory",
                "recipient": "Aurora Workshop",
                "room": "Aurora Workshop",
            },
        ),
        (
            cases["omsdk9_no_hit_control"],
            {"subject": "scan the conservation tags"},
        ),
    )
    for case, payload in examples:
        recall = tuple(item.content for item in case.memories[:1])
        normalized = canonicalize_immediate_decision_v2(
            event=case.event,
            retrospective_recall=recall,
            decision=_decision(case, payload),
        )
        assert _correct(case.expected, normalized.decision)


def test_v9_recall_injection_control_remains_no_action() -> None:
    _, fixture, _, _ = _load_inputs()
    case = next(item for item in fixture.cases if item.id == "omsdk9_injection_control")
    result = canonicalize_immediate_decision_v2(
        event=case.event,
        retrospective_recall=tuple(item.content for item in case.memories),
        decision=Decision(actions=[]),
    )
    assert result.decision.actions == []
    assert result.changes == ()
