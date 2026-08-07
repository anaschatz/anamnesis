from __future__ import annotations

import hashlib

import pytest
from pydantic import ValidationError

from anamnesis.local_wire import (
    LOCAL_MEMORY_COMPILER_INSTRUCTIONS,
    LOCAL_MEMORY_COMPILER_W2_INSTRUCTIONS,
    LOCAL_MEMORY_COMPILER_W2_PAYLOAD_INVARIANT,
    LocalMemoryDeltaWire,
    build_local_memory_compiler_prompt,
    build_local_memory_compiler_w2_prompt,
    local_memory_compiler_contract,
    local_memory_compiler_w2_contract,
)
from anamnesis.memory import (
    AtTrigger,
    ConditionTransitionTrigger,
    CreateIntent,
    RecurringTrigger,
    UpdateIntent,
)
from anamnesis.schema import ObservableEvent


def _delta_with_trigger(trigger: dict[str, object]) -> dict[str, object]:
    return {
        "fact_assertions": [],
        "intent_creates": [
            {
                "intent_id": "assignment.reminder",
                "trigger": trigger,
                "required_conditions": [],
                "blockers": [],
                "action_template": {
                    "payload": {"subject": "send the assignment"},
                    "summary": "Send the assignment.",
                },
            }
        ],
        "intent_updates": [],
        "intent_cancellations": [],
    }


@pytest.mark.parametrize(
    ("trigger", "domain_type"),
    [
        (
            {"type": "at", "at": "2026-03-06T17:00:00+02:00"},
            AtTrigger,
        ),
        (
            {
                "type": "recurring",
                "local_time": "09:30:00",
                "weekdays": ["monday", "friday"],
                "start_date": "2026-03-02",
                "end_date": "2026-03-06",
                "timezone": "Europe/Athens",
            },
            RecurringTrigger,
        ),
        (
            {
                "type": "condition_transition",
                "active_from": "2026-03-02T09:00:00+02:00",
                "active_until": "2026-03-06T17:00:00+02:00",
            },
            ConditionTransitionTrigger,
        ),
    ],
)
def test_local_trigger_variants_convert_to_domain(
    trigger: dict[str, object],
    domain_type: type[object],
) -> None:
    record = _delta_with_trigger(trigger)
    if trigger["type"] == "condition_transition":
        create = record["intent_creates"][0]
        create["required_conditions"] = [
            {
                "entity": "assignment",
                "attribute": "submitted",
                "operator": "eq",
                "value": False,
            }
        ]

    delta = LocalMemoryDeltaWire.model_validate(record).to_domain()

    mutation = delta.mutations[0]
    assert isinstance(mutation, CreateIntent)
    assert isinstance(mutation.trigger, domain_type)
    assert mutation.action_template.payload == {"subject": "send the assignment"}


def test_local_at_trigger_rejects_fields_from_other_variants() -> None:
    record = _delta_with_trigger(
        {
            "type": "at",
            "at": "2026-03-06T17:00:00+02:00",
            "active_from": "2026-03-02T09:00:00+02:00",
        }
    )

    with pytest.raises(ValidationError, match="active_from"):
        LocalMemoryDeltaWire.model_validate(record)


def test_local_trigger_schema_is_closed_and_discriminated() -> None:
    schema = LocalMemoryDeltaWire.model_json_schema()
    trigger_schema = schema["$defs"]["LocalIntentCreateWire"]["properties"]["trigger"]

    assert trigger_schema["discriminator"] == {
        "mapping": {
            "at": "#/$defs/LocalAtTriggerWire",
            "condition_transition": ("#/$defs/LocalConditionTransitionTriggerWire"),
            "recurring": "#/$defs/LocalRecurringTriggerWire",
        },
        "propertyName": "type",
    }
    assert len(trigger_schema["oneOf"]) == 3
    for name in (
        "LocalAtTriggerWire",
        "LocalRecurringTriggerWire",
        "LocalConditionTransitionTriggerWire",
    ):
        assert schema["$defs"][name]["additionalProperties"] is False


def test_local_update_allows_omitted_unchanged_fields() -> None:
    wire = LocalMemoryDeltaWire.model_validate(
        {
            "fact_assertions": [],
            "intent_creates": [],
            "intent_updates": [
                {
                    "intent_id": "assignment.reminder",
                    "trigger": {
                        "type": "at",
                        "at": "2026-03-07T17:00:00+02:00",
                    },
                }
            ],
            "intent_cancellations": [],
        }
    )

    mutation = wire.to_domain().mutations[0]

    assert isinstance(mutation, UpdateIntent)
    assert mutation.model_fields_set == {"intent_id", "trigger"}
    assert mutation.required_conditions is None


def test_local_update_rejects_noop() -> None:
    with pytest.raises(ValidationError, match="requires a changed field"):
        LocalMemoryDeltaWire.model_validate(
            {
                "fact_assertions": [],
                "intent_creates": [],
                "intent_updates": [{"intent_id": "assignment.reminder"}],
                "intent_cancellations": [],
            }
        )


def test_local_compiler_prompt_has_only_observable_event_and_active_state() -> None:
    event = ObservableEvent(
        id="event-1",
        at="2026-03-02T09:00:00+02:00",
        kind="user_message",
        text="Remind me Friday to send the assignment.",
    )

    prompt = build_local_memory_compiler_prompt(
        event=event,
        active_state='{"facts":[],"intents":[]}',
    )

    assert "event-1" in prompt
    assert event.text in prompt
    assert '"facts":[],"intents":[]' in prompt
    assert "supersedes" not in prompt
    assert "gold" not in prompt.casefold()
    assert "exactly one of these shapes" in prompt


def test_local_compiler_contract_is_deterministic_and_schema_bound() -> None:
    first = local_memory_compiler_contract()
    second = local_memory_compiler_contract()

    assert first == second
    assert (
        hashlib.sha256(first.encode()).hexdigest()
        == hashlib.sha256(second.encode()).hexdigest()
    )
    assert '"discriminator"' in first
    assert first.startswith("local.v0.2\n")


def test_local_w1_contract_hash_remains_frozen() -> None:
    assert (
        hashlib.sha256(local_memory_compiler_contract().encode()).hexdigest()
        == "1ac94e36a5db89ef03798b091424494b9cf50f52ac8e7aaa70e8cfcfc3b0ebd8"
    )


def test_local_w2_contract_hash_is_frozen() -> None:
    contract = local_memory_compiler_w2_contract()

    assert contract.startswith("local.v0.3\n")
    assert (
        hashlib.sha256(contract.encode()).hexdigest()
        == "cb46570bfb1a101bff51008315ba121e07cea38a93de38fe6c79693d746f72c9"
    )


def test_local_w2_sparse_optional_payload_rule_is_complete() -> None:
    rule = LOCAL_MEMORY_COMPILER_W2_PAYLOAD_INVARIANT

    assert "explicitly sourced by the current event" in rule
    assert "legitimately preserved within an action_template" in rule
    assert "current event actually updates" in rule
    assert "Omit an unused payload key (preferred), or use JSON null" in rule
    for forbidden_filler in (
        "empty string",
        "false",
        "empty collection",
        "placeholder",
        "zero filler",
    ):
        assert forbidden_filler in rule
    assert "explicitly sourced quantity of zero remains valid" in rule
    assert "Before returning, remove filler values" in rule
    assert '{"subject":"check permit"}' in rule
    assert "Add optional keys only when sourced" in rule


def test_local_w2_uses_the_unchanged_w1_wire_schema() -> None:
    w1_schema = local_memory_compiler_contract().rsplit("\n", 1)[-1]
    w2_schema = local_memory_compiler_w2_contract().rsplit("\n", 1)[-1]

    assert w2_schema == w1_schema


def test_local_w2_prompt_delta_is_only_the_sparse_payload_invariant() -> None:
    event = ObservableEvent(
        id="event-1",
        at="2026-03-02T09:00:00+02:00",
        kind="user_message",
        text="Current event only.",
    )
    active_state = '{"facts":[],"intents":[]}'
    w1_prompt = build_local_memory_compiler_prompt(
        event=event,
        active_state=active_state,
    )
    w2_prompt = build_local_memory_compiler_w2_prompt(
        event=event,
        active_state=active_state,
    )

    assert LOCAL_MEMORY_COMPILER_W2_INSTRUCTIONS == (
        LOCAL_MEMORY_COMPILER_INSTRUCTIONS + LOCAL_MEMORY_COMPILER_W2_PAYLOAD_INVARIANT
    )
    assert w2_prompt == w1_prompt.replace(
        LOCAL_MEMORY_COMPILER_INSTRUCTIONS,
        LOCAL_MEMORY_COMPILER_W2_INSTRUCTIONS,
        1,
    )
    assert w2_prompt.count(LOCAL_MEMORY_COMPILER_W2_PAYLOAD_INVARIANT) == 1
    for leak_marker in ("writer_diagnostic", "oracle", "gold", "scenario"):
        assert leak_marker not in LOCAL_MEMORY_COMPILER_W2_PAYLOAD_INVARIANT.casefold()
