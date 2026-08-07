"""Local-model compiler contract with type-specific, closed trigger shapes.

The hosted experiment intentionally keeps :mod:`anamnesis.wire` frozen around
the flat, required-but-nullable schema accepted by OpenAI strict structured
outputs.  Smaller local models have no reason to emit fields belonging to a
different trigger variant, so this module exposes a separate discriminated
transport contract.  Both contracts reduce to the same provider-neutral
``MemoryDelta`` domain model.
"""

from __future__ import annotations

import json
from datetime import date, datetime, time
from typing import Annotated, Literal

from pydantic import Field, model_validator

from anamnesis.memory import (
    ActionTemplate,
    AtTrigger,
    CancelIntent,
    Condition,
    ConditionTransitionTrigger,
    CreateIntent,
    FactKey,
    FactKeyTemplate,
    MemoryDelta,
    RecurringTrigger,
    SetFact,
    UpdateIntent,
    Weekday,
)
from anamnesis.schema import ActionValue, ObservableEvent, StrictModel

LOCAL_MEMORY_COMPILER_VERSION = "local.v0.1"


class LocalPayloadWire(StrictModel):
    """Closed payload whose unused local wire slots may be omitted."""

    subject: str
    address: str | None = None
    build: str | None = None
    date: str | None = None
    flight: str | None = None
    greenhouse: str | None = None
    item: str | None = None
    project: str | None = None
    quantity: int | float | None = None
    recipient: str | None = None
    room: str | None = None
    shipment: str | None = None
    tank: str | None = None
    trip: str | None = None

    def to_payload(self) -> dict[str, ActionValue]:
        return self.model_dump(exclude_none=True)


class LocalAtTriggerWire(StrictModel):
    type: Literal["at"]
    at: datetime

    def to_domain(self) -> AtTrigger:
        return AtTrigger(at=self.at)


class LocalRecurringTriggerWire(StrictModel):
    type: Literal["recurring"]
    local_time: time
    weekdays: list[Weekday]
    start_date: date
    end_date: date
    timezone: str

    def to_domain(self) -> RecurringTrigger:
        return RecurringTrigger(
            local_time=self.local_time,
            weekdays=tuple(self.weekdays),
            start_date=self.start_date,
            end_date=self.end_date,
            timezone=self.timezone,
        )


class LocalConditionTransitionTriggerWire(StrictModel):
    type: Literal["condition_transition"]
    active_from: datetime
    active_until: datetime

    def to_domain(self) -> ConditionTransitionTrigger:
        return ConditionTransitionTrigger(
            active_from=self.active_from,
            active_until=self.active_until,
        )


LocalTriggerWire = Annotated[
    LocalAtTriggerWire
    | LocalRecurringTriggerWire
    | LocalConditionTransitionTriggerWire,
    Field(discriminator="type"),
]


class LocalConditionWire(StrictModel):
    entity: str
    attribute: str
    key_template: bool = False
    operator: Literal["eq", "gte", "lte"]
    value: ActionValue
    unit: str | None = None

    def to_domain(self) -> Condition:
        key = (
            FactKeyTemplate(entity=self.entity, attribute=self.attribute)
            if self.key_template
            else FactKey(entity=self.entity, attribute=self.attribute)
        )
        return Condition(
            key=key,
            operator=self.operator,
            value=self.value,
            unit=self.unit,
        )


class LocalActionTemplateWire(StrictModel):
    kind: Literal["reminder"] = "reminder"
    payload: LocalPayloadWire
    summary: str

    def to_domain(self) -> ActionTemplate:
        return ActionTemplate(
            kind=self.kind,
            payload=self.payload.to_payload(),
            summary=self.summary,
        )


class LocalFactAssertionWire(StrictModel):
    entity: str
    attribute: str
    value: ActionValue
    unit: str | None = None

    def to_domain(self) -> SetFact:
        return SetFact(
            key=FactKey(entity=self.entity, attribute=self.attribute),
            value=self.value,
            unit=self.unit,
        )


class LocalIntentCreateWire(StrictModel):
    intent_id: str
    trigger: LocalTriggerWire
    required_conditions: list[LocalConditionWire]
    blockers: list[LocalConditionWire]
    action_template: LocalActionTemplateWire

    def to_domain(self) -> CreateIntent:
        return CreateIntent(
            intent_id=self.intent_id,
            trigger=self.trigger.to_domain(),
            required_conditions=tuple(
                condition.to_domain() for condition in self.required_conditions
            ),
            blockers=tuple(condition.to_domain() for condition in self.blockers),
            action_template=self.action_template.to_domain(),
        )


class LocalIntentUpdateWire(StrictModel):
    """A partial update; unchanged compound fields are omitted or null."""

    intent_id: str
    trigger: LocalTriggerWire | None = None
    required_conditions: list[LocalConditionWire] | None = None
    blockers: list[LocalConditionWire] | None = None
    action_template: LocalActionTemplateWire | None = None

    @model_validator(mode="after")
    def validate_changed_field(self) -> LocalIntentUpdateWire:
        if all(
            value is None
            for value in (
                self.trigger,
                self.required_conditions,
                self.blockers,
                self.action_template,
            )
        ):
            raise ValueError("local intent update requires a changed field")
        return self

    def to_domain(self) -> UpdateIntent:
        updates: dict[str, object] = {"intent_id": self.intent_id}
        if self.trigger is not None:
            updates["trigger"] = self.trigger.to_domain()
        if self.required_conditions is not None:
            updates["required_conditions"] = tuple(
                condition.to_domain() for condition in self.required_conditions
            )
        if self.blockers is not None:
            updates["blockers"] = tuple(
                condition.to_domain() for condition in self.blockers
            )
        if self.action_template is not None:
            updates["action_template"] = self.action_template.to_domain()
        return UpdateIntent.model_validate(updates)


class LocalIntentCancellationWire(StrictModel):
    intent_id: str

    def to_domain(self) -> CancelIntent:
        return CancelIntent(intent_id=self.intent_id)


class LocalMemoryDeltaWire(StrictModel):
    """Local structured-output envelope reduced atomically by the same store."""

    fact_assertions: list[LocalFactAssertionWire]
    intent_creates: list[LocalIntentCreateWire]
    intent_updates: list[LocalIntentUpdateWire]
    intent_cancellations: list[LocalIntentCancellationWire]

    def to_domain(self) -> MemoryDelta:
        mutations = [item.to_domain() for item in self.fact_assertions]
        mutations.extend(item.to_domain() for item in self.intent_creates)
        mutations.extend(item.to_domain() for item in self.intent_updates)
        mutations.extend(item.to_domain() for item in self.intent_cancellations)
        return MemoryDelta(mutations=tuple(mutations))


LOCAL_MEMORY_COMPILER_INSTRUCTIONS = (
    "You are the memory compiler for a simulated personal assistant. Convert "
    "only the current observable event into one strict local MemoryDelta.\n\n"
    "Rules:\n"
    "- Return all four arrays: fact_assertions, intent_creates, "
    "intent_updates, and intent_cancellations. Use [] when an array has no "
    "mutation.\n"
    "- For irrelevant conversation return all four arrays empty.\n"
    "- Use a fact assertion for a current-world fact, completion, or factual "
    "correction.\n"
    "- Use an intent create for a new prospective reminder. Choose a stable, "
    "normalized intent_id. The deterministic store supplies action_key and "
    "provenance.\n"
    "- Update or cancel only an intent_id listed in active state. In an update, "
    "omit unchanged top-level fields. Include the entire current compound field "
    "when trigger, conditions, blockers, or action_template changes.\n"
    "- A trigger must have exactly one of these shapes and no fields from the "
    "other shapes:\n"
    '  at: {"type":"at","at":"ISO datetime with UTC offset"}\n'
    '  recurring: {"type":"recurring","local_time":"HH:MM:SS",'
    '"weekdays":["monday"],"start_date":"YYYY-MM-DD",'
    '"end_date":"YYYY-MM-DD","timezone":"IANA timezone"}\n'
    '  condition transition: {"type":"condition_transition",'
    '"active_from":"ISO datetime with UTC offset",'
    '"active_until":"ISO datetime with UTC offset"}\n'
    "- A condition transition needs at least one required condition. A fact-key "
    "template is allowed only for recurring triggers and must contain {date} or "
    "{weekday}.\n"
    "- Conditions use only eq, gte, or lte. Do not invent operators, provenance "
    "IDs, cron expressions, hidden facts, or unsupported Boolean logic.\n"
    "- Resolve explicit temporal language against the current event timestamp "
    "and preserve its UTC offset.\n"
    "- Action payload subject is a lowercase imperative verb phrase. Omit every "
    "unused optional payload slot. Recurring actions use an ISO date, not a "
    "weekday.\n"
    "- Do not copy historical state unless the current event actually changes "
    "it. Return only schema-matching JSON and no prose.\n"
)


def build_local_memory_compiler_prompt(
    *,
    event: ObservableEvent,
    active_state: str,
) -> str:
    """Render the local compiler prompt without raw history or hidden gold."""

    return (
        f"{LOCAL_MEMORY_COMPILER_INSTRUCTIONS}\n"
        f"Current event: [{event.id}] {event.at.isoformat()} | "
        f"{event.kind} | {event.text}\n\n"
        "Active compact state (canonical JSON):\n"
        f"{active_state}\n"
    )


def local_memory_compiler_contract() -> str:
    """Return the local compiler prompt and schema for reproducibility hashes."""

    sentinel = ObservableEvent(
        id="<event-id>",
        at=datetime.fromisoformat("2000-01-01T00:00:00+00:00"),
        kind="user_message",
        text="<event-text>",
    )
    rendered = build_local_memory_compiler_prompt(
        event=sentinel,
        active_state='{"facts":[],"intents":[]}',
    )
    schema = json.dumps(
        LocalMemoryDeltaWire.model_json_schema(),
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"{LOCAL_MEMORY_COMPILER_VERSION}\n{rendered}\n{schema}"


__all__ = [
    "LOCAL_MEMORY_COMPILER_VERSION",
    "LocalAtTriggerWire",
    "LocalConditionTransitionTriggerWire",
    "LocalMemoryDeltaWire",
    "LocalRecurringTriggerWire",
    "LocalTriggerWire",
    "build_local_memory_compiler_prompt",
    "local_memory_compiler_contract",
]
