from anamnesis.action_canonicalizer import canonicalize_immediate_decision
from anamnesis.action_canonicalizer_v2 import canonicalize_immediate_decision_v2
from anamnesis.schema import Decision, ObservableEvent, ProposedAction


def _event(text: str) -> ObservableEvent:
    return ObservableEvent(
        id="event_1",
        at="2042-01-01T09:00:00+02:00",
        kind="user_message",
        text=text,
    )


def _decision(payload: dict[str, str]) -> Decision:
    return Decision(
        actions=[
            ProposedAction(
                kind="reminder",
                action_key="event_1",
                payload=payload,
                summary="Execute requested action",
                evidence_event_ids=["event_1"],
            )
        ]
    )


def test_moves_grounded_address_from_wrong_optional_slot() -> None:
    event = _event("Act now: send the permit to my usual agent.")
    result = canonicalize_immediate_decision(
        event=event,
        retrospective_recall=("The agent is at 44 Amber Crescent.",),
        decision=_decision(
            {
                "subject": "send permit",
                "recipient": "Lumen Office",
                "room": "44 Amber Crescent",
            }
        ),
    )
    assert result.decision.actions[0].payload == {
        "subject": "send permit",
        "recipient": "Lumen Office",
        "address": "44 Amber Crescent",
    }


def test_does_not_move_ungrounded_address_like_value() -> None:
    event = _event("Act now: send the permit.")
    result = canonicalize_immediate_decision(
        event=event,
        retrospective_recall=None,
        decision=_decision({"subject": "send permit", "room": "44 Amber Crescent"}),
    )
    assert result.decision.actions[0].payload["room"] == "44 Amber Crescent"


def test_track_shipment_uses_source_grounded_item_as_subject_object() -> None:
    event = _event("Act now: track the shipment for my prism lenses.")
    result = canonicalize_immediate_decision(
        event=event,
        retrospective_recall=("The shipment identifier is PRISM-72.",),
        decision=_decision(
            {
                "subject": "track shipment",
                "item": "prism lenses",
                "shipment": "PRISM-72",
                "recipient": "my",
            }
        ),
    )
    assert result.decision.actions[0].payload == {
        "subject": "track prism lenses",
        "shipment": "PRISM-72",
    }


def test_upload_report_rewrites_only_from_explicit_event_shape() -> None:
    event = _event("Act now: upload the latest report for my river sediment study.")
    result = canonicalize_immediate_decision(
        event=event,
        retrospective_recall=("The study is Delta Silt Review.",),
        decision=_decision(
            {"subject": "upload report", "project": "Delta Silt Review"}
        ),
    )
    assert result.decision.actions[0].payload == {
        "subject": "upload river sediment report",
        "project": "Delta Silt Review",
    }


def test_no_action_and_wrong_provenance_are_unchanged() -> None:
    event = _event("Act now: track the shipment for my prism lenses.")
    assert not canonicalize_immediate_decision(
        event=event, retrospective_recall=(), decision=Decision(actions=[])
    ).changes
    wrong = _decision({"subject": "track shipment", "item": "prism lenses"})
    wrong.actions[0].evidence_event_ids = ["other"]
    result = canonicalize_immediate_decision(
        event=event, retrospective_recall=(), decision=wrong
    )
    assert result.decision == wrong


def test_drops_only_event_grounded_location_that_duplicates_recipient() -> None:
    event = _event("Act now: send the lens inventory to Aurora Workshop.")
    duplicate = canonicalize_immediate_decision_v2(
        event=event,
        retrospective_recall=(),
        decision=_decision(
            {
                "subject": "send lens inventory",
                "recipient": "Aurora Workshop",
                "room": "Aurora Workshop",
            }
        ),
    )
    assert duplicate.decision.actions[0].payload == {
        "subject": "send lens inventory",
        "recipient": "Aurora Workshop",
    }

    distinct = canonicalize_immediate_decision_v2(
        event=event,
        retrospective_recall=(),
        decision=_decision(
            {
                "subject": "send lens inventory",
                "recipient": "Aurora Workshop",
                "room": "Bench 7",
            }
        ),
    )
    assert distinct.decision.actions[0].payload["room"] == "Bench 7"


def test_removes_simple_article_only_from_event_quoted_imperative() -> None:
    event = _event("Act now: photograph the archive seals.")
    result = canonicalize_immediate_decision_v2(
        event=event,
        retrospective_recall=(),
        decision=_decision({"subject": "photograph the archive seals"}),
    )
    assert result.decision.actions[0].payload == {"subject": "photograph archive seals"}

    ungrounded = canonicalize_immediate_decision_v2(
        event=_event("Act now: photograph archive seals."),
        retrospective_recall=(),
        decision=_decision({"subject": "photograph the archive seals"}),
    )
    assert ungrounded.decision.actions[0].payload["subject"] == (
        "photograph the archive seals"
    )


def test_composes_generic_upload_subject_only_with_recalled_project() -> None:
    event = _event("Act now: upload the field summary for my mountain lichen survey.")
    result = canonicalize_immediate_decision_v2(
        event=event,
        retrospective_recall=("The mountain lichen project is Silver Ridge.",),
        decision=_decision(
            {
                "subject": "upload field summary",
                "item": "mountain lichen survey",
                "project": "Silver Ridge",
            }
        ),
    )
    assert result.decision.actions[0].payload == {
        "subject": "upload mountain lichen field summary",
        "project": "Silver Ridge",
    }

    no_recall = canonicalize_immediate_decision_v2(
        event=event,
        retrospective_recall=(),
        decision=_decision(
            {
                "subject": "upload field summary",
                "item": "mountain lichen survey",
                "project": "Silver Ridge",
            }
        ),
    )
    assert no_recall.decision.actions[0].payload["subject"] == "upload field summary"


def test_v2_canonicalization_is_idempotent() -> None:
    event = _event("Act now: scan the conservation tags.")
    first = canonicalize_immediate_decision_v2(
        event=event,
        retrospective_recall=(),
        decision=_decision({"subject": "scan the conservation tags"}),
    )
    second = canonicalize_immediate_decision_v2(
        event=event,
        retrospective_recall=(),
        decision=first.decision,
    )
    assert second.decision == first.decision
    assert second.changes == ()
