from __future__ import annotations

from datetime import datetime

import pytest

from anamnesis.lifecycle_filter import (
    DeterministicLifecycleFilter,
    LifecycleDirective,
    LifecycleFilterError,
)
from anamnesis.memory_benchmark import BenchmarkHit


def _directive(
    source: str,
    key: str,
    operation: str = "upsert",
    supersedes: tuple[str, ...] = (),
) -> LifecycleDirective:
    return LifecycleDirective(
        source_event_id=source,
        key=key,
        operation=operation,
        supersedes_event_ids=supersedes,
    )


def _hit(source: str, text: str) -> BenchmarkHit:
    return BenchmarkHit(
        adapter="mem0",
        handle=f"opaque-{source}",
        text=text,
        score=0.9,
        kind="profile",
        observed_at=datetime.fromisoformat("2045-01-01T00:00:00+02:00"),
        source_event_ids=(source,),
        action_evidence_ids=(),
    )


def test_correction_keeps_only_explicit_replacement() -> None:
    lifecycle = DeterministicLifecycleFilter()
    lifecycle.apply(_directive("e1", "profile.report_language"))
    lifecycle.apply(_directive("e2", "profile.report_language", supersedes=("e1",)))
    assert lifecycle.active_source_event_ids == ("e2",)
    assert lifecycle.invalidated_source_event_ids == ("e1",)
    assert lifecycle.filter_active_hits(
        (_hit("e1", "Italian"), _hit("e2", "German"))
    ) == (_hit("e2", "German"),)


def test_cancellation_removes_obligation_and_marker_from_active_view() -> None:
    lifecycle = DeterministicLifecycleFilter()
    lifecycle.apply(_directive("e3", "obligation.key_return"))
    lifecycle.apply(
        _directive(
            "e4",
            "obligation.key_return",
            operation="cancel",
            supersedes=("e3",),
        )
    )
    assert lifecycle.active_source_event_ids == ()
    assert lifecycle.invalidated_source_event_ids == ("e3", "e4")
    assert (
        lifecycle.filter_active_hits(
            (_hit("e3", "return key"), _hit("e4", "cancelled"))
        )
        == ()
    )


@pytest.mark.parametrize(
    ("directives", "message"),
    [
        (
            (
                _directive("e1", "profile.language"),
                _directive("e2", "profile.language"),
            ),
            "explicitly supersede",
        ),
        (
            (_directive("e2", "profile.language", supersedes=("missing",)),),
            "unknown event",
        ),
        (
            (
                _directive("e1", "profile.language"),
                _directive("e2", "project.owner", supersedes=("e1",)),
            ),
            "different key",
        ),
        (
            (_directive("e1", "obligation.key", operation="cancel", supersedes=()),),
            "no active value",
        ),
    ],
)
def test_invalid_lifecycle_directives_fail_closed(
    directives: tuple[LifecycleDirective, ...], message: str
) -> None:
    lifecycle = DeterministicLifecycleFilter()
    with pytest.raises(LifecycleFilterError, match=message):
        for directive in directives:
            lifecycle.apply(directive)


def test_unknown_or_ambiguous_hit_provenance_fails_closed() -> None:
    lifecycle = DeterministicLifecycleFilter()
    lifecycle.apply(_directive("e1", "profile.language"))
    with pytest.raises(LifecycleFilterError, match="unknown lifecycle"):
        lifecycle.filter_active_hits((_hit("unknown", "value"),))
    ambiguous = _hit("e1", "value").model_copy(
        update={"source_event_ids": ("e1", "e2")}
    )
    with pytest.raises(LifecycleFilterError, match="exactly one source"):
        lifecycle.filter_active_hits((ambiguous,))


def test_filter_never_adds_action_evidence() -> None:
    lifecycle = DeterministicLifecycleFilter()
    lifecycle.apply(_directive("e1", "project.owner"))
    hit = _hit("e1", "Mira owns the budget")
    assert lifecycle.filter_active_hits((hit,))[0].action_evidence_ids == ()
