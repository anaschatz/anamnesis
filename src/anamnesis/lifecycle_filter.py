"""Deterministic active-state projection over non-authoritative recall hits."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from anamnesis.memory_benchmark import BenchmarkHit
from anamnesis.schema import StrictModel


class LifecycleFilterError(RuntimeError):
    """A directive or recall hit violated the causal lifecycle contract."""


class LifecycleDirective(StrictModel):
    """Authored lifecycle fact released at one observable source event."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_event_id: str = Field(
        min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._:-]+$"
    )
    key: str = Field(
        min_length=3,
        max_length=256,
        pattern=r"^[a-z0-9]+(?:[._-][a-z0-9]+)+$",
    )
    operation: Literal["upsert", "cancel"]
    supersedes_event_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def source_cannot_supersede_itself(self) -> LifecycleDirective:
        if self.source_event_id in self.supersedes_event_ids:
            raise ValueError("lifecycle directive cannot supersede itself")
        if len(set(self.supersedes_event_ids)) != len(self.supersedes_event_ids):
            raise ValueError("lifecycle supersedes IDs must be unique")
        return self


class DeterministicLifecycleFilter:
    """Tracks one scope's active source event for every lifecycle key."""

    def __init__(self) -> None:
        self._key_by_source: dict[str, str] = {}
        self._active_by_key: dict[str, str] = {}
        self._invalidated: set[str] = set()

    @property
    def active_source_event_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._active_by_key.values()))

    @property
    def invalidated_source_event_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._invalidated))

    def apply(self, directive: LifecycleDirective) -> None:
        source = directive.source_event_id
        if source in self._key_by_source:
            raise LifecycleFilterError(f"duplicate lifecycle source event: {source}")
        for superseded in directive.supersedes_event_ids:
            known_key = self._key_by_source.get(superseded)
            if known_key is None:
                raise LifecycleFilterError(
                    f"lifecycle directive supersedes unknown event: {superseded}"
                )
            if known_key != directive.key:
                raise LifecycleFilterError(
                    "lifecycle directive cannot supersede a different key"
                )

        current = self._active_by_key.get(directive.key)
        if current is not None and current not in directive.supersedes_event_ids:
            raise LifecycleFilterError(
                "lifecycle replacement must explicitly supersede the active event"
            )
        if directive.operation == "cancel" and current is None:
            raise LifecycleFilterError("lifecycle cancellation has no active value")

        self._key_by_source[source] = directive.key
        self._invalidated.update(directive.supersedes_event_ids)
        if current is not None:
            self._invalidated.add(current)

        if directive.operation == "upsert":
            self._active_by_key[directive.key] = source
        else:
            self._invalidated.add(source)
            self._active_by_key.pop(directive.key, None)

    def filter_active_hits(
        self, hits: Iterable[BenchmarkHit]
    ) -> tuple[BenchmarkHit, ...]:
        active: list[BenchmarkHit] = []
        for hit in hits:
            if len(hit.source_event_ids) != 1:
                raise LifecycleFilterError(
                    "lifecycle-filtered hits require exactly one source event"
                )
            source = hit.source_event_ids[0]
            key = self._key_by_source.get(source)
            if key is None:
                raise LifecycleFilterError(
                    f"recall hit has unknown lifecycle source event: {source}"
                )
            if self._active_by_key.get(key) == source:
                active.append(hit)
        return tuple(active)


__all__ = [
    "DeterministicLifecycleFilter",
    "LifecycleDirective",
    "LifecycleFilterError",
]
