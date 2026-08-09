"""Provider-neutral contracts for external memory benchmark adapters."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Protocol, runtime_checkable

from pydantic import ConfigDict, Field, field_validator, model_validator

from anamnesis.schema import StrictModel

MemoryAdapterName = Literal["mem0", "letta", "graphiti"]
MemoryKind = Literal["profile", "decision", "project", "prospective_obligation"]


class _Frozen(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class BenchmarkScope(_Frozen):
    """Exact benchmark partition; no adapter may search outside it."""

    user_id: str = Field(min_length=1, max_length=128, pattern=r"^[a-z0-9._-]+$")
    session_id: str = Field(min_length=1, max_length=128, pattern=r"^[a-z0-9._-]+$")
    project_id: str | None = Field(
        default=None, max_length=128, pattern=r"^[a-z0-9._-]+$"
    )


class BenchmarkMemoryInput(_Frozen):
    """Observable input released to every adapter at the same checkpoint."""

    source_event_id: str = Field(
        min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._:-]+$"
    )
    kind: MemoryKind
    text: str = Field(min_length=1, max_length=8192)
    observed_at: datetime


class BenchmarkQuery(_Frozen):
    scope: BenchmarkScope
    text: str = Field(min_length=1, max_length=8192)
    at: datetime
    top_k: int = Field(ge=1, le=100)


class BenchmarkHit(_Frozen):
    """Normalized retrieval output with provider identity kept opaque."""

    adapter: MemoryAdapterName
    handle: str = Field(min_length=1, max_length=256)
    text: str = Field(min_length=1, max_length=8192)
    score: float | None = None
    kind: MemoryKind | None = None
    observed_at: datetime | None = None
    valid_from: datetime | None = None
    valid_until: datetime | None = None
    source_event_ids: tuple[str, ...] = ()
    action_evidence_ids: tuple[str, ...] = ()

    @field_validator("action_evidence_ids")
    @classmethod
    def provider_hits_cannot_supply_action_evidence(
        cls, value: tuple[str, ...]
    ) -> tuple[str, ...]:
        if value:
            raise ValueError("external memory hits cannot supply action evidence")
        return value

    @model_validator(mode="after")
    def valid_window_is_ordered(self) -> BenchmarkHit:
        if (
            self.valid_from is not None
            and self.valid_until is not None
            and self.valid_until < self.valid_from
        ):
            raise ValueError("memory validity window is reversed")
        return self


class AdapterCapabilities(_Frozen):
    automatic_fact_extraction: bool = False
    deduplication: bool = False
    user_session_scoping: bool = False
    vector_retrieval: bool = False
    agent_managed_core_memory: bool = False
    agent_managed_archival_memory: bool = False
    temporal_knowledge_graph: bool = False
    entity_relation_model: bool = False
    validity_windows: bool = False
    episode_provenance: bool = False


class AdapterIdentity(_Frozen):
    name: MemoryAdapterName
    upstream_revision: str = Field(min_length=7, max_length=128)
    package_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    capabilities: AdapterCapabilities
    production_dependency: Literal[False] = False
    authoritative_temporal_state: Literal[False] = False
    supports_action_evidence: Literal[False] = False


ADAPTER_PROFILES: dict[MemoryAdapterName, AdapterCapabilities] = {
    "mem0": AdapterCapabilities(
        automatic_fact_extraction=True,
        deduplication=True,
        user_session_scoping=True,
        vector_retrieval=True,
    ),
    "letta": AdapterCapabilities(
        user_session_scoping=True,
        vector_retrieval=True,
        agent_managed_core_memory=True,
        agent_managed_archival_memory=True,
    ),
    "graphiti": AdapterCapabilities(
        vector_retrieval=True,
        temporal_knowledge_graph=True,
        entity_relation_model=True,
        validity_windows=True,
        episode_provenance=True,
    ),
}


@runtime_checkable
class MemoryBenchmarkAdapter(Protocol):
    """Async lifecycle implemented by isolated Mem0, Letta, or Graphiti cells."""

    @property
    def identity(self) -> AdapterIdentity: ...

    async def begin(self, scope: BenchmarkScope) -> None: ...

    async def ingest(self, item: BenchmarkMemoryInput) -> None: ...

    async def search(self, query: BenchmarkQuery) -> tuple[BenchmarkHit, ...]: ...

    async def close(self, scope: BenchmarkScope) -> None: ...


def validate_adapter_identity(identity: AdapterIdentity) -> None:
    """Fail closed when a cell claims capabilities outside its frozen profile."""

    if identity.capabilities != ADAPTER_PROFILES[identity.name]:
        raise ValueError(f"{identity.name} capability profile differs")


__all__ = [
    "ADAPTER_PROFILES",
    "AdapterCapabilities",
    "AdapterIdentity",
    "BenchmarkHit",
    "BenchmarkMemoryInput",
    "BenchmarkQuery",
    "BenchmarkScope",
    "MemoryAdapterName",
    "MemoryBenchmarkAdapter",
    "MemoryKind",
    "validate_adapter_identity",
]
