from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from anamnesis.memory_benchmark import (
    ADAPTER_PROFILES,
    AdapterIdentity,
    BenchmarkHit,
    BenchmarkMemoryInput,
    BenchmarkQuery,
    BenchmarkScope,
    MemoryBenchmarkAdapter,
    validate_adapter_identity,
)


def _scope() -> BenchmarkScope:
    return BenchmarkScope(user_id="user-1", session_id="session-1", project_id="p-1")


def test_profiles_capture_three_distinct_baseline_architectures() -> None:
    mem0 = ADAPTER_PROFILES["mem0"]
    letta = ADAPTER_PROFILES["letta"]
    graphiti = ADAPTER_PROFILES["graphiti"]
    assert mem0.automatic_fact_extraction and mem0.deduplication
    assert letta.agent_managed_core_memory and letta.agent_managed_archival_memory
    assert graphiti.temporal_knowledge_graph and graphiti.validity_windows
    assert len({mem0, letta, graphiti}) == 3


@pytest.mark.parametrize("name", ["mem0", "letta", "graphiti"])
def test_adapter_identity_is_non_authoritative_and_not_a_production_dependency(
    name: str,
) -> None:
    identity = AdapterIdentity(
        name=name,
        upstream_revision="abcdef0123456789",
        package_identity_sha256="1" * 64,
        capabilities=ADAPTER_PROFILES[name],
    )
    validate_adapter_identity(identity)
    assert not identity.production_dependency
    assert not identity.authoritative_temporal_state
    assert not identity.supports_action_evidence


def test_external_hit_cannot_become_action_evidence() -> None:
    with pytest.raises(ValidationError, match="cannot supply action evidence"):
        BenchmarkHit(
            adapter="mem0",
            handle="opaque-1",
            text="User prefers concise reports.",
            action_evidence_ids=("event-1",),
        )


def test_temporal_hit_rejects_reversed_validity_window() -> None:
    with pytest.raises(ValidationError, match="validity window is reversed"):
        BenchmarkHit(
            adapter="graphiti",
            handle="opaque-2",
            text="Project Atlas uses the north lab.",
            valid_from="2040-02-02T00:00:00Z",
            valid_until="2040-02-01T00:00:00Z",
        )


def test_scope_and_inputs_are_closed_and_time_explicit() -> None:
    now = datetime(2040, 1, 1, tzinfo=UTC)
    item = BenchmarkMemoryInput(
        source_event_id="event:1",
        kind="prospective_obligation",
        text="Submit the permit next Friday.",
        observed_at=now,
    )
    query = BenchmarkQuery(scope=_scope(), text="What is due?", at=now, top_k=3)
    assert item.observed_at == query.at
    with pytest.raises(ValidationError):
        BenchmarkScope(user_id="User With Spaces", session_id="session-1")


def test_protocol_accepts_an_injected_async_fake_without_third_party_imports() -> None:
    class FakeAdapter:
        identity = AdapterIdentity(
            name="mem0",
            upstream_revision="abcdef0",
            package_identity_sha256="2" * 64,
            capabilities=ADAPTER_PROFILES["mem0"],
        )

        async def begin(self, scope):
            return None

        async def ingest(self, item):
            return None

        async def search(self, query):
            return ()

        async def close(self, scope):
            return None

    assert isinstance(FakeAdapter(), MemoryBenchmarkAdapter)
    source = Path("src/anamnesis/memory_benchmark.py").read_text()
    assert "import mem0" not in source
    assert "import letta" not in source
    assert "import graphiti" not in source


def test_claimed_capability_drift_fails_closed() -> None:
    identity = AdapterIdentity(
        name="mem0",
        upstream_revision="abcdef0",
        package_identity_sha256="3" * 64,
        capabilities=ADAPTER_PROFILES["graphiti"],
    )
    with pytest.raises(ValueError, match="capability profile differs"):
        validate_adapter_identity(identity)
