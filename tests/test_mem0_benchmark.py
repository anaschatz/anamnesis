from __future__ import annotations

import asyncio
import functools
from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from anamnesis.mem0_benchmark import (
    Mem0AdapterConfig,
    Mem0BenchmarkAdapter,
    Mem0ProtocolError,
    Mem0ScopeError,
    mem0_package_identity_sha256,
)
from anamnesis.memory_benchmark import (
    BenchmarkMemoryInput,
    BenchmarkQuery,
    BenchmarkScope,
    MemoryBenchmarkAdapter,
)

REVISION = "12c47f524935692e27ad48d829f35fa1e4417181"
SOURCE_SHA = "1" * 64


def async_test(function):
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        return asyncio.run(function(*args, **kwargs))

    return wrapper


class FakeMem0:
    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}
        self.calls: list[tuple[str, object]] = []
        self.next_id = 1
        self.search_override: object | None = None
        self.get_override: dict[str, object] = {}

    def add(self, messages, *, user_id, metadata, infer):
        self.calls.append(("add", {"infer": infer, "user_id": user_id}))
        memory_id = f"memory-{self.next_id}"
        self.next_id += 1
        self.rows[memory_id] = {
            "id": memory_id,
            "memory": messages,
            "user_id": user_id,
            "metadata": dict(metadata),
        }
        return {"results": [{"id": memory_id, "memory": messages, "event": "ADD"}]}

    def search(self, query, *, top_k, filters, threshold, rerank, explain):
        self.calls.append(
            (
                "search",
                {
                    "query": query,
                    "top_k": top_k,
                    "filters": dict(filters),
                    "threshold": threshold,
                    "rerank": rerank,
                    "explain": explain,
                },
            )
        )
        if self.search_override is not None:
            return self.search_override
        results = []
        for row in self.rows.values():
            metadata = row["metadata"]
            if row["user_id"] != filters["user_id"]:
                continue
            if any(
                metadata.get(key) != value
                for key, value in filters.items()
                if key != "user_id"
            ):
                continue
            results.append({"id": row["id"], "memory": row["memory"], "score": 0.91})
        return {"results": results[:top_k]}

    def get(self, memory_id):
        self.calls.append(("get", memory_id))
        if memory_id in self.get_override:
            return self.get_override[memory_id]
        row = self.rows.get(memory_id)
        return dict(row) if row is not None else None

    def update(self, memory_id, *, text, metadata):
        self.calls.append(("update", memory_id))
        row = self.rows[memory_id]
        row["memory"] = text
        row["metadata"] = dict(metadata)
        return {"message": "Memory updated successfully!"}

    def delete(self, memory_id):
        self.calls.append(("delete", memory_id))
        self.rows.pop(memory_id)
        return {"message": "Memory deleted successfully!"}


def _scope(**overrides: str | None) -> BenchmarkScope:
    values: dict[str, str | None] = {
        "user_id": "user-1",
        "session_id": "session-1",
        "project_id": "project-1",
    }
    values.update(overrides)
    return BenchmarkScope.model_validate(values)


def _item(
    *,
    event_id: str = "event-1",
    text: str = "Submit the permit by Friday.",
) -> BenchmarkMemoryInput:
    return BenchmarkMemoryInput(
        source_event_id=event_id,
        kind="prospective_obligation",
        text=text,
        observed_at=datetime(2041, 4, 1, 9, tzinfo=UTC),
    )


def _query(scope: BenchmarkScope, *, top_k: int = 3) -> BenchmarkQuery:
    return BenchmarkQuery(
        scope=scope,
        text="What permit is due?",
        at=datetime(2041, 4, 1, 10, tzinfo=UTC),
        top_k=top_k,
    )


def _adapter(client: FakeMem0) -> Mem0BenchmarkAdapter:
    return Mem0BenchmarkAdapter(
        client=client,
        upstream_revision=REVISION,
        package_identity_sha256=mem0_package_identity_sha256(
            upstream_revision=REVISION,
            package_version="2.0.17",
            source_tree_sha256=SOURCE_SHA,
        ),
        config=Mem0AdapterConfig(infer=False),
    )


@async_test
async def test_real_shape_lifecycle_is_scoped_opaque_and_cleanup_verified() -> None:
    client = FakeMem0()
    adapter = _adapter(client)
    scope = _scope()
    await adapter.begin(scope)
    ingested = await adapter.ingest_with_handle(_item())
    assert ingested.events == ("ADD",)
    assert len(ingested.handles) == 1
    assert "memory-1" not in ingested.handles[0]

    hits = await adapter.search(_query(scope))
    assert len(hits) == 1
    assert hits[0].handle == ingested.handles[0]
    assert hits[0].text == "Submit the permit by Friday."
    assert hits[0].source_event_ids == ("event-1",)
    assert hits[0].action_evidence_ids == ()
    search_call = next(value for name, value in client.calls if name == "search")
    assert search_call["filters"] == {
        "user_id": "user-1",
        "_anamnesis_session_id": "session-1",
        "_anamnesis_project_id": "project-1",
    }

    await adapter.close(scope)
    assert client.rows == {}
    assert [name for name, _ in client.calls[-3:]] == ["get", "delete", "get"]


@async_test
async def test_update_preserves_provider_identity_and_rebinds_provenance() -> None:
    client = FakeMem0()
    adapter = _adapter(client)
    scope = _scope()
    await adapter.begin(scope)
    result = await adapter.ingest_with_handle(_item())
    handle = result.handles[0]
    await adapter.update(
        handle,
        _item(event_id="event-2", text="Submit the revised permit by Monday."),
    )
    hits = await adapter.search(_query(scope))
    assert hits[0].handle == handle
    assert hits[0].text == "Submit the revised permit by Monday."
    assert hits[0].source_event_ids == ("event-2",)
    await adapter.close(scope)


@async_test
async def test_raw_cell_forwards_false_and_rejects_inference_configuration() -> None:
    client = FakeMem0()
    adapter = _adapter(client)
    await adapter.begin(_scope())
    await adapter.ingest(_item())
    add_call = next(value for name, value in client.calls if name == "add")
    assert add_call["infer"] is False
    with pytest.raises(ValidationError):
        Mem0AdapterConfig(infer=True)


@async_test
async def test_query_and_close_reject_cross_scope_calls_before_provider_use() -> None:
    client = FakeMem0()
    adapter = _adapter(client)
    scope = _scope()
    other = _scope(session_id="session-2")
    await adapter.begin(scope)
    before = list(client.calls)
    with pytest.raises(Mem0ScopeError, match="query scope differs"):
        await adapter.search(_query(other))
    with pytest.raises(Mem0ScopeError, match="close scope differs"):
        await adapter.close(other)
    assert client.calls == before


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("user_id", "user-2", "user partition"),
        ("_anamnesis_session_id", "session-2", "session partition"),
        ("_anamnesis_project_id", "project-2", "project partition"),
    ],
)
@async_test
async def test_hydrated_cross_scope_records_fail_closed(field, value, message) -> None:
    client = FakeMem0()
    adapter = _adapter(client)
    scope = _scope()
    await adapter.begin(scope)
    result = await adapter.ingest_with_handle(_item())
    row = dict(client.rows["memory-1"])
    if field == "user_id":
        row[field] = value
    else:
        row["metadata"] = dict(row["metadata"], **{field: value})
    client.get_override["memory-1"] = row
    with pytest.raises(Mem0ScopeError, match=message):
        await adapter.search(_query(scope))
    assert result.handles


@async_test
async def test_search_rejects_unverified_record_even_when_scope_metadata_is_valid() -> (
    None
):
    client = FakeMem0()
    adapter = _adapter(client)
    scope = _scope()
    await adapter.begin(scope)
    await adapter.ingest(_item())
    client.rows["foreign"] = {
        **client.rows["memory-1"],
        "id": "foreign",
        "memory": "Injected text",
    }
    client.search_override = {
        "results": [{"id": "foreign", "memory": "Injected text", "score": 1.0}]
    }
    with pytest.raises(Mem0ProtocolError, match="unverified provider record"):
        await adapter.search(_query(scope))


@async_test
async def test_search_rejects_text_not_bound_to_hydrated_record() -> None:
    client = FakeMem0()
    adapter = _adapter(client)
    scope = _scope()
    await adapter.begin(scope)
    await adapter.ingest(_item())
    client.search_override = {
        "results": [{"id": "memory-1", "memory": "Injected text", "score": 1.0}]
    }
    with pytest.raises(Mem0ProtocolError, match="text differs"):
        await adapter.search(_query(scope))


@async_test
async def test_search_rejects_over_limit_before_hydration() -> None:
    client = FakeMem0()
    adapter = _adapter(client)
    scope = _scope()
    await adapter.begin(scope)
    client.search_override = {
        "results": [
            {"id": "a", "memory": "A"},
            {"id": "b", "memory": "B"},
        ]
    }
    with pytest.raises(Mem0ProtocolError, match="more than the limit"):
        await adapter.search(_query(scope, top_k=1))
    assert not any(name == "get" for name, _ in client.calls)


@async_test
async def test_close_never_deletes_an_unverified_search_identifier() -> None:
    client = FakeMem0()
    adapter = _adapter(client)
    scope = _scope()
    await adapter.begin(scope)
    await adapter.ingest(_item())
    client.rows["foreign"] = {
        **client.rows["memory-1"],
        "id": "foreign",
    }
    client.search_override = {
        "results": [
            {
                "id": "foreign",
                "memory": client.rows["foreign"]["memory"],
                "score": 1.0,
            }
        ]
    }
    with pytest.raises(Mem0ProtocolError):
        await adapter.search(_query(scope))
    await adapter.close(scope)
    assert "foreign" in client.rows
    assert ("delete", "foreign") not in client.calls


@async_test
async def test_adapter_is_common_protocol_and_has_no_mem0_import() -> None:
    adapter = _adapter(FakeMem0())
    assert isinstance(adapter, MemoryBenchmarkAdapter)
    source = __import__("pathlib").Path("src/anamnesis/mem0_benchmark.py").read_text()
    assert "import mem0" not in source


def test_package_identity_binds_revision_version_and_source_tree() -> None:
    first = mem0_package_identity_sha256(
        upstream_revision=REVISION,
        package_version="2.0.17",
        source_tree_sha256=SOURCE_SHA,
    )
    second = mem0_package_identity_sha256(
        upstream_revision=REVISION,
        package_version="2.0.17",
        source_tree_sha256="2" * 64,
    )
    assert len(first) == 64
    assert first != second
