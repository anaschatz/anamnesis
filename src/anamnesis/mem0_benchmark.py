"""Isolated Mem0 translation adapter for the external-memory benchmark SPI.

The adapter deliberately receives an already-constructed client.  It never
imports Mem0 and never lets provider identifiers or metadata become action
evidence.  Every returned record is hydrated through ``get`` and checked
against the active user/session/project partition before it is normalized.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any, Literal, Protocol

from pydantic import ConfigDict, Field

from anamnesis.memory_benchmark import (
    ADAPTER_PROFILES,
    AdapterIdentity,
    BenchmarkHit,
    BenchmarkMemoryInput,
    BenchmarkQuery,
    BenchmarkScope,
    MemoryKind,
    validate_adapter_identity,
)
from anamnesis.schema import StrictModel

_SCOPE_SESSION = "_anamnesis_session_id"
_SCOPE_PROJECT = "_anamnesis_project_id"
_SOURCE_EVENT = "_anamnesis_source_event_id"
_MEMORY_KIND = "_anamnesis_kind"
_OBSERVED_AT = "_anamnesis_observed_at"
_HANDLE_PREFIX = "mem0-"
_MAX_RESULTS = 100


class Mem0AdapterError(RuntimeError):
    """Base error for the optional Mem0 boundary."""


class Mem0ProviderError(Mem0AdapterError):
    """The injected client failed before its response could be validated."""


class Mem0ProtocolError(Mem0AdapterError):
    """The Mem0 response violated the frozen adapter contract."""


class Mem0ScopeError(Mem0AdapterError):
    """A call attempted to cross the active benchmark partition."""


class _Frozen(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Mem0AdapterConfig(_Frozen):
    """Frozen behavior for one diagnostic cell."""

    infer: Literal[False] = False
    search_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    rerank: Literal[False] = False
    explain: Literal[False] = False


class Mem0IngestResult(_Frozen):
    handles: tuple[str, ...] = ()
    events: tuple[Literal["ADD", "UPDATE", "DELETE", "NONE"], ...] = ()


class Mem0ClientProtocol(Protocol):
    """The sync surface of the pinned Mem0 OSS ``Memory`` class."""

    def add(
        self,
        messages: str,
        *,
        user_id: str,
        metadata: dict[str, object],
        infer: bool,
    ) -> object: ...

    def search(
        self,
        query: str,
        *,
        top_k: int,
        filters: dict[str, object],
        threshold: float,
        rerank: bool,
        explain: bool,
    ) -> object: ...

    def get(self, memory_id: str) -> object: ...

    def update(
        self,
        memory_id: str,
        *,
        text: str,
        metadata: dict[str, object],
    ) -> object: ...

    def delete(self, memory_id: str) -> object: ...


class Mem0BenchmarkAdapter:
    """Fail-closed Mem0 adapter with opaque handles and exact scope checks."""

    def __init__(
        self,
        *,
        client: Mem0ClientProtocol,
        upstream_revision: str,
        package_identity_sha256: str,
        config: Mem0AdapterConfig,
    ) -> None:
        self._client = client
        self._config = config
        self._scope: BenchmarkScope | None = None
        self._handles: dict[str, str] = {}
        self._identity = AdapterIdentity(
            name="mem0",
            upstream_revision=upstream_revision,
            package_identity_sha256=package_identity_sha256,
            capabilities=ADAPTER_PROFILES["mem0"],
        )
        validate_adapter_identity(self._identity)

    @property
    def identity(self) -> AdapterIdentity:
        return self._identity

    @property
    def config(self) -> Mem0AdapterConfig:
        return self._config

    async def begin(self, scope: BenchmarkScope) -> None:
        if self._scope is not None:
            raise Mem0ScopeError("Mem0 adapter already has an active scope")
        self._scope = scope
        self._handles.clear()

    async def ingest(self, item: BenchmarkMemoryInput) -> None:
        await self.ingest_with_handle(item)

    async def ingest_with_handle(self, item: BenchmarkMemoryInput) -> Mem0IngestResult:
        scope = self._require_scope()
        metadata = _metadata_for(scope, item)
        raw = await self._provider_call(
            "add",
            self._client.add,
            item.text,
            user_id=scope.user_id,
            metadata=metadata,
            infer=self._config.infer,
        )
        results = _result_list(raw, operation="add", limit=_MAX_RESULTS)
        if not results:
            return Mem0IngestResult()
        if len(results) != 1:
            raise Mem0ProtocolError(
                "raw Mem0 storage must resolve to exactly one record"
            )

        handles: list[str] = []
        events: list[Literal["ADD", "UPDATE", "DELETE", "NONE"]] = []
        for result in results:
            provider_id = _provider_id(result, operation="add")
            event = _event(result)
            record = await self._provider_call("get", self._client.get, provider_id)
            _validate_record(
                record,
                provider_id=provider_id,
                scope=scope,
                expected_source_event_id=item.source_event_id,
            )
            handle = self._handle_for_provider_id(provider_id, create=True)
            handles.append(handle)
            events.append(event)
        return Mem0IngestResult(handles=tuple(handles), events=tuple(events))

    async def update(
        self,
        handle: str,
        item: BenchmarkMemoryInput,
    ) -> None:
        scope = self._require_scope()
        provider_id = self._provider_id_for_handle(handle)
        before = await self._provider_call("get", self._client.get, provider_id)
        _validate_record(before, provider_id=provider_id, scope=scope)
        metadata = _metadata_for(scope, item)
        await self._provider_call(
            "update",
            self._client.update,
            provider_id,
            text=item.text,
            metadata=metadata,
        )
        after = await self._provider_call("get", self._client.get, provider_id)
        _validate_record(
            after,
            provider_id=provider_id,
            scope=scope,
            expected_source_event_id=item.source_event_id,
            expected_text=item.text,
        )

    async def search(self, query: BenchmarkQuery) -> tuple[BenchmarkHit, ...]:
        scope = self._require_scope()
        if query.scope != scope:
            raise Mem0ScopeError("Mem0 query scope differs from active scope")
        filters = _scope_filters(scope)
        raw = await self._provider_call(
            "search",
            self._client.search,
            query.text,
            top_k=query.top_k,
            filters=filters,
            threshold=self._config.search_threshold,
            rerank=self._config.rerank,
            explain=self._config.explain,
        )
        results = _result_list(raw, operation="search", limit=query.top_k)
        hits: list[BenchmarkHit] = []
        for result in results:
            provider_id = _provider_id(result, operation="search")
            record = await self._provider_call("get", self._client.get, provider_id)
            normalized = _validate_record(
                record,
                provider_id=provider_id,
                scope=scope,
            )
            returned_text = _memory_text(result, operation="search")
            stored_text = _memory_text(normalized, operation="get")
            if returned_text != stored_text:
                raise Mem0ProtocolError("Mem0 search text differs from hydrated record")
            metadata = _metadata(normalized)
            handle = self._handle_for_provider_id(provider_id)
            hits.append(
                BenchmarkHit(
                    adapter="mem0",
                    handle=handle,
                    text=stored_text,
                    score=_score(result),
                    kind=_kind(metadata),
                    observed_at=_datetime(metadata.get(_OBSERVED_AT)),
                    source_event_ids=(_source_event_id(metadata),),
                    action_evidence_ids=(),
                )
            )
        return tuple(hits)

    async def close(self, scope: BenchmarkScope) -> None:
        active = self._require_scope()
        if scope != active:
            raise Mem0ScopeError("Mem0 close scope differs from active scope")
        provider_ids = tuple(dict.fromkeys(self._handles.values()))
        for provider_id in provider_ids:
            record = await self._provider_call("get", self._client.get, provider_id)
            _validate_record(record, provider_id=provider_id, scope=scope)
            await self._provider_call("delete", self._client.delete, provider_id)
            absent = await self._provider_call("get", self._client.get, provider_id)
            if absent is not None:
                raise Mem0ProtocolError("Mem0 record remained after delete")
        self._handles.clear()
        self._scope = None

    def _require_scope(self) -> BenchmarkScope:
        if self._scope is None:
            raise Mem0ScopeError("Mem0 adapter has no active scope")
        return self._scope

    def _provider_id_for_handle(self, handle: str) -> str:
        try:
            return self._handles[handle]
        except KeyError as error:
            raise Mem0ScopeError("unknown or expired Mem0 handle") from error

    def _handle_for_provider_id(self, provider_id: str, *, create: bool = False) -> str:
        for handle, known_id in self._handles.items():
            if known_id == provider_id:
                return handle
        if create:
            handle = _new_handle()
            self._handles[handle] = provider_id
            return handle
        raise Mem0ProtocolError("Mem0 search returned an unverified provider record")

    async def _provider_call(
        self, operation: str, call: Any, *args: Any, **kwargs: Any
    ) -> object:
        try:
            return await asyncio.to_thread(call, *args, **kwargs)
        except Mem0AdapterError:
            raise
        except Exception as error:
            raise Mem0ProviderError(f"Mem0 {operation} failed") from error


def mem0_package_identity_sha256(
    *, upstream_revision: str, package_version: str, source_tree_sha256: str
) -> str:
    """Canonical identity used by the common adapter contract."""

    value = {
        "package": "mem0ai",
        "package_version": package_version,
        "source_tree_sha256": source_tree_sha256,
        "upstream_revision": upstream_revision,
    }
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _metadata_for(
    scope: BenchmarkScope, item: BenchmarkMemoryInput
) -> dict[str, object]:
    metadata: dict[str, object] = {
        _SCOPE_SESSION: scope.session_id,
        _SOURCE_EVENT: item.source_event_id,
        _MEMORY_KIND: item.kind,
        _OBSERVED_AT: item.observed_at.isoformat(),
    }
    if scope.project_id is not None:
        metadata[_SCOPE_PROJECT] = scope.project_id
    return metadata


def _scope_filters(scope: BenchmarkScope) -> dict[str, object]:
    filters: dict[str, object] = {
        "user_id": scope.user_id,
        _SCOPE_SESSION: scope.session_id,
    }
    if scope.project_id is not None:
        filters[_SCOPE_PROJECT] = scope.project_id
    return filters


def _result_list(
    raw: object, *, operation: str, limit: int
) -> list[Mapping[str, object]]:
    envelope = _mapping(raw, operation=operation)
    values = envelope.get("results")
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        raise Mem0ProtocolError(f"Mem0 {operation} results must be a list")
    if len(values) > limit:
        raise Mem0ProtocolError(f"Mem0 {operation} returned more than the limit")
    return [_mapping(value, operation=operation) for value in values]


def _mapping(value: object, *, operation: str) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, Mapping):
            return dumped
    raise Mem0ProtocolError(f"Mem0 {operation} response must be an object")


def _provider_id(record: Mapping[str, object], *, operation: str) -> str:
    value = record.get("id")
    if not isinstance(value, str) or not value.strip() or len(value) > 512:
        raise Mem0ProtocolError(f"Mem0 {operation} record has no valid id")
    return value


def _event(record: Mapping[str, object]) -> Literal["ADD", "UPDATE", "DELETE", "NONE"]:
    value = record.get("event", "NONE")
    if value not in {"ADD", "UPDATE", "DELETE", "NONE"}:
        raise Mem0ProtocolError("Mem0 add returned an unknown event")
    return value


def _metadata(record: Mapping[str, object]) -> Mapping[str, object]:
    value = record.get("metadata", {})
    if not isinstance(value, Mapping):
        raise Mem0ProtocolError("Mem0 record metadata must be an object")
    return value


def _validate_record(
    raw: object,
    *,
    provider_id: str,
    scope: BenchmarkScope,
    expected_source_event_id: str | None = None,
    expected_text: str | None = None,
) -> Mapping[str, object]:
    record = _mapping(raw, operation="get")
    if _provider_id(record, operation="get") != provider_id:
        raise Mem0ProtocolError("Mem0 get returned a different record")
    if record.get("user_id") != scope.user_id:
        raise Mem0ScopeError("Mem0 record crossed the user partition")
    metadata = _metadata(record)
    if metadata.get(_SCOPE_SESSION) != scope.session_id:
        raise Mem0ScopeError("Mem0 record crossed the session partition")
    if metadata.get(_SCOPE_PROJECT) != scope.project_id:
        raise Mem0ScopeError("Mem0 record crossed the project partition")
    if (
        expected_source_event_id is not None
        and metadata.get(_SOURCE_EVENT) != expected_source_event_id
    ):
        raise Mem0ProtocolError("Mem0 source-event metadata drifted")
    if (
        expected_text is not None
        and _memory_text(record, operation="get") != expected_text
    ):
        raise Mem0ProtocolError("Mem0 updated text drifted")
    _kind(metadata)
    _source_event_id(metadata)
    _datetime(metadata.get(_OBSERVED_AT))
    return record


def _memory_text(record: Mapping[str, object], *, operation: str) -> str:
    value = record.get("memory")
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value.encode("utf-8")) > 8192
    ):
        raise Mem0ProtocolError(f"Mem0 {operation} memory text is invalid")
    return value


def _kind(metadata: Mapping[str, object]) -> MemoryKind:
    value = metadata.get(_MEMORY_KIND)
    if value not in {"profile", "decision", "project", "prospective_obligation"}:
        raise Mem0ProtocolError("Mem0 memory kind metadata is invalid")
    return value


def _source_event_id(metadata: Mapping[str, object]) -> str:
    value = metadata.get(_SOURCE_EVENT)
    if not isinstance(value, str) or not value:
        raise Mem0ProtocolError("Mem0 source-event metadata is invalid")
    return value


def _datetime(value: object) -> datetime:
    if not isinstance(value, str):
        raise Mem0ProtocolError("Mem0 observed-at metadata is invalid")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise Mem0ProtocolError("Mem0 observed-at metadata is invalid") from error
    if parsed.tzinfo is None:
        raise Mem0ProtocolError("Mem0 observed-at metadata must be timezone-aware")
    return parsed


def _score(record: Mapping[str, object]) -> float | None:
    value = record.get("score")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Mem0ProtocolError("Mem0 search score is invalid")
    score = float(value)
    if not math.isfinite(score):
        raise Mem0ProtocolError("Mem0 search score is not finite")
    return score


def _new_handle() -> str:
    return f"{_HANDLE_PREFIX}{uuid.uuid4().hex}"


__all__ = [
    "Mem0AdapterConfig",
    "Mem0AdapterError",
    "Mem0BenchmarkAdapter",
    "Mem0ClientProtocol",
    "Mem0IngestResult",
    "Mem0ProtocolError",
    "Mem0ProviderError",
    "Mem0ScopeError",
    "mem0_package_identity_sha256",
]
