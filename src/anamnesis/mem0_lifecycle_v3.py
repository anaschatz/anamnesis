"""One-attempt real Mem0 recall versus deterministic lifecycle filtering."""

from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from anamnesis.lifecycle_filter import (
    DeterministicLifecycleFilter,
    LifecycleDirective,
)
from anamnesis.mem0_inference_diagnostic import (
    LlmCallAudit,
    _attest_server,
    _close_memory,
    _construct_memory,
    _loopback_only,
    _records,
    _require_local_environment,
    _scope_filters,
    _serialize_llm_calls,
    _sha256_file,
    _verify_ollama_artifact,
)
from anamnesis.memory_benchmark import BenchmarkHit, BenchmarkScope

PROTOCOL_SCHEMA_VERSION = "mem0_lifecycle_protocol.v3"
RESULT_SCHEMA_VERSION = "mem0_lifecycle_result.v3"
PROTOCOL_SHA256 = "4e9d63ebd6d66b2c76175f94d55262ab74b0045d025297e2f813e07073aaef9a"

_SESSION_KEY = "_anamnesis_session_id"
_PROJECT_KEY = "_anamnesis_project_id"
_SOURCE_KEY = "_anamnesis_source_event_id"
_KIND_KEY = "_anamnesis_kind"
_OBSERVED_AT_KEY = "_anamnesis_observed_at"


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class LifecycleQueryResult(_Frozen):
    query_id: str
    scope: Literal["a", "b"]
    raw_source_event_ids: tuple[str, ...]
    filtered_source_event_ids: tuple[str, ...]
    raw_memories: tuple[str, ...]
    filtered_memories: tuple[str, ...]
    required_raw_stale_source_event_ids: tuple[str, ...]
    expected_active_source_event_ids: tuple[str, ...]
    raw_stale_present: bool
    filtered_exact: bool
    filtered_stale_hits: int = Field(ge=0)
    search_latency_ms: float = Field(ge=0.0)
    filter_latency_ms: float = Field(ge=0.0)


class Mem0LifecycleResult(_Frozen):
    schema_version: Literal["mem0_lifecycle_result.v3"]
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    passed: bool
    integrity_passed: bool
    semantic_passed: bool
    hypothesis_test_eligible: Literal[False]
    upstream_revision: str
    model_name: str
    model_manifest_sha256: str
    model_blob_sha256: str
    ollama_version: str
    localhost_model_calls: int = Field(ge=0)
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    provider_api_cost_usd: Literal[0.0]
    external_network_calls: Literal[0]
    all_calls_finished: bool
    usage_complete: bool
    scope_isolation_passed: bool
    cleanup_passed: bool
    extracted_records: int = Field(ge=0)
    raw_stale_recall_opportunities: int = Field(ge=0)
    filtered_query_exact: int = Field(ge=0)
    filtered_stale_hits: int = Field(ge=0)
    query_results: tuple[LifecycleQueryResult, ...]
    llm_calls: tuple[LlmCallAudit, ...]
    integrity_error: str | None = None


def load_protocol(path: Path) -> dict[str, Any]:
    if _sha256_file(path) != PROTOCOL_SHA256:
        raise RuntimeError("Mem0 lifecycle protocol bytes drifted")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != PROTOCOL_SCHEMA_VERSION:
        raise RuntimeError("unexpected Mem0 lifecycle protocol schema")
    if value.get("preregistered_before_implementation_and_model_calls") is not True:
        raise RuntimeError("Mem0 lifecycle protocol lacks preregistration")
    return value


def _scope_metadata(scope: BenchmarkScope, event: Mapping[str, Any]) -> dict[str, str]:
    metadata = {
        _SESSION_KEY: scope.session_id,
        _SOURCE_KEY: str(event["id"]),
        _KIND_KEY: str(event["kind"]),
        _OBSERVED_AT_KEY: str(event["observed_at"]),
    }
    if scope.project_id is not None:
        metadata[_PROJECT_KEY] = scope.project_id
    return metadata


def _mapping(value: object, *, operation: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"Mem0 lifecycle {operation} response is invalid")
    return value


def _search_hits(
    memory: object,
    *,
    query: Mapping[str, Any],
    scope: BenchmarkScope,
    event_scope: Mapping[str, str],
) -> tuple[BenchmarkHit, ...]:
    raw = memory.search(
        str(query["text"]),
        top_k=int(query["top_k"]),
        filters=_scope_filters(scope),
        threshold=0.0,
        rerank=False,
        explain=False,
    )
    envelope = _mapping(raw, operation="search")
    values = envelope.get("results")
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise RuntimeError("Mem0 lifecycle search results are invalid")
    if len(values) > int(query["top_k"]):
        raise RuntimeError("Mem0 lifecycle search exceeded top_k")
    hits: list[BenchmarkHit] = []
    for rank, raw_item in enumerate(values):
        item = _mapping(raw_item, operation="search item")
        provider_id = item.get("id")
        if not isinstance(provider_id, str) or not provider_id:
            raise RuntimeError("Mem0 lifecycle search item has no ID")
        record = _mapping(memory.get(provider_id), operation="get")
        if record.get("id") != provider_id or record.get("user_id") != scope.user_id:
            raise RuntimeError("Mem0 lifecycle hydrated record crossed scope")
        metadata = _mapping(record.get("metadata"), operation="metadata")
        if metadata.get(_SESSION_KEY) != scope.session_id:
            raise RuntimeError("Mem0 lifecycle record crossed session scope")
        if metadata.get(_PROJECT_KEY) != scope.project_id:
            raise RuntimeError("Mem0 lifecycle record crossed project scope")
        source = metadata.get(_SOURCE_KEY)
        if not isinstance(source, str) or event_scope.get(source) != query["scope"]:
            raise RuntimeError("Mem0 lifecycle source event crossed scope")
        text = record.get("memory")
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("Mem0 lifecycle record text is invalid")
        returned_text = item.get("memory")
        if returned_text != text:
            raise RuntimeError("Mem0 lifecycle search text differs from stored text")
        score = item.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise RuntimeError("Mem0 lifecycle search score is invalid")
        kind = metadata.get(_KIND_KEY)
        if kind not in {"profile", "decision", "project", "prospective_obligation"}:
            raise RuntimeError("Mem0 lifecycle memory kind is invalid")
        observed_at = metadata.get(_OBSERVED_AT_KEY)
        if not isinstance(observed_at, str):
            raise RuntimeError("Mem0 lifecycle observed_at is invalid")
        hits.append(
            BenchmarkHit(
                adapter="mem0",
                handle=f"mem0-lifecycle-{query['id']}-{rank}",
                text=text,
                score=float(score),
                kind=kind,
                observed_at=datetime.fromisoformat(observed_at),
                source_event_ids=(source,),
                action_evidence_ids=(),
            )
        )
    return tuple(hits)


async def run_mem0_lifecycle_v3(
    *,
    protocol_path: Path,
    embedding_snapshot: Path,
    models_root: Path,
    source_commit: str,
) -> Mem0LifecycleResult:
    protocol = load_protocol(protocol_path)
    if len(source_commit) != 40:
        raise RuntimeError("Mem0 lifecycle source commit must be full length")
    _require_local_environment(context_length=protocol["model"]["context_length"])
    from anamnesis.baselines import _directory_sha256

    if (
        _directory_sha256(embedding_snapshot)
        != protocol["storage"]["embedding_artifact_sha256"]
    ):
        raise RuntimeError("Mem0 lifecycle embedding artifact drifted")
    _verify_ollama_artifact(protocol, models_root)

    scopes = {
        name: BenchmarkScope.model_validate(value)
        for name, value in protocol["scopes"].items()
    }
    lifecycle = {name: DeterministicLifecycleFilter() for name in scopes}
    event_scope = {event["id"]: event["scope"] for event in protocol["events"]}
    query_results: list[LifecycleQueryResult] = []
    calls: tuple[object, ...] = ()
    cleanup_passed = False
    scope_isolation = False
    integrity_error: str | None = None
    ollama_version = ""
    extracted_records = 0

    with tempfile.TemporaryDirectory(prefix="anamnesis-mem0-life-v3-") as directory:
        memory: object | None = None
        with _loopback_only():
            try:
                ollama_version = _attest_server(protocol, require_resident=False)
                memory = _construct_memory(
                    protocol,
                    embedding_snapshot=embedding_snapshot,
                    runtime_root=Path(directory),
                    collection_name="anamnesis_mem0_lifecycle_v3",
                )
                for event in protocol["events"]:
                    scope = scopes[event["scope"]]
                    before = len(memory.llm.calls)
                    raw = memory.add(
                        event["text"],
                        user_id=scope.user_id,
                        metadata=_scope_metadata(scope, event),
                        infer=True,
                    )
                    envelope = _mapping(raw, operation="add")
                    if not isinstance(envelope.get("results"), list):
                        raise RuntimeError("Mem0 lifecycle add results are invalid")
                    if len(memory.llm.calls) != before + 1:
                        raise RuntimeError("Mem0 lifecycle event call count drifted")
                    directive = event["directive"]
                    lifecycle[event["scope"]].apply(
                        LifecycleDirective(
                            source_event_id=event["id"],
                            key=directive["key"],
                            operation=directive["operation"],
                            supersedes_event_ids=tuple(
                                directive["supersedes_event_ids"]
                            ),
                        )
                    )

                for query in protocol["queries"]:
                    scope = scopes[query["scope"]]
                    search_started = perf_counter()
                    raw_hits = _search_hits(
                        memory,
                        query=query,
                        scope=scope,
                        event_scope=event_scope,
                    )
                    search_latency_ms = (perf_counter() - search_started) * 1000
                    filter_started = perf_counter()
                    filtered = lifecycle[query["scope"]].filter_active_hits(raw_hits)
                    filter_latency_ms = (perf_counter() - filter_started) * 1000
                    raw_sources = tuple(hit.source_event_ids[0] for hit in raw_hits)
                    filtered_sources = tuple(
                        hit.source_event_ids[0] for hit in filtered
                    )
                    stale_required = tuple(query["required_raw_stale_source_event_ids"])
                    expected = tuple(query["expected_active_source_event_ids"])
                    stale_present = all(item in raw_sources for item in stale_required)
                    stale_set = set(stale_required)
                    query_results.append(
                        LifecycleQueryResult(
                            query_id=query["id"],
                            scope=query["scope"],
                            raw_source_event_ids=raw_sources,
                            filtered_source_event_ids=filtered_sources,
                            raw_memories=tuple(hit.text for hit in raw_hits),
                            filtered_memories=tuple(hit.text for hit in filtered),
                            required_raw_stale_source_event_ids=stale_required,
                            expected_active_source_event_ids=expected,
                            raw_stale_present=stale_present,
                            filtered_exact=filtered_sources == expected,
                            filtered_stale_hits=sum(
                                source in stale_set for source in filtered_sources
                            ),
                            search_latency_ms=search_latency_ms,
                            filter_latency_ms=filter_latency_ms,
                        )
                    )

                _attest_server(protocol, require_resident=True)
                calls = tuple(memory.llm.calls)
                all_records = {
                    name: _records(memory, scope) for name, scope in scopes.items()
                }
                extracted_records = sum(len(items) for items in all_records.values())
                scope_isolation = all(
                    all(
                        event_scope[item["metadata"][_SOURCE_KEY]] == name
                        for item in items
                    )
                    for name, items in all_records.items()
                )
                for name, records in all_records.items():
                    for record in records:
                        memory.delete(record["id"])
                    if _records(memory, scopes[name]):
                        raise RuntimeError("Mem0 lifecycle cleanup left records")
                cleanup_passed = True
            except Exception as error:
                integrity_error = f"{type(error).__name__}: {error}"
            finally:
                if memory is not None:
                    calls = tuple(getattr(memory.llm, "calls", ()))
                    if not cleanup_passed:
                        try:
                            for scope in scopes.values():
                                for record in _records(memory, scope):
                                    memory.delete(record["id"])
                                if _records(memory, scope):
                                    raise RuntimeError(
                                        "Mem0 lifecycle cleanup left records"
                                    )
                            cleanup_passed = True
                        except Exception as cleanup_error:
                            detail = f"{type(cleanup_error).__name__}: {cleanup_error}"
                            integrity_error = (
                                detail
                                if integrity_error is None
                                else f"{integrity_error}; cleanup failed: {detail}"
                            )
                    _close_memory(memory)

    all_calls_finished = bool(calls) and all(
        call.done_reason == "stop" for call in calls
    )
    usage_complete = bool(calls) and all(
        call.prompt_tokens > 0 and call.completion_tokens > 0 for call in calls
    )
    expected_calls = protocol["model"]["expected_model_calls"]
    integrity_passed = (
        integrity_error is None
        and len(calls) == expected_calls
        and len(query_results) == len(protocol["queries"])
        and all_calls_finished
        and usage_complete
        and scope_isolation
        and cleanup_passed
    )
    raw_stale = sum(query.raw_stale_present for query in query_results)
    exact = sum(query.filtered_exact for query in query_results)
    filtered_stale = sum(query.filtered_stale_hits for query in query_results)
    semantic_passed = (
        integrity_passed
        and raw_stale == protocol["gate"]["raw_stale_recall_opportunities"]
        and exact == protocol["gate"]["filtered_query_exact"]
        and filtered_stale == protocol["gate"]["filtered_stale_hits"]
    )
    return Mem0LifecycleResult(
        schema_version=RESULT_SCHEMA_VERSION,
        protocol_sha256=PROTOCOL_SHA256,
        source_commit=source_commit,
        passed=semantic_passed,
        integrity_passed=integrity_passed,
        semantic_passed=semantic_passed,
        hypothesis_test_eligible=False,
        upstream_revision=protocol["upstream"]["revision"],
        model_name=protocol["model"]["name"],
        model_manifest_sha256=protocol["model"]["manifest_sha256"],
        model_blob_sha256=protocol["model"]["model_blob_sha256"],
        ollama_version=ollama_version,
        localhost_model_calls=len(calls),
        prompt_tokens=sum(call.prompt_tokens for call in calls),
        completion_tokens=sum(call.completion_tokens for call in calls),
        provider_api_cost_usd=0.0,
        external_network_calls=0,
        all_calls_finished=all_calls_finished,
        usage_complete=usage_complete,
        scope_isolation_passed=scope_isolation,
        cleanup_passed=cleanup_passed,
        extracted_records=extracted_records,
        raw_stale_recall_opportunities=raw_stale,
        filtered_query_exact=exact,
        filtered_stale_hits=filtered_stale,
        query_results=tuple(query_results),
        llm_calls=_serialize_llm_calls(calls),
        integrity_error=integrity_error,
    )


def _write_result(path: Path, result: Mem0LifecycleResult) -> None:
    if path.exists():
        raise FileExistsError("refusing to overwrite Mem0 lifecycle result")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")


def mem0_lifecycle_v3_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--embedding-snapshot", type=Path, required=True)
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = asyncio.run(
        run_mem0_lifecycle_v3(
            protocol_path=args.protocol,
            embedding_snapshot=args.embedding_snapshot,
            models_root=args.models_root,
            source_commit=args.source_commit,
        )
    )
    _write_result(args.output, result)
    return 0 if result.integrity_passed else 2


if __name__ == "__main__":
    raise SystemExit(mem0_lifecycle_v3_main())


__all__ = [
    "LifecycleQueryResult",
    "Mem0LifecycleResult",
    "PROTOCOL_SHA256",
    "load_protocol",
    "mem0_lifecycle_v3_main",
    "run_mem0_lifecycle_v3",
]
