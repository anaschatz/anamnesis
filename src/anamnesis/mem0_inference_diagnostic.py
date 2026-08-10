"""One-attempt Mem0 ``infer=true`` diagnostic over a pinned local Ollama model."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import socket
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from typing import Any, Literal
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field

from anamnesis.memory_benchmark import BenchmarkScope

PROTOCOL_SCHEMA_VERSION = "mem0_inference_protocol.v1"
RESULT_SCHEMA_VERSION = "mem0_inference_result.v1"
PROTOCOL_SHA256 = "7bc9532c599397414ddf856fa3e74dbfdf6039af39b1d7dae3656454261be5d1"

_SESSION_KEY = "_anamnesis_session_id"
_PROJECT_KEY = "_anamnesis_project_id"
_SOURCE_KEY = "_anamnesis_source_event_id"
_KIND_KEY = "_anamnesis_kind"


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class LlmCallAudit(_Frozen):
    index: int = Field(ge=0)
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    response_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    response_text: str
    prompt_tokens: int = Field(gt=0)
    completion_tokens: int = Field(gt=0)
    latency_ms: float = Field(ge=0.0)
    done_reason: str


class EventDiagnostic(_Frozen):
    event_id: str
    scope: Literal["a", "b"]
    sdk_events: tuple[str, ...]
    record_count: int = Field(ge=0)
    memories: tuple[str, ...]
    model_call_indexes: tuple[int, ...]
    latency_ms: float = Field(ge=0.0)
    assertion_type: str
    assertion_passed: bool
    assertion_reason: str


class Mem0InferenceResult(_Frozen):
    schema_version: Literal["mem0_inference_result.v1"]
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
    event_results: tuple[EventDiagnostic, ...]
    llm_calls: tuple[LlmCallAudit, ...]
    integrity_error: str | None = None


class PinnedMem0OllamaConfig:
    """Factory-compatible config whose full behavior comes from the protocol."""

    def __init__(
        self,
        *,
        model: str,
        ollama_base_url: str,
        seed: int,
        temperature: float,
        top_p: float,
        top_k: int,
        max_tokens: int,
        num_ctx: int,
        timeout_seconds: float,
        **kwargs: object,
    ) -> None:
        del kwargs
        self.model = model
        self.ollama_base_url = ollama_base_url
        self.seed = seed
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.num_ctx = num_ctx
        self.timeout_seconds = timeout_seconds


class PinnedMem0OllamaLlm:
    """No-retry local transport that preserves Mem0's exact prompt messages."""

    def __init__(self, config: PinnedMem0OllamaConfig) -> None:
        from ollama import Client

        if config.ollama_base_url != "http://127.0.0.1:11434":
            raise ValueError("Mem0 inference requires exact loopback Ollama URL")
        self.config = config
        self.client = Client(
            host=config.ollama_base_url,
            timeout=config.timeout_seconds,
        )
        self.calls: list[LlmCallAudit] = []

    def generate_response(
        self,
        messages: Sequence[Mapping[str, str]],
        response_format: Mapping[str, object] | None = None,
        tools: object = None,
        **kwargs: object,
    ) -> str:
        del kwargs
        if tools is not None:
            raise ValueError("Mem0 inference diagnostic forbids tools")
        if response_format != {"type": "json_object"}:
            raise ValueError("Mem0 inference diagnostic requires JSON mode")
        normalized_messages = [dict(message) for message in messages]
        request_body = {
            "model": self.config.model,
            "messages": normalized_messages,
            "format": "json",
            "think": False,
            "options": {
                "seed": self.config.seed,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "top_k": self.config.top_k,
                "num_predict": self.config.max_tokens,
                "num_ctx": self.config.num_ctx,
            },
        }
        request_sha256 = _canonical_sha256(request_body)
        started = perf_counter()
        response = self.client.chat(
            model=self.config.model,
            messages=normalized_messages,
            format="json",
            think=False,
            options=request_body["options"],
            stream=False,
        )
        latency_ms = (perf_counter() - started) * 1000
        content = response.message.content
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("Ollama returned empty Mem0 extraction content")
        prompt_tokens = response.prompt_eval_count
        completion_tokens = response.eval_count
        if not isinstance(prompt_tokens, int) or prompt_tokens <= 0:
            raise RuntimeError("Ollama omitted positive prompt token usage")
        if not isinstance(completion_tokens, int) or completion_tokens <= 0:
            raise RuntimeError("Ollama omitted positive completion token usage")
        done_reason = response.done_reason or ""
        self.calls.append(
            LlmCallAudit(
                index=len(self.calls),
                request_sha256=request_sha256,
                response_sha256=_sha256_text(content),
                response_text=content,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
                done_reason=done_reason,
            )
        )
        return content


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _serialize_llm_calls(calls: Sequence[object]) -> tuple[dict[str, object], ...]:
    """Cross the Mem0 factory import boundary without relying on class identity.

    ``python -m`` executes this module as ``__main__``, while Mem0's factory imports
    the configured provider through its canonical package name.  Consequently the
    two otherwise identical ``LlmCallAudit`` classes have different identities.
    Revalidate their JSON-compatible data at the result boundary instead.
    """

    serialized: list[dict[str, object]] = []
    for call in calls:
        model_dump = getattr(call, "model_dump", None)
        if not callable(model_dump):
            raise TypeError("Mem0 LLM call audit does not support model_dump")
        value = model_dump(mode="json")
        if not isinstance(value, dict):
            raise TypeError("Mem0 LLM call audit did not serialize to an object")
        serialized.append(value)
    return tuple(serialized)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_protocol(
    path: Path,
    *,
    expected_sha256: str = PROTOCOL_SHA256,
    expected_schema_version: str = PROTOCOL_SCHEMA_VERSION,
) -> dict[str, Any]:
    if _sha256_file(path) != expected_sha256:
        raise RuntimeError("Mem0 inference protocol bytes drifted")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != expected_schema_version:
        raise RuntimeError("unexpected Mem0 inference protocol schema")
    if value.get("hypothesis_test_eligible") is not False:
        raise RuntimeError("Mem0 inference protocol cannot be hypothesis eligible")
    if value.get("preregistered_before_model_calls") is not True:
        raise RuntimeError("Mem0 inference protocol lacks preregistration attestation")
    return value


def _require_local_environment(*, context_length: int = 8192) -> None:
    expected = {
        "OLLAMA_NO_CLOUD": "1",
        "OLLAMA_HOST": "127.0.0.1:11434",
        "OLLAMA_CONTEXT_LENGTH": str(context_length),
        "OLLAMA_NUM_PARALLEL": "1",
        "OLLAMA_MAX_LOADED_MODELS": "1",
    }
    for name, value in expected.items():
        if os.environ.get(name) != value:
            raise RuntimeError(f"Mem0 inference requires {name}={value}")


def _verify_ollama_artifact(protocol: Mapping[str, Any], models_root: Path) -> None:
    model = protocol["model"]
    manifest = (
        models_root
        / "manifests"
        / "registry.ollama.ai"
        / "library"
        / "qwen3.5"
        / "9b-q4_K_M"
    )
    if _sha256_file(manifest) != model["manifest_sha256"]:
        raise RuntimeError("Ollama model manifest drifted")
    value = json.loads(manifest.read_text(encoding="utf-8"))
    descriptors = [value["config"], *value["layers"]]
    for descriptor in descriptors:
        digest = descriptor["digest"]
        if not isinstance(digest, str) or not digest.startswith("sha256:"):
            raise RuntimeError("Ollama manifest contains an invalid digest")
        blob_hash = digest.removeprefix("sha256:")
        blob = models_root / "blobs" / f"sha256-{blob_hash}"
        if blob.stat().st_size != descriptor["size"] or _sha256_file(blob) != blob_hash:
            raise RuntimeError("Ollama model blob drifted")
    model_layer = next(
        item for item in value["layers"] if item["mediaType"].endswith("image.model")
    )
    if model_layer["digest"].removeprefix("sha256:") != model["model_blob_sha256"]:
        raise RuntimeError("Ollama model layer differs from protocol")
    if model_layer["size"] != model["model_blob_bytes"]:
        raise RuntimeError("Ollama model layer size differs from protocol")


def _http_json(url: str, *, body: Mapping[str, object] | None = None) -> object:
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = Request(url, data=data, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=10) as response:  # noqa: S310 - exact loopback URL
        return json.load(response)


def _attest_server(protocol: Mapping[str, Any], *, require_resident: bool) -> str:
    model = protocol["model"]
    version = _http_json(f"{model['base_url']}/api/version")
    if (
        not isinstance(version, Mapping)
        or version.get("version") != model["ollama_version"]
    ):
        raise RuntimeError("Ollama server version drifted")
    if require_resident:
        process = _http_json(f"{model['base_url']}/api/ps")
        if not isinstance(process, Mapping) or not isinstance(
            process.get("models"), list
        ):
            raise RuntimeError("Ollama resident-model response is invalid")
        resident = [
            item
            for item in process["models"]
            if isinstance(item, Mapping) and item.get("name") == model["name"]
        ]
        if len(resident) != 1:
            raise RuntimeError("pinned Ollama model is not uniquely resident")
        digest = resident[0].get("digest")
        if digest != model["manifest_sha256"]:
            raise RuntimeError("resident Ollama model digest drifted")
        if resident[0].get("context_length") != model["context_length"]:
            raise RuntimeError("resident Ollama context length drifted")
    return str(version["version"])


@contextmanager
def _loopback_only() -> Iterator[None]:
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex

    def checked_connect(sock: socket.socket, address: object) -> object:
        if (
            isinstance(address, tuple)
            and len(address) >= 2
            and address[0] == "127.0.0.1"
            and address[1] == 11434
        ):
            return original_connect(sock, address)
        raise RuntimeError("external network call blocked by Mem0 inference diagnostic")

    def checked_connect_ex(sock: socket.socket, address: object) -> int:
        checked_connect(sock, address)
        return 0

    socket.socket.connect = checked_connect  # type: ignore[method-assign]
    socket.socket.connect_ex = checked_connect_ex  # type: ignore[method-assign]
    try:
        yield
    finally:
        socket.socket.connect = original_connect  # type: ignore[method-assign]
        socket.socket.connect_ex = original_connect_ex  # type: ignore[method-assign]


def _construct_memory(
    protocol: Mapping[str, Any],
    *,
    embedding_snapshot: Path,
    runtime_root: Path,
    collection_name: str = "anamnesis_mem0_inference_v1",
) -> object:
    os.environ["MEM0_TELEMETRY"] = "False"
    os.environ["MEM0_DIR"] = str(runtime_root / "mem0-home")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from mem0 import Memory
    from mem0.utils.factory import EmbedderFactory, LlmFactory

    EmbedderFactory.provider_to_class["fastembed"] = (
        "anamnesis.mem0_sdk_smoke.PinnedFastEmbedEmbedding"
    )
    LlmFactory.provider_to_class["ollama"] = (
        "anamnesis.mem0_inference_diagnostic.PinnedMem0OllamaLlm",
        PinnedMem0OllamaConfig,
    )
    model = protocol["model"]
    storage = protocol["storage"]
    memory = Memory.from_config(
        {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": collection_name,
                    "embedding_model_dims": 384,
                    "path": str(runtime_root / "qdrant"),
                    "on_disk": True,
                },
            },
            "embedder": {
                "provider": "fastembed",
                "config": {
                    "model": str(embedding_snapshot),
                    "embedding_dims": 384,
                },
            },
            "llm": {
                "provider": "ollama",
                "config": {
                    "model": model["name"],
                    "ollama_base_url": model["base_url"],
                    "seed": model["seed"],
                    "temperature": model["temperature"],
                    "top_p": model["top_p"],
                    "top_k": model["top_k"],
                    "max_tokens": model["max_output_tokens"],
                    "num_ctx": model["context_length"],
                    "timeout_seconds": model["timeout_seconds"],
                },
            },
            "history_db_path": str(runtime_root / "history.db"),
            "version": "v1.1",
            "custom_instructions": None,
        }
    )
    if storage["vector_store"] != "qdrant_embedded":
        raise RuntimeError("unexpected Mem0 inference vector store")
    memory.vector_store._has_bm25_slot = False
    memory.vector_store.keyword_search = lambda **kwargs: None
    return memory


def _scope_filters(scope: BenchmarkScope) -> dict[str, str]:
    filters = {"user_id": scope.user_id, _SESSION_KEY: scope.session_id}
    if scope.project_id is not None:
        filters[_PROJECT_KEY] = scope.project_id
    return filters


def _scope_metadata(scope: BenchmarkScope, event_id: str) -> dict[str, str]:
    metadata = {
        _SESSION_KEY: scope.session_id,
        _SOURCE_KEY: event_id,
        _KIND_KEY: "prospective_obligation",
    }
    if scope.project_id is not None:
        metadata[_PROJECT_KEY] = scope.project_id
    return metadata


def _records(memory: object, scope: BenchmarkScope) -> list[Mapping[str, object]]:
    raw = memory.get_all(filters=_scope_filters(scope), top_k=100)
    if not isinstance(raw, Mapping) or not isinstance(raw.get("results"), list):
        raise RuntimeError("Mem0 get_all response is invalid")
    records: list[Mapping[str, object]] = []
    for item in raw["results"]:
        if not isinstance(item, Mapping):
            raise RuntimeError("Mem0 get_all record is invalid")
        if item.get("user_id") != scope.user_id:
            raise RuntimeError("Mem0 record crossed user scope")
        metadata = item.get("metadata")
        if not isinstance(metadata, Mapping):
            raise RuntimeError("Mem0 record metadata is invalid")
        if metadata.get(_SESSION_KEY) != scope.session_id:
            raise RuntimeError("Mem0 record crossed session scope")
        if metadata.get(_PROJECT_KEY) != scope.project_id:
            raise RuntimeError("Mem0 record crossed project scope")
        text = item.get("memory")
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("Mem0 record text is invalid")
        records.append(item)
    return records


def _normalized(value: str) -> str:
    return " ".join(value.casefold().split())


def _contains_all(text: str, terms: Sequence[str]) -> bool:
    normalized = _normalized(text)
    return all(_normalized(term) in normalized for term in terms)


def evaluate_assertion(
    assertion: Mapping[str, Any], memories: Sequence[str]
) -> tuple[bool, str]:
    kind = assertion["type"]
    if kind == "contains_fact":
        passed = any(
            _contains_all(text, assertion["required_terms"]) for text in memories
        )
        return passed, "required fact present" if passed else "required fact missing"
    if kind == "deduplicated_fact":
        count = sum(
            _contains_all(text, assertion["required_terms"]) for text in memories
        )
        expected = assertion["matching_record_count"]
        return count == expected, f"matching records: {count}, expected: {expected}"
    if kind == "corrected_fact":
        corrected = sum(
            _contains_all(text, assertion["required_terms"]) for text in memories
        )
        stale = sum(
            _contains_all(text, assertion["forbidden_active_terms"])
            for text in memories
        )
        expected = assertion["matching_record_count"]
        passed = corrected == expected and stale == 0
        return passed, f"corrected records: {corrected}, stale records: {stale}"
    if kind == "cancelled_or_absent":
        subject = [
            text for text in memories if _contains_all(text, assertion["subject_terms"])
        ]
        stale = [
            text
            for text in subject
            if _contains_all(text, assertion["active_obligation_terms"])
            and not any(
                _normalized(term) in _normalized(text)
                for term in assertion["cancellation_terms"]
            )
        ]
        return not stale, f"stale active obligation records: {len(stale)}"
    if kind == "no_hard_obligation":
        subject = [
            text for text in memories if _contains_all(text, assertion["subject_terms"])
        ]
        hard = [
            text
            for text in subject
            if any(
                _normalized(term) in _normalized(text)
                for term in assertion["hard_obligation_terms"]
            )
        ]
        return not hard, f"hard obligation records: {len(hard)}"
    raise RuntimeError(f"unknown Mem0 inference assertion type: {kind}")


def _global_scope_check(
    protocol: Mapping[str, Any],
    records_a: Sequence[Mapping[str, object]],
    records_b: Sequence[Mapping[str, object]],
) -> bool:
    assertions = protocol["global_assertions"]
    text_a = "\n".join(str(item["memory"]) for item in records_a)
    text_b = "\n".join(str(item["memory"]) for item in records_b)
    return not any(
        _normalized(term) in _normalized(text_a)
        for term in assertions["scope_a_forbidden_terms"]
    ) and not any(
        _normalized(term) in _normalized(text_b)
        for term in assertions["scope_b_forbidden_terms"]
    )


def _close_memory(memory: object) -> None:
    memory.close()
    client = getattr(getattr(memory, "vector_store", None), "client", None)
    close = getattr(client, "close", None)
    if callable(close):
        close()


async def run_mem0_inference_diagnostic(
    *,
    protocol_path: Path,
    embedding_snapshot: Path,
    models_root: Path,
    source_commit: str,
    protocol_sha256: str = PROTOCOL_SHA256,
    protocol_schema_version: str = PROTOCOL_SCHEMA_VERSION,
    collection_name: str = "anamnesis_mem0_inference_v1",
) -> Mem0InferenceResult:
    protocol = _load_protocol(
        protocol_path,
        expected_sha256=protocol_sha256,
        expected_schema_version=protocol_schema_version,
    )
    _require_local_environment(context_length=protocol["model"]["context_length"])
    if len(source_commit) != 40:
        raise RuntimeError("Mem0 inference source commit must be full length")
    from anamnesis.baselines import _directory_sha256

    if (
        _directory_sha256(embedding_snapshot)
        != protocol["storage"]["embedding_artifact_sha256"]
    ):
        raise RuntimeError("Mem0 inference embedding artifact drifted")
    _verify_ollama_artifact(protocol, models_root)
    scope_a = BenchmarkScope.model_validate(protocol["scope_a"])
    scope_b = BenchmarkScope.model_validate(protocol["scope_b"])
    events_out: list[EventDiagnostic] = []
    integrity_error: str | None = None
    cleanup_passed = False
    scope_isolation = False
    calls: tuple[LlmCallAudit, ...] = ()
    ollama_version = ""
    with tempfile.TemporaryDirectory(prefix="anamnesis-mem0-infer-") as directory:
        memory: object | None = None
        with _loopback_only():
            try:
                ollama_version = _attest_server(protocol, require_resident=False)
                memory = _construct_memory(
                    protocol,
                    embedding_snapshot=embedding_snapshot,
                    runtime_root=Path(directory),
                    collection_name=collection_name,
                )
                llm = memory.llm
                for event in protocol["events"]:
                    scope = scope_a if event["scope"] == "a" else scope_b
                    before_call_count = len(llm.calls)
                    started = perf_counter()
                    raw = memory.add(
                        event["text"],
                        user_id=scope.user_id,
                        metadata=_scope_metadata(scope, event["id"]),
                        infer=True,
                    )
                    latency_ms = (perf_counter() - started) * 1000
                    if not isinstance(raw, Mapping) or not isinstance(
                        raw.get("results"), list
                    ):
                        raise RuntimeError("Mem0 infer add response is invalid")
                    sdk_events = tuple(
                        str(item.get("event", ""))
                        for item in raw["results"]
                        if isinstance(item, Mapping)
                    )
                    records = _records(memory, scope)
                    memories = tuple(str(item["memory"]) for item in records)
                    passed, reason = evaluate_assertion(event["assertion"], memories)
                    events_out.append(
                        EventDiagnostic(
                            event_id=event["id"],
                            scope=event["scope"],
                            sdk_events=sdk_events,
                            record_count=len(records),
                            memories=memories,
                            model_call_indexes=tuple(
                                range(before_call_count, len(llm.calls))
                            ),
                            latency_ms=latency_ms,
                            assertion_type=event["assertion"]["type"],
                            assertion_passed=passed,
                            assertion_reason=reason,
                        )
                    )
                _attest_server(protocol, require_resident=True)
                calls = tuple(llm.calls)
                records_a = _records(memory, scope_a)
                records_b = _records(memory, scope_b)
                scope_isolation = _global_scope_check(protocol, records_a, records_b)
                for scope, records in ((scope_a, records_a), (scope_b, records_b)):
                    for record in records:
                        memory.delete(record["id"])
                    if _records(memory, scope):
                        raise RuntimeError("Mem0 inference cleanup left scoped records")
                cleanup_passed = True
            except Exception as error:
                integrity_error = f"{type(error).__name__}: {error}"
            finally:
                if memory is not None:
                    calls = tuple(getattr(memory.llm, "calls", ()))
                    if not cleanup_passed:
                        try:
                            for scope in (scope_a, scope_b):
                                for record in _records(memory, scope):
                                    memory.delete(record["id"])
                                if _records(memory, scope):
                                    raise RuntimeError(
                                        "Mem0 inference cleanup left scoped records"
                                    )
                            cleanup_passed = True
                        except Exception as cleanup_error:
                            detail = f"{type(cleanup_error).__name__}: {cleanup_error}"
                            if integrity_error is None:
                                integrity_error = detail
                            else:
                                integrity_error = (
                                    f"{integrity_error}; cleanup failed: {detail}"
                                )
                    _close_memory(memory)

    expected_calls = protocol["model"]["expected_model_calls"]
    all_calls_finished = bool(calls) and all(
        call.done_reason == "stop" for call in calls
    )
    usage_complete = bool(calls) and all(
        call.prompt_tokens > 0 and call.completion_tokens > 0 for call in calls
    )
    integrity_passed = (
        integrity_error is None
        and len(events_out) == len(protocol["events"])
        and len(calls) == expected_calls
        and all_calls_finished
        and usage_complete
        and cleanup_passed
    )
    semantic_passed = (
        integrity_passed
        and scope_isolation
        and all(event.assertion_passed for event in events_out)
    )
    return Mem0InferenceResult(
        schema_version=RESULT_SCHEMA_VERSION,
        protocol_sha256=protocol_sha256,
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
        event_results=tuple(events_out),
        llm_calls=_serialize_llm_calls(calls),
        integrity_error=integrity_error,
    )


def _write_result(path: Path, result: Mem0InferenceResult) -> None:
    if path.exists():
        raise FileExistsError("refusing to overwrite Mem0 inference result")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")


def mem0_inference_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--embedding-snapshot", type=Path, required=True)
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = asyncio.run(
        run_mem0_inference_diagnostic(
            protocol_path=args.protocol,
            embedding_snapshot=args.embedding_snapshot,
            models_root=args.models_root,
            source_commit=args.source_commit,
        )
    )
    _write_result(args.output, result)
    return 0 if result.integrity_passed else 2


if __name__ == "__main__":
    raise SystemExit(mem0_inference_main())


__all__ = [
    "EventDiagnostic",
    "LlmCallAudit",
    "Mem0InferenceResult",
    "PinnedMem0OllamaConfig",
    "PinnedMem0OllamaLlm",
    "PROTOCOL_SHA256",
    "evaluate_assertion",
    "mem0_inference_main",
    "run_mem0_inference_diagnostic",
]
