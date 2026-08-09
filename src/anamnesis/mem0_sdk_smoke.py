"""One-shot local contract smoke against a byte-pinned Mem0 OSS SDK."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
import socket
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from anamnesis.baselines import _directory_sha256
from anamnesis.mem0_benchmark import (
    Mem0AdapterConfig,
    Mem0BenchmarkAdapter,
    mem0_package_identity_sha256,
)
from anamnesis.memory_benchmark import (
    BenchmarkMemoryInput,
    BenchmarkQuery,
    BenchmarkScope,
)
from anamnesis.openmemory_sdk_smoke import python_source_tree_identity

PIN_SCHEMA_VERSION = "mem0_sdk_smoke.v1"
RESULT_SCHEMA_VERSION = "mem0_sdk_smoke_result.v1"


class Mem0SdkPin(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["mem0_sdk_smoke.v1"]
    purpose: Literal["real_mem0_storage_retrieval_contract"]
    hypothesis_test_eligible: Literal[False]
    upstream_repository: str
    upstream_tag: str
    upstream_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    package_name: Literal["mem0ai"]
    package_version: str
    python_source_tree_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    python_source_file_count: int = Field(gt=0)
    python_source_bytes: int = Field(gt=0)
    pyproject_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    embedding_repository: str
    embedding_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    embedding_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    embedding_dimensions: int = Field(gt=0)
    vector_store: Literal["qdrant_embedded"]
    infer: Literal[False]
    network_calls_allowed: Literal[False]
    runtime_packages: dict[str, str]
    scope: dict[str, str]
    initial_memory: str
    updated_memory: str
    query: str


class Mem0SdkSmokeResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["mem0_sdk_smoke_result.v1"]
    passed: bool
    hypothesis_test_eligible: Literal[False]
    upstream_revision: str
    package_version: str
    python_version: str
    source_tree_sha256: str
    pin_sha256: str
    pyproject_sha256: str
    package_identity_sha256: str
    runtime_packages_sha256: str
    embedding_artifact_sha256: str
    vector_store: Literal["qdrant_embedded"]
    provider_api_cost_usd: Literal[0.0]
    network_calls: Literal[0]
    infer: Literal[False]
    automatic_fact_extraction_tested: Literal[False]
    automatic_deduplication_tested: Literal[False]
    add_verified: bool
    scoped_vector_search_verified: bool
    update_verified: bool
    delete_verified: bool
    cleanup_verified: bool
    opaque_provider_ids: bool
    action_evidence_ids_empty: bool


class PinnedFastEmbedEmbedding:
    """Mem0 provider shim that can load an exact local FastEmbed snapshot."""

    def __init__(self, config: object) -> None:
        from fastembed import TextEmbedding

        snapshot = getattr(config, "model", None)
        if not isinstance(snapshot, str):
            raise ValueError("Mem0 smoke requires an exact embedding snapshot path")
        expected_dims = getattr(config, "embedding_dims", None)
        self._model = TextEmbedding(
            model_name="BAAI/bge-small-en-v1.5",
            specific_model_path=snapshot,
            local_files_only=True,
        )
        if self._model.embedding_size != expected_dims:
            raise ValueError("Mem0 embedding dimension drifted")

    def embed(self, text: str, memory_action: str | None = None) -> list[float]:
        del memory_action
        normalized = text.replace("\n", " ")
        return list(self._model.embed([normalized]))[0].tolist()


class NoCallLlm:
    """Sentinel: the raw-storage cell must never invoke Mem0's LLM path."""

    def __init__(self, config: object) -> None:
        del config

    def generate_response(self, *args: object, **kwargs: object) -> object:
        del args, kwargs
        raise RuntimeError("Mem0 raw-storage smoke attempted an LLM call")


def _canonical_sha256(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def load_mem0_sdk_pin(path: Path) -> Mem0SdkPin:
    return Mem0SdkPin.model_validate_json(path.read_text(encoding="utf-8"))


def _installed_runtime_packages(pin: Mem0SdkPin) -> dict[str, str]:
    actual: dict[str, str] = {}
    for name, expected in sorted(pin.runtime_packages.items()):
        value = importlib.metadata.version(name)
        if value != expected:
            raise RuntimeError(
                f"Mem0 runtime package drift: {name} expected {expected}, got {value}"
            )
        actual[name] = value
    return actual


@contextmanager
def _network_blocked() -> Iterator[None]:
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex

    def blocked(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise RuntimeError("network call blocked by Mem0 SDK smoke")

    socket.socket.connect = blocked  # type: ignore[method-assign]
    socket.socket.connect_ex = blocked  # type: ignore[method-assign]
    try:
        yield
    finally:
        socket.socket.connect = original_connect  # type: ignore[method-assign]
        socket.socket.connect_ex = original_connect_ex  # type: ignore[method-assign]


def _construct_memory(
    *, pin: Mem0SdkPin, embedding_snapshot: Path, runtime_root: Path
) -> object:
    os.environ["MEM0_TELEMETRY"] = "False"
    os.environ["MEM0_DIR"] = str(runtime_root / "mem0-home")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from mem0 import Memory
    from mem0.configs.llms.ollama import OllamaConfig
    from mem0.utils.factory import EmbedderFactory, LlmFactory

    EmbedderFactory.provider_to_class["fastembed"] = (
        "anamnesis.mem0_sdk_smoke.PinnedFastEmbedEmbedding"
    )
    LlmFactory.provider_to_class["ollama"] = (
        "anamnesis.mem0_sdk_smoke.NoCallLlm",
        OllamaConfig,
    )
    memory = Memory.from_config(
        {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": "anamnesis_mem0_sdk_smoke",
                    "embedding_model_dims": pin.embedding_dimensions,
                    "path": str(runtime_root / "qdrant"),
                    "on_disk": True,
                },
            },
            "embedder": {
                "provider": "fastembed",
                "config": {
                    "model": str(embedding_snapshot),
                    "embedding_dims": pin.embedding_dimensions,
                },
            },
            "llm": {"provider": "ollama", "config": {}},
            "history_db_path": str(runtime_root / "history.db"),
            "version": "v1.1",
        }
    )
    # This cell measures dense vector retrieval only.  Mem0's Qdrant adapter
    # otherwise lazily downloads a separate BM25 model, which would violate the
    # frozen no-network contract and confound the pinned dense embedder.
    memory.vector_store._has_bm25_slot = False
    memory.vector_store.keyword_search = lambda **kwargs: None
    return memory


async def run_mem0_sdk_smoke(
    *,
    pin: Mem0SdkPin,
    pin_sha256: str,
    package_root: Path,
    upstream_pyproject: Path,
    embedding_snapshot: Path,
) -> Mem0SdkSmokeResult:
    source_hash, source_count, source_bytes = python_source_tree_identity(package_root)
    if (
        source_hash != pin.python_source_tree_sha256
        or source_count != pin.python_source_file_count
        or source_bytes != pin.python_source_bytes
    ):
        raise RuntimeError("installed Mem0 source tree drifted")
    if _file_sha256(upstream_pyproject) != pin.pyproject_sha256:
        raise RuntimeError("Mem0 upstream pyproject drifted")
    embedding_hash = _directory_sha256(embedding_snapshot)
    if embedding_hash != pin.embedding_artifact_sha256:
        raise RuntimeError("Mem0 embedding artifact drifted")
    packages = _installed_runtime_packages(pin)
    scope = BenchmarkScope.model_validate(pin.scope)
    package_identity = mem0_package_identity_sha256(
        upstream_revision=pin.upstream_revision,
        package_version=pin.package_version,
        source_tree_sha256=source_hash,
    )
    with tempfile.TemporaryDirectory(prefix="anamnesis-mem0-sdk-") as directory:
        runtime_root = Path(directory)
        with _network_blocked():
            memory = _construct_memory(
                pin=pin,
                embedding_snapshot=embedding_snapshot,
                runtime_root=runtime_root,
            )
            adapter = Mem0BenchmarkAdapter(
                client=memory,  # type: ignore[arg-type]
                upstream_revision=pin.upstream_revision,
                package_identity_sha256=package_identity,
                config=Mem0AdapterConfig(infer=False),
            )
            await adapter.begin(scope)
            first = BenchmarkMemoryInput(
                source_event_id="mem0-smoke-e1",
                kind="prospective_obligation",
                text=pin.initial_memory,
                observed_at="2042-06-01T09:00:00Z",
            )
            ingested = await adapter.ingest_with_handle(first)
            if len(ingested.handles) != 1 or ingested.events != ("ADD",):
                raise RuntimeError("Mem0 add did not create one verified raw record")
            handle = ingested.handles[0]
            query = BenchmarkQuery(
                scope=scope,
                text=pin.query,
                at="2042-06-01T09:01:00Z",
                top_k=3,
            )
            before = await adapter.search(query)
            if len(before) != 1 or before[0].text != pin.initial_memory:
                raise RuntimeError("Mem0 initial scoped vector search failed")
            updated = BenchmarkMemoryInput(
                source_event_id="mem0-smoke-e2",
                kind="prospective_obligation",
                text=pin.updated_memory,
                observed_at="2042-06-01T09:02:00Z",
            )
            await adapter.update(handle, updated)
            after = await adapter.search(query)
            if len(after) != 1 or after[0].text != pin.updated_memory:
                raise RuntimeError("Mem0 updated scoped vector search failed")
            if after[0].source_event_ids != ("mem0-smoke-e2",):
                raise RuntimeError("Mem0 update provenance metadata drifted")
            await adapter.close(scope)
            remaining = memory.get_all(filters={"user_id": scope.user_id}, top_k=20)
            if remaining != {"results": []}:
                raise RuntimeError("Mem0 scoped cleanup left records")
            memory.close()
            client = getattr(getattr(memory, "vector_store", None), "client", None)
            close = getattr(client, "close", None)
            if callable(close):
                close()

    return Mem0SdkSmokeResult(
        schema_version=RESULT_SCHEMA_VERSION,
        passed=True,
        hypothesis_test_eligible=False,
        upstream_revision=pin.upstream_revision,
        package_version=pin.package_version,
        python_version=sys.version.split()[0],
        source_tree_sha256=source_hash,
        pin_sha256=pin_sha256,
        pyproject_sha256=pin.pyproject_sha256,
        package_identity_sha256=package_identity,
        runtime_packages_sha256=_canonical_sha256(packages),
        embedding_artifact_sha256=embedding_hash,
        vector_store=pin.vector_store,
        provider_api_cost_usd=0.0,
        network_calls=0,
        infer=False,
        automatic_fact_extraction_tested=False,
        automatic_deduplication_tested=False,
        add_verified=True,
        scoped_vector_search_verified=True,
        update_verified=True,
        delete_verified=True,
        cleanup_verified=True,
        opaque_provider_ids=True,
        action_evidence_ids_empty=all(
            not hit.action_evidence_ids for hit in (*before, *after)
        ),
    )


def _write_result(path: Path, result: Mem0SdkSmokeResult) -> None:
    if path.exists():
        raise FileExistsError("refusing to overwrite Mem0 SDK smoke result")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mem0_sdk_smoke_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pin", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--upstream-pyproject", type=Path, required=True)
    parser.add_argument("--embedding-snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    pin = load_mem0_sdk_pin(args.pin)
    result = asyncio.run(
        run_mem0_sdk_smoke(
            pin=pin,
            pin_sha256=_file_sha256(args.pin),
            package_root=args.package_root,
            upstream_pyproject=args.upstream_pyproject,
            embedding_snapshot=args.embedding_snapshot,
        )
    )
    _write_result(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(mem0_sdk_smoke_main())


__all__ = [
    "Mem0SdkPin",
    "Mem0SdkSmokeResult",
    "NoCallLlm",
    "PIN_SCHEMA_VERSION",
    "PinnedFastEmbedEmbedding",
    "RESULT_SCHEMA_VERSION",
    "load_mem0_sdk_pin",
    "mem0_sdk_smoke_main",
    "run_mem0_sdk_smoke",
]
