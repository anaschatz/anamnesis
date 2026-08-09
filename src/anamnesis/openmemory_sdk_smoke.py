"""One-shot contract smoke against a byte-pinned upstream OpenMemory SDK."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
import sqlite3
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from anamnesis.openmemory_recall import (
    OpenMemoryMainClientAdapter,
    OpenMemoryRecallIndex,
)

PIN_SCHEMA_VERSION = "openmemory_sdk_smoke.v1"
RESULT_SCHEMA_VERSION = "openmemory_sdk_smoke_result.v1"


class OpenMemorySdkPin(BaseModel):
    """Closed protocol pin loaded before importing the optional SDK."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str
    purpose: str
    hypothesis_test_eligible: bool
    upstream_repository: str
    upstream_tag: str
    upstream_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    package_name: str
    package_version: str
    python_source_tree_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    python_source_file_count: int = Field(gt=0)
    python_source_bytes: int = Field(gt=0)
    pyproject_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    embedding_provider: str
    database_backend: str
    network_calls_allowed: bool
    namespace: str
    user_id: str
    content: str
    query: str
    search_limit: int = Field(ge=1, le=100)
    runtime_packages: dict[str, str]
    compatibility_shims: list[str]
    success_criteria: list[str]


class OpenMemorySdkSmokeResult(BaseModel):
    """Provider-ID-free, path-free result suitable for tracked provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str
    passed: bool
    hypothesis_test_eligible: bool
    upstream_revision: str
    package_version: str
    python_version: str
    source_tree_sha256: str
    runtime_packages_sha256: str
    embedding_provider: str
    database_backend: str
    database_sha256: str
    database_bytes: int
    add_scope_verified: bool
    search_exact_match: bool
    search_match_count: int
    get_exact_match: bool
    delete_verified: bool
    handle_expired: bool
    authoritative: bool
    supports_action_evidence: bool
    mutates_anamnesis: bool


def _canonical_json_sha256(value: object) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def python_source_tree_identity(root: Path) -> tuple[str, int, int]:
    """Hash sorted Python and migration identities independent of install path."""

    root = root.resolve(strict=True)
    rows: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in {".py", ".sql"}:
            continue
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return (
        _canonical_json_sha256(rows),
        len(rows),
        sum(int(row["size"]) for row in rows),
    )


def load_openmemory_sdk_pin(path: Path) -> OpenMemorySdkPin:
    pin = OpenMemorySdkPin.model_validate_json(path.read_text(encoding="utf-8"))
    if pin.schema_version != PIN_SCHEMA_VERSION:
        raise ValueError("unexpected OpenMemory SDK pin schema")
    if pin.purpose != "real_upstream_sdk_contract":
        raise ValueError("unexpected OpenMemory SDK pin purpose")
    if pin.hypothesis_test_eligible:
        raise ValueError("OpenMemory SDK smoke cannot be hypothesis-test eligible")
    if pin.embedding_provider != "synthetic":
        raise ValueError("OpenMemory SDK smoke requires synthetic embeddings")
    if pin.database_backend != "sqlite":
        raise ValueError("OpenMemory SDK smoke requires SQLite")
    if pin.network_calls_allowed:
        raise ValueError("OpenMemory SDK smoke must forbid network calls")
    return pin


def _installed_runtime_packages(pin: OpenMemorySdkPin) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name, expected in sorted(pin.runtime_packages.items()):
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as error:
            raise RuntimeError(
                f"required runtime package is missing: {name}"
            ) from error
        if actual != expected:
            raise RuntimeError(
                f"runtime package drift for {name}: expected {expected}, got {actual}"
            )
        versions[name] = actual
    return versions


def _installed_package_root(package_name: str) -> Path:
    spec = importlib.util.find_spec(package_name)
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError(f"installed package cannot be located: {package_name}")
    locations = list(spec.submodule_search_locations)
    if len(locations) != 1:
        raise RuntimeError("OpenMemory package must have exactly one source location")
    return Path(locations[0]).resolve(strict=True)


def _prepare_database(path: Path) -> Path:
    if not path.is_absolute():
        raise ValueError("database path must be absolute")
    resolved_parent = path.parent.resolve(strict=True)
    resolved = resolved_parent / path.name
    if resolved.exists():
        raise FileExistsError("OpenMemory SDK smoke requires a fresh database path")
    for suffix in ("-shm", "-wal"):
        if Path(f"{resolved}{suffix}").exists():
            raise FileExistsError("OpenMemory SDK smoke found stale SQLite sidecars")
    return resolved


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError("refusing to overwrite SDK smoke result")
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


async def run_openmemory_sdk_smoke(
    *,
    pin: OpenMemorySdkPin,
    database_path: Path,
    package_root: Path,
    memory_factory: Callable[..., object],
) -> OpenMemorySdkSmokeResult:
    """Run one lifecycle through the production Anamnesis recall boundary."""

    source_hash, source_count, source_bytes = python_source_tree_identity(package_root)
    if (
        source_hash != pin.python_source_tree_sha256
        or source_count != pin.python_source_file_count
        or source_bytes != pin.python_source_bytes
    ):
        raise RuntimeError("installed OpenMemory Python source tree drifted")
    packages = _installed_runtime_packages(pin)

    memory = memory_factory(user=f"anamnesis::{pin.namespace}::{pin.user_id}")
    adapter = OpenMemoryMainClientAdapter(
        memory,
        upstream_revision=pin.upstream_revision,
        database_path=str(database_path),
        embedding_provider=pin.embedding_provider,
    )
    index = OpenMemoryRecallIndex(
        namespace=pin.namespace,
        user_id=pin.user_id,
        client=adapter,
        local_only=True,
    )
    handle = await index.add(pin.content, metadata={"purpose": pin.purpose})
    search = await index.search(pin.query, limit=pin.search_limit)
    exact_matches = [match for match in search.matches if match.content == pin.content]
    if len(exact_matches) != 1:
        raise RuntimeError("OpenMemory search did not return one exact scoped match")
    document = await index.get(handle)
    if document.content != pin.content:
        raise RuntimeError("OpenMemory get content drifted")
    deletion = await index.delete(handle)
    if not deletion.deleted:
        raise RuntimeError("OpenMemory delete was not acknowledged")
    try:
        await index.get(handle)
    except ValueError as error:
        handle_expired = str(error) == "unknown or expired recall handle"
    else:
        handle_expired = False
    if not handle_expired:
        raise RuntimeError("deleted OpenMemory handle remained usable")

    with sqlite3.connect(database_path) as connection:
        remaining = connection.execute(
            "SELECT COUNT(*) FROM memories WHERE user_id = ?",
            (index.scoped_user_id,),
        ).fetchone()
    if remaining is None or remaining[0] != 0:
        raise RuntimeError("OpenMemory scoped record remained after delete")

    return OpenMemorySdkSmokeResult(
        schema_version=RESULT_SCHEMA_VERSION,
        passed=True,
        hypothesis_test_eligible=False,
        upstream_revision=pin.upstream_revision,
        package_version=packages[pin.package_name],
        python_version=sys.version.split()[0],
        source_tree_sha256=source_hash,
        runtime_packages_sha256=_canonical_json_sha256(packages),
        embedding_provider=pin.embedding_provider,
        database_backend=pin.database_backend,
        database_sha256=_sha256(database_path),
        database_bytes=database_path.stat().st_size,
        add_scope_verified=True,
        search_exact_match=True,
        search_match_count=len(search.matches),
        get_exact_match=True,
        delete_verified=True,
        handle_expired=True,
        authoritative=index.authoritative,
        supports_action_evidence=index.supports_action_evidence,
        mutates_anamnesis=index.mutates_anamnesis,
    )


def openmemory_sdk_smoke_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run one byte-pinned real OpenMemory SDK lifecycle"
    )
    parser.add_argument("--pin", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    pin = load_openmemory_sdk_pin(args.pin.resolve(strict=True))
    database_path = _prepare_database(args.database)

    os.environ["OM_DB_URL"] = f"sqlite:///{database_path}"
    os.environ["OM_EMBED_KIND"] = pin.embedding_provider
    if Path.cwd().joinpath("openmemory.toml").exists():
        raise RuntimeError("openmemory.toml would override the frozen environment")

    packages = _installed_runtime_packages(pin)
    if packages[pin.package_name] != pin.package_version:
        raise RuntimeError("OpenMemory package version drifted")
    package_root = _installed_package_root(pin.package_name)
    module = importlib.import_module("openmemory.client")
    memory_factory = getattr(module, "Memory", None)
    if not callable(memory_factory):
        raise RuntimeError("openmemory.client.Memory is unavailable")
    result = asyncio.run(
        run_openmemory_sdk_smoke(
            pin=pin,
            database_path=database_path,
            package_root=package_root,
            memory_factory=memory_factory,
        )
    )
    _write_json_atomic(args.output, result.model_dump(mode="json"))
    print(result.model_dump_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(openmemory_sdk_smoke_main())
