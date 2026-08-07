"""Stable component pins included in every Anamnesis system fingerprint."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache

from pydantic import BaseModel

from anamnesis.memory import (
    ExecutionRecord,
    FactRevision,
    IntentRevision,
    MemoryDelta,
    MemorySelection,
    Occurrence,
)
from anamnesis.schema import MemoryView

# Bump the relevant identifier whenever its behavior changes. The experiment
# manifest also pins the git commit; these identifiers make component drift an
# explicit part of the per-system configuration fingerprint.
ANAMNESIS_REDUCER_VERSION = "anamnesis.reducer.v0.1.0"
ANAMNESIS_TRIGGER_ENGINE_VERSION = "anamnesis.trigger-engine.v0.1.0"
ANAMNESIS_RENDERER_VERSION = "anamnesis.memory-view-renderer.v0.1.0"
ANAMNESIS_MEMORY_SCHEMA_VERSION = "anamnesis.memory-schema.v0.1.0"

_MEMORY_SCHEMA_MODELS: tuple[type[BaseModel], ...] = (
    MemoryDelta,
    FactRevision,
    IntentRevision,
    Occurrence,
    ExecutionRecord,
    MemorySelection,
    MemoryView,
)


@lru_cache(maxsize=1)
def memory_schema_sha256() -> str:
    """Hash the complete reducer input/state/view schema bundle."""

    schema_bundle = {
        model.__name__: model.model_json_schema() for model in _MEMORY_SCHEMA_MODELS
    }
    serialized = json.dumps(schema_bundle, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()


def anamnesis_runtime_contract() -> dict[str, str]:
    """Return the current explicit deterministic-memory component pins."""

    return {
        "memory_schema_version": ANAMNESIS_MEMORY_SCHEMA_VERSION,
        "memory_schema_sha256": memory_schema_sha256(),
        "reducer_version": ANAMNESIS_REDUCER_VERSION,
        "trigger_engine_version": ANAMNESIS_TRIGGER_ENGINE_VERSION,
        "renderer_version": ANAMNESIS_RENDERER_VERSION,
    }


__all__ = [
    "ANAMNESIS_MEMORY_SCHEMA_VERSION",
    "ANAMNESIS_REDUCER_VERSION",
    "ANAMNESIS_RENDERER_VERSION",
    "ANAMNESIS_TRIGGER_ENGINE_VERSION",
    "anamnesis_runtime_contract",
    "memory_schema_sha256",
]
