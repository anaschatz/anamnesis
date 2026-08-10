from __future__ import annotations

from pathlib import Path

import pytest

from anamnesis.mem0_inference_diagnostic import (
    _load_protocol,
    _require_local_environment,
)
from anamnesis.mem0_inference_diagnostic_v2 import (
    PROTOCOL_SCHEMA_VERSION,
    PROTOCOL_SHA256,
)


def test_v2_loader_is_cross_locked_to_v2_bytes_and_schema() -> None:
    protocol = _load_protocol(
        Path("eval/mem0_inference_v2.protocol.json"),
        expected_sha256=PROTOCOL_SHA256,
        expected_schema_version=PROTOCOL_SCHEMA_VERSION,
    )
    assert protocol["model"]["context_length"] == 32768
    with pytest.raises(RuntimeError, match="bytes drifted"):
        _load_protocol(
            Path("eval/mem0_inference_v1.protocol.json"),
            expected_sha256=PROTOCOL_SHA256,
            expected_schema_version=PROTOCOL_SCHEMA_VERSION,
        )


def test_v2_environment_requires_32768_context(monkeypatch) -> None:
    expected = {
        "OLLAMA_NO_CLOUD": "1",
        "OLLAMA_HOST": "127.0.0.1:11434",
        "OLLAMA_CONTEXT_LENGTH": "32768",
        "OLLAMA_NUM_PARALLEL": "1",
        "OLLAMA_MAX_LOADED_MODELS": "1",
    }
    for name, value in expected.items():
        monkeypatch.setenv(name, value)
    _require_local_environment(context_length=32768)
    monkeypatch.setenv("OLLAMA_CONTEXT_LENGTH", "8192")
    with pytest.raises(RuntimeError, match="OLLAMA_CONTEXT_LENGTH=32768"):
        _require_local_environment(context_length=32768)


def test_v1_default_environment_contract_remains_8192(monkeypatch) -> None:
    expected = {
        "OLLAMA_NO_CLOUD": "1",
        "OLLAMA_HOST": "127.0.0.1:11434",
        "OLLAMA_CONTEXT_LENGTH": "8192",
        "OLLAMA_NUM_PARALLEL": "1",
        "OLLAMA_MAX_LOADED_MODELS": "1",
    }
    for name, value in expected.items():
        monkeypatch.setenv(name, value)
    _require_local_environment()
