from __future__ import annotations

import hashlib
import json
import socket
from pathlib import Path

import pytest

from anamnesis.mem0_sdk_smoke import (
    Mem0SdkSmokeResult,
    NoCallLlm,
    _network_blocked,
    load_mem0_sdk_pin,
)

PIN = Path("eval/mem0_v2.0.17.pin.json")
RESULT = Path("results/mem0_sdk_smoke.v1.json")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_pin_and_result_are_closed_and_cryptographically_bound() -> None:
    pin = load_mem0_sdk_pin(PIN)
    result = Mem0SdkSmokeResult.model_validate_json(RESULT.read_text())
    assert result.passed
    assert result.pin_sha256 == _sha256(PIN)
    assert result.upstream_revision == pin.upstream_revision
    assert result.source_tree_sha256 == pin.python_source_tree_sha256
    assert result.pyproject_sha256 == pin.pyproject_sha256
    assert result.embedding_artifact_sha256 == pin.embedding_artifact_sha256


def test_result_claims_only_the_exercised_raw_storage_cell() -> None:
    result = Mem0SdkSmokeResult.model_validate_json(RESULT.read_text())
    assert not result.infer
    assert not result.automatic_fact_extraction_tested
    assert not result.automatic_deduplication_tested
    assert result.scoped_vector_search_verified
    assert result.update_verified
    assert result.delete_verified
    assert result.cleanup_verified
    assert result.network_calls == 0
    assert result.provider_api_cost_usd == 0.0


def test_pin_uses_exact_upstream_and_local_embedding_identities() -> None:
    pin = load_mem0_sdk_pin(PIN)
    assert pin.upstream_tag == "v2.0.17"
    assert pin.upstream_revision == "12c47f524935692e27ad48d829f35fa1e4417181"
    assert pin.embedding_revision == "52398278842ec682c6f32300af41344b1c0b0bb2"
    assert pin.vector_store == "qdrant_embedded"
    assert pin.runtime_packages["mem0ai"] == "2.0.17"
    assert pin.runtime_packages["fastembed"] == "0.7.4"


def test_no_call_llm_fails_if_raw_cell_reaches_inference() -> None:
    sentinel = NoCallLlm(object())
    with pytest.raises(RuntimeError, match="attempted an LLM call"):
        sentinel.generate_response("should never run")


def test_network_guard_rejects_socket_connect_and_restores_it() -> None:
    original = socket.socket.connect
    with _network_blocked(), pytest.raises(RuntimeError, match="network call blocked"):
        socket.socket().connect(("127.0.0.1", 9))
    assert socket.socket.connect is original


def test_smoke_source_has_no_module_level_mem0_dependency() -> None:
    source = Path("src/anamnesis/mem0_sdk_smoke.py").read_text()
    prefix = source.split("def _construct_memory", maxsplit=1)[0]
    assert "from mem0" not in prefix
    assert "local_files_only=True" in source
    assert 'os.environ["MEM0_TELEMETRY"] = "False"' in source


def test_result_is_provider_id_and_path_free() -> None:
    payload = json.loads(RESULT.read_text())
    encoded = json.dumps(payload)
    assert "/private/" not in encoded
    assert "/Users/" not in encoded
    assert "memory_id" not in encoded
