from __future__ import annotations

import hashlib
import json
import socket
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from anamnesis.mem0_inference_diagnostic import (
    PROTOCOL_SHA256,
    PinnedMem0OllamaConfig,
    PinnedMem0OllamaLlm,
    _canonical_sha256,
    _load_protocol,
    _loopback_only,
    _require_local_environment,
    _serialize_llm_calls,
    _verify_ollama_artifact,
    evaluate_assertion,
)


def test_llm_call_audits_cross_factory_module_identity_boundary() -> None:
    class FactoryLoadedAudit:
        def model_dump(self, *, mode: str) -> dict[str, object]:
            assert mode == "json"
            return {
                "index": 0,
                "request_sha256": "a" * 64,
                "response_sha256": "b" * 64,
                "response_text": '{"memory": []}',
                "prompt_tokens": 123,
                "completion_tokens": 7,
                "latency_ms": 1.5,
                "done_reason": "stop",
            }

    assert _serialize_llm_calls((FactoryLoadedAudit(),)) == (
        {
            "index": 0,
            "request_sha256": "a" * 64,
            "response_sha256": "b" * 64,
            "response_text": '{"memory": []}',
            "prompt_tokens": 123,
            "completion_tokens": 7,
            "latency_ms": 1.5,
            "done_reason": "stop",
        },
    )


def test_runtime_source_attests_resident_context_length() -> None:
    source = Path("src/anamnesis/mem0_inference_diagnostic.py").read_text()
    assert 'resident[0].get("context_length") != model["context_length"]' in source


def test_runtime_requires_all_five_local_environment_pins(monkeypatch) -> None:
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
    monkeypatch.setenv("OLLAMA_NO_CLOUD", "0")
    with pytest.raises(RuntimeError, match="OLLAMA_NO_CLOUD=1"):
        _require_local_environment()


@pytest.mark.parametrize(
    ("assertion", "memories", "expected"),
    [
        (
            {"type": "contains_fact", "required_terms": ["Argo", "permit"]},
            ["User must submit the Argo permit."],
            True,
        ),
        (
            {
                "type": "deduplicated_fact",
                "required_terms": ["quarterly", "Greek"],
                "matching_record_count": 1,
            },
            ["User prefers Greek for quarterly reports."],
            True,
        ),
        (
            {
                "type": "deduplicated_fact",
                "required_terms": ["quarterly", "Greek"],
                "matching_record_count": 1,
            },
            [
                "User prefers Greek for quarterly reports.",
                "Greek is preferred for quarterly reports.",
            ],
            False,
        ),
        (
            {
                "type": "corrected_fact",
                "required_terms": ["quarterly", "English"],
                "forbidden_active_terms": ["prefer", "quarterly", "Greek"],
                "matching_record_count": 1,
            },
            ["User prefers English for quarterly reports."],
            True,
        ),
        (
            {
                "type": "corrected_fact",
                "required_terms": ["quarterly", "English"],
                "forbidden_active_terms": ["prefer", "quarterly", "Greek"],
                "matching_record_count": 1,
            },
            [
                "User prefers Greek for quarterly reports.",
                "User prefers English for quarterly reports.",
            ],
            False,
        ),
        (
            {
                "type": "cancelled_or_absent",
                "subject_terms": ["Argo", "permit"],
                "active_obligation_terms": ["need", "submit"],
                "cancellation_terms": ["cancelled", "no longer"],
            },
            ["The Argo permit submission was cancelled."],
            True,
        ),
        (
            {
                "type": "cancelled_or_absent",
                "subject_terms": ["Argo", "permit"],
                "active_obligation_terms": ["need", "submit"],
                "cancellation_terms": ["cancelled", "no longer"],
            },
            ["User needs to submit the Argo permit."],
            False,
        ),
        (
            {
                "type": "no_hard_obligation",
                "subject_terms": ["amber", "museum"],
                "hard_obligation_terms": ["must", "need", "due"],
            },
            ["User may visit the amber museum someday."],
            True,
        ),
        (
            {
                "type": "no_hard_obligation",
                "subject_terms": ["amber", "museum"],
                "hard_obligation_terms": ["must", "need", "due"],
            },
            ["User must visit the amber museum."],
            False,
        ),
    ],
)
def test_assertion_projection_is_frozen(assertion, memories, expected) -> None:
    passed, reason = evaluate_assertion(assertion, memories)
    assert passed is expected
    assert reason


def test_protocol_loader_requires_exact_frozen_bytes(tmp_path: Path) -> None:
    protocol = Path("eval/mem0_inference_v1.protocol.json")
    assert hashlib.sha256(protocol.read_bytes()).hexdigest() == PROTOCOL_SHA256
    assert _load_protocol(protocol)["model"]["expected_model_calls"] == 7
    altered = tmp_path / "protocol.json"
    altered.write_bytes(protocol.read_bytes() + b"\n")
    with pytest.raises(RuntimeError, match="bytes drifted"):
        _load_protocol(altered)


def test_local_transport_emits_exact_json_request_and_usage(monkeypatch) -> None:
    captured: list[dict] = []

    class FakeClient:
        def __init__(self, *, host, timeout):
            assert host == "http://127.0.0.1:11434"
            assert timeout == 180

        def chat(self, **kwargs):
            captured.append(kwargs)
            return SimpleNamespace(
                message=SimpleNamespace(content='{"memory": []}'),
                prompt_eval_count=123,
                eval_count=7,
                done_reason="stop",
            )

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(Client=FakeClient))
    config = PinnedMem0OllamaConfig(
        model="qwen3.5:9b-q4_K_M",
        ollama_base_url="http://127.0.0.1:11434",
        seed=101,
        temperature=0.0,
        top_p=1.0,
        top_k=1,
        max_tokens=1024,
        num_ctx=8192,
        timeout_seconds=180,
    )
    llm = PinnedMem0OllamaLlm(config)
    messages = [
        {"role": "system", "content": "extract"},
        {"role": "user", "content": "input"},
    ]
    assert (
        llm.generate_response(
            messages,
            response_format={"type": "json_object"},
        )
        == '{"memory": []}'
    )
    assert captured == [
        {
            "model": "qwen3.5:9b-q4_K_M",
            "messages": messages,
            "format": "json",
            "think": False,
            "options": {
                "seed": 101,
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 1,
                "num_predict": 1024,
                "num_ctx": 8192,
            },
            "stream": False,
        }
    ]
    request = {key: value for key, value in captured[0].items() if key != "stream"}
    assert llm.calls[0].request_sha256 == _canonical_sha256(request)
    assert llm.calls[0].prompt_tokens == 123
    assert llm.calls[0].completion_tokens == 7
    assert llm.calls[0].done_reason == "stop"


def test_local_transport_rejects_tools_and_non_json(monkeypatch) -> None:
    class FakeClient:
        def __init__(self, **kwargs):
            pass

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(Client=FakeClient))
    config = PinnedMem0OllamaConfig(
        model="model",
        ollama_base_url="http://127.0.0.1:11434",
        seed=101,
        temperature=0,
        top_p=1,
        top_k=1,
        max_tokens=100,
        num_ctx=8192,
        timeout_seconds=180,
    )
    llm = PinnedMem0OllamaLlm(config)
    with pytest.raises(ValueError, match="forbids tools"):
        llm.generate_response([], response_format={"type": "json_object"}, tools=[])
    with pytest.raises(ValueError, match="requires JSON"):
        llm.generate_response([], response_format=None)


def test_artifact_attestation_hashes_every_manifest_blob(tmp_path: Path) -> None:
    root = tmp_path / "models"
    manifest = (
        root / "manifests" / "registry.ollama.ai" / "library" / "qwen3.5" / "9b-q4_K_M"
    )
    manifest.parent.mkdir(parents=True)
    blobs = root / "blobs"
    blobs.mkdir()
    config = b"config"
    model = b"model"
    config_sha = hashlib.sha256(config).hexdigest()
    model_sha = hashlib.sha256(model).hexdigest()
    (blobs / f"sha256-{config_sha}").write_bytes(config)
    (blobs / f"sha256-{model_sha}").write_bytes(model)
    manifest_value = {
        "config": {"digest": f"sha256:{config_sha}", "size": len(config)},
        "layers": [
            {
                "mediaType": "application/vnd.ollama.image.model",
                "digest": f"sha256:{model_sha}",
                "size": len(model),
            }
        ],
    }
    manifest.write_text(json.dumps(manifest_value, separators=(",", ":")))
    protocol = {
        "model": {
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "model_blob_sha256": model_sha,
            "model_blob_bytes": len(model),
        }
    }
    _verify_ollama_artifact(protocol, root)
    (blobs / f"sha256-{model_sha}").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="blob drifted"):
        _verify_ollama_artifact(protocol, root)


def test_network_guard_blocks_non_pinned_destinations_and_restores() -> None:
    original = socket.socket.connect
    with _loopback_only(), pytest.raises(RuntimeError, match="external network call"):
        socket.socket().connect(("127.0.0.1", 9999))
    assert socket.socket.connect is original
