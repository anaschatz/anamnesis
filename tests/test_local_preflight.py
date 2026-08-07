from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from inspect_ai.event import ModelEvent
from inspect_ai.log import (
    EvalConfig,
    EvalDataset,
    EvalLog,
    EvalPlan,
    EvalRevision,
    EvalSample,
    EvalSpec,
    write_eval_log,
)
from inspect_ai.model import (
    ChatMessageUser,
    GenerateConfig,
    ModelCall,
    ModelOutput,
    ModelUsage,
)
from inspect_ai.model._openai import openai_completion_params

from anamnesis.experiment import ArtifactPin
from anamnesis.local_experiment import LOCAL_PRICING_SHA256
from anamnesis.local_preflight import (
    LOCAL_PREFLIGHT_SAMPLE_ID,
    local_preflight_prompts,
    validate_local_preflight_artifact,
    validate_local_preflight_log,
)
from anamnesis.local_runtime import (
    LOCAL_MODEL_PREFLIGHT_PURPOSE,
    LOCAL_MODEL_PREFLIGHT_TASK_VERSION,
    LOCAL_OLLAMA_BASE_URL,
    LOCAL_OLLAMA_CONTEXT_LENGTH,
    LOCAL_OLLAMA_FAMILY,
    LOCAL_OLLAMA_MANIFEST_SHA256,
    LOCAL_OLLAMA_MODEL,
    LOCAL_OLLAMA_PARAMETER_SIZE,
    LOCAL_OLLAMA_QUANTIZATION,
    LOCAL_OLLAMA_SERVICE_MODEL,
    LocalLoadedModelAttestation,
    LocalModelPreflightResult,
    LocalOllamaRuntimeAttestation,
    _local_decision_schema,
    _local_memory_delta_schema,
)
from anamnesis.schema import Usage

COMMIT = "a" * 40


def _compiler_completion() -> str:
    return json.dumps(
        {
            "fact_assertions": [],
            "intent_creates": [
                {
                    "intent_id": "remind-compatibility-check-1700",
                    "trigger": {
                        "type": "at",
                        "at": "2026-01-05T17:00:00Z",
                    },
                    "required_conditions": [],
                    "blockers": [],
                    "action_template": {
                        "payload": {"subject": "perform compatibility check"},
                        "summary": "Perform compatibility check.",
                    },
                }
            ],
            "intent_updates": [],
            "intent_cancellations": [],
        }
    )


def _model_event(prompt: str, schema, completion: str) -> ModelEvent:
    usage = ModelUsage(
        input_tokens=100,
        output_tokens=10,
        total_tokens=110,
        total_cost=0.0,
    )
    config = GenerateConfig(
        temperature=0.0,
        seed=101,
        cache=False,
        max_retries=0,
        max_connections=1,
        adaptive_connections=False,
        response_schema=schema,
    )
    request = openai_completion_params(
        LOCAL_OLLAMA_SERVICE_MODEL,
        config,
        tools=False,
    )
    request.update(
        messages=[{"role": "user", "content": prompt}],
        tools=None,
        tool_choice=None,
        extra_headers={"x-irid": "local-test-request"},
    )
    return ModelEvent.model_construct(
        event="model",
        model=LOCAL_OLLAMA_MODEL,
        input=[ChatMessageUser(content=prompt)],
        tools=[],
        tool_choice="none",
        config=config,
        output=ModelOutput(
            model=LOCAL_OLLAMA_SERVICE_MODEL,
            completion=completion,
            usage=usage,
        ),
        call=ModelCall(
            request=request,
            response={"model": LOCAL_OLLAMA_SERVICE_MODEL},
        ),
        cache=None,
        error=None,
        retries=None,
        timestamp=datetime(2026, 1, 5, 9, 0, tzinfo=UTC),
        working_start=0.0,
    )


def _valid_log() -> EvalLog:
    compiler_prompt, decision_prompt = local_preflight_prompts()
    events = [
        _model_event(
            compiler_prompt,
            _local_memory_delta_schema(LOCAL_OLLAMA_MODEL),
            _compiler_completion(),
        ),
        _model_event(
            decision_prompt,
            _local_decision_schema(LOCAL_OLLAMA_MODEL),
            '{"mode":"no_action","actions":[]}',
        ),
    ]
    compiler_usage = Usage(
        input_tokens=100,
        uncached_input_tokens=100,
        output_tokens=10,
        cost_usd=0.0,
    )
    result = LocalModelPreflightResult(
        model=LOCAL_OLLAMA_MODEL,
        runtime=LocalOllamaRuntimeAttestation(
            model=LOCAL_OLLAMA_MODEL,
            base_url=LOCAL_OLLAMA_BASE_URL,
            no_cloud="1",
            context_length=LOCAL_OLLAMA_CONTEXT_LENGTH,
            host="127.0.0.1:11434",
            num_parallel=1,
            max_loaded_models=1,
        ),
        loaded_model=LocalLoadedModelAttestation(
            model=LOCAL_OLLAMA_SERVICE_MODEL,
            digest=LOCAL_OLLAMA_MANIFEST_SHA256,
            family=LOCAL_OLLAMA_FAMILY,
            parameter_size=LOCAL_OLLAMA_PARAMETER_SIZE,
            quantization_level=LOCAL_OLLAMA_QUANTIZATION,
            context_length=LOCAL_OLLAMA_CONTEXT_LENGTH,
            size_vram=3_169_761_361,
            ollama_version="0.31.1",
        ),
        same_model_for_compiler_and_decision=True,
        compiler_parse_error=False,
        decision_parse_error=False,
        compiler_semantic_valid=True,
        decision_semantic_valid=True,
        compiler_usage=compiler_usage,
        decision_usage=compiler_usage,
        compiler_usage_complete=True,
        decision_usage_complete=True,
        compiler_cost_complete=True,
        decision_cost_complete=True,
        compiler_latency_ms=1.0,
        decision_latency_ms=1.0,
        residency_probe_latency_ms=1.0,
        passed=True,
    )
    sample = EvalSample.model_construct(
        id=LOCAL_PREFLIGHT_SAMPLE_ID,
        epoch=1,
        input="Check local compatibility.",
        target="pass",
        events=events,
        error=None,
        invalidation=None,
        error_retries=[],
        output=ModelOutput.from_content(
            model=LOCAL_OLLAMA_MODEL,
            content=result.model_dump_json(),
        ),
    )
    spec = EvalSpec.model_construct(
        created="2026-01-05T09:00:00+00:00",
        task="local_model_preflight",
        task_registry_name="local_model_preflight",
        task_version=LOCAL_MODEL_PREFLIGHT_TASK_VERSION,
        metadata={
            "purpose": LOCAL_MODEL_PREFLIGHT_PURPOSE,
            "pricing_config_sha256": LOCAL_PRICING_SHA256,
        },
        model=LOCAL_OLLAMA_MODEL,
        model_base_url=LOCAL_OLLAMA_BASE_URL,
        model_args={},
        model_generate_config=GenerateConfig(
            temperature=0.0,
            seed=101,
            max_retries=0,
            max_connections=1,
            adaptive_connections=False,
        ),
        config=EvalConfig(
            max_samples=1,
            max_tasks=1,
            epochs=1,
            log_model_api=True,
        ),
        revision=EvalRevision(
            type="git",
            origin="test",
            commit=COMMIT[:7],
            dirty=False,
        ),
        dataset=EvalDataset(),
    )
    return EvalLog.model_construct(
        status="success",
        invalidated=False,
        config_updates=None,
        log_updates=None,
        eval=spec,
        plan=EvalPlan.model_construct(config=GenerateConfig(cache=False)),
        samples=[sample],
    )


def test_local_preflight_log_accepts_exact_semantic_two_call_gate() -> None:
    validate_local_preflight_log(
        _valid_log(),
        expected_git_commit=COMMIT,
        expected_pricing_sha256=LOCAL_PRICING_SHA256,
    )


@pytest.mark.parametrize(
    ("tamper", "expected"),
    [
        ("dirty", "frozen clean commit"),
        ("commit", "frozen clean commit"),
        ("route", "provider route"),
        ("pricing", "pricing hash"),
        ("retry", "cached, retried, or failed"),
        ("response", "raw local response model"),
        ("early_action", "false reminder"),
        ("wrong_time", "compiler preflight semantics"),
    ],
)
def test_local_preflight_log_rejects_drift(tamper: str, expected: str) -> None:
    log = _valid_log()
    if tamper == "dirty":
        log.eval.revision.dirty = True
    elif tamper == "commit":
        log.eval.revision.commit = "b" * 7
    elif tamper == "route":
        log.eval.model_base_url = "https://proxy.example/v1"
    elif tamper == "pricing":
        log.eval.metadata["pricing_config_sha256"] = "0" * 64
    elif tamper == "retry":
        log.samples[0].events[0].retries = 1
    elif tamper == "response":
        log.samples[0].events[0].call.response = None
    elif tamper == "early_action":
        log.samples[0].events[1].output.completion = json.dumps(
            {
                "mode": "emit",
                "actions": [
                    {
                        "action_key": "local-preflight-event",
                        "payload": {"subject": "perform compatibility check"},
                        "summary": "Perform compatibility check.",
                        "evidence_event_ids": ["local-preflight-event"],
                    }
                ],
            }
        )
    else:
        completion = json.loads(log.samples[0].events[0].output.completion)
        completion["intent_creates"][0]["trigger"]["at"] = "2026-01-05T09:00:00Z"
        log.samples[0].events[0].output.completion = json.dumps(completion)

    with pytest.raises(ValueError, match=expected):
        validate_local_preflight_log(
            log,
            expected_git_commit=COMMIT,
            expected_pricing_sha256=LOCAL_PRICING_SHA256,
        )


def test_local_preflight_artifact_binds_exact_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "model_preflight.eval"
    path.write_bytes(b"pinned local Inspect log")
    artifact = ArtifactPin(
        path=str(path),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        "anamnesis.local_preflight.read_eval_log",
        lambda *args, **kwargs: _valid_log(),
    )

    validate_local_preflight_artifact(
        artifact,
        expected_git_commit=COMMIT,
        expected_pricing_sha256=LOCAL_PRICING_SHA256,
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_local_preflight_artifact(
            artifact.model_copy(update={"sha256": "0" * 64}),
            expected_git_commit=COMMIT,
            expected_pricing_sha256=LOCAL_PRICING_SHA256,
        )


def test_serialized_eval_roundtrip_uses_attachment_resolution(
    tmp_path: Path,
) -> None:
    path = tmp_path / "model_preflight.eval"
    write_eval_log(_valid_log(), path, format="eval")
    artifact = ArtifactPin(
        path=str(path),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )

    validate_local_preflight_artifact(
        artifact,
        expected_git_commit=COMMIT,
        expected_pricing_sha256=LOCAL_PRICING_SHA256,
    )
