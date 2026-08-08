from __future__ import annotations

import asyncio
import hashlib
import json
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
)
from inspect_ai.model import (
    ChatMessageUser,
    GenerateConfig,
    ModelCall,
    ModelOutput,
    ModelUsage,
)
from inspect_ai.model._openai import openai_completion_params

from anamnesis.local_experiment import (
    LOCAL_W3_M2_MODEL_ARTIFACT_PATH,
    LOCAL_W3_M2_MODEL_ID,
    LOCAL_W3_M2_PRICING_PATH,
    LOCAL_W3_M2_PRICING_SHA256,
    LOCAL_W3_M2_PROTOCOL_PATH,
    LOCAL_W3_M2_PROTOCOL_SHA256,
    load_ollama_artifact_pin,
    validate_zero_api_pricing,
)
from anamnesis.local_preflight import validate_local_w3_m2_preflight_log
from anamnesis.local_runtime import (
    LOCAL_MODEL_PREFLIGHT_W3_M2_PURPOSE,
    LOCAL_MODEL_PREFLIGHT_W3_M2_SAMPLE_ID,
    LOCAL_MODEL_PREFLIGHT_W3_M2_TASK_VERSION,
    LOCAL_OLLAMA_BASE_URL,
    LOCAL_OLLAMA_CONTEXT_LENGTH,
    LOCAL_OLLAMA_MODEL,
    LOCAL_W3_M2_OLLAMA_FAMILY,
    LOCAL_W3_M2_OLLAMA_MANIFEST_SHA256,
    LOCAL_W3_M2_OLLAMA_MODEL,
    LOCAL_W3_M2_OLLAMA_PARAMETER_SIZE,
    LOCAL_W3_M2_OLLAMA_SERVICE_MODEL,
    LocalLoadedModelAttestation,
    LocalModelPreflightW3CaseResult,
    LocalModelPreflightW3Result,
    LocalOllamaRuntimeAttestation,
    _local_decision_schema,
    _local_memory_delta_schema,
    load_local_w3_preflight_fixture,
    local_w3_preflight_prompts,
    run_local_model_preflight_w3,
)
from anamnesis.runner import DecisionCall, DecisionRequest
from anamnesis.schema import Decision, Usage

ROOT = Path(__file__).resolve().parents[1]


def test_w3_m2_artifacts_and_protocol_are_exactly_pinned() -> None:
    pin = load_ollama_artifact_pin(ROOT / LOCAL_W3_M2_MODEL_ARTIFACT_PATH)
    assert pin.model == LOCAL_W3_M2_MODEL_ID
    assert pin.manifest_sha256 == LOCAL_W3_M2_OLLAMA_MANIFEST_SHA256
    assert [blob.role for blob in pin.blobs] == [
        "config",
        "model",
        "license",
        "params",
    ]
    assert (
        validate_zero_api_pricing(
            ROOT / LOCAL_W3_M2_PRICING_PATH,
            LOCAL_W3_M2_MODEL_ID,
        )
        == LOCAL_W3_M2_PRICING_SHA256
    )
    assert (
        hashlib.sha256((ROOT / LOCAL_W3_M2_PROTOCOL_PATH).read_bytes()).hexdigest()
        == LOCAL_W3_M2_PROTOCOL_SHA256
    )


def test_w3_m2_uses_byte_identical_w3_response_schemas() -> None:
    assert (
        _local_memory_delta_schema(LOCAL_W3_M2_OLLAMA_MODEL).model_dump_json()
        == _local_memory_delta_schema(LOCAL_OLLAMA_MODEL).model_dump_json()
    )
    assert (
        _local_decision_schema(LOCAL_W3_M2_OLLAMA_MODEL).model_dump_json()
        == _local_decision_schema(LOCAL_OLLAMA_MODEL).model_dump_json()
    )


def test_w3_m2_runs_the_unchanged_nine_case_semantic_gate() -> None:
    fixture = load_local_w3_preflight_fixture(
        ROOT / "eval/preflight/local_writer_w3.v1.json"
    )
    compiler_cases = fixture["compiler_cases"]
    assert isinstance(compiler_cases, list)
    completions = [json.dumps(case["valid_wire_example"]) for case in compiler_cases]

    class FakeModel:
        name = LOCAL_W3_M2_OLLAMA_MODEL
        runtime_attestation = LocalOllamaRuntimeAttestation(
            model=LOCAL_W3_M2_OLLAMA_MODEL,
            base_url=LOCAL_OLLAMA_BASE_URL,
            no_cloud="1",
            context_length=LOCAL_OLLAMA_CONTEXT_LENGTH,
            host="127.0.0.1:11434",
            num_parallel=1,
            max_loaded_models=1,
        )

        def __init__(self) -> None:
            self.index = 0

        async def complete_structured(self, *, prompt, response_schema):
            completion = completions[self.index]
            self.index += 1
            return (
                ModelOutput(
                    model=LOCAL_W3_M2_OLLAMA_SERVICE_MODEL,
                    completion=completion,
                    usage=ModelUsage(input_tokens=100, output_tokens=10),
                ),
                1.0,
            )

        async def decide(self, request: DecisionRequest) -> DecisionCall:
            return DecisionCall(
                decision=Decision(),
                usage=Usage(
                    input_tokens=100,
                    uncached_input_tokens=100,
                    output_tokens=10,
                    cost_usd=0.0,
                ),
                latency_ms=1.0,
                raw_completion='{"mode":"no_action","actions":[]}',
                usage_complete=True,
                cost_complete=True,
            )

    loaded = LocalLoadedModelAttestation(
        model=LOCAL_W3_M2_OLLAMA_SERVICE_MODEL,
        digest=LOCAL_W3_M2_OLLAMA_MANIFEST_SHA256,
        family=LOCAL_W3_M2_OLLAMA_FAMILY,
        parameter_size=LOCAL_W3_M2_OLLAMA_PARAMETER_SIZE,
        quantization_level="Q4_K_M",
        context_length=LOCAL_OLLAMA_CONTEXT_LENGTH,
        size_vram=8_000_000_000,
        ollama_version="0.31.1",
    )
    result = asyncio.run(
        run_local_model_preflight_w3(  # type: ignore[arg-type]
            FakeModel(),
            fixture=fixture,
            residency_probe=lambda: loaded,
        )
    )
    assert result.model == LOCAL_W3_M2_OLLAMA_MODEL
    assert result.passed
    assert [case.case_id for case in result.cases] == [
        "C1",
        "C2",
        "C3",
        "C4",
        "C5",
        "C6",
        "C7",
        "C8",
        "D1",
    ]


def _m2_event(prompt: str, schema, completion: str) -> ModelEvent:
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
        LOCAL_W3_M2_OLLAMA_SERVICE_MODEL,
        config,
        tools=False,
    )
    request.update(
        messages=[{"role": "user", "content": prompt}],
        tools=None,
        tool_choice=None,
        extra_headers={"x-irid": "w3-m2-test"},
    )
    return ModelEvent.model_construct(
        event="model",
        model=LOCAL_W3_M2_OLLAMA_MODEL,
        input=[ChatMessageUser(content=prompt)],
        tools=[],
        tool_choice="none",
        config=config,
        output=ModelOutput(
            model=LOCAL_W3_M2_OLLAMA_SERVICE_MODEL,
            completion=completion,
            usage=ModelUsage(
                input_tokens=100,
                output_tokens=10,
                total_cost=0.0,
            ),
        ),
        call=ModelCall(
            request=request,
            response={"model": LOCAL_W3_M2_OLLAMA_SERVICE_MODEL},
        ),
        cache=None,
        error=None,
        retries=None,
    )


def _valid_m2_log() -> tuple[EvalLog, dict]:
    fixture = load_local_w3_preflight_fixture(
        ROOT / "eval/preflight/local_writer_w3.v1.json"
    )
    compiler_cases = fixture["compiler_cases"]
    decision_cases = fixture["decision_cases"]
    assert isinstance(compiler_cases, list)
    assert isinstance(decision_cases, list)
    prompts = local_w3_preflight_prompts(fixture)
    completions = [
        *(json.dumps(case["valid_wire_example"]) for case in compiler_cases),
        json.dumps(decision_cases[0]["valid_wire_example"]),
    ]
    schemas = [
        *(_local_memory_delta_schema(LOCAL_W3_M2_OLLAMA_MODEL) for _ in range(8)),
        _local_decision_schema(LOCAL_W3_M2_OLLAMA_MODEL),
    ]
    events = [
        _m2_event(prompt, schema, completion)
        for prompt, schema, completion in zip(
            prompts,
            schemas,
            completions,
            strict=True,
        )
    ]
    usage = Usage(
        input_tokens=100,
        uncached_input_tokens=100,
        output_tokens=10,
        cost_usd=0.0,
    )
    result = LocalModelPreflightW3Result(
        model=LOCAL_W3_M2_OLLAMA_MODEL,
        runtime=LocalOllamaRuntimeAttestation(
            model=LOCAL_W3_M2_OLLAMA_MODEL,
            base_url=LOCAL_OLLAMA_BASE_URL,
            no_cloud="1",
            context_length=LOCAL_OLLAMA_CONTEXT_LENGTH,
            host="127.0.0.1:11434",
            num_parallel=1,
            max_loaded_models=1,
        ),
        loaded_model=LocalLoadedModelAttestation(
            model=LOCAL_W3_M2_OLLAMA_SERVICE_MODEL,
            digest=LOCAL_W3_M2_OLLAMA_MANIFEST_SHA256,
            family=LOCAL_W3_M2_OLLAMA_FAMILY,
            parameter_size=LOCAL_W3_M2_OLLAMA_PARAMETER_SIZE,
            quantization_level="Q4_K_M",
            context_length=LOCAL_OLLAMA_CONTEXT_LENGTH,
            size_vram=8_000_000_000,
            ollama_version="0.31.1",
        ),
        same_model_for_compiler_and_decision=True,
        cases=[
            LocalModelPreflightW3CaseResult(
                case_id=case_id,  # type: ignore[arg-type]
                role="decision" if case_id == "D1" else "compiler",
                parse_error=False,
                semantic_valid=True,
                usage=usage,
                usage_complete=True,
                cost_complete=True,
                latency_ms=1.0,
            )
            for case_id in (*[f"C{index}" for index in range(1, 9)], "D1")
        ],
        residency_probe_latency_ms=1.0,
        fixture_sha256=(
            "5628c3c1d7f8e1a5da43d6e567d55ac8e4fbabd8b9c4054325de6f4def1da30c"
        ),
        protocol_sha256=(
            "7f63c156a8af74ced2d5e5530b3e8083da95c7f46e14e1afafaaf864b3ce1915"
        ),
        passed=True,
    )
    serialized = result.model_dump(mode="json")
    sample = EvalSample.model_construct(
        id=LOCAL_MODEL_PREFLIGHT_W3_M2_SAMPLE_ID,
        epoch=1,
        events=events,
        metadata={"anamnesis.local_preflight_w3_m2": serialized},
        store={"anamnesis.local_preflight_w3_m2": serialized},
        error=None,
        invalidation=None,
        error_retries=[],
        output=ModelOutput.from_content(
            model=LOCAL_W3_M2_OLLAMA_MODEL,
            content=result.model_dump_json(),
        ),
    )
    commit = "a" * 40
    spec = EvalSpec.model_construct(
        task="local_model_preflight_w3_m2",
        task_registry_name="local_model_preflight_w3_m2",
        task_version=LOCAL_MODEL_PREFLIGHT_W3_M2_TASK_VERSION,
        metadata={
            "purpose": LOCAL_MODEL_PREFLIGHT_W3_M2_PURPOSE,
            "track": "local_zero_api_cost",
            "hypothesis_test_eligible": False,
            "intervention": "model_only",
            "parent_prompt_cell": "W3",
            "pricing_config_sha256": LOCAL_W3_M2_PRICING_SHA256,
            "preflight_fixture_sha256": (
                "5628c3c1d7f8e1a5da43d6e567d55ac8e4fbabd8b9c4054325de6f4def1da30c"
            ),
            "preflight_protocol_sha256": (
                "7f63c156a8af74ced2d5e5530b3e8083da95c7f46e14e1afafaaf864b3ce1915"
            ),
            "model_artifact_sha256": LOCAL_W3_M2_OLLAMA_MANIFEST_SHA256,
            "model_only_protocol_sha256": LOCAL_W3_M2_PROTOCOL_SHA256,
        },
        model=LOCAL_W3_M2_OLLAMA_MODEL,
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
            commit=commit[:7],
            dirty=False,
        ),
        dataset=EvalDataset(),
    )
    return (
        EvalLog.model_construct(
            status="success",
            invalidated=False,
            eval=spec,
            plan=EvalPlan.model_construct(config=GenerateConfig(cache=False)),
            samples=[sample],
        ),
        fixture,
    )


def test_w3_m2_strict_log_validator_binds_model_and_task_identity() -> None:
    log, fixture = _valid_m2_log()
    result = validate_local_w3_m2_preflight_log(
        log,
        fixture=fixture,
        expected_git_commit="a" * 40,
        expected_pricing_sha256=LOCAL_W3_M2_PRICING_SHA256,
    )
    assert result.passed
    log.eval.model = LOCAL_OLLAMA_MODEL
    with pytest.raises(ValueError, match="different model"):
        validate_local_w3_m2_preflight_log(
            log,
            fixture=fixture,
            expected_git_commit="a" * 40,
            expected_pricing_sha256=LOCAL_W3_M2_PRICING_SHA256,
        )
