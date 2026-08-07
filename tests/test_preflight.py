from __future__ import annotations

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
    ModelCost,
    ModelOutput,
    ModelUsage,
)

from anamnesis.experiment import ArtifactPin
from anamnesis.inspect_adapter import (
    ModelPreflightResult,
    _decision_schema,
    _memory_delta_schema,
)
from anamnesis.preflight import (
    MODEL_PREFLIGHT_PURPOSE,
    MODEL_PREFLIGHT_TASK_VERSION,
    _expected_prompts,
    validate_model_preflight_artifact,
    validate_model_preflight_log,
)

MODEL = "openai/frozen-snapshot"
MODEL_COST = ModelCost(
    input=1.0,
    output=2.0,
    input_cache_write=1.2,
    input_cache_read=0.5,
)


def _model_event(prompt: str, schema, completion: str) -> ModelEvent:
    usage = ModelUsage(
        input_tokens=100,
        output_tokens=10,
        total_tokens=110,
        total_cost=0.00012,
    )
    return ModelEvent.model_construct(
        event="model",
        model=MODEL,
        input=[ChatMessageUser(content=prompt)],
        config=GenerateConfig(response_schema=schema),
        output=ModelOutput(model=MODEL, completion=completion, usage=usage),
        cache=None,
        error=None,
        retries=None,
    )


def _valid_preflight_log() -> EvalLog:
    compiler_prompt, decision_prompt = _expected_prompts()
    events = [
        _model_event(
            compiler_prompt,
            _memory_delta_schema(MODEL),
            json.dumps(
                {
                    "fact_assertions": [],
                    "intent_creates": [],
                    "intent_updates": [],
                    "intent_cancellations": [],
                }
            ),
        ),
        _model_event(
            decision_prompt,
            _decision_schema(MODEL),
            '{"actions":[]}',
        ),
    ]
    result = ModelPreflightResult(
        model=MODEL,
        strict_schema_supported=True,
        compiler_parse_error=False,
        decision_parse_error=False,
        compiler_usage_complete=True,
        decision_usage_complete=True,
        compiler_cost_complete=True,
        decision_cost_complete=True,
        passed=True,
    )
    sample = EvalSample.model_construct(
        id="model-preflight-v0",
        epoch=1,
        events=events,
        error=None,
        invalidation=None,
        error_retries=None,
        output=ModelOutput.from_content(
            model=MODEL,
            content=result.model_dump_json(),
        ),
    )
    spec = EvalSpec.model_construct(
        task="model_preflight",
        task_registry_name="model_preflight",
        task_version=MODEL_PREFLIGHT_TASK_VERSION,
        metadata={"purpose": MODEL_PREFLIGHT_PURPOSE},
        model=MODEL,
        model_generate_config=GenerateConfig(
            temperature=0.0,
            seed=101,
            cache=False,
            max_connections=1,
            adaptive_connections=False,
        ),
        config=EvalConfig(max_samples=1, max_tasks=1, epochs=1),
        revision=EvalRevision(
            type="git",
            origin="test",
            commit="a" * 40,
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
        plan=EvalPlan.model_construct(config=GenerateConfig()),
        samples=[sample],
    )


def test_semantic_preflight_accepts_exact_two_call_attestation() -> None:
    validate_model_preflight_log(
        _valid_preflight_log(),
        model_name=MODEL,
        pricing=MODEL_COST,
    )


@pytest.mark.parametrize("tamper", ["prompt", "schema", "cost", "cache"])
def test_semantic_preflight_rejects_tampered_model_events(tamper: str) -> None:
    log = _valid_preflight_log()
    first = log.samples[0].events[0]
    assert isinstance(first, ModelEvent)
    if tamper == "prompt":
        first.input = [ChatMessageUser(content="different compiler prompt")]
        expected = "prompt differs"
    elif tamper == "schema":
        first.config.response_schema = _decision_schema(MODEL)
        expected = "strict schema"
    elif tamper == "cost":
        assert first.output.usage is not None
        first.output.usage.total_cost = 1.0
        expected = "pinned pricing"
    else:
        first.cache = "read"
        expected = "cached"

    with pytest.raises(ValueError, match=expected):
        validate_model_preflight_log(log, model_name=MODEL, pricing=MODEL_COST)


def test_preflight_artifact_binds_bytes_and_pricing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preflight_path = tmp_path / "preflight.eval"
    preflight_path.write_bytes(b"pinned Inspect log bytes")
    pricing_path = tmp_path / "pricing.json"
    pricing_path.write_text(
        json.dumps({MODEL: MODEL_COST.model_dump()}),
        encoding="utf-8",
    )
    preflight = ArtifactPin(
        path=str(preflight_path),
        sha256=hashlib.sha256(preflight_path.read_bytes()).hexdigest(),
    )
    pricing = ArtifactPin(
        path=str(pricing_path),
        sha256=hashlib.sha256(pricing_path.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        "anamnesis.preflight.read_eval_log",
        lambda _: _valid_preflight_log(),
    )

    validate_model_preflight_artifact(
        preflight,
        model_name=MODEL,
        pricing=pricing,
    )

    with pytest.raises(ValueError, match="hash mismatch"):
        validate_model_preflight_artifact(
            preflight.model_copy(update={"sha256": "0" * 64}),
            model_name=MODEL,
            pricing=pricing,
        )
