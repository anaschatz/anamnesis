"""Validation of the frozen local Ollama semantic-preflight artifact."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from inspect_ai.event import ModelEvent
from inspect_ai.log import EvalLog, read_eval_log
from inspect_ai.model import ChatMessageUser, ResponseSchema
from inspect_ai.model._openai import openai_completion_params
from pydantic import ValidationError

from anamnesis.experiment import ArtifactPin
from anamnesis.local_runtime import (
    LOCAL_MODEL_PREFLIGHT_PURPOSE,
    LOCAL_MODEL_PREFLIGHT_TASK_VERSION,
    LOCAL_OLLAMA_BASE_URL,
    LOCAL_OLLAMA_MODEL,
    LOCAL_OLLAMA_SERVICE_MODEL,
    LocalDecisionWire,
    LocalModelPreflightResult,
    _compiler_preflight_semantics,
    _local_decision_schema,
    _local_memory_delta_schema,
    _local_usage_from_output,
    build_local_decision_prompt,
)
from anamnesis.local_wire import (
    LocalMemoryDeltaWire,
    build_local_memory_compiler_prompt,
)
from anamnesis.memory import CompilerCall
from anamnesis.schema import ObservableEvent, Usage

LOCAL_PREFLIGHT_SAMPLE_ID = "local-model-preflight-v0"
LOCAL_PREFLIGHT_EVENT_ID = "local-preflight-event"
LOCAL_PREFLIGHT_ACTIVE_STATE = '{"facts":[],"intents":[]}'
INSPECT_REQUEST_ID_HEADER = "x-irid"


def local_preflight_event() -> ObservableEvent:
    return ObservableEvent(
        id=LOCAL_PREFLIGHT_EVENT_ID,
        at="2026-01-05T09:00:00+00:00",
        kind="user_message",
        text="At 17:00 today remind me to run the compatibility check.",
    )


def local_preflight_prompts() -> tuple[str, str]:
    event = local_preflight_event()
    return (
        build_local_memory_compiler_prompt(
            event=event,
            active_state=LOCAL_PREFLIGHT_ACTIVE_STATE,
        ),
        build_local_decision_prompt(
            now=event.at.isoformat(),
            current_event_id=event.id,
            context_events=[event],
            decision_history=[],
            memory_view=None,
        ),
    )


def _canonical_schema(schema: ResponseSchema) -> str:
    return schema.model_dump_json(exclude_none=False)


def _validate_local_model_event(
    event: ModelEvent,
    *,
    prompt: str,
    schema: ResponseSchema,
    seed: int,
) -> None:
    if event.model != LOCAL_OLLAMA_MODEL:
        raise ValueError("preflight model event differs from the pinned local model")
    if (
        event.cache is not None
        or event.retries not in (None, 0)
        or event.error is not None
        or event.output.error is not None
    ):
        raise ValueError("local model event was cached, retried, or failed")
    if event.output.model not in {LOCAL_OLLAMA_MODEL, LOCAL_OLLAMA_SERVICE_MODEL}:
        raise ValueError("local response model differs from the pinned model")
    if event.output.usage is None or event.output.usage.input_tokens <= 0:
        raise ValueError("local model event is missing positive input-token usage")
    if event.output.usage.total_cost not in (None, 0, 0.0):
        raise ValueError("local model event reports a non-zero provider cost")
    if len(event.input) != 1 or not isinstance(event.input[0], ChatMessageUser):
        raise ValueError("local model event must contain one exact user prompt")
    if event.input[0].content != prompt:
        raise ValueError("local model event prompt differs from the frozen prompt")
    if event.config.response_schema is None or _canonical_schema(
        event.config.response_schema
    ) != _canonical_schema(schema):
        raise ValueError("local model event uses an unpinned response schema")
    effective_config = event.config.model_dump(mode="json", exclude_none=True)
    effective_config.pop("response_schema", None)
    if effective_config != {
        "max_retries": 0,
        "max_connections": 1,
        "adaptive_connections": False,
        "temperature": 0.0,
        "seed": seed,
        "cache": False,
    }:
        raise ValueError("local model event uses unpinned generation settings")

    call = event.call
    if call is None or call.error:
        raise ValueError("local model event is missing a successful raw API call")
    expected_params = openai_completion_params(
        LOCAL_OLLAMA_SERVICE_MODEL,
        event.config,
        tools=False,
    )
    request = call.request
    expected_keys = set(expected_params) | {
        "messages",
        "tools",
        "tool_choice",
        "extra_headers",
    }
    if set(request) != expected_keys:
        raise ValueError("raw local request contains unpinned fields")
    if any(request.get(key) != value for key, value in expected_params.items()):
        raise ValueError("raw local request differs from effective model settings")
    if request.get("messages") != [{"role": "user", "content": prompt}]:
        raise ValueError("raw local request messages differ from the prompt")
    if request.get("tools") is not None or request.get("tool_choice") is not None:
        raise ValueError("raw local request contains tools")
    headers = request.get("extra_headers")
    if (
        not isinstance(headers, dict)
        or set(headers) != {INSPECT_REQUEST_ID_HEADER}
        or not isinstance(headers[INSPECT_REQUEST_ID_HEADER], str)
        or not headers[INSPECT_REQUEST_ID_HEADER]
    ):
        raise ValueError("raw local request contains unpinned headers")
    response = call.response
    if not isinstance(response, dict) or response.get("model") not in {
        LOCAL_OLLAMA_MODEL,
        LOCAL_OLLAMA_SERVICE_MODEL,
    }:
        raise ValueError("raw local response model differs from the pin")


def validate_local_preflight_log(
    log: EvalLog,
    *,
    expected_git_commit: str,
    expected_pricing_sha256: str,
    seed: int = 101,
) -> None:
    """Validate an exact two-call semantic local compatibility attestation."""

    if log.status != "success" or log.invalidated:
        raise ValueError("local preflight log is not a valid successful evaluation")
    spec = log.eval
    if (
        spec.task_registry_name != "local_model_preflight"
        or spec.task_version != LOCAL_MODEL_PREFLIGHT_TASK_VERSION
        or (spec.metadata or {}).get("purpose") != LOCAL_MODEL_PREFLIGHT_PURPOSE
    ):
        raise ValueError("local preflight task identity differs from the protocol")
    if (spec.metadata or {}).get("pricing_config_sha256") != (expected_pricing_sha256):
        raise ValueError("local preflight pricing hash differs from the pin")
    if spec.model != LOCAL_OLLAMA_MODEL:
        raise ValueError("local preflight used a different model")
    if str(spec.model_base_url).rstrip("/") != LOCAL_OLLAMA_BASE_URL:
        raise ValueError("local preflight used a different provider route")
    if spec.model_args != {}:
        raise ValueError("local preflight used unpinned provider arguments")
    effective_model_config = spec.model_generate_config.merge(
        log.plan.config
    ).model_dump(
        mode="json",
        exclude_none=True,
    )
    if effective_model_config != {
        "max_retries": 0,
        "max_connections": 1,
        "adaptive_connections": False,
        "temperature": 0.0,
        "seed": seed,
        "cache": False,
    }:
        raise ValueError("local preflight model configuration differs from the pin")
    revision = spec.revision
    revision_commit = revision.commit if revision is not None else None
    if (
        revision is None
        or revision.dirty is not False
        or not isinstance(revision_commit, str)
        or re.fullmatch(r"[0-9a-f]{7,40}", revision_commit) is None
        or not expected_git_commit.startswith(revision_commit)
    ):
        raise ValueError("local preflight was not run from the frozen clean commit")
    if spec.config.log_model_api is not True:
        raise ValueError("local preflight did not retain raw model API calls")
    if (
        spec.config.max_samples != 1
        or spec.config.max_tasks != 1
        or spec.config.epochs != 1
    ):
        raise ValueError("local preflight has an invalid sample/task policy")
    if log.samples is None or len(log.samples) != 1:
        raise ValueError("local preflight requires exactly one synthetic sample")
    sample = log.samples[0]
    if (
        sample.id != LOCAL_PREFLIGHT_SAMPLE_ID
        or sample.epoch != 1
        or sample.error is not None
        or sample.invalidation is not None
        or bool(sample.error_retries)
        or sample.output is None
    ):
        raise ValueError("local preflight sample failed or has the wrong identity")
    if sample.output.model != LOCAL_OLLAMA_MODEL:
        raise ValueError("local preflight final output has the wrong model identity")
    result = LocalModelPreflightResult.model_validate_json(sample.output.completion)
    if not result.passed or result.loaded_model is None:
        raise ValueError("local semantic preflight result did not pass")
    if not all(
        (
            result.same_model_for_compiler_and_decision,
            not result.compiler_parse_error,
            not result.decision_parse_error,
            result.compiler_semantic_valid,
            result.decision_semantic_valid,
            result.compiler_usage_complete,
            result.decision_usage_complete,
            result.compiler_cost_complete,
            result.decision_cost_complete,
        )
    ):
        raise ValueError("local semantic preflight flags are internally inconsistent")
    if result.compiler_usage.cost_usd != 0.0 or result.decision_usage.cost_usd != 0.0:
        raise ValueError("local semantic preflight did not record zero API cost")

    events = [event for event in sample.events if isinstance(event, ModelEvent)]
    if len(events) != 2:
        raise ValueError("local preflight must contain exactly two model calls")
    compiler_prompt, decision_prompt = local_preflight_prompts()
    _validate_local_model_event(
        events[0],
        prompt=compiler_prompt,
        schema=_local_memory_delta_schema(LOCAL_OLLAMA_MODEL),
        seed=seed,
    )
    if result.compiler_usage != _local_usage_from_output(events[0].output):
        raise ValueError("compiler usage differs from the raw local model event")
    if result.decision_usage != _local_usage_from_output(events[1].output):
        raise ValueError("decision usage differs from the raw local model event")
    _validate_local_model_event(
        events[1],
        prompt=decision_prompt,
        schema=_local_decision_schema(LOCAL_OLLAMA_MODEL),
        seed=seed,
    )

    try:
        delta = LocalMemoryDeltaWire.model_validate_json(
            events[0].output.completion
        ).to_domain()
    except (ValidationError, ValueError) as error:
        raise ValueError("local compiler preflight output is invalid") from error
    compiler_call = CompilerCall(
        delta=delta,
        usage=Usage(cost_usd=0.0),
        raw_completion=events[0].output.completion,
    )
    if not _compiler_preflight_semantics(compiler_call):
        raise ValueError("local compiler preflight semantics are invalid")
    try:
        decision = LocalDecisionWire.model_validate_json(
            events[1].output.completion
        ).to_domain()
    except (ValidationError, ValueError) as error:
        raise ValueError("local decision preflight output is invalid") from error
    if decision.actions:
        raise ValueError("local decision preflight emitted a false reminder")


def validate_local_preflight_artifact(
    artifact: ArtifactPin,
    *,
    expected_git_commit: str,
    expected_pricing_sha256: str,
    seed: int = 101,
) -> None:
    """Bind local preflight semantics to the exact tracked Inspect log bytes."""

    path = Path(artifact.path)
    if path.suffix != ".eval" or not path.is_file():
        raise ValueError("local model preflight artifact must be an Inspect .eval log")
    content = path.read_bytes()
    if artifact.sha256 != hashlib.sha256(content).hexdigest():
        raise ValueError("local model preflight artifact hash mismatch")
    validate_local_preflight_log(
        read_eval_log(path, resolve_attachments=True),
        expected_git_commit=expected_git_commit,
        expected_pricing_sha256=expected_pricing_sha256,
        seed=seed,
    )


__all__ = [
    "LOCAL_PREFLIGHT_ACTIVE_STATE",
    "LOCAL_PREFLIGHT_EVENT_ID",
    "LOCAL_PREFLIGHT_SAMPLE_ID",
    "local_preflight_event",
    "local_preflight_prompts",
    "validate_local_preflight_artifact",
    "validate_local_preflight_log",
]
