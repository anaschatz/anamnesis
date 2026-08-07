"""Isolated Ollama runtime for zero-API-cost diagnostic experiments.

The hosted OpenAI adapter remains the preregistered headline contract.  This
module deliberately uses separate prompts, wire schemas, endpoint checks and
accounting so a local compatibility concession cannot silently weaken that
contract.
"""

from __future__ import annotations

import asyncio
import hashlib
import http.client
import json
import os
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, Self
from urllib.parse import urlsplit

from inspect_ai.dataset import Sample
from inspect_ai.model import (
    ChatMessageUser,
    ModelCost,
    ModelOutput,
    ResponseSchema,
    get_model,
    get_model_info,
)
from inspect_ai.scorer import Score, Scorer, Target, scorer
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import JSONSchema, json_schema
from pydantic import Field, ValidationError, model_validator

from anamnesis.baselines import (
    AnamnesisMemoryStrategy,
    FastEmbedVectorizer,
    Vectorizer,
    create_strategy,
)
from anamnesis.inspect_adapter import (
    SCENARIO_RUN_STORE_KEY,
    _scenario_from_state,
    _scenario_sha256_from_state,
)
from anamnesis.local_experiment import (
    LOCAL_BASE_URL,
    LOCAL_MODEL_ID,
    LOCAL_OLLAMA_VERSION,
    require_local_only_environment,
)
from anamnesis.local_wire import (
    LOCAL_MEMORY_COMPILER_VERSION,
    LocalAtTriggerWire,
    LocalConditionTransitionTriggerWire,
    LocalMemoryDeltaWire,
    LocalPayloadWire,
    LocalRecurringTriggerWire,
    build_local_memory_compiler_prompt,
)
from anamnesis.memory import (
    AtTrigger,
    CompilerCall,
    CompilerRequest,
    CreateIntent,
)
from anamnesis.oracle import (
    ORACLE_COMPILER_VERSION,
    ORACLE_SYSTEM_NAME,
    OracleAnamnesisMemoryStrategy,
    OracleCompiler,
    OracleCompilerArtifact,
)
from anamnesis.prompts import ACTION_OUTPUT_GUIDE, build_decision_prompt
from anamnesis.runner import (
    DecisionCall,
    DecisionModel,
    DecisionRequest,
    run_scenario,
)
from anamnesis.runtime_contract import anamnesis_runtime_contract
from anamnesis.schema import (
    Decision,
    ObservableEvent,
    ProposedAction,
    RuntimeScenario,
    ScenarioRun,
    StrictModel,
    Usage,
)

LOCAL_RUNTIME_VERSION = "ollama.local.v0.1"
LOCAL_DECISION_VERSION = "ollama.decision.v0.2"
LOCAL_OLLAMA_MODEL = LOCAL_MODEL_ID
LOCAL_OLLAMA_SERVICE_MODEL = "qwen3:4b-instruct"
LOCAL_OLLAMA_BASE_URL = LOCAL_BASE_URL
LOCAL_OLLAMA_PS_URL = "http://127.0.0.1:11434/api/ps"
LOCAL_OLLAMA_MANIFEST_SHA256 = (
    "0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0"
)
LOCAL_OLLAMA_MODEL_BLOB_SHA256 = (
    "85e4a5b7b8ef0e48af0e8658f5aaab9c2324c76c1641493f4d1e25fce54b18b9"
)
LOCAL_OLLAMA_CONTEXT_LENGTH = 4096
LOCAL_OLLAMA_FAMILY = "qwen3"
LOCAL_OLLAMA_PARAMETER_SIZE = "4.0B"
LOCAL_OLLAMA_QUANTIZATION = "Q4_K_M"
LOCAL_NO_CLOUD_ENV = "OLLAMA_NO_CLOUD"
LOCAL_CONTEXT_ENV = "OLLAMA_CONTEXT_LENGTH"
LOCAL_HOST_ENV = "OLLAMA_HOST"
LOCAL_NUM_PARALLEL_ENV = "OLLAMA_NUM_PARALLEL"
LOCAL_MAX_LOADED_MODELS_ENV = "OLLAMA_MAX_LOADED_MODELS"
LOCAL_PREFLIGHT_METADATA_KEY = "anamnesis.local_preflight"
LOCAL_PREFLIGHT_STORE_KEY = "anamnesis.local_preflight"
LOCAL_MODEL_PREFLIGHT_TASK_VERSION = "local.0.1"
LOCAL_MODEL_PREFLIGHT_PURPOSE = "local-model-semantic-preflight"
LOCAL_SCENARIO_TASK_VERSION = "local.0.1"
LOCAL_ZERO_MODEL_COST = ModelCost(
    input=0.0,
    output=0.0,
    input_cache_write=0.0,
    input_cache_read=0.0,
)
LocalSystemName = Literal[
    "no_memory",
    "full_context",
    "vector_rag",
    "anamnesis",
    "anamnesis_oracle_compiler",
]

_HOSTED_NO_ACTION_SENTENCE = (
    '- Return JSON matching the supplied schema. Return {"actions": []} when no '
    "action is due.\n"
)
_LOCAL_NO_ACTION_SENTENCE = (
    "- Return JSON matching the supplied schema. Set mode=no_action with an "
    "empty actions array when no action is due. Set mode=emit only when at "
    "least one reminder is due now.\n"
)
_HOSTED_UNUSED_PAYLOAD_SENTENCE = (
    "- The only optional payload slots are address, build, date, flight, "
    "greenhouse, item, project, quantity, recipient, room, shipment, tank, "
    "and trip. Use null for every unused wire slot.\n"
)
_LOCAL_UNUSED_PAYLOAD_SENTENCE = (
    "- The only optional payload slots are address, build, date, flight, "
    "greenhouse, item, project, quantity, recipient, room, shipment, tank, "
    "and trip. Omit every unused optional payload slot.\n"
)
LOCAL_ACTION_OUTPUT_GUIDE = (
    "Choose exactly one explicit decision mode:\n"
    '- no action: {"mode":"no_action","actions":[]}\n'
    '- emit: {"mode":"emit","actions":[{"kind":"reminder",'
    '"action_key":"creating-event-id","payload":{'
    '"subject":"send the assignment"},"summary":"what to remind",'
    '"evidence_event_ids":["event-id"]}]}\n'
    "For emit, payload.subject is required. Optional payload slots are address, "
    "build, date, flight, greenhouse, item, project, quantity, recipient, room, "
    "shipment, tank, and trip; omit unused slots.\n"
)
LOCAL_STRUCTURED_MEMORY_PRECEDENCE = (
    "Structured-memory precedence (D1):\n"
    "- When Structured memory view is provided by this system, it is "
    "authoritative.\n"
    "- If it contains zero DUE_CANDIDATE blocks, set mode=no_action with an "
    "empty actions array regardless of wording in Available context.\n"
    "- For each DUE_CANDIDATE block, emit exactly one action and copy kind, "
    "action_key, payload, and summary value-for-value from that block's JSON.\n"
    "- Set evidence_event_ids to exactly the block's evidence IDs in displayed "
    "order, followed by Current decision event if it is not already present; "
    "include no other IDs.\n"
    "- A prior EXECUTION suppresses only a DUE_CANDIDATE with the same "
    "occurrence_id. A different occurrence_id or date is a distinct recurring "
    "occurrence even when action_key is the same.\n"
    "- When Structured memory view is not provided by this system, use the "
    "general Rules above.\n"
)


class LocalProposedActionWire(StrictModel):
    """Smaller local action envelope mapped to the unchanged domain action."""

    kind: Literal["reminder"] = "reminder"
    action_key: str
    payload: LocalPayloadWire
    summary: str
    evidence_event_ids: list[str]

    def to_domain(self) -> ProposedAction:
        return ProposedAction(
            kind=self.kind,
            action_key=self.action_key,
            payload=self.payload.to_payload(),
            summary=self.summary,
            evidence_event_ids=self.evidence_event_ids,
        )


class LocalDecisionWire(StrictModel):
    """Decision envelope with an explicit no-action/emit discriminator."""

    mode: Literal["no_action", "emit"]
    actions: list[LocalProposedActionWire]

    @model_validator(mode="after")
    def validate_mode(self) -> Self:
        if self.mode == "no_action" and self.actions:
            raise ValueError("no_action mode requires an empty actions array")
        if self.mode == "emit" and not self.actions:
            raise ValueError("emit mode requires at least one action")
        return self

    def to_domain(self) -> Decision:
        return Decision(actions=[action.to_domain() for action in self.actions])


class LocalOllamaRuntimeAttestation(StrictModel):
    """Static client-side proof that Inspect targets the pinned local route."""

    model: Literal[LOCAL_OLLAMA_MODEL]
    base_url: Literal[LOCAL_OLLAMA_BASE_URL]
    no_cloud: Literal["1"]
    context_length: Literal[LOCAL_OLLAMA_CONTEXT_LENGTH]
    host: Literal["127.0.0.1:11434"]
    num_parallel: Literal[1]
    max_loaded_models: Literal[1]


class LocalLoadedModelAttestation(StrictModel):
    """Evidence from Ollama that the pinned artifact is resident locally."""

    model: Literal[LOCAL_OLLAMA_SERVICE_MODEL]
    digest: Literal[LOCAL_OLLAMA_MANIFEST_SHA256]
    family: Literal[LOCAL_OLLAMA_FAMILY]
    parameter_size: Literal[LOCAL_OLLAMA_PARAMETER_SIZE]
    quantization_level: Literal[LOCAL_OLLAMA_QUANTIZATION]
    context_length: Literal[LOCAL_OLLAMA_CONTEXT_LENGTH]
    size_vram: int = Field(gt=0)
    ollama_version: Literal[LOCAL_OLLAMA_VERSION]


class LocalModelPreflightResult(StrictModel):
    """Parse, semantic, accounting and local-residency compatibility gate."""

    model: Literal[LOCAL_OLLAMA_MODEL]
    runtime: LocalOllamaRuntimeAttestation
    loaded_model: LocalLoadedModelAttestation | None = None
    same_model_for_compiler_and_decision: bool
    compiler_parse_error: bool
    decision_parse_error: bool
    compiler_semantic_valid: bool
    decision_semantic_valid: bool
    compiler_usage: Usage
    decision_usage: Usage
    compiler_usage_complete: bool
    decision_usage_complete: bool
    compiler_cost_complete: bool
    decision_cost_complete: bool
    compiler_latency_ms: float = Field(ge=0)
    decision_latency_ms: float = Field(ge=0)
    residency_probe_latency_ms: float = Field(ge=0)
    passed: bool

    @property
    def setup_latency_ms(self) -> float:
        return (
            self.compiler_latency_ms
            + self.decision_latency_ms
            + self.residency_probe_latency_ms
        )


def build_local_decision_prompt(
    *,
    now: str,
    current_event_id: str,
    context_events: list[ObservableEvent],
    decision_history: list[Any] | None = None,
    memory_view: Any | None = None,
) -> str:
    """Render the common decision information with a local-only wire guide."""

    hosted = build_decision_prompt(
        now=now,
        current_event_id=current_event_id,
        context_events=context_events,
        decision_history=decision_history,
        memory_view=memory_view,
    )
    if (
        _HOSTED_NO_ACTION_SENTENCE not in hosted
        or _HOSTED_UNUSED_PAYLOAD_SENTENCE not in hosted
        or ACTION_OUTPUT_GUIDE not in hosted
    ):
        raise RuntimeError("hosted decision prompt changed without local prompt review")
    return (
        hosted.replace(
            _HOSTED_NO_ACTION_SENTENCE,
            f"{_LOCAL_NO_ACTION_SENTENCE}\n{LOCAL_STRUCTURED_MEMORY_PRECEDENCE}",
        )
        .replace(
            _HOSTED_UNUSED_PAYLOAD_SENTENCE,
            _LOCAL_UNUSED_PAYLOAD_SENTENCE,
        )
        .replace(ACTION_OUTPUT_GUIDE, LOCAL_ACTION_OUTPUT_GUIDE)
    )


def local_decision_contract() -> str:
    """Return the complete local decision prompt/schema fingerprint input."""

    prompt = local_decision_prompt_contract()
    schema = local_decision_schema_contract()
    return f"{LOCAL_DECISION_VERSION}\n{prompt}\n{schema}"


def local_decision_prompt_contract() -> str:
    """Render the static local decision prompt sentinel used for hashing."""

    sentinel = ObservableEvent(
        id="<event-id>",
        at=datetime.fromisoformat("2000-01-01T00:00:00+00:00"),
        kind="user_message",
        text="<event-text>",
    )
    return build_local_decision_prompt(
        now="<current-time>",
        current_event_id="<current-event-id>",
        context_events=[sentinel],
        decision_history=[],
        memory_view=None,
    )


def local_decision_schema_contract() -> str:
    """Serialize the exact Inspect response schema sent to Ollama."""

    return json.dumps(
        _local_decision_schema(LOCAL_OLLAMA_MODEL).model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    )


def local_memory_compiler_prompt_contract() -> str:
    """Render the static local compiler prompt sentinel used for hashing."""

    sentinel = ObservableEvent(
        id="<event-id>",
        at=datetime.fromisoformat("2000-01-01T00:00:00+00:00"),
        kind="user_message",
        text="<event-text>",
    )
    return build_local_memory_compiler_prompt(
        event=sentinel,
        active_state='{"facts":[],"intents":[]}',
    )


def local_memory_compiler_schema_contract() -> str:
    """Serialize the exact Inspect compiler response schema sent to Ollama."""

    return json.dumps(
        _local_memory_delta_schema(LOCAL_OLLAMA_MODEL).model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    )


def local_memory_compiler_transport_contract() -> str:
    """Bind the local compiler prompt to its actual Inspect wire schema."""

    return (
        f"{LOCAL_MEMORY_COMPILER_VERSION}\n"
        f"{local_memory_compiler_prompt_contract()}\n"
        f"{local_memory_compiler_schema_contract()}"
    )


def _local_decision_schema(model_name: str) -> ResponseSchema:
    if model_name != LOCAL_OLLAMA_MODEL:
        raise ValueError("local decision schema requires the pinned Ollama model")
    return ResponseSchema(
        name="anamnesis_local_decision",
        json_schema=json_schema(LocalDecisionWire),
        strict=None,
    )


def _local_memory_delta_schema(model_name: str) -> ResponseSchema:
    if model_name != LOCAL_OLLAMA_MODEL:
        raise ValueError("local compiler schema requires the pinned Ollama model")
    compiler_schema = json_schema(LocalMemoryDeltaWire)
    if compiler_schema.properties is None:
        raise RuntimeError("local compiler schema has no root properties")

    trigger_variants: list[JSONSchema] = []
    for model, trigger_type in (
        (LocalAtTriggerWire, "at"),
        (LocalRecurringTriggerWire, "recurring"),
        (LocalConditionTransitionTriggerWire, "condition_transition"),
    ):
        variant = json_schema(model)
        if variant.properties is None or "type" not in variant.properties:
            raise RuntimeError("local trigger schema is missing its discriminator")
        variant.properties["type"].enum = [trigger_type]
        trigger_variants.append(variant)
    trigger_schema = JSONSchema(anyOf=trigger_variants)

    creates = compiler_schema.properties["intent_creates"].items
    updates = compiler_schema.properties["intent_updates"].items
    if (
        creates is None
        or creates.properties is None
        or updates is None
        or updates.properties is None
    ):
        raise RuntimeError("local compiler intent schemas are incomplete")
    creates.properties["trigger"] = trigger_schema
    updates.properties["trigger"] = JSONSchema(
        anyOf=[*trigger_variants, JSONSchema(type="null")]
    )
    return ResponseSchema(
        name="anamnesis_local_memory_delta",
        json_schema=compiler_schema,
        strict=None,
    )


def _active_model_base_url(active_model: object) -> str:
    api = getattr(active_model, "api", None)
    base_url = getattr(api, "base_url", None)
    client = getattr(api, "client", None)
    client_base_url = getattr(client, "base_url", None)
    if base_url is None or client_base_url is None:
        raise ValueError("active Ollama model is missing its endpoint")
    normalized_api = str(base_url).rstrip("/")
    normalized_client = str(client_base_url).rstrip("/")
    if normalized_api != normalized_client:
        raise ValueError("Ollama provider and client endpoints differ")
    return normalized_api


def _verify_local_ollama_runtime(
    active_model: object,
    model_name: str,
    *,
    environ: Mapping[str, str] | None = None,
) -> LocalOllamaRuntimeAttestation:
    """Fail closed unless Inspect targets the exact local, no-cloud runtime."""

    if model_name != LOCAL_OLLAMA_MODEL:
        raise ValueError("local track requires the pinned Ollama model")
    api = getattr(active_model, "api", None)
    if getattr(api, "service", None) != "Ollama":
        raise ValueError("active model is not the Inspect Ollama provider")
    base_url = _active_model_base_url(active_model)
    parsed = urlsplit(base_url)
    if (
        base_url != LOCAL_OLLAMA_BASE_URL
        or parsed.scheme != "http"
        or parsed.hostname != "127.0.0.1"
        or parsed.port != 11434
        or parsed.path.rstrip("/") != "/v1"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("active Ollama endpoint is not the pinned localhost route")

    active_environ = os.environ if environ is None else environ
    require_local_only_environment(active_environ)
    return LocalOllamaRuntimeAttestation(
        model=LOCAL_OLLAMA_MODEL,
        base_url=LOCAL_OLLAMA_BASE_URL,
        no_cloud="1",
        context_length=LOCAL_OLLAMA_CONTEXT_LENGTH,
        host="127.0.0.1:11434",
        num_parallel=1,
        max_loaded_models=1,
    )


def _verify_local_output_model(output: ModelOutput) -> None:
    if output.model not in {LOCAL_OLLAMA_MODEL, LOCAL_OLLAMA_SERVICE_MODEL}:
        raise ValueError("Ollama response model differs from the pinned local model")


def _verify_effective_zero_model_cost(active_model: object) -> None:
    info = get_model_info(active_model)  # type: ignore[arg-type]
    if info is None or info.cost != LOCAL_ZERO_MODEL_COST:
        raise ValueError("active Inspect model lacks the pinned all-zero local pricing")


def _local_usage_from_output(output: ModelOutput) -> Usage:
    """Account local inference as complete API cost $0, never as unknown cost."""

    _verify_local_output_model(output)
    model_usage = output.usage
    if model_usage is None:
        return Usage(cost_usd=0.0)
    reported_cost = model_usage.total_cost
    if reported_cost not in (None, 0, 0.0):
        raise ValueError("verified local Ollama output reported a non-zero API cost")
    uncached = int(model_usage.input_tokens)
    cache_read = int(getattr(model_usage, "input_tokens_cache_read", 0) or 0)
    cache_write = int(getattr(model_usage, "input_tokens_cache_write", 0) or 0)
    return Usage(
        input_tokens=uncached + cache_read + cache_write,
        uncached_input_tokens=uncached,
        cache_read_input_tokens=cache_read,
        cache_write_input_tokens=cache_write,
        output_tokens=int(model_usage.output_tokens),
        cost_usd=0.0,
    )


def _local_usage_is_complete(output: ModelOutput) -> bool:
    usage = _local_usage_from_output(output)
    return (
        output.usage is not None and usage.input_tokens > 0 and usage.output_tokens > 0
    )


class LocalInspectDecisionModel(DecisionModel):
    """Inspect adapter that can only talk to the pinned local Ollama runtime."""

    def __init__(self, state: TaskState, generate: Generate) -> None:
        active_model = get_model()
        self.name = str(state.model)
        self.runtime_attestation = _verify_local_ollama_runtime(
            active_model,
            self.name,
        )
        _verify_effective_zero_model_cost(active_model)
        self.state = state
        self._generate = generate
        self._response_schema = _local_decision_schema(self.name)

    async def complete_structured(
        self,
        *,
        prompt: str,
        response_schema: ResponseSchema,
    ) -> tuple[ModelOutput, float]:
        """Make one schema-constrained call; the task config forbids retries."""

        self.state.messages = [ChatMessageUser(content=prompt)]
        started = perf_counter()
        self.state = await self._generate(
            self.state,
            tool_calls="none",
            response_schema=response_schema,
        )
        output = self.state.output
        _verify_local_output_model(output)
        return output, max(0.0, (perf_counter() - started) * 1000)

    async def decide(self, request: DecisionRequest) -> DecisionCall:
        output, latency_ms = await self.complete_structured(
            prompt=request.prompt,
            response_schema=self._response_schema,
        )
        try:
            decision = LocalDecisionWire.model_validate_json(
                output.completion
            ).to_domain()
            parse_error = False
        except (ValidationError, ValueError):
            decision = Decision()
            parse_error = True
        usage = _local_usage_from_output(output)
        usage_complete = _local_usage_is_complete(output)
        return DecisionCall(
            decision=decision,
            usage=usage,
            latency_ms=latency_ms,
            parse_error=parse_error,
            raw_completion=output.completion,
            usage_complete=usage_complete,
            cost_complete=usage_complete,
        )


class LocalInspectMemoryCompiler:
    """Local compiler sharing the exact decision model and Inspect session."""

    def __init__(self, model: LocalInspectDecisionModel) -> None:
        self.name = model.name
        self._model = model
        self._response_schema = _local_memory_delta_schema(self.name)

    async def compile(self, request: CompilerRequest) -> CompilerCall:
        output, latency_ms = await self._model.complete_structured(
            prompt=build_local_memory_compiler_prompt(
                event=request.event,
                active_state=request.active_state,
            ),
            response_schema=self._response_schema,
        )
        try:
            delta = LocalMemoryDeltaWire.model_validate_json(
                output.completion
            ).to_domain()
            parse_error = False
        except (ValidationError, ValueError):
            delta = None
            parse_error = True
        usage = _local_usage_from_output(output)
        usage_complete = _local_usage_is_complete(output)
        return CompilerCall(
            delta=delta,
            usage=usage,
            latency_ms=latency_ms,
            parse_error=parse_error,
            raw_completion=output.completion,
            usage_complete=usage_complete,
            cost_complete=usage_complete,
        )


def _loaded_model_from_ps(
    payload: object,
    *,
    ollama_version: str = LOCAL_OLLAMA_VERSION,
) -> LocalLoadedModelAttestation:
    if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
        raise ValueError("Ollama /api/ps returned an invalid response")
    matches = [
        item
        for item in payload["models"]
        if isinstance(item, dict)
        and item.get("model", item.get("name")) == LOCAL_OLLAMA_SERVICE_MODEL
    ]
    if len(matches) != 1:
        raise ValueError("pinned Ollama model is not uniquely loaded locally")
    model = matches[0]
    details = model.get("details")
    if not isinstance(details, dict):
        raise ValueError("Ollama /api/ps omitted loaded-model details")
    return LocalLoadedModelAttestation(
        model=LOCAL_OLLAMA_SERVICE_MODEL,
        digest=model.get("digest"),
        family=details.get("family"),
        parameter_size=details.get("parameter_size"),
        quantization_level=details.get("quantization_level"),
        context_length=model.get("context_length"),
        size_vram=model.get("size_vram"),
        ollama_version=ollama_version,
    )


def probe_loaded_local_model() -> LocalLoadedModelAttestation:
    """Query only Ollama's loopback process endpoint for residency evidence."""

    # HTTPConnection is intentionally used instead of a URL opener so proxy
    # environment variables cannot route this attestation away from loopback.
    connection = http.client.HTTPConnection("127.0.0.1", 11434, timeout=5)
    try:
        connection.request(
            "GET",
            "/api/version",
            headers={"Accept": "application/json"},
        )
        version_response = connection.getresponse()
        if version_response.status != 200:
            raise ValueError("Ollama version probe did not return HTTP 200")
        version_payload = json.loads(version_response.read())
        if not isinstance(version_payload, dict):
            raise ValueError("Ollama version probe returned invalid JSON")
        ollama_version = version_payload.get("version")
        connection.request("GET", "/api/ps", headers={"Accept": "application/json"})
        response = connection.getresponse()
        if response.status != 200:
            raise ValueError("Ollama residency probe did not return HTTP 200")
        payload = json.loads(response.read())
    finally:
        connection.close()
    return _loaded_model_from_ps(payload, ollama_version=ollama_version)


def _compiler_preflight_semantics(call: CompilerCall) -> bool:
    delta = call.delta
    if delta is None or len(delta.mutations) != 1:
        return False
    mutation = delta.mutations[0]
    if not isinstance(mutation, CreateIntent):
        return False
    expected_at = datetime.fromisoformat("2026-01-05T17:00:00+00:00")
    return bool(
        isinstance(mutation.trigger, AtTrigger)
        and mutation.trigger.at == expected_at
        and not mutation.required_conditions
        and not mutation.blockers
        and mutation.action_template.kind == "reminder"
        and mutation.action_template.payload
        == {"subject": "perform compatibility check"}
    )


async def run_local_model_preflight(
    model: LocalInspectDecisionModel,
    *,
    residency_probe: Callable[[], LocalLoadedModelAttestation] = (
        probe_loaded_local_model
    ),
) -> LocalModelPreflightResult:
    """Make one compiler and one contemporaneous decision call, without retry."""

    compiler = LocalInspectMemoryCompiler(model)
    event = ObservableEvent(
        id="local-preflight-event",
        at=datetime.fromisoformat("2026-01-05T09:00:00+00:00"),
        kind="user_message",
        text="At 17:00 today remind me to perform compatibility check.",
    )
    compiler_call = await compiler.compile(
        CompilerRequest(event=event, active_state='{"facts":[],"intents":[]}')
    )
    decision_call = await model.decide(
        DecisionRequest(
            event=event,
            prompt=build_local_decision_prompt(
                now=event.at.isoformat(),
                current_event_id=event.id,
                context_events=[event],
                decision_history=[],
                memory_view=None,
            ),
        )
    )
    probe_started = perf_counter()
    try:
        loaded_model = await asyncio.to_thread(residency_probe)
    except (OSError, ValueError, ValidationError, json.JSONDecodeError):
        loaded_model = None
    probe_latency_ms = max(0.0, (perf_counter() - probe_started) * 1000)
    compiler_semantic_valid = _compiler_preflight_semantics(compiler_call)
    decision_semantic_valid = (
        not decision_call.parse_error and not decision_call.decision.actions
    )
    same_model = compiler.name == model.name and compiler._model is model
    passed = all(
        (
            loaded_model is not None,
            same_model,
            not compiler_call.parse_error,
            not decision_call.parse_error,
            compiler_semantic_valid,
            decision_semantic_valid,
            compiler_call.usage_complete,
            decision_call.usage_complete,
            compiler_call.cost_complete,
            decision_call.cost_complete,
            compiler_call.usage.cost_usd == 0.0,
            decision_call.usage.cost_usd == 0.0,
        )
    )
    return LocalModelPreflightResult(
        model=LOCAL_OLLAMA_MODEL,
        runtime=model.runtime_attestation,
        loaded_model=loaded_model,
        same_model_for_compiler_and_decision=same_model,
        compiler_parse_error=compiler_call.parse_error,
        decision_parse_error=decision_call.parse_error,
        compiler_semantic_valid=compiler_semantic_valid,
        decision_semantic_valid=decision_semantic_valid,
        compiler_usage=compiler_call.usage,
        decision_usage=decision_call.usage,
        compiler_usage_complete=compiler_call.usage_complete,
        decision_usage_complete=decision_call.usage_complete,
        compiler_cost_complete=compiler_call.cost_complete,
        decision_cost_complete=decision_call.cost_complete,
        compiler_latency_ms=compiler_call.latency_ms,
        decision_latency_ms=decision_call.latency_ms,
        residency_probe_latency_ms=probe_latency_ms,
        passed=passed,
    )


class _TaskLocalPreflight:
    """Concurrency-safe live gate performed before any scenario model call."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._result: LocalModelPreflightResult | None = None

    async def ensure(
        self,
        model: LocalInspectDecisionModel,
    ) -> tuple[LocalModelPreflightResult, bool]:
        if self._result is not None:
            return self._result, False
        async with self._lock:
            if self._result is None:
                result = await run_local_model_preflight(model)
                if not result.passed:
                    raise ValueError("live local semantic preflight failed")
                self._result = result
                return result, True
        return self._result, False


def local_model_preflight_sample() -> Sample:
    """Return one synthetic local compatibility case, never benchmark data."""

    return Sample(
        id="local-model-preflight-v0",
        input="Check local structured output, semantics, residency and accounting.",
        target="pass",
    )


@solver
def local_model_preflight_solver() -> Solver:
    """Standalone diagnostic gate that exposes failure instead of raising."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        model = LocalInspectDecisionModel(state, generate)
        result = await run_local_model_preflight(model)
        state = model.state
        serialized = result.model_dump(mode="json")
        state.metadata[LOCAL_PREFLIGHT_METADATA_KEY] = serialized
        state.store.set(LOCAL_PREFLIGHT_STORE_KEY, serialized)
        state.output = ModelOutput.from_content(
            model=model.name,
            content=result.model_dump_json(),
        )
        return state

    return solve


@scorer(metrics=[])
def local_model_preflight_scorer() -> Scorer:
    """Score local compatibility only, never reminder quality."""

    async def score(state: TaskState, target: Target) -> Score:
        result = LocalModelPreflightResult.model_validate_json(state.output.completion)
        return Score(
            value=1 if result.passed else 0,
            answer=result.model,
            explanation=(
                "local compiler semantics, no-action decision, residency and "
                "zero-cost accounting passed"
                if result.passed
                else "local model failed one or more semantic preflight checks"
            ),
        )

    return score


def local_system_config_sha256(
    *,
    system: LocalSystemName,
    top_k: int = 5,
    embedding_model: str = "BAAI/bge-small-en-v1.5",
    embedding_repository: str = "qdrant/bge-small-en-v1.5-onnx-q",
    embedding_revision: str | None = None,
    embedding_artifact_sha256: str | None = None,
    pricing_config_sha256: str | None = None,
    oracle_annotations_sha256: str | None = None,
) -> str:
    """Fingerprint all local knobs that can change a diagnostic result."""

    payload: dict[str, object] = {
        "runtime_version": LOCAL_RUNTIME_VERSION,
        "model": LOCAL_OLLAMA_MODEL,
        "model_manifest_sha256": LOCAL_OLLAMA_MANIFEST_SHA256,
        "model_blob_sha256": LOCAL_OLLAMA_MODEL_BLOB_SHA256,
        "ollama_version": LOCAL_OLLAMA_VERSION,
        "model_family": LOCAL_OLLAMA_FAMILY,
        "model_parameter_size": LOCAL_OLLAMA_PARAMETER_SIZE,
        "model_quantization": LOCAL_OLLAMA_QUANTIZATION,
        "base_url": LOCAL_OLLAMA_BASE_URL,
        "no_cloud": "1",
        "server_bind": "127.0.0.1:11434",
        "context_length": LOCAL_OLLAMA_CONTEXT_LENGTH,
        "num_parallel": 1,
        "max_loaded_models": 1,
        "temperature": 0.0,
        "system": system,
        "decision_contract_sha256": hashlib.sha256(
            local_decision_contract().encode()
        ).hexdigest(),
        "pricing_config_sha256": pricing_config_sha256,
    }
    if system == "vector_rag":
        payload.update(
            top_k=top_k,
            embedding_model=embedding_model,
            embedding_repository=embedding_repository,
            embedding_revision=embedding_revision,
            embedding_artifact_sha256=embedding_artifact_sha256,
        )
    if system in {"anamnesis", ORACLE_SYSTEM_NAME}:
        payload["deterministic_memory"] = anamnesis_runtime_contract()
    if system == "anamnesis":
        payload["memory_compiler_sha256"] = hashlib.sha256(
            local_memory_compiler_transport_contract().encode()
        ).hexdigest()
    if system == ORACLE_SYSTEM_NAME:
        if oracle_annotations_sha256 is None or (
            len(oracle_annotations_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in oracle_annotations_sha256
            )
        ):
            raise ValueError(
                "oracle system hash requires a valid oracle_annotations_sha256"
            )
        payload.update(
            compiler_mode="oracle",
            oracle_compiler_version=ORACLE_COMPILER_VERSION,
            oracle_annotations_sha256=oracle_annotations_sha256,
            same_model_for_compiler_and_decision=False,
            shared_decision_contract_version=LOCAL_DECISION_VERSION,
        )
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()


def _validate_oracle_scenario_run(
    scenario: RuntimeScenario,
    run: ScenarioRun,
) -> None:
    """Fail closed unless oracle compilation stayed accepted and token-free."""

    if run.system != ORACLE_SYSTEM_NAME:
        raise ValueError("oracle runtime produced the wrong system identity")
    if len(run.checkpoints) != len(scenario.events):
        raise ValueError("oracle runtime checkpoint coverage differs from scenario")
    if run.compiler_parse_errors:
        raise ValueError("oracle runtime recorded a compiler parse error")
    if not run.usage_complete or not run.cost_complete:
        raise ValueError("oracle runtime usage and cost must be complete")
    zero_usage = Usage(cost_usd=0.0)
    if run.compiler_usage != zero_usage:
        raise ValueError("oracle compiler usage must be exactly zero and complete")
    if run.usage != run.decision_usage:
        raise ValueError("oracle total usage must equal decision-only usage")
    for event, checkpoint in zip(scenario.events, run.checkpoints, strict=True):
        expected_call = event.kind != "clock_tick"
        if checkpoint.compiler_called is not expected_call:
            raise ValueError("oracle compiler call coverage differs from scenario")
        if not expected_call:
            continue
        if checkpoint.compiler_parse_error:
            raise ValueError("oracle checkpoint recorded a compiler parse error")
        if checkpoint.memory_delta_accepted is not True:
            raise ValueError("oracle memory delta was not accepted")
        if checkpoint.compiler_usage != zero_usage:
            raise ValueError("oracle checkpoint compiler usage must be exactly zero")


@solver
def local_scenario_solver(
    system: LocalSystemName,
    *,
    repetition: int = 1,
    seed: int | None = None,
    top_k: int = 5,
    embedding_model: str = "BAAI/bge-small-en-v1.5",
    embedding_repository: str = "qdrant/bge-small-en-v1.5-onnx-q",
    embedding_revision: str | None = None,
    embedding_snapshot_path: str | None = None,
    expected_model: str = LOCAL_OLLAMA_MODEL,
    expected_system_config_sha256: str | None = None,
    expected_embedding_artifact_sha256: str | None = None,
    manifest_sha256: str | None = None,
    pricing_config_sha256: str | None = None,
    oracle_annotations_path: str | None = None,
    oracle_annotations_sha256: str | None = None,
) -> Solver:
    """Run one local diagnostic system after a live semantic preflight."""

    if repetition < 1:
        raise ValueError("repetition must be at least 1")
    if expected_model != LOCAL_OLLAMA_MODEL:
        raise ValueError("local solver expected_model must be the pinned Ollama tag")
    if system == ORACLE_SYSTEM_NAME:
        if oracle_annotations_path is None or oracle_annotations_sha256 is None:
            raise ValueError("local oracle solver requires a pinned oracle artifact")
        oracle_path = Path(oracle_annotations_path)
        if not oracle_path.is_absolute() or not oracle_path.is_file():
            raise ValueError("local oracle solver requires an absolute artifact path")
    elif oracle_annotations_path is not None or oracle_annotations_sha256 is not None:
        raise ValueError("oracle inputs are only valid for the oracle system")

    shared_vectorizer: Vectorizer | None = None
    if system == "vector_rag":
        if embedding_revision is None:
            raise ValueError("local vector_rag requires an exact embedding revision")
        shared_vectorizer = FastEmbedVectorizer(
            model_name=embedding_model,
            repository=embedding_repository,
            revision=embedding_revision,
            snapshot_path=embedding_snapshot_path,
        )
    task_preflight = _TaskLocalPreflight()

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        decision_model = LocalInspectDecisionModel(state, generate)
        if decision_model.name != expected_model:
            raise ValueError("active model differs from the frozen local manifest")
        preflight, performed_here = await task_preflight.ensure(decision_model)
        state = decision_model.state
        serialized_preflight = preflight.model_dump(mode="json")
        state.metadata[LOCAL_PREFLIGHT_METADATA_KEY] = serialized_preflight
        state.store.set(LOCAL_PREFLIGHT_STORE_KEY, serialized_preflight)
        setup_latency_ms = preflight.setup_latency_ms if performed_here else 0.0

        scenario = _scenario_from_state(state)
        if isinstance(shared_vectorizer, FastEmbedVectorizer):
            setup_latency_ms += shared_vectorizer.warmup()
            if (
                expected_embedding_artifact_sha256 is not None
                and shared_vectorizer.artifact_sha256
                != expected_embedding_artifact_sha256
            ):
                raise ValueError(
                    "local embedding snapshot differs from the frozen manifest"
                )
        oracle_compiler: OracleCompiler | None = None
        if system == "anamnesis":
            strategy = AnamnesisMemoryStrategy(
                compiler=LocalInspectMemoryCompiler(decision_model)
            )
        elif system == ORACLE_SYSTEM_NAME:
            if oracle_annotations_path is None or oracle_annotations_sha256 is None:
                raise ValueError("oracle artifact was not bound at task construction")
            oracle_bytes = Path(oracle_annotations_path).read_bytes()
            if hashlib.sha256(oracle_bytes).hexdigest() != oracle_annotations_sha256:
                raise ValueError(
                    "oracle artifact bytes changed after task construction"
                )
            oracle_artifact = OracleCompilerArtifact.model_validate_json(oracle_bytes)
            oracle_compiler = OracleCompiler(oracle_artifact, scenario)
            strategy = OracleAnamnesisMemoryStrategy(compiler=oracle_compiler)
        else:
            strategy = create_strategy(
                system,
                vectorizer=shared_vectorizer,
                top_k=top_k,
            )
        system_config_sha256 = local_system_config_sha256(
            system=system,
            top_k=top_k,
            embedding_model=embedding_model,
            embedding_repository=embedding_repository,
            embedding_revision=embedding_revision,
            embedding_artifact_sha256=(
                shared_vectorizer.artifact_sha256
                if isinstance(shared_vectorizer, FastEmbedVectorizer)
                else None
            ),
            pricing_config_sha256=pricing_config_sha256,
            oracle_annotations_sha256=oracle_annotations_sha256,
        )
        if (
            expected_system_config_sha256 is not None
            and system_config_sha256 != expected_system_config_sha256
        ):
            raise ValueError(
                "runtime system configuration differs from the local manifest"
            )
        run = await run_scenario(
            scenario=scenario,
            strategy=strategy,
            model=decision_model,
            repetition=repetition,
            seed=seed,
            scenario_sha256_override=_scenario_sha256_from_state(state),
            system_config_sha256=system_config_sha256,
            manifest_sha256=manifest_sha256,
            pricing_config_sha256=pricing_config_sha256,
            setup_latency_ms=setup_latency_ms,
            decision_prompt_builder=build_local_decision_prompt,
            decision_prompt_contract=local_decision_contract(),
            decision_prompt_version=LOCAL_DECISION_VERSION,
        )
        if oracle_compiler is not None:
            oracle_compiler.assert_complete()
            _validate_oracle_scenario_run(scenario, run)
        if run.usage.cost_usd != 0.0 or not run.cost_complete:
            raise ValueError("local run did not preserve complete zero-cost accounting")
        state = decision_model.state
        state.store.set(SCENARIO_RUN_STORE_KEY, run.model_dump(mode="json"))
        state.output = ModelOutput.from_content(
            model=decision_model.name,
            content=run.model_dump_json(),
        )
        return state

    return solve


def verify_zero_local_pricing(path: str | Path) -> str:
    """Verify the tracked pricing file declares exactly one all-zero local entry."""

    pricing_path = Path(path)
    content = pricing_path.read_bytes()
    raw = json.loads(content)
    if not isinstance(raw, dict) or set(raw) != {LOCAL_OLLAMA_MODEL}:
        raise ValueError("local pricing config must contain exactly the pinned model")
    entry = raw[LOCAL_OLLAMA_MODEL]
    expected_fields = {
        "input",
        "output",
        "input_cache_write",
        "input_cache_read",
    }
    if not isinstance(entry, dict) or set(entry) != expected_fields:
        raise ValueError("local pricing config has an invalid Inspect cost shape")
    if any(value != 0 and value != 0.0 for value in entry.values()):
        raise ValueError("local pricing config contains a non-zero API price")
    return hashlib.sha256(content).hexdigest()


__all__ = [
    "LOCAL_CONTEXT_ENV",
    "LOCAL_DECISION_VERSION",
    "LOCAL_STRUCTURED_MEMORY_PRECEDENCE",
    "LOCAL_MODEL_PREFLIGHT_PURPOSE",
    "LOCAL_MODEL_PREFLIGHT_TASK_VERSION",
    "LOCAL_NO_CLOUD_ENV",
    "LOCAL_OLLAMA_BASE_URL",
    "LOCAL_OLLAMA_CONTEXT_LENGTH",
    "LOCAL_OLLAMA_MANIFEST_SHA256",
    "LOCAL_OLLAMA_MODEL",
    "LOCAL_OLLAMA_MODEL_BLOB_SHA256",
    "LOCAL_PREFLIGHT_METADATA_KEY",
    "LOCAL_SCENARIO_TASK_VERSION",
    "LocalDecisionWire",
    "LocalInspectDecisionModel",
    "LocalInspectMemoryCompiler",
    "LocalSystemName",
    "LocalLoadedModelAttestation",
    "LocalModelPreflightResult",
    "build_local_decision_prompt",
    "local_decision_contract",
    "local_decision_prompt_contract",
    "local_decision_schema_contract",
    "local_memory_compiler_prompt_contract",
    "local_memory_compiler_schema_contract",
    "local_memory_compiler_transport_contract",
    "local_model_preflight_sample",
    "local_model_preflight_scorer",
    "local_model_preflight_solver",
    "local_scenario_solver",
    "local_system_config_sha256",
    "probe_loaded_local_model",
    "run_local_model_preflight",
    "verify_zero_local_pricing",
]
