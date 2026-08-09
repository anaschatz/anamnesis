"""Larger prospective real-SDK generalization diagnostic."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib
import json
import os
import sqlite3
from pathlib import Path
from typing import Literal, Self

from pydantic import ConfigDict, Field, model_validator

from anamnesis.action_canonicalizer_v2 import canonicalize_immediate_decision_v2
from anamnesis.openmemory_diagnostic import ExpectedRecallDecision
from anamnesis.openmemory_recall import (
    OpenMemoryMainClientAdapter,
    OpenMemoryRecallIndex,
)
from anamnesis.openmemory_sdk_smoke import (
    _canonical_json_sha256,
    _installed_package_root,
    _installed_runtime_packages,
    load_openmemory_sdk_pin,
    python_source_tree_identity,
)
from anamnesis.openmemory_vllm import (
    HttpOperatorVllmProbe,
    VllmDecisionRuntimePin,
    VllmOpenMemoryAlignedDecisionModel,
    build_openmemory_vllm_user_envelope,
    openmemory_vllm_aligned_decision_contract_sha256,
    openmemory_vllm_aligned_schema_sha256,
)
from anamnesis.openmemory_vllm_run import REPO_ROOT, _verify_source_commit
from anamnesis.openmemory_vllm_v5 import BASE_PIN_PATH
from anamnesis.openmemory_vllm_v6 import _correct
from anamnesis.openmemory_vllm_v8 import V8Arm, V8CaseResult, V8Metrics, _fresh_database
from anamnesis.runner import DecisionRequest
from anamnesis.schema import ObservableEvent, StrictModel, Usage
from anamnesis.vllm_runtime import (
    OpenAIExternalVllmClient,
    VllmProbeSnapshot,
    canonical_json_sha256,
    verify_vllm_artifact,
)

FIXTURE_PATH = REPO_ROOT / "eval/openmemory/real_sdk_generalization_v1.json"
PIN_PATH = REPO_ROOT / "eval/openmemory/real_sdk_generalization_v1.pin.json"
SDK_PIN_PATH = REPO_ROOT / "eval/openmemory_sdk_v1.3.0.pin.json"
CANONICALIZER_PATH = REPO_ROOT / "src/anamnesis/action_canonicalizer_v2.py"
ADAPTER_PATH = REPO_ROOT / "src/anamnesis/openmemory_recall.py"


class _Frozen(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class V10Memory(_Frozen):
    id: str = Field(pattern=r"^omsdk10_mem_[a-z0-9_]+$")
    content: str = Field(min_length=1, max_length=4096)


class V10Case(_Frozen):
    id: str = Field(pattern=r"^omsdk10_[a-z0-9_]+$")
    event: ObservableEvent
    memories: tuple[V10Memory, ...]
    expected_retrieved_memory_ids: tuple[str, ...]
    expected: ExpectedRecallDecision
    helpful_opportunity: bool

    @model_validator(mode="after")
    def valid_identity(self) -> Self:
        ids = tuple(item.id for item in self.memories)
        if len(ids) != len(set(ids)) or any(
            item not in ids for item in self.expected_retrieved_memory_ids
        ):
            raise ValueError("v10 memory identity differs")
        if len(self.expected_retrieved_memory_ids) > 1:
            raise ValueError("v10 freezes top_k=1")
        return self


class V10Fixture(_Frozen):
    schema_version: Literal[1] = 1
    purpose: Literal["openmemory_real_sdk_generalization_v1"]
    hypothesis_test_eligible: Literal[False] = False
    cases: tuple[V10Case, ...]

    @model_validator(mode="after")
    def valid_matrix(self) -> Self:
        if len(self.cases) != 12 or sum(x.helpful_opportunity for x in self.cases) != 6:
            raise ValueError("v10 requires 12 cases and six helpful opportunities")
        return self


class V10Pin(_Frozen):
    schema_version: Literal[1] = 1
    purpose: Literal["openmemory_real_sdk_generalization_v1"]
    hypothesis_test_eligible: Literal[False] = False
    fixture_raw_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    fixture_canonical_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    base_runtime_pin_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    openmemory_sdk_pin_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    openmemory_adapter_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    canonicalizer_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    base_url: Literal["http://127.0.0.1:18006/v1"]
    served_model: Literal["anamnesis-openmemory-v10"]
    server_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    decision_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    response_schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    upstream_revision: Literal["b04bf6e245577d0a024ea37cc02f4187ca7b0ffc"]
    openmemory_source_tree_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    case_count: Literal[12]
    helpful_count: Literal[6]
    top_k: Literal[1]
    expected_model_calls: Literal[24]
    seed: Literal[101]
    temperature: Literal[0.0]
    max_tokens: Literal[256]
    retries_repairs_cache: Literal[0]
    stopping_rule: Literal["one_fresh_paired_real_sdk_v10_run"]


class V10Run(_Frozen):
    schema_version: Literal[1] = 1
    purpose: Literal["openmemory_real_sdk_generalization_v1"] = (
        "openmemory_real_sdk_generalization_v1"
    )
    hypothesis_test_eligible: Literal[False] = False
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    fixture_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pin_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sdk_source_tree_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sdk_runtime_packages_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    cases: tuple[V8CaseResult, ...]
    metrics: V8Metrics
    usage: Usage
    passed: bool

    @model_validator(mode="after")
    def valid_result(self) -> Self:
        usage = Usage()
        for case in self.cases:
            usage = usage.plus(case.baseline.usage).plus(case.recall.usage)
        if len(self.cases) != 12 or usage != self.usage:
            raise ValueError("v10 matrix or usage differs")
        if self.passed != self.metrics.gate_passed:
            raise ValueError("v10 pass flag differs")
        return self


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _server_config(runtime: VllmDecisionRuntimePin) -> dict[str, object]:
    return {
        "artifact_manifest_sha256": runtime.artifact.manifest_sha256,
        "generation_config": "vllm",
        "host": "127.0.0.1",
        "log_deltas": False,
        "log_outputs": False,
        "max_model_len": 4096,
        "max_num_seqs": 1,
        "multimodal_mode": "text-only-compat",
        "paged_attention": True,
        "port": 18006,
        "served_model": runtime.served_model,
        "speculative_decoding": False,
        "structured_output_backend": "xgrammar",
    }


def _load_inputs():
    pin = V10Pin.model_validate_json(PIN_PATH.read_text())
    raw = json.loads(FIXTURE_PATH.read_text())
    sdk = load_openmemory_sdk_pin(SDK_PIN_PATH)
    checks = (
        (_sha256(FIXTURE_PATH), pin.fixture_raw_sha256),
        (canonical_json_sha256(raw), pin.fixture_canonical_sha256),
        (_sha256(BASE_PIN_PATH), pin.base_runtime_pin_sha256),
        (_sha256(SDK_PIN_PATH), pin.openmemory_sdk_pin_sha256),
        (_sha256(ADAPTER_PATH), pin.openmemory_adapter_sha256),
        (_sha256(CANONICALIZER_PATH), pin.canonicalizer_source_sha256),
        (
            openmemory_vllm_aligned_decision_contract_sha256(),
            pin.decision_contract_sha256,
        ),
        (openmemory_vllm_aligned_schema_sha256(), pin.response_schema_sha256),
        (sdk.upstream_revision, pin.upstream_revision),
        (sdk.python_source_tree_sha256, pin.openmemory_source_tree_sha256),
    )
    if any(actual != expected for actual, expected in checks):
        raise ValueError("v10 frozen input differs")
    base = VllmDecisionRuntimePin.model_validate_json(BASE_PIN_PATH.read_text())
    runtime = VllmDecisionRuntimePin.model_validate(
        base.model_dump(mode="json")
        | {
            "base_url": pin.base_url,
            "served_model": pin.served_model,
            "server_config_sha256": pin.server_config_sha256,
            "decision_contract_sha256": pin.decision_contract_sha256,
            "response_schema_sha256": pin.response_schema_sha256,
            "max_tokens": pin.max_tokens,
        }
    )
    return pin, V10Fixture.model_validate(raw), runtime, sdk


def _probe(runtime: VllmDecisionRuntimePin) -> VllmProbeSnapshot:
    return VllmProbeSnapshot(
        health_ok=False,
        base_url=runtime.base_url,
        vllm_version=runtime.vllm_server_version,
        model_ids=(runtime.served_model,),
        model_artifact_manifest_sha256=runtime.artifact.manifest_sha256,
        server_config=_server_config(runtime),
        runtime_packages={item.name: item.version for item in runtime.runtime_packages},
        structured_output_backend="xgrammar",
        generation_config="vllm",
        max_model_len=4096,
        max_num_seqs=1,
        speculative_decoding=False,
    )


async def run_v10(
    *, artifact_root: Path, database_path: Path, api_key: str, source_commit: str
) -> V10Run:
    _verify_source_commit(REPO_ROOT, source_commit)
    pin, fixture, runtime, sdk = _load_inputs()
    if canonical_json_sha256(_server_config(runtime)) != pin.server_config_sha256:
        raise ValueError("v10 server configuration differs")
    verify_vllm_artifact(artifact_root, runtime.artifact)
    database_path = _fresh_database(database_path)
    os.environ["OM_DB_URL"] = f"sqlite:///{database_path}"
    os.environ["OM_EMBED_KIND"] = sdk.embedding_provider
    packages = _installed_runtime_packages(sdk)
    root = _installed_package_root(sdk.package_name)
    source_hash, source_count, source_bytes = python_source_tree_identity(root)
    if (source_hash, source_count, source_bytes) != (
        sdk.python_source_tree_sha256,
        sdk.python_source_file_count,
        sdk.python_source_bytes,
    ):
        raise RuntimeError("v10 installed OpenMemory source differs")
    memory = importlib.import_module("openmemory.client").Memory(user="anamnesis-v10")
    adapter = OpenMemoryMainClientAdapter(
        memory,
        upstream_revision=sdk.upstream_revision,
        database_path=str(database_path),
        embedding_provider=sdk.embedding_provider,
    )
    client = OpenAIExternalVllmClient(
        base_url=runtime.base_url,
        api_key=api_key,
        request_timeout_seconds=runtime.request_timeout_seconds,
    )
    model = VllmOpenMemoryAlignedDecisionModel(
        pin=runtime,
        api_key=api_key,
        artifact_root=artifact_root,
        client=client,
        probe=HttpOperatorVllmProbe(
            declared=_probe(runtime),
            api_key=api_key,
            timeout_seconds=runtime.request_timeout_seconds,
        ),
    )
    results: list[V8CaseResult] = []
    try:
        for case in fixture.cases:
            index = OpenMemoryRecallIndex(
                namespace=case.id, user_id="prospective-v10", client=adapter
            )
            handles = []
            cleanup = False
            try:
                for record in case.memories:
                    handles.append(
                        await index.add(
                            record.content, metadata={"fixture_id": record.id}
                        )
                    )
                recalled = await index.search(case.event.text, limit=1)
                contents = tuple(match.content for match in recalled.matches)
                ids = tuple(
                    next(item.id for item in case.memories if item.content == content)
                    for content in contents
                )
                arms: list[V8Arm] = []
                for arm, recall in (("baseline", None), ("recall", contents)):
                    prompt = build_openmemory_vllm_user_envelope(
                        now=case.event.at.isoformat(),
                        current_event_id=case.event.id,
                        context_events=[case.event],
                        decision_history=[],
                        memory_view=None,
                        retrospective_recall=recall,
                    )
                    call = await model.decide(
                        DecisionRequest(event=case.event, prompt=prompt)
                    )
                    normalized = canonicalize_immediate_decision_v2(
                        event=case.event,
                        retrospective_recall=recall,
                        decision=call.decision,
                    )
                    arms.append(
                        V8Arm(
                            arm=arm,
                            prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                            raw_decision=call.decision,
                            canonical_decision=normalized.decision,
                            changes=normalized.changes,
                            correct=_correct(case.expected, normalized.decision),
                            usage=call.usage,
                            latency_ms=call.latency_ms,
                            audit=model.audits[-1],
                        )
                    )
            finally:
                for handle in handles:
                    if not (await index.delete(handle)).deleted:
                        raise RuntimeError("v10 cleanup was not acknowledged")
                with sqlite3.connect(database_path) as connection:
                    remaining = connection.execute(
                        "SELECT COUNT(*) FROM memories WHERE user_id = ?",
                        (index.scoped_user_id,),
                    ).fetchone()
                cleanup = remaining is not None and remaining[0] == 0
                if not cleanup:
                    raise RuntimeError("v10 scoped cleanup failed")
            results.append(
                V8CaseResult(
                    case_id=case.id,
                    retrieved_memory_ids=ids,
                    retrieved_contents=contents,
                    retrieval_correct=ids == case.expected_retrieved_memory_ids,
                    cleanup_verified=cleanup,
                    baseline=arms[0],
                    recall=arms[1],
                )
            )
    finally:
        await client.aclose()
    baseline = sum(item.baseline.correct for item in results)
    recall = sum(item.recall.correct for item in results)
    gains = sum(
        case.helpful_opportunity
        and not result.baseline.correct
        and result.recall.correct
        for case, result in zip(fixture.cases, results, strict=True)
    )
    regressions = sum(
        not case.helpful_opportunity
        and result.baseline.correct
        and not result.recall.correct
        for case, result in zip(fixture.cases, results, strict=True)
    )
    retrieved = sum(item.retrieval_correct for item in results)
    cleaned = sum(item.cleanup_verified for item in results)
    accepted = sum(
        item.baseline.audit.validation.accepted + item.recall.audit.validation.accepted
        for item in results
    )
    changes = sum(len(x.baseline.changes) + len(x.recall.changes) for x in results)
    passed = (
        retrieved == 12
        and cleaned == 12
        and accepted == 24
        and gains >= 5
        and regressions == 0
        and recall > baseline
    )
    metrics = V8Metrics(
        baseline_correct=baseline,
        recall_correct=recall,
        helpful_gain=gains,
        safety_regressions=regressions,
        retrieval_correct=retrieved,
        cleanup_verified=cleaned,
        accepted_calls=accepted,
        canonicalizer_changes=changes,
        gate_passed=passed,
    )
    usage = Usage()
    for result in results:
        usage = usage.plus(result.baseline.usage).plus(result.recall.usage)
    return V10Run(
        source_commit=source_commit,
        fixture_sha256=pin.fixture_raw_sha256,
        pin_sha256=_sha256(PIN_PATH),
        sdk_source_tree_sha256=source_hash,
        sdk_runtime_packages_sha256=_canonical_json_sha256(packages),
        cases=tuple(results),
        metrics=metrics,
        usage=usage,
        passed=passed,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = asyncio.run(
        run_v10(
            artifact_root=args.artifact_root,
            database_path=args.database,
            api_key=args.api_key,
            source_commit=args.source_commit,
        )
    )
    output = args.output.resolve()
    allowed = (REPO_ROOT / "results/runs/local/openmemory_vllm_v10").resolve()
    if (
        not output.is_relative_to(allowed)
        or output.suffix != ".json"
        or output.exists()
    ):
        raise ValueError("v10 output path differs or already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(result.model_dump_json(indent=2) + "\n")
    return 0 if result.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
