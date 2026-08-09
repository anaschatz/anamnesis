"""Strict report for the canonicalizer-v2 real OpenMemory SDK diagnostic."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import subprocess
from pathlib import Path

from anamnesis.action_canonicalizer_v2 import canonicalize_immediate_decision_v2
from anamnesis.openmemory_vllm import (
    build_openmemory_vllm_aligned_request,
    build_openmemory_vllm_user_envelope,
)
from anamnesis.openmemory_vllm_run import REPO_ROOT
from anamnesis.openmemory_vllm_v6 import _correct
from anamnesis.openmemory_vllm_v9 import (
    FIXTURE_PATH,
    PIN_PATH,
    SDK_PIN_PATH,
    V9Run,
    _load_inputs,
)
from anamnesis.runner import DecisionRequest
from anamnesis.vllm_runtime import canonical_json_sha256

TITLE = "Real OpenMemory SDK v9 canonicalizer-v2 diagnostic — not a hypothesis test"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_run(path: Path) -> V9Run:
    result = V9Run.model_validate_json(path.read_text())
    bound_paths = (
        "eval/openmemory/real_sdk_canonicalizer_v2.v1.json",
        "eval/openmemory/real_sdk_canonicalizer_v2.v1.pin.json",
        "eval/openmemory_sdk_v1.3.0.pin.json",
        "src/anamnesis/action_canonicalizer_v2.py",
        "src/anamnesis/openmemory_recall.py",
        "src/anamnesis/openmemory_vllm_v9.py",
    )
    source_check = subprocess.run(
        ["git", "cat-file", "-e", f"{result.source_commit}^{{commit}}"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
    )
    drift_check = subprocess.run(
        ["git", "diff", "--quiet", result.source_commit, "--", *bound_paths],
        cwd=REPO_ROOT,
        check=False,
    )
    if source_check.returncode or drift_check.returncode:
        raise ValueError("v9 source commit or bound source bytes differ")
    pin, fixture, runtime, sdk_pin = _load_inputs()
    if result.pin_sha256 != _sha256(PIN_PATH):
        raise ValueError("v9 pin hash differs")
    if result.sdk_source_tree_sha256 != sdk_pin.python_source_tree_sha256:
        raise ValueError("v9 SDK source identity differs")
    for case, observed in zip(fixture.cases, result.cases, strict=True):
        if observed.case_id != case.id:
            raise ValueError("v9 case order differs")
        if observed.retrieved_memory_ids != case.expected_retrieved_memory_ids:
            raise ValueError("v9 retrieval differs")
        expected_contents = tuple(
            next(item.content for item in case.memories if item.id == memory_id)
            for memory_id in case.expected_retrieved_memory_ids
        )
        if observed.retrieved_contents != expected_contents:
            raise ValueError("v9 retrieved content differs")
        if not observed.retrieval_correct or not observed.cleanup_verified:
            raise ValueError("v9 retrieval or SDK cleanup failed")
        for arm, recall in (
            (observed.baseline, None),
            (observed.recall, observed.retrieved_contents),
        ):
            prompt = build_openmemory_vllm_user_envelope(
                now=case.event.at.isoformat(),
                current_event_id=case.event.id,
                context_events=[case.event],
                decision_history=[],
                memory_view=None,
                retrospective_recall=recall,
            )
            request = build_openmemory_vllm_aligned_request(
                runtime, DecisionRequest(event=case.event, prompt=prompt)
            )
            normalized = canonicalize_immediate_decision_v2(
                event=case.event,
                retrospective_recall=recall,
                decision=arm.raw_decision,
            )
            checks = (
                arm.prompt_sha256 == hashlib.sha256(prompt.encode()).hexdigest(),
                arm.audit.request_sha256 == canonical_json_sha256(request),
                arm.canonical_decision == normalized.decision,
                arm.changes == normalized.changes,
                arm.correct == _correct(case.expected, normalized.decision),
                arm.audit.usage == arm.usage,
                arm.audit.validation.accepted,
            )
            if not all(checks):
                raise ValueError("v9 request, normalization, score, or usage differs")
    if result.metrics.cleanup_verified != 6:
        raise ValueError("v9 aggregate SDK cleanup differs")
    return result


def _csv(result: V9Run) -> bytes:
    fields = [
        "passed",
        "cases",
        "retrieval_correct",
        "cleanup_verified",
        "baseline_correct",
        "recall_correct",
        "helpful_gain",
        "safety_regressions",
        "accepted_calls",
        "canonicalizer_changes",
        "input_tokens",
        "output_tokens",
        "provider_api_cost_usd",
    ]
    row = result.metrics.model_dump() | {
        "passed": str(result.passed).lower(),
        "cases": len(result.cases),
        "input_tokens": result.usage.input_tokens,
        "output_tokens": result.usage.output_tokens,
        "provider_api_cost_usd": result.usage.cost_usd,
    }
    row.pop("gate_passed")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerow(row)
    return stream.getvalue().encode()


def _markdown(result: V9Run) -> bytes:
    rows = "\n".join(
        f"| {case.case_id} | {','.join(case.retrieved_memory_ids) or 'none'} | "
        f"{str(case.baseline.correct).lower()} | {str(case.recall.correct).lower()} | "
        f"{str(case.cleanup_verified).lower()} | "
        f"{len(case.baseline.changes)}/{len(case.recall.changes)} |"
        for case in result.cases
    )
    return f"""# {TITLE}

The one authorized fresh paired run **{"PASS" if result.passed else "FAIL"}**.
Both arms used the same frozen model, schema, prompt, and additive canonicalizer
v2. Recall used the official CaviraOSS OpenMemory Python SDK through local SQLite
and deterministic synthetic embeddings. OpenMemory remained non-authoritative;
provider identifiers stayed opaque and every scoped record was verified deleted.

| Case | Retrieved memory | Baseline correct | Recall correct | Cleanup | Changes B/R |
|---|---|---:|---:|---:|---:|
{rows}

Retrieval {result.metrics.retrieval_correct}/6; cleanup
{result.metrics.cleanup_verified}/6; baseline {result.metrics.baseline_correct}/6;
recall {result.metrics.recall_correct}/6; helpful gain
{result.metrics.helpful_gain}/3; safety regressions
{result.metrics.safety_regressions}; accepted calls
{result.metrics.accepted_calls}/12; canonicalizer transformations
{result.metrics.canonicalizer_changes}. Usage: {result.usage.input_tokens} input
and {result.usage.output_tokens} output tokens at `${result.usage.cost_usd}`
provider API cost. OpenMemory synthetic-embedding operations report no token or
cost accounting; electricity, hardware, and human effort are unmeasured.

This is a prospective development diagnostic, not a hypothesis test, latency
benchmark, or promotion of OpenMemory to authoritative temporal memory.
""".encode()


def write_report(
    *, run_path: Path, csv_path: Path, markdown_path: Path, provenance_path: Path
) -> bool:
    run_path = run_path.resolve()
    if not run_path.is_relative_to(
        (REPO_ROOT / "results/runs/local/openmemory_vllm_v9").resolve()
    ):
        raise ValueError("v9 run path differs")
    result = validate_run(run_path)
    csv_data, markdown_data = _csv(result), _markdown(result)
    for output in (csv_path, markdown_path, provenance_path):
        resolved = output.resolve()
        if not resolved.is_relative_to((REPO_ROOT / "results").resolve()):
            raise ValueError("v9 report outputs must stay under results")
        if resolved.exists():
            raise FileExistsError("refusing to overwrite v9 report output")
    provenance = {
        "schema_version": 1,
        "title": TITLE,
        "hypothesis_test_eligible": False,
        "source_commit": result.source_commit,
        "openmemory": {
            "upstream_revision": "b04bf6e245577d0a024ea37cc02f4187ca7b0ffc",
            "source_tree_sha256": result.sdk_source_tree_sha256,
            "runtime_packages_sha256": result.sdk_runtime_packages_sha256,
            "authoritative": False,
            "supports_action_evidence": False,
            "scoped_cleanup_verified": result.metrics.cleanup_verified,
        },
        "inputs": {
            "run": {
                "path": run_path.relative_to(REPO_ROOT).as_posix(),
                "sha256": _sha256(run_path),
            },
            "fixture": {
                "path": FIXTURE_PATH.relative_to(REPO_ROOT).as_posix(),
                "sha256": _sha256(FIXTURE_PATH),
            },
            "pin": {
                "path": PIN_PATH.relative_to(REPO_ROOT).as_posix(),
                "sha256": _sha256(PIN_PATH),
            },
            "sdk_pin": {
                "path": SDK_PIN_PATH.relative_to(REPO_ROOT).as_posix(),
                "sha256": _sha256(SDK_PIN_PATH),
            },
        },
        "outputs": {
            "csv": {
                "path": csv_path.resolve().relative_to(REPO_ROOT).as_posix(),
                "sha256": hashlib.sha256(csv_data).hexdigest(),
            },
            "markdown": {
                "path": markdown_path.resolve().relative_to(REPO_ROOT).as_posix(),
                "sha256": hashlib.sha256(markdown_data).hexdigest(),
            },
        },
    }
    csv_path.write_bytes(csv_data)
    markdown_path.write_bytes(markdown_data)
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    return result.passed


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("run", "csv", "markdown", "provenance"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    return (
        0
        if write_report(
            run_path=args.run,
            csv_path=args.csv,
            markdown_path=args.markdown,
            provenance_path=args.provenance,
        )
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
