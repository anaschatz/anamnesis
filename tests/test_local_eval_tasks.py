from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest
from inspect_ai.model import get_model_info

from anamnesis.local_experiment import LOCAL_PRICING_SHA256, LocalExperimentManifest
from anamnesis.local_runtime import LOCAL_OLLAMA_MODEL, LOCAL_ZERO_MODEL_COST


def test_local_task_registry_names_are_isolated_and_frozen() -> None:
    module = runpy.run_path("eval/anamnesis_local_eval.py")
    expected = {
        "local_model_preflight",
        "local_no_memory",
        "local_full_context",
        "local_vector_rag",
        "local_anamnesis",
    }

    assert {
        module[name].__registry_info__.name  # type: ignore[attr-defined]
        for name in expected
    } == expected
    task = module["local_model_preflight"]()
    model_info = get_model_info(LOCAL_OLLAMA_MODEL)
    assert model_info is not None
    assert model_info.cost == LOCAL_ZERO_MODEL_COST
    assert task.version == "local.0.1"
    assert task.config.temperature == 0.0
    assert task.config.seed == 101
    assert task.config.cache is False
    assert task.config.max_retries == 0
    assert task.config.max_connections == 1
    assert task.config.adaptive_connections is False
    assert task.metadata["hypothesis_test_eligible"] is False
    assert task.metadata["pricing_config_sha256"] == LOCAL_PRICING_SHA256


def test_local_scenario_tasks_fail_closed_without_frozen_manifest() -> None:
    module = runpy.run_path("eval/anamnesis_local_eval.py")

    for name in (
        "local_no_memory",
        "local_full_context",
        "local_vector_rag",
        "local_anamnesis",
    ):
        with pytest.raises(ValueError, match="frozen local manifest"):
            module[name]()


def test_local_scenario_task_metadata_binds_ordered_smoke_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = runpy.run_path("eval/anamnesis_local_eval.py")
    raw = json.loads(
        Path("eval/local_experiment_manifest.template.json").read_text(encoding="utf-8")
    )
    raw["system_config_sha256"] = {
        "no_memory": "a" * 64,
        "full_context": "b" * 64,
        "vector_rag": "c" * 64,
        "anamnesis": "d" * 64,
    }
    manifest = LocalExperimentManifest.model_validate(raw)
    dataset_path = Path("eval/scenarios/smoke.jsonl").resolve()
    system_task = module["_system_task"]
    monkeypatch.setitem(
        system_task.__globals__,  # type: ignore[attr-defined]
        "_validated_local_manifest",
        lambda *args, **kwargs: (manifest, "e" * 64, dataset_path),
    )

    task = system_task(
        "no_memory",
        seed=101,
        repetition=1,
        manifest="ignored-by-test.json",
        ollama_models_dir="/ignored/by/test",
    )
    expected_ids = [
        json.loads(line)["id"]
        for line in dataset_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert task.metadata["dataset_scenario_count"] == 10
    assert task.metadata["dataset_sample_ids"] == expected_ids
    assert task.metadata["dataset_split"] == "smoke"
    assert task.metadata["claim_scope"] == "diagnostic_development_only"
    assert task.metadata["hypothesis_test_eligible"] is False
    assert task.metadata["provider_api_cost_usd"] == 0.0
    assert task.metadata["live_semantic_preflight_required"] is True
