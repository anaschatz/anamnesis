"""Reproducibility contract for the zero-provider-cost local experiment track."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator

from anamnesis.experiment import (
    FINAL_SYSTEMS,
    SIMPLE_SYSTEMS,
    ArtifactPin,
    EmbeddingPin,
)
from anamnesis.schema import StrictModel

LOCAL_MODEL_ID = "ollama/qwen3:4b-instruct"
LOCAL_BASE_URL = "http://127.0.0.1:11434/v1"
LOCAL_MODEL_ARTIFACT_PATH = "eval/ollama_qwen3_4b_instruct.pin.json"
LOCAL_PRICING_PATH = "eval/local_model_costs.json"
LOCAL_PRICING_SHA256 = (
    "c185e2fad06d6bd2abaaf0be81a1720fc245555fa2a477c1b1bea558b28c2f74"
)
LOCAL_OLLAMA_VERSION = "0.31.1"

SHA256_PATTERN = r"^[0-9a-f]{64}$"
LOCAL_SMOKE_SYSTEMS = FINAL_SYSTEMS
LOCAL_SMOKE_SEEDS = [101]


class OllamaBlobPin(StrictModel):
    """One content-addressed object referenced by an Ollama manifest."""

    role: Literal["config", "model", "template", "license", "params"]
    media_type: str = Field(min_length=1)
    sha256: str = Field(pattern=SHA256_PATTERN)
    size_bytes: int = Field(ge=0)


class OllamaArtifactPin(StrictModel):
    """Portable identity for a local Ollama model without a machine path."""

    schema_version: Literal[1] = 1
    model: str = Field(min_length=1)
    manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    blobs: list[OllamaBlobPin] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_blob_set(self) -> Self:
        roles = [blob.role for blob in self.blobs]
        expected = {"config", "model", "template", "license", "params"}
        if set(roles) != expected or len(roles) != len(expected):
            raise ValueError(
                "Ollama artifact must pin config, model, template, license, "
                "and params exactly once"
            )
        digests = [blob.sha256 for blob in self.blobs]
        if len(digests) != len(set(digests)):
            raise ValueError("Ollama artifact blob hashes must be unique")
        return self


class LocalProviderRoute(StrictModel):
    """The only provider route permitted for the local experiment track."""

    provider: Literal["ollama"] = "ollama"
    base_url: Literal["http://127.0.0.1:11434/v1"] = LOCAL_BASE_URL
    cloud_disabled_environment: Literal["OLLAMA_NO_CLOUD=1"] = "OLLAMA_NO_CLOUD=1"
    server_bind_environment: Literal["OLLAMA_HOST=127.0.0.1:11434"] = (
        "OLLAMA_HOST=127.0.0.1:11434"
    )
    api_key_required: Literal[False] = False


class LocalModelPin(StrictModel):
    """Exact local model, provider route, artifact and preflight inputs."""

    snapshot: Literal["ollama/qwen3:4b-instruct"] = LOCAL_MODEL_ID
    artifact: ArtifactPin
    schema_constrained_output: Literal[True] = True
    live_preflight_required: Literal[True] = True
    same_model_for_compiler_and_decision: Literal[True] = True
    provider: LocalProviderRoute = Field(default_factory=LocalProviderRoute)
    pricing: ArtifactPin
    preflight: ArtifactPin


class LocalHardwarePin(StrictModel):
    """Non-identifying hardware class used for comparable latency results."""

    architecture: Literal["arm64"] = "arm64"
    chip: Literal["Apple M3"] = "Apple M3"
    memory_bytes: Literal[17179869184] = 17179869184


class LocalRuntimePin(StrictModel):
    """Local inference runtime settings that may affect outputs or latency."""

    ollama_version: Literal["0.31.1"] = LOCAL_OLLAMA_VERSION
    inference_backend: Literal["Metal"] = "Metal"
    context_window_tokens: Literal[4096] = 4096
    context_length_environment: Literal["OLLAMA_CONTEXT_LENGTH=4096"] = (
        "OLLAMA_CONTEXT_LENGTH=4096"
    )
    num_parallel_environment: Literal["OLLAMA_NUM_PARALLEL=1"] = "OLLAMA_NUM_PARALLEL=1"
    max_loaded_models_environment: Literal["OLLAMA_MAX_LOADED_MODELS=1"] = (
        "OLLAMA_MAX_LOADED_MODELS=1"
    )
    process_attestation_endpoint: Literal["http://127.0.0.1:11434/api/ps"] = (
        "http://127.0.0.1:11434/api/ps"
    )
    process_attestation_required: Literal[True] = True
    local_only: Literal[True] = True
    setup_latency_reported_separately: Literal[True] = True


class LocalCostPolicy(StrictModel):
    """Scope of the zero-cost claim for local inference."""

    headline_metric: Literal["provider_api_cost_usd"] = "provider_api_cost_usd"
    provider_api_price_usd: Literal[0.0] = 0.0
    electricity_measured: Literal[False] = False
    hardware_amortization_measured: Literal[False] = False


class LocalExecutionPolicy(StrictModel):
    """Frozen execution settings for local diagnostic or measured matrices."""

    temperature: float = Field(default=0.0, ge=0, le=0)
    seeds: list[int] = Field(default_factory=lambda: list(LOCAL_SMOKE_SEEDS))
    repetitions: int = Field(default=1, ge=1, le=3)
    response_cache: Literal[False] = False
    max_samples: Literal[1] = 1
    concurrency: Literal[1] = 1
    max_retries: Literal[0] = 0
    structured_output_repair_calls: Literal[0] = 0
    log_model_api: Literal[True] = True
    warmup_policy: Literal["one_unmeasured_call_per_schema"] = (
        "one_unmeasured_call_per_schema"
    )
    warmup_latency_in_headline: Literal[False] = False

    @model_validator(mode="after")
    def validate_seeds(self) -> Self:
        if len(self.seeds) != self.repetitions or len(set(self.seeds)) != len(
            self.seeds
        ):
            raise ValueError("execution seeds must be unique and match repetitions")
        return self


class LocalExperimentManifest(StrictModel):
    """A local-only experiment declaration kept separate from the hosted track."""

    schema_version: Literal[1] = 1
    track: Literal["local_zero_api_cost"] = "local_zero_api_cost"
    claim_scope: Literal["diagnostic_development_only"] = "diagnostic_development_only"
    hypothesis_test_eligible: Literal[False] = False
    status: Literal["draft", "frozen"] = "draft"
    phase: Literal["smoke", "baseline"]
    dataset: ArtifactPin
    scenario_count: int = Field(ge=1)
    sealed_opened: bool
    systems: list[str] = Field(min_length=1)
    model: LocalModelPin
    hardware: LocalHardwarePin = Field(default_factory=LocalHardwarePin)
    runtime: LocalRuntimePin = Field(default_factory=LocalRuntimePin)
    cost_policy: LocalCostPolicy = Field(default_factory=LocalCostPolicy)
    embedding: EmbeddingPin
    execution: LocalExecutionPolicy = Field(default_factory=LocalExecutionPolicy)
    git_commit: str | None = Field(default=None, pattern=r"^[0-9a-f]{40}$")
    dependency_lock: ArtifactPin
    research_contract: ArtifactPin
    architecture_contract: ArtifactPin
    decision_prompt_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    decision_schema_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    memory_compiler_prompt_sha256: str | None = Field(
        default=None, pattern=SHA256_PATTERN
    )
    memory_compiler_schema_sha256: str | None = Field(
        default=None, pattern=SHA256_PATTERN
    )
    system_config_sha256: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_local_matrix(self) -> Self:
        matrix = {
            "smoke": {
                "systems": LOCAL_SMOKE_SYSTEMS,
                "count": 10,
                "sealed": False,
                "seeds": LOCAL_SMOKE_SEEDS,
                "dataset": "eval/scenarios/smoke.jsonl",
            },
            "baseline": {
                "systems": SIMPLE_SYSTEMS,
                "count": 35,
                "sealed": False,
                "seeds": [101],
                "dataset": "eval/scenarios/dev.jsonl",
            },
        }[self.phase]

        expected_systems = matrix["systems"]
        if set(self.systems) != expected_systems or len(self.systems) != len(
            expected_systems
        ):
            raise ValueError(
                f"local {self.phase} phase requires systems "
                f"{sorted(expected_systems)} exactly"
            )
        if self.scenario_count != matrix["count"]:
            raise ValueError(
                f"local {self.phase} phase requires {matrix['count']} scenarios"
            )
        if self.sealed_opened is not matrix["sealed"]:
            raise ValueError(
                f"sealed_opened must be {matrix['sealed']} for local {self.phase}"
            )
        if self.dataset.path != matrix["dataset"]:
            raise ValueError(
                f"local {self.phase} phase requires dataset {matrix['dataset']}"
            )
        if self.execution.seeds != matrix["seeds"]:
            raise ValueError(
                f"local {self.phase} phase requires seeds {matrix['seeds']} exactly"
            )
        if self.execution.repetitions != len(matrix["seeds"]):
            raise ValueError(
                f"local {self.phase} phase requires {len(matrix['seeds'])} repetitions"
            )

        if self.model.artifact.path != LOCAL_MODEL_ARTIFACT_PATH:
            raise ValueError(
                f"local model artifact path must be {LOCAL_MODEL_ARTIFACT_PATH}"
            )
        if self.model.pricing.path != LOCAL_PRICING_PATH:
            raise ValueError(f"local pricing path must be {LOCAL_PRICING_PATH}")

        invalid_hashes = {
            name: digest
            for name, digest in self.system_config_sha256.items()
            if not _is_sha256(digest)
        }
        if invalid_hashes:
            raise ValueError(f"invalid system configuration hashes: {invalid_hashes}")

        if self.status == "frozen":
            missing = self._missing_freeze_inputs(expected_systems)
            if missing:
                raise ValueError(
                    "frozen local manifest is missing reproducibility inputs: "
                    + ", ".join(missing)
                )
        return self

    def _missing_freeze_inputs(self, expected_systems: set[str]) -> list[str]:
        missing: list[str] = []
        for name, artifact in (
            ("dataset", self.dataset),
            ("model.artifact", self.model.artifact),
            ("model.pricing", self.model.pricing),
            ("model.preflight", self.model.preflight),
            ("dependency_lock", self.dependency_lock),
            ("research_contract", self.research_contract),
            ("architecture_contract", self.architecture_contract),
        ):
            if artifact.sha256 is None:
                missing.append(f"{name}.sha256")
        for name in ("git_commit", "decision_prompt_sha256", "decision_schema_sha256"):
            if getattr(self, name) is None:
                missing.append(name)
        if "anamnesis" in expected_systems:
            if self.memory_compiler_prompt_sha256 is None:
                missing.append("memory_compiler_prompt_sha256")
            if self.memory_compiler_schema_sha256 is None:
                missing.append("memory_compiler_schema_sha256")
        if self.embedding.revision is None:
            missing.append("embedding.revision")
        if self.embedding.artifact_sha256 is None:
            missing.append("embedding.artifact_sha256")
        if set(self.system_config_sha256) != expected_systems:
            missing.append("system_config_sha256")
        return missing


def load_ollama_artifact_pin(path: Path) -> OllamaArtifactPin:
    """Load the portable model pin without accessing an Ollama server."""

    return OllamaArtifactPin.model_validate_json(path.read_text(encoding="utf-8"))


def verify_ollama_artifact(
    pin: OllamaArtifactPin,
    *,
    manifest_path: Path,
    blobs_dir: Path,
) -> int:
    """Verify the manifest and every referenced local blob; return bytes hashed."""

    if _sha256_file(manifest_path) != pin.manifest_sha256:
        raise ValueError("Ollama manifest SHA-256 does not match the tracked pin")

    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual_descriptors = _ollama_descriptors(raw)
    expected_descriptors = [blob.model_dump() for blob in pin.blobs]
    if actual_descriptors != expected_descriptors:
        raise ValueError("Ollama manifest descriptors do not match the tracked pin")

    bytes_hashed = 0
    for blob in pin.blobs:
        blob_path = blobs_dir / f"sha256-{blob.sha256}"
        if not blob_path.is_file():
            raise ValueError(f"missing pinned Ollama blob for role {blob.role}")
        size = blob_path.stat().st_size
        if size != blob.size_bytes:
            raise ValueError(f"Ollama {blob.role} blob size does not match the pin")
        if _sha256_file(blob_path) != blob.sha256:
            raise ValueError(f"Ollama {blob.role} blob SHA-256 does not match the pin")
        bytes_hashed += size
    return bytes_hashed


def validate_zero_api_pricing(path: Path, model: str = LOCAL_MODEL_ID) -> str:
    """Require one exact all-zero Inspect pricing entry and return its file hash."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    zero_rates = {
        "input": 0.0,
        "output": 0.0,
        "input_cache_write": 0.0,
        "input_cache_read": 0.0,
    }
    if raw != {model: zero_rates}:
        raise ValueError("local pricing must contain one exact all-zero model entry")
    return _sha256_file(path)


def require_local_only_environment(environment: Mapping[str, str]) -> None:
    """Fail unless the Ollama process is local-only with the pinned context."""

    required = {
        "OLLAMA_NO_CLOUD": "1",
        "OLLAMA_HOST": "127.0.0.1:11434",
        "OLLAMA_CONTEXT_LENGTH": "4096",
        "OLLAMA_NUM_PARALLEL": "1",
        "OLLAMA_MAX_LOADED_MODELS": "1",
    }
    mismatches = {
        name: expected
        for name, expected in required.items()
        if environment.get(name) != expected
    }
    if mismatches:
        expected_values = ", ".join(
            f"{name}={value}" for name, value in sorted(mismatches.items())
        )
        raise ValueError(f"local runs require {expected_values}")


def verify_static_local_inputs(
    manifest: LocalExperimentManifest, *, repo_root: Path
) -> None:
    """Verify tracked local inputs whose bytes exist before a model run."""

    artifacts = (
        manifest.model.artifact,
        manifest.model.pricing,
        manifest.dependency_lock,
        manifest.research_contract,
        manifest.architecture_contract,
        manifest.dataset,
    )
    for artifact in artifacts:
        if artifact.sha256 is None:
            if manifest.status == "frozen":
                raise ValueError(f"frozen artifact has no SHA-256: {artifact.path}")
            continue
        path = _repo_file(repo_root, artifact.path)
        if _sha256_file(path) != artifact.sha256:
            raise ValueError(f"tracked artifact SHA-256 mismatch: {artifact.path}")

    pin = load_ollama_artifact_pin(_repo_file(repo_root, LOCAL_MODEL_ARTIFACT_PATH))
    if pin.model != manifest.model.snapshot:
        raise ValueError("Ollama artifact model does not match manifest snapshot")
    pricing_digest = validate_zero_api_pricing(
        _repo_file(repo_root, LOCAL_PRICING_PATH), manifest.model.snapshot
    )
    if pricing_digest != manifest.model.pricing.sha256:
        raise ValueError("local pricing SHA-256 does not match manifest pin")


def _ollama_descriptors(raw: object) -> list[dict[str, object]]:
    if not isinstance(raw, dict):
        raise ValueError("Ollama manifest must be a JSON object")
    config = raw.get("config")
    layers = raw.get("layers")
    if not isinstance(config, dict) or not isinstance(layers, list):
        raise ValueError("Ollama manifest requires config and layers")

    descriptors: list[tuple[str, object]] = [("config", config)]
    role_by_media_type = {
        "application/vnd.ollama.image.model": "model",
        "application/vnd.ollama.image.template": "template",
        "application/vnd.ollama.image.license": "license",
        "application/vnd.ollama.image.params": "params",
    }
    for layer in layers:
        if not isinstance(layer, dict):
            raise ValueError("Ollama manifest layer must be an object")
        media_type = layer.get("mediaType")
        try:
            role = role_by_media_type[str(media_type)]
        except KeyError as error:
            raise ValueError(
                f"unsupported Ollama layer media type: {media_type}"
            ) from error
        descriptors.append((role, layer))

    normalized: list[dict[str, object]] = []
    for role, descriptor in descriptors:
        if not isinstance(descriptor, dict):
            raise ValueError("Ollama descriptor must be an object")
        digest = descriptor.get("digest")
        media_type = descriptor.get("mediaType")
        size = descriptor.get("size")
        if (
            not isinstance(digest, str)
            or not digest.startswith("sha256:")
            or not isinstance(media_type, str)
            or not isinstance(size, int)
        ):
            raise ValueError("invalid Ollama descriptor")
        normalized.append(
            {
                "role": role,
                "media_type": media_type,
                "sha256": digest.removeprefix("sha256:"),
                "size_bytes": size,
            }
        )
    return normalized


def _repo_file(repo_root: Path, relative_path: str) -> Path:
    candidate = Path(relative_path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("manifest artifact paths must stay relative to repo root")
    resolved_root = repo_root.resolve()
    resolved = (resolved_root / candidate).resolve()
    if not resolved.is_relative_to(resolved_root):
        raise ValueError("manifest artifact path escapes repo root")
    if not resolved.is_file():
        raise ValueError(f"manifest artifact does not exist: {relative_path}")
    return resolved


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


__all__ = [
    "LOCAL_BASE_URL",
    "LOCAL_MODEL_ARTIFACT_PATH",
    "LOCAL_MODEL_ID",
    "LOCAL_OLLAMA_VERSION",
    "LOCAL_PRICING_PATH",
    "LOCAL_PRICING_SHA256",
    "LocalExperimentManifest",
    "OllamaArtifactPin",
    "load_ollama_artifact_pin",
    "require_local_only_environment",
    "validate_zero_api_pricing",
    "verify_ollama_artifact",
    "verify_static_local_inputs",
]
