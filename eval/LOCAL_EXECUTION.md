# Local zero-provider-cost execution track

This track runs the compiler and decision model through Ollama on the local
machine. It is a diagnostic development track, not a replacement for the
preregistered hosted experiment. It may use only the 10 smoke scenarios or the
35 visible development scenarios. The manifest schema deliberately has no
`final` phase and always sets `hypothesis_test_eligible` to `false`; the current
15-scenario holdout candidate remains ineligible.

“Zero cost” means **USD 0 in provider API charges**. The tracked Inspect pricing
entry has four explicit zero rates. Electricity, hardware purchase or
amortization, and developer time are not measured and must not be described as
zero.

## Frozen local inputs

The initial candidate is `ollama/qwen3:4b-instruct` under these exact inputs:

| Input | Pin |
|---|---|
| Ollama | `0.31.1` |
| Ollama manifest SHA-256 | `0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0` |
| Quantized model blob SHA-256 | `85e4a5b7b8ef0e48af0e8658f5aaab9c2324c76c1641493f4d1e25fce54b18b9` |
| Model blob size | `2,497,280,480` bytes |
| Context window | `4096` tokens |
| Provider route | `http://127.0.0.1:11434/v1` |
| Latency hardware class | Apple M3, 16 GiB, arm64 |

[`ollama_qwen3_4b_instruct.pin.json`](ollama_qwen3_4b_instruct.pin.json)
also pins the config, prompt template, license and parameter blobs. A measured
run must hash all five local blobs before warm-up; an Ollama tag or the short ID
shown by `ollama list` is not enough. The pin stores no user path, serial number,
hardware UUID or machine ID.

The portable verifier is:

```python
from pathlib import Path

from anamnesis.local_experiment import (
    load_ollama_artifact_pin,
    verify_ollama_artifact,
)

models = Path.home() / ".ollama" / "models"
pin = load_ollama_artifact_pin(Path("eval/ollama_qwen3_4b_instruct.pin.json"))
verified_bytes = verify_ollama_artifact(
    pin,
    manifest_path=(models / "manifests/registry.ollama.ai/library/qwen3/4b-instruct"),
    blobs_dir=models / "blobs",
)
print(f"verified {verified_bytes} bytes")
```

On the preregistered machine this must print `2497293803` verified bytes.
Artifact hashing is setup work and is excluded from model latency.

## Local-only server

Start Ollama in a dedicated shell with all five raw values set exactly:

```bash
OLLAMA_NO_CLOUD=1 \
OLLAMA_HOST=127.0.0.1:11434 \
OLLAMA_CONTEXT_LENGTH=4096 \
OLLAMA_NUM_PARALLEL=1 \
OLLAMA_MAX_LOADED_MODELS=1 \
ollama serve
```

Keep the same variables exported in the shell that launches Inspect. The task
records and enforces those five launcher/client declarations, but it cannot
introspect the environment of an independently started Ollama server. The exact
startup command above is therefore a required manual procedure, not an
API-attested fact. No OpenAI API key is needed.

Actual local inference is checked independently: the client route must be the
exact loopback URL, and proxy-bypassing calls to `/api/version` and `/api/ps`
must attest Ollama 0.31.1 plus the resident model digest, family, quantization
and active context length of 4096. `/api/ps` does not prove the server's
no-cloud or parallelism environment. Hardware class and Metal backend remain
manual run declarations. Setup attestation is outside measured latency.

## Gate order

1. Sync the locked environment and run the full test suite.
2. Confirm Ollama version, hardware class and every artifact hash.
3. Commit the source and require a clean worktree.
4. Run only the synthetic local model preflight. It must parse both constrained
   schemas without a retry or repair call, report positive token usage, report
   exact zero provider cost, and preserve the effective runtime settings.
5. Record the resulting `.eval` path and byte SHA-256 in a copy of
   [`local_experiment_manifest.template.json`](local_experiment_manifest.template.json).
   Fill the clean Git commit, prompt/schema hashes and all four system config
   hashes, then change `status` to `frozen`.
6. Only after that frozen preflight passes may the four systems run on the 10
   smoke scenarios. A failed preflight ends the 4B candidate attempt; it does
   not authorize scenario calls or output repair.

Store the filled run manifest under `results/runs/local/` so it remains a run
artifact rather than a self-referential source commit. Archive its exact bytes
with the Inspect logs.

The common locked execution flags are:

```text
--model ollama/qwen3:4b-instruct
--model-base-url http://127.0.0.1:11434/v1
--temperature 0 --seed 101 --cache false --epochs 1 --max-retries 0
--max-samples 1 --max-tasks 1 --max-connections 1
--adaptive-connections false --log-model-api --log-format eval
```

Do not pass `--model-cost-config` on this local track. Inspect 0.3.252 applies
that CLI file before importing the task module and therefore rejects an Ollama
tag absent from its built-in model database. The isolated local task instead
validates the exact tracked `local_model_costs.json` bytes and pinned SHA-256 at
module import, registers that model with four zero rates, records the pricing
hash in task metadata, and checks the effective `ModelInfo.cost` again at every
model construction. The frozen manifest and preflight artifact bind the same
pricing hash.

Use the local task definitions in `eval/anamnesis_local_eval.py`. First run
`local_model_preflight`; after it passes, run `local_no_memory`,
`local_full_context`, `local_vector_rag`, and `local_anamnesis` sequentially
against the same frozen smoke manifest. The vector task still requires the
already pinned local FastEmbed snapshot and must not download it during a
measured run.

In the Inspect shell, export the same five declarations and run the synthetic
gate first:

```bash
export OLLAMA_NO_CLOUD=1
export OLLAMA_HOST=127.0.0.1:11434
export OLLAMA_CONTEXT_LENGTH=4096
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_MAX_LOADED_MODELS=1

inspect eval eval/anamnesis_local_eval.py@local_model_preflight \
  --model ollama/qwen3:4b-instruct \
  --model-base-url http://127.0.0.1:11434/v1 \
  --temperature 0 --seed 101 --cache false --epochs 1 --max-retries 0 \
  --max-samples 1 --max-tasks 1 --max-connections 1 \
  --adaptive-connections false --log-model-api \
  --log-format eval --log-dir results/runs/local/preflight --json \
  -T seed=101
```

After that clean preflight log is pinned and the smoke manifest is frozen, set
only task-specific path variables; do not change the five declarations above:

```bash
ANAMNESIS_LOCAL_MANIFEST="$PWD/results/runs/local/experiment.smoke.json"
ANAMNESIS_OLLAMA_MODELS="$HOME/.ollama/models"
ANAMNESIS_FASTEMBED_SNAPSHOT="$PWD/results/runs/fastembed/bge-small-en-v1.5-onnx-q/52398278842ec682c6f32300af41344b1c0b0bb2"
```

Run the four tasks sequentially. The global `--seed` controls generation; the
separate `-T seed=101` is also mandatory because it binds the task-function
argument recorded in the log.

```bash
inspect eval eval/anamnesis_local_eval.py@local_no_memory \
  --model ollama/qwen3:4b-instruct \
  --model-base-url http://127.0.0.1:11434/v1 \
  --temperature 0 --seed 101 --cache false --epochs 1 --max-retries 0 \
  --max-samples 1 --max-tasks 1 --max-connections 1 \
  --adaptive-connections false --log-model-api \
  --log-format eval --log-dir results/runs/local/smoke --json \
  -T seed=101 -T repetition=1 \
  -T manifest="$ANAMNESIS_LOCAL_MANIFEST" \
  -T ollama_models_dir="$ANAMNESIS_OLLAMA_MODELS"

inspect eval eval/anamnesis_local_eval.py@local_full_context \
  --model ollama/qwen3:4b-instruct \
  --model-base-url http://127.0.0.1:11434/v1 \
  --temperature 0 --seed 101 --cache false --epochs 1 --max-retries 0 \
  --max-samples 1 --max-tasks 1 --max-connections 1 \
  --adaptive-connections false --log-model-api \
  --log-format eval --log-dir results/runs/local/smoke --json \
  -T seed=101 -T repetition=1 \
  -T manifest="$ANAMNESIS_LOCAL_MANIFEST" \
  -T ollama_models_dir="$ANAMNESIS_OLLAMA_MODELS"

inspect eval eval/anamnesis_local_eval.py@local_vector_rag \
  --model ollama/qwen3:4b-instruct \
  --model-base-url http://127.0.0.1:11434/v1 \
  --temperature 0 --seed 101 --cache false --epochs 1 --max-retries 0 \
  --max-samples 1 --max-tasks 1 --max-connections 1 \
  --adaptive-connections false --log-model-api \
  --log-format eval --log-dir results/runs/local/smoke --json \
  -T seed=101 -T repetition=1 \
  -T manifest="$ANAMNESIS_LOCAL_MANIFEST" \
  -T ollama_models_dir="$ANAMNESIS_OLLAMA_MODELS" \
  -T embedding_snapshot_path="$ANAMNESIS_FASTEMBED_SNAPSHOT"

inspect eval eval/anamnesis_local_eval.py@local_anamnesis \
  --model ollama/qwen3:4b-instruct \
  --model-base-url http://127.0.0.1:11434/v1 \
  --temperature 0 --seed 101 --cache false --epochs 1 --max-retries 0 \
  --max-samples 1 --max-tasks 1 --max-connections 1 \
  --adaptive-connections false --log-model-api \
  --log-format eval --log-dir results/runs/local/smoke --json \
  -T seed=101 -T repetition=1 \
  -T manifest="$ANAMNESIS_LOCAL_MANIFEST" \
  -T ollama_models_dir="$ANAMNESIS_OLLAMA_MODELS"
```

Keep the four exact `log_location` values printed by the JSON `done` records.
The strict local reporter accepts no glob containing old/repeated runs:

```bash
ANAMNESIS_NO_MEMORY_LOG="/replace/with/no-memory-log.eval"
ANAMNESIS_FULL_CONTEXT_LOG="/replace/with/full-context-log.eval"
ANAMNESIS_VECTOR_RAG_LOG="/replace/with/vector-rag-log.eval"
ANAMNESIS_TREATMENT_LOG="/replace/with/anamnesis-log.eval"

anamnesis-local-report \
  --manifest "$ANAMNESIS_LOCAL_MANIFEST" \
  --scenarios eval/scenarios/smoke.jsonl \
  --runs "$ANAMNESIS_NO_MEMORY_LOG" "$ANAMNESIS_FULL_CONTEXT_LOG" \
    "$ANAMNESIS_VECTOR_RAG_LOG" "$ANAMNESIS_TREATMENT_LOG" \
  --csv results/local_smoke.csv \
  --markdown results/local_smoke.md \
  --provenance results/local_smoke.provenance.json
```

The provenance argument is optional and defaults to
`results/local_smoke.provenance.json`. The reporter writes it last, after the
CSV and Markdown exist. It binds the source Git commit; frozen manifest,
scenario dataset, and four exact `.eval` logs; and both rendered result files
to their SHA-256 digests. All of those paths must resolve inside the repository
and are stored as repository-relative paths, so the sidecar cannot disclose a
user directory or silently refer to an external artifact.

Any 10-scenario table must be titled **Local smoke diagnostic — not a hypothesis
test**. It may report precision, recall, F1, false alarms, obsolete-memory
errors, input/output tokens, zero provider API cost and prewarmed latency. It
must keep setup latency separate and state that electricity and hardware cost
are unmeasured. Do not feed these logs into the hosted strict baseline/final
reporter by weakening its manifest checks.

## Oracle-compiler ceiling

The next smoke-only diagnostic uses the frozen, manually annotated
[`oracle/smoke_memory_deltas.v1.json`](oracle/smoke_memory_deltas.v1.json).
It is gold-assisted and tests only the deterministic store, trigger engine,
renderer, and unchanged `ollama.decision.v0.1` decision policy. It is not an
Anamnesis compiler result, is not hypothesis-test eligible, and its zero
compiler tokens are a decision-only lower bound with unmeasured human
annotation effort.

The offline sanitized replay must first remain exactly TP 8, FP 0, FN 0, with
provenance 8/8. Then commit the oracle implementation and require a clean
worktree. Because the preflight is bound to the source commit, rerun
`local_model_preflight` from that commit and place its exact path and SHA-256,
the clean Git SHA, current decision prompt/schema hashes, and the singleton
oracle system hash into a copy of
[`local_oracle_manifest.template.json`](local_oracle_manifest.template.json).
Keep the filled manifest under `results/runs/local/oracle/` and change its
status to `frozen`.

The setup preflight still makes one compiler-schema compatibility call and one
decision call. The compiler call is setup-only and is never used by the oracle
scenario execution. Every non-clock scenario checkpoint replays one local
annotation record with zero model calls, tokens, and provider cost; every
checkpoint still makes exactly one decision-model call.

With the same five Ollama environment declarations and common Inspect flags
shown above:

```bash
ANAMNESIS_ORACLE_MANIFEST="$PWD/results/runs/local/oracle/experiment.oracle.json"
ANAMNESIS_ORACLE_ANNOTATIONS="$PWD/eval/oracle/smoke_memory_deltas.v1.json"
ANAMNESIS_OLLAMA_MODELS="$HOME/.ollama/models"

inspect eval eval/anamnesis_local_eval.py@local_anamnesis_oracle_compiler \
  --model ollama/qwen3:4b-instruct \
  --model-base-url http://127.0.0.1:11434/v1 \
  --temperature 0 --seed 101 --cache false --epochs 1 --max-retries 0 \
  --max-samples 1 --max-tasks 1 --max-connections 1 \
  --adaptive-connections false --log-model-api \
  --log-format eval --log-dir results/runs/local/oracle/smoke --json \
  -T seed=101 -T repetition=1 \
  -T manifest="$ANAMNESIS_ORACLE_MANIFEST" \
  -T ollama_models_dir="$ANAMNESIS_OLLAMA_MODELS" \
  -T oracle_annotations_path="$ANAMNESIS_ORACLE_ANNOTATIONS"
```

Pass the single successful `.eval` path to the separate fail-closed reporter:

```bash
ANAMNESIS_ORACLE_LOG="/replace/with/oracle-log.eval"

anamnesis-oracle-report \
  --manifest "$ANAMNESIS_ORACLE_MANIFEST" \
  --scenarios eval/scenarios/smoke.jsonl \
  --oracle-artifact "$ANAMNESIS_ORACLE_ANNOTATIONS" \
  --run "$ANAMNESIS_ORACLE_LOG" \
  --csv results/local_oracle_smoke.csv \
  --markdown results/local_oracle_smoke.md \
  --provenance results/local_oracle_smoke.provenance.json
```

The report title must be **Local oracle-compiler ceiling — diagnostic only**.
It cannot be merged into the four-system smoke table or used for token-efficiency
claims. The strict reporter requires exactly one 10-sample log, no scenario
compiler ModelEvents, exact frozen deltas, complete zero-cost accounting, and
one raw decision call per authored checkpoint.
