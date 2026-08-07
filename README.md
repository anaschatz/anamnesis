# Anamnesis

An evaluation-first research prototype for temporal and prospective memory in
LLM agents. It tests whether explicit, provenance-aware state helps an agent
remember what is true now and execute an intended action at the right moment
with fewer errors and fewer input tokens than simple memory baselines.

The research question and pass/fail thresholds are frozen in
[`RESEARCH.md`](RESEARCH.md). The treatment and fairness boundary are specified
in [`ARCHITECTURE.md`](ARCHITECTURE.md).

## Current status

The v0 harness and in-memory treatment are implemented. The repository now
contains:

- 35 visible development scenarios and 15 holdout-shaped harness scenarios;
- no-memory, full-context, exact top-k vector-RAG, and Anamnesis systems;
- an online LLM memory compiler plus a deterministic versioned store, closed
  trigger DSL, occurrence state, compact memory view, and execution ledger;
- deterministic action/provenance scoring and strict baseline/final reports;
- component-level token, cost, latency, parse-error, and audit accounting;
- frozen-manifest, dataset-release, model-pricing, prompt, git, and embedding
  artifact guards; and
- a synthetic hosted-model compatibility preflight.

No empirical result is claimed yet. A frozen hosted model and pricing file have
not been selected, so the required 35-scenario development baseline table has
not been run. The existing 15-record holdout was inspected during development
and is deliberately marked ineligible for hypothesis testing. A fresh,
externally held and independently reviewed replacement is required after the
system freeze. See [`eval/SCENARIO_PLAN.md`](eval/SCENARIO_PLAN.md).

## Architecture

```text
ObservableEvent(id, at, kind, text)
  -> LLM MemoryCompiler
  -> strict MemoryDelta
  -> deterministic temporal store and trigger engine
  -> compact MemoryView
  -> shared decision LLM
  -> occurrence/execution ledger
```

Every system receives the same sanitized event stream and invokes the same
decision model once per authored checkpoint. Anamnesis alone invokes the same
frozen model once more for each non-clock event to compile memory. All compiler
usage is included in its headline totals. Gold actions, forbidden traps,
`supersedes`, scenario annotations, and future events never cross the runtime
boundary.

The store is pure in-memory Python. It keeps immutable fact and intent
revisions, stable action keys, per-occurrence recurring state, raw-event
provenance, and an execution ledger. It does not use a database, embeddings,
LangChain/LlamaIndex, a scheduler service, PyTorch, a UI, or model training.

## Open-source stack

| Role | Tool |
|---|---|
| Evaluation tasks and raw logs | [Inspect AI](https://inspect.aisi.org.uk/) |
| Strict data contracts | [Pydantic](https://docs.pydantic.dev/) |
| Vector-RAG baseline only | [FastEmbed](https://github.com/qdrant/fastembed) |
| Retrieval and report math | [NumPy](https://numpy.org/) |
| Tests and linting | [pytest](https://pytest.org/) and [Ruff](https://docs.astral.sh/ruff/) |
| Reproducible environment | [uv](https://docs.astral.sh/uv/) |

AutoResearch is intentionally not a core dependency. After the development
split and experiment configuration are frozen, its experiment-loop pattern may
be used for declared prompt/config ablations on the 35 development scenarios
only. It must never receive the final sealed data.

## Setup and local verification

Use Python 3.11, 3.12, or 3.13:

```bash
uv sync --extra dev --no-editable
source .venv/bin/activate

anamnesis-validate eval/scenarios/dev.jsonl
pytest
ruff check src tests eval
ruff format --check src tests eval
inspect list tasks eval/anamnesis_eval.py
```

The current candidate validates as 35 development scenarios with 219
checkpoints, and 50 combined scenarios with 296 checkpoints. Dataset manifests
record canonical and per-record hashes, family counts, origin, and review
state. All current cases are locally authored; no LongMemEval or TriggerBench
content is silently included.

## Freeze before measuring

1. Run the synthetic `model_preflight` task on a hosted immutable snapshot. It
   must pass both strict schemas and complete usage/cost accounting without a
   repair call. Pin the resulting `.eval` file and its byte SHA-256 as
   `model.preflight` in the experiment manifest.
2. Pin an Inspect model-cost JSON/YAML file for that exact model identifier.
3. Pin the FastEmbed Hugging Face repository to a 40-character commit SHA and
   record the hash of the downloaded artifact tree.
4. Commit the source tree, then copy
   [`eval/experiment_manifest.template.json`](eval/experiment_manifest.template.json)
   to the ignored local path `eval/experiment.baseline.json` (or
   `eval/experiment.final.json` for the final phase). Fill every required
   hash/configuration field, set `git_commit` to that clean `HEAD`, and set
   `status` to `frozen`. Do not add this generated manifest to the commit it
   identifies: a Git commit cannot contain its own hash. Once the first measured
   task starts, do not modify it; archive and publish those exact frozen bytes,
   rather than a regenerated copy, alongside the resulting `.eval` logs and
   result table.
5. Keep temperature at zero, response cache disabled, `max_samples=1`,
   `max_tasks=1`, and model concurrency at one for every compared task.

Model-selection rules are detailed in
[`eval/MODEL_SELECTION.md`](eval/MODEL_SELECTION.md). The task factory and
strict report fail closed when the active model pricing, prompt/schema, dataset,
git revision, RAG configuration, or embedding artifact differs from the frozen
manifest. They also re-open the pinned preflight log and verify its exact two
prompts, strict schemas, non-cached/no-retry calls, token usage, and cost against
the pinned pricing table.

Example preflight:

```bash
inspect eval eval/anamnesis_eval.py@model_preflight \
  --model provider/immutable-snapshot \
  --model-cost-config eval/model_costs.json \
  --cache false --max-samples 1 --max-tasks 1 --max-connections 1 \
  -T seed=101
```

## Development baseline gate

The first measured milestone is exactly three systems × 35 development
scenarios × one declared repetition (`seed=101`). Run all tasks from the same
clean frozen baseline manifest:

```bash
inspect eval eval/anamnesis_eval.py@no_memory \
  --model provider/immutable-snapshot \
  --model-cost-config eval/model_costs.json \
  --cache false --max-samples 1 --max-tasks 1 --max-connections 1 \
  -T manifest=eval/experiment.baseline.json -T seed=101 -T repetition=1

inspect eval eval/anamnesis_eval.py@full_context \
  --model provider/immutable-snapshot \
  --model-cost-config eval/model_costs.json \
  --cache false --max-samples 1 --max-tasks 1 --max-connections 1 \
  -T manifest=eval/experiment.baseline.json -T seed=101 -T repetition=1

inspect eval eval/anamnesis_eval.py@vector_rag \
  --model provider/immutable-snapshot \
  --model-cost-config eval/model_costs.json \
  --cache false --max-samples 1 --max-tasks 1 --max-connections 1 \
  -T manifest=eval/experiment.baseline.json -T seed=101 -T repetition=1 \
  -T embedding_revision=0123456789abcdef0123456789abcdef01234567
```

Then generate the explicitly non-final development table:

```bash
anamnesis-report \
  --mode baseline \
  --manifest eval/experiment.baseline.json \
  --scenarios eval/scenarios/dev.jsonl \
  --runs logs/*.eval \
  --csv results/development.csv \
  --markdown results/development.md
```

Only after this table exists should the Anamnesis compiler be evaluated or
tuned on the same development split. The repository contains the implementation
scaffolding now, but this chronological research gate has not yet been
satisfied and must not be rewritten as if it had.

## Final experiment

After freezing the compiler prompt, schema, reducer, trigger behavior, and
system configuration, replace the contaminated holdout candidate with a fresh
15-scenario independently reviewed set. Its adjacent release manifest must mark
human review as passed and `preregistered_final_eligible` as true; otherwise
the `dataset=all` tasks refuse to start.

Run all four systems over all 50 scenarios for repetitions 1/2/3 with seeds
101/202/303, respectively. A strict final report requires the complete matrix
and applies the preregistered gate independently to each repetition:

- Anamnesis F1 exceeds the best simple baseline by at least 0.05;
- Anamnesis total logical input tokens are at least 30% below full-context; and
- Anamnesis has no more false-alarm checkpoints than that same comparator.

`--allow-incomplete` exists only for diagnostics. It labels output as incomplete
and suppresses every hypothesis-support claim. No LLM judge participates in
headline scoring.
