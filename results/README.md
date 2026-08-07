# Results

No preregistered benchmark result is claimed yet. A strict 10-scenario
zero-provider-cost [local smoke diagnostic](local_smoke.md) is available with
its [CSV](local_smoke.csv), [failure analysis](local_smoke_analysis.md), and
[provenance sidecar](local_smoke.provenance.json). It is explicitly not a
hypothesis test and it rejected the current Qwen 4B compiler/prompt
configuration before expansion to the development set.

Raw Inspect logs belong in the ignored `results/runs/` directory. A strict
development report requires one complete 3-system × 35-scenario matrix from a
frozen baseline manifest and is titled “Development baseline — not a final
hypothesis test.” A strict final report requires four systems × 50 scenarios ×
three repetitions from an eligible post-freeze dataset.

Incomplete or exploratory inputs are accepted only with `--allow-incomplete`.
Those reports are visibly labeled diagnostic and never print the preregistered
success gate. Provider-neutral JSONL remains useful for such diagnostics;
headline reports use Inspect `.eval` logs so effective execution settings can
be audited.

The first hosted experiment artifact to place here is `development.md`, reporting the
full-context baseline's recall, precision, tokens, cost, and latency alongside
the other two simple baselines. Until a hosted model snapshot, pricing file,
pinned successful preflight log, embedding commit/artifact, and clean experiment
manifest are frozen, any table would be illustrative rather than measured
evidence.

The frozen experiment manifest is a run artifact. Publish its exact bytes next
to the `.eval` logs and tables, even though the local baseline/final manifest
paths are Git-ignored to let `git_commit` identify an already-clean source
revision without a self-referential commit hash.
