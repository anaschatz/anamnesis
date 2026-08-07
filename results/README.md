# Results

No preregistered benchmark result is claimed yet. A strict 10-scenario
zero-provider-cost [local smoke diagnostic](local_smoke.md) is available with
its [CSV](local_smoke.csv), [failure analysis](local_smoke_analysis.md), and
[provenance sidecar](local_smoke.provenance.json). It is explicitly not a
hypothesis test and it rejected the current Qwen 4B compiler/prompt
configuration before expansion to the development set.

A separate gold-assisted [oracle-compiler ceiling](local_oracle_smoke.md), its
[CSV](local_oracle_smoke.csv),
[forensic analysis](local_oracle_smoke_analysis.md), and
[provenance sidecar](local_oracle_smoke.provenance.json) isolate the writer from
the deterministic memory and shared decision path. It achieved F1 87.5% (7 TP,
1 FP, 1 FN) with zero obsolete-memory errors. It is a diagnostic, reports a
decision-only token lower bound, and is neither a baseline nor a hypothesis
test.

The post-hoc, preregistered [D1 shared decision-prompt ablation](local_smoke_d1.md)
is published with its [CSV](local_smoke_d1.csv),
[analysis](local_smoke_d1_analysis.md), and
[provenance sidecar](local_smoke_d1.provenance.json). All four systems still
scored F1 0. The simple baselines produced no false alarms, but Anamnesis kept 7
false reminders and 2 invalid compiler outputs and failed the frozen promotion
gate. D1 is rejected; it is not a hypothesis test and does not authorize a
35-scenario run.

The subsequent [W1 writer preflight record](local_writer_w1_preflight_failure.md)
documents a stricter compiler prompt tested against the same pinned local model
only on a synthetic compatibility request. Its compiler output failed domain
validation because unused optional payload fields were emitted as empty values,
including `date: ""`. The frozen stopping rule therefore prevented all 10 fresh
writer scenarios from running. The accompanying
[provenance sidecar](local_writer_w1_preflight_failure.provenance.json) binds the
source commit and exact ignored `.eval` bytes. This is a preflight rejection,
not a scenario result or a hypothesis test.

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
