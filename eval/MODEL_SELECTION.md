# Hosted model selection gate

No model snapshot is selected by the repository and no measured result may use
the placeholder experiment manifest.

The primary v0 model must be a hosted, immutable snapshot addressable by an
exact Inspect model identifier. Before any development baseline result is
recorded, run a compatibility check on a synthetic case that is not one of the
50 evaluation scenarios. The candidate passes only when it:

1. accepts strict JSON schemas for both `Decision` and `MemoryDelta`;
2. returns valid structured output without a repair call;
3. reports input, cache, and output tokens for every request;
4. has a pinned pricing configuration from which USD cost is reproducible;
5. supports temperature zero and the preregistered seeds; and
6. can be identified by an immutable snapshot rather than a moving alias.

Run the check with Inspect response caching disabled, `max_samples=1`,
`max_tasks=1`, and `max_connections=1`; use those same settings in every
measured task. Strict reports read the resulting `.eval` metadata rather than
trusting a declared command line.

The successful preflight `.eval` is itself a required manifest artifact. Its
byte hash is pinned, and measured task construction revalidates the exact two
model calls, current prompts and wire schemas, no-cache/no-retry policy, token
usage, and cost recomputed from the pinned pricing entry. A boolean declaration
of structured-output support is not sufficient.

This is a compatibility gate, not a model leaderboard. Select the first
preregistered candidate that passes; do not compare candidate F1 on the
development or sealed scenarios. Then fill and freeze a copy of
`experiment_manifest.template.json` after committing the source tree. Store the
generated baseline/final manifest at its ignored local path, set `git_commit` to
the clean source `HEAD`, and treat it as immutable once the first measured task
starts. Archive and publish those exact frozen bytes, rather than a regenerated
copy, with the `.eval` logs. It must not be committed into the revision it
identifies because that would make the Git hash self-referential. The same
snapshot must serve as both the Anamnesis memory compiler and the final decision
model, and it must be used for all four systems.

Changing the model, pricing configuration, generation settings, or schema
after a measured development run creates a new experiment and requires a new
manifest. The 15 sealed scenarios must never be used for compatibility testing.

The vector baseline has a separate immutable-input gate. Record the exact
FastEmbed Hugging Face repository, a 40-character commit SHA, `top_k`, and the
canonical hash of the downloaded artifact tree. The runtime loads that snapshot
locally and rejects any tree whose content hash differs from the manifest.
