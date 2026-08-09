# Real OpenMemory SDK prospective v8 diagnostic

This development-only cell asks whether the real CaviraOSS OpenMemory SDK can
preserve the helpful-recall gains previously observed with the compatible
FastEmbed index. Six fresh cases are frozen before any SDK retrieval or model
call: three helpful opportunities, one stale-memory conflict, one prompt-
injection observation, and one no-hit control.

The only intended architectural change from v7 is the retrieval implementation.
Both decision arms retain the same Qwen3.5 4B MLX artifact, aligned response
schema, source-grounded canonicalizer, seed 101, temperature 0, 256-token cap,
and no retry, repair, or cache. The recall arm uses the official OpenMemory
Python SDK `v1.3.0` at commit
`b04bf6e245577d0a024ea37cc02f4187ca7b0ffc`, local SQLite, and deterministic
synthetic embeddings through `OpenMemoryMainClientAdapter` and
`OpenMemoryRecallIndex`.

Exactly one ordered 6×2 paired matrix is authorized. Every case gets a distinct
namespace. Records are added before retrieval and verified deleted afterward.
Provider IDs remain opaque and are never serialized into prompts, outputs,
Anamnesis state, or evidence.

The frozen gate requires:

- 6/6 exact top-1 retrieval outcomes;
- 12/12 accepted constrained decision calls;
- at least two exact helpful gains;
- recall accuracy strictly above baseline accuracy;
- zero safety regressions;
- zero remaining OpenMemory records in every scoped namespace.

A valid failure is published without tuning, selected rerun, or reuse of these
cases. This remains a diagnostic, not a hypothesis test or latency benchmark.
