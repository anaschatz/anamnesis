# Memory baseline architecture

Anamnesis remains the research core and evaluation harness. It owns authored
checkpoints, causal prefix release, temporal/prospective scoring, action evidence,
failure taxonomy, manifests, raw-call reconciliation, and publication gates.
External memory systems are experimental competitors behind
`MemoryBenchmarkAdapter`; none is an authoritative temporal store or production
dependency.

## Normalized benchmark roles

| Adapter | What the benchmark measures | What remains outside its authority |
|---|---|---|
| Mem0 | Automatic fact extraction, deduplication, user/session scoping, vector recall | Trigger truth, pending-obligation lifecycle, action evidence |
| Letta | Agent-managed core memory and archival memory, including what the agent chooses to retain or search | Gold temporal state, scorer state, execution ledger |
| Graphiti | Entities, relationships, temporal validity windows, episode provenance, hybrid graph retrieval | Anamnesis trigger engine and action-evidence contract |

Each adapter must be installed in a separately pinned environment and injected
into the harness. The repository does not import or lock all three packages.
Every experimental cell freezes the upstream revision, package/source identity,
backend configuration, model/embedding dependencies, scope, and cleanup policy.

## Fair comparison

All systems receive the same observable event prefix at the same checkpoint.
The harness normalizes inputs into four semantic categories: profile, decision,
project, and prospective obligation. Retrieval output is normalized to opaque
handles and bounded text. Provider IDs never become evidence IDs.

Headline dimensions are:

1. prospective-action precision, recall, false alarms, obsolete actions, and
   exact causal provenance;
2. fact/project retrieval precision and stale-fact rate;
3. storage and retrieval model calls, tokens, cost, and latency;
4. cross-user/session leakage, deletion, deterministic replay, and failure rate;
5. operational complexity, including databases, background services, and model
   dependencies.

The first phase uses separate diagnostic cells rather than a combined install:
Mem0 fact-recall, Letta agent-managed retention, and Graphiti temporal-graph
retrieval. Only after each cell passes its own integrity preflight will the
Anamnesis reporter compare normalized outputs on one newly frozen dataset. The
simplest architecture that wins the relevant Anamnesis metrics is the promotion
candidate; feature count is not a success criterion.

## First real Mem0 integration result

The first provider cell uses the official Mem0 `v2.0.17` source at commit
`12c47f524935692e27ad48d829f35fa1e4417181`, installed only in a separate
environment. The tracked pin binds its Python source tree and upstream
`pyproject.toml`, all material runtime package versions, and the existing
byte-pinned FastEmbed `bge-small-en-v1.5` artifact. Mem0 runs with embedded
Qdrant, telemetry disabled, a Python socket guard, and no provider API access.

The real SDK smoke passed `add -> scoped dense-vector search -> update ->
search -> delete -> empty-scope verification`. The Anamnesis adapter hydrated
every provider result before trusting it, retained Mem0 identifiers only behind
process-local opaque handles, and emitted no action evidence. The immutable
result is [`results/mem0_sdk_smoke.v1.json`](results/mem0_sdk_smoke.v1.json).

This is intentionally a storage/retrieval contract result with `infer=false`.
It does **not** claim that Mem0's automatic fact extraction or automatic
deduplication works on the Anamnesis benchmark. Those features require a second,
separately preregistered cell with a pinned local writer model; combining them
with the SDK plumbing test would hide whether failures came from storage,
retrieval, or LLM extraction.

That second cell was attempted exactly once and stopped on infrastructure
integrity rather than memory quality. Mem0 completed seven `infer=true` model
calls, but the runner lost the final artifact at a dual-module Pydantic class
boundary. Server logs also showed that accumulated Mem0 prompts exceeded the
effective input budget and were truncated to 4098 tokens despite an
8192-token resident context. The failure is published in
[`results/mem0_inference_v1_failure.md`](results/mem0_inference_v1_failure.md).
The runner defect is fixed, but the same seven events are not eligible for a
retry; the next cell needs fresh events and a context-fit preflight.

The fresh v2 cell completed with integrity under a 32768-token context and is
published in [`results/mem0_inference_v2.json`](results/mem0_inference_v2.json).
It passed 4/7 frozen event gates: fact extraction, paraphrase deduplication,
speculation safety, and user/session isolation worked. Correction and
cancellation did not supersede the old active records; the additive pipeline
stored both the newer statement and stale state. This directly supports the
architecture boundary: Mem0 can supply non-authoritative recall candidates,
but Anamnesis must own temporal validity, cancellation, pending obligations,
and action evidence.

## Why the boundary is strict

Mem0's official examples extract memories from message lists and search them
with user-scoped filters. Letta describes stateful agents whose core and archival
memory are managed through the agent lifecycle. Graphiti represents evolving
entities and facts with temporal validity and episode lineage. These are useful,
different hypotheses—not interchangeable implementations. Loading all three
inside production would multiply storage semantics, hidden model calls, privacy
surfaces, and failure modes before the evaluation establishes which capability
is actually needed.
