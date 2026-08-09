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

## Why the boundary is strict

Mem0's official examples extract memories from message lists and search them
with user-scoped filters. Letta describes stateful agents whose core and archival
memory are managed through the agent lifecycle. Graphiti represents evolving
entities and facts with temporal validity and episode lineage. These are useful,
different hypotheses—not interchangeable implementations. Loading all three
inside production would multiply storage semantics, hidden model calls, privacy
surfaces, and failure modes before the evaluation establishes which capability
is actually needed.
