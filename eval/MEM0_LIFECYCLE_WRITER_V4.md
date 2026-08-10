# Mem0 lifecycle writer v4 — frozen diagnostic protocol

This development-only cell measures the previously untested boundary from
observable user text to deterministic lifecycle directives. It does not rerun
Mem0 extraction, embeddings, or vector retrieval, and it is not eligible for a
hypothesis claim.

The nine fresh events are frozen before the writer prompt, implementation, and
model calls. The writer receives only the current event and the causally prior
active-state view for that scope. Expected operations, key relations, and gate
fields remain evaluator-only.

## Fair matching

For a new memory, the gate accepts any normalized lifecycle key because a
single exact key spelling would be arbitrary. For a correction, reschedule, or
cancellation, the writer must copy the exact key of the referenced active event
and identify the exact source event it supersedes. The negative control must be
ignored with no key and no superseded IDs.

The deterministic filter then replays every accepted non-ignore directive. Its
final active source-event sets must be exactly the frozen sets for scopes `a`
and `b`; external memory never contributes action evidence.

## Execution and stopping

- exact model: `qwen3.5:9b-q4_K_M` on loopback Ollama;
- seed 101, temperature 0, top-k 1, 32768 context;
- exactly nine calls, no cache, retries, repair, tools, or external network;
- one measured attempt only;
- integrity failure stops interpretation;
- a valid run publishes all directives whether the semantic gate passes or
  fails;
- no writer-prompt tuning is allowed on these events.

The success gate is 9/9 wire-valid outputs, 9/9 exact directive projections,
eight accepted lifecycle mutations, one ignored event, and the exact final
active sets. A failure moves any new writer revision to fresh events.
