# Automatic lifecycle writer v4 — result analysis

The one allowed local run completed with protocol integrity but failed the
semantic gate. The writer is not ready to supply authoritative lifecycle
directives.

## Integrity

- exact source commit: `5aa8b081d38396da3d11aee6d158993e63cbbc90`;
- 9/9 loopback model calls completed with `stop`;
- 3,414 prompt tokens and 562 completion tokens;
- complete positive usage, `$0.00` provider API cost, and zero external calls;
- pinned `qwen3.5:9b-q4_K_M`, Ollama 0.31.1, seed 101, temperature 0;
- no retries, repairs, cache, or second attempt.

The byte-exact raw artifact is
[`mem0_lifecycle_writer_v4.raw.json`](mem0_lifecycle_writer_v4.raw.json), with
SHA-256
`e0bab6f607daecbcf2892a6b4827cb3d0bc3c029bc244f862f2eaeb6d64d532f`.

## Frozen gate

| Gate | Required | Observed |
|---|---:|---:|
| Wire-valid outputs | 9 | 3 |
| Exact directive projections | 9 | 2 |
| Filter-accepted mutations | 8 | 2 |
| Correct ignored controls | 1 | 0 |
| Final active sources, scope a | e2, e5, e9 | e5, e8 |
| Final active sources, scope b | e7 | empty |

## Failure taxonomy

Six outputs violated the supplied closed JSON Schema. Three added an
unauthorized `value` field, three omitted the required `source_event_id`, and
some did both. The request contained `additionalProperties: false` and required
`source_event_id`, so Ollama's schema-format path did not reliably constrain the
Qwen output under this exact runtime.

Only the Project Meridian fact and the initial Borealis permit obligation were
exact and accepted. The Borealis reschedule copied the correct active key but
omitted `supersedes_event_ids=["mw4-e8"]`; the deterministic filter correctly
rejected that causal no-op replacement. The correction and cancellation cases
never reached lifecycle evaluation because their predecessor creates were
wire-invalid and therefore absent from active state.

The negative brainstorming control selected `ignore`, but omitted the required
source event ID and therefore remained invalid. This distinction matters: the
semantic choice was directionally right, while the transport contract was not.

## Architectural conclusion

The previous v3 result remains valid: deterministic lifecycle filtering works
when directives are correct. This v4 result shows that the current 9B local
writer cannot yet generate those directives reliably, even with a closed schema
request and a minimal causal active-state view.

The safe architecture therefore remains:

- Mem0: non-authoritative extraction, embeddings, scoped vector recall;
- deterministic Anamnesis: active versions, supersession, cancellation, and
  action-evidence policy;
- model writer: untrusted proposal source whose output must pass both wire and
  reducer validation before state mutation.

No prompt correction will be made on these nine events. A next writer revision
requires fresh cases and should first isolate transport enforcement from
semantic lifecycle reasoning, for example with a server whose grammar backend
is independently attested or with deterministic envelope construction around a
smaller model prediction surface.
