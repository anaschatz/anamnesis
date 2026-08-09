# Mem0 automatic-memory diagnostic v1

This development-only cell tests the part that the raw Mem0 SDK smoke did not:
`infer=true` automatic extraction, paraphrase deduplication, correction, and
deletion/update decisions. It is not a hypothesis test and cannot promote Mem0
to production.

The exact protocol is
[`mem0_inference_v1.protocol.json`](mem0_inference_v1.protocol.json). It was
frozen before any model generation. The cell keeps official Mem0 `v2.0.17`,
embedded Qdrant, and the existing pinned FastEmbed artifact. Mem0's own default
extraction and update prompts are used unchanged. A small Anamnesis transport
shim only pins local Ollama request settings and records prompts, responses,
tokens, finish state, and latency.

Mem0 `v2.0.17` uses its additive v3 pipeline here: one JSON extraction call per
event, seven calls total. Thinking is disabled; this is not a hidden extra call.

Seven events exercise:

1. profile fact extraction;
2. paraphrase deduplication;
3. correction from Greek to English;
4. a dated prospective obligation;
5. cancellation of that obligation;
6. speculative language that must not become a hard obligation;
7. a second user/session partition.

The run uses one attempt, seed 101, temperature 0, no cache, retry, or repair.
Every event result and failure is published. If integrity fails, metrics are not
interpreted. If the run is valid but semantic gates fail, the failure is still
the result; no prompt is tuned on these seven events.

This cell remains non-authoritative. Mem0 records and IDs never become
Anamnesis trigger state, execution-ledger state, or action evidence.
