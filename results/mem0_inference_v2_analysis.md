# Mem0 automatic-memory diagnostic v2 — analysis

The one preregistered v2 run was operationally valid but failed its semantic
gate: **4 of 7 event assertions passed**. This is development evidence, not a
hypothesis test or a production promotion.

## Integrity and accounting

- official Mem0 `v2.0.17` at commit `12c47f5`;
- local Qwen3.5 9B Q4_K_M through Ollama 0.31.1;
- exactly 7/7 model calls, all `done_reason=stop`;
- 58,103 prompt tokens and 373 completion tokens;
- 155.725 seconds total model-call latency, including cold load;
- complete user/session isolation and cleanup;
- `$0.00` provider API cost and zero external network calls.

The 32768-token runtime removed the v1 context failure. Actual prompt usage was
8,221–8,428 tokens per call, and the contemporaneous server terminal contained
no `truncating input prompt` warning. The operator observation is not a
separately archived raw log, so it is disclosed rather than presented as a
cryptographic attestation.

## What worked

1. **Extraction:** the first preference became one concise memory.
2. **Semantic deduplication:** the paraphrase generated no new record and the
   store remained at one matching fact.
3. **Speculation safety:** “Maybe I could…” remained a possible future activity,
   not a hard obligation.
4. **Scope isolation:** the second user's Korean preference remained separate;
   no cross-user terms leaked.

## What failed

The important failure is lifecycle semantics, not JSON parsing.

- **Correction:** Mem0 added a linked “Spanish to French” correction but retained
  the active Spanish preference. The store held both the stale fact and its
  correction.
- **Cancellation:** Mem0 added a cancellation memory but retained the active
  “needs to renew” obligation. A naive retriever can therefore return the stale
  obligation after cancellation.
- **Date surface form:** the obligation was extracted with the correct semantic
  date as “September 12, 2044,” while the frozen assertion required the literal
  ISO token `2044-09-12`. The gate correctly remains failed; post-hoc rescoring is
  not allowed. Architecturally, this is a benchmark canonicalization gap rather
  than missing memory content.

This behavior matches the evaluated Mem0 additive pipeline: extraction produced
`ADD` records (and optional links) but did not make the linked correction or
cancellation authoritative over the prior records.

## Architecture decision

Mem0 is useful as a **non-authoritative fact extraction, deduplication, scoping,
and vector-recall baseline**. It must not own pending obligations, cancellation,
trigger truth, or action evidence. Anamnesis keeps those temporal lifecycle
semantics in its deterministic store and evaluation core.

The next useful comparison is not another prompt tweak on these seven cases. It
is a fresh paired retrieval test that measures whether a lifecycle-aware
Anamnesis filter can use Mem0 recall while deterministically suppressing stale
superseded/cancelled records.
