# Mem0 + deterministic lifecycle filter v3 — analysis

The real paired cell supports the intended architecture: raw Mem0 retrieval
returned stale correction and cancellation records in both preregistered
opportunities, while the deterministic Anamnesis layer produced the exact active
set in **4/4 queries with zero stale hits**.

## Raw evidence

- 6/6 local Mem0 extraction calls completed with `stop`;
- 49,784 prompt and 354 completion tokens;
- six extracted records, complete scope isolation and cleanup;
- `$0.00` provider API cost and zero external network calls;
- prompt lengths reached 8,412 tokens with no observed truncation warning.

The raw Mem0 preference query returned both Italian and its German correction.
The filtered view retained only the German correction. The raw obligation query
returned both “needs to return the key” and its cancellation. The filtered
active view returned neither. Unrelated project memory and the second user's
preference remained intact.

## Reporter correction

The immutable [raw result](mem0_lifecycle_v3.raw.json) says
`semantic_passed=false` because the original runner used Python's vacuous
`all([]) == true` for queries with no required stale IDs. It therefore counted
four stale opportunities instead of the two frozen in the protocol.

No model call was repeated and no retrieval output was changed. The
[machine-readable recomputation](mem0_lifecycle_v3.json) counts an opportunity
only when the preregistered stale-ID list is nonempty, yielding the frozen gate:

| Gate | Result |
|---|---:|
| Raw stale opportunities | 2 / 2 |
| Exact filtered queries | 4 / 4 |
| Stale hits after filtering | 0 |
| Scope isolation | Pass |
| Cleanup | Pass |

The runner now contains the corrected non-vacuous calculation and a regression
test. This is an offline deterministic reporter correction over byte-pinned raw
evidence, not a selected rerun.

## What this establishes

The hybrid boundary works when lifecycle directives are correct:

- Mem0 performs extraction, embedding, scoped retrieval, and retrospective
  storage.
- Anamnesis owns source-event identity, active-version selection, supersession,
  cancellation, and the prohibition on action evidence from external memory.

It does **not** establish that lifecycle directives can yet be extracted
reliably from arbitrary user text. The next independent test should measure
that writer boundary on fresh cases rather than retuning this filter.
