# Real OpenMemory SDK v8 forensic analysis

## Verdict

The frozen v8 gate passed, but the important conclusion is narrower than
“memory works.” The official OpenMemory SDK successfully performed the complete
indexed-memory lifecycle and produced useful decision gains. The remaining
accuracy ceiling is now primarily in decision payload normalization rather than
retrieval.

- OpenMemory top-1 retrieval: **6/6**.
- Scoped SDK cleanup after each case: **6/6**; zero retained records.
- Accepted constrained model calls: **12/12**.
- Baseline exact decisions: **1/6**.
- Recall exact decisions: **3/6**.
- Exact helpful gains: **2/3 opportunities**.
- Recall-induced safety regressions: **0**.
- Provider API cost: **$0.00**; electricity and hardware unmeasured.

The “zero safety regressions” metric is paired and relative: recall did not turn
a correct control into an incorrect one. It does not mean every control action
was exact. Two action controls were already noncanonical in both arms.

## What OpenMemory solved

1. **Harbor coordinator.** Baseline used the unresolved phrase “regular port
   coordinator.” OpenMemory retrieved the exact record, and recall produced the
   correct recipient `Meridian Harbor Office` and address `18 Lantern Quay`.
2. **Glaze shipment.** Baseline omitted the tracking identifier. OpenMemory
   retrieved `GLAZE-842`, and recall produced the exact shipment action.
3. **Lichen survey.** OpenMemory retrieved the correct project name
   `Silver Ridge`, and the model placed it in the correct `project` slot. The
   action still failed exact scoring because the subject remained the generic
   `upload field summary` and retained a redundant `item` slot instead of the
   canonical `upload mountain lichen field summary`.

Thus recall supplied the missing source-grounded value in all three helpful
cases, and two became exact actions. The third is a canonical composition miss,
not a retrieval miss.

## Residual weaknesses

- **Duplicate/unsourced location slot.** Both arms for the explicit Nova Lab
  instruction emitted `recipient: Nova Lab` and the redundant `room: Nova Lab`.
  Current canonicalization does not remove a location slot that merely
  duplicates an already-grounded recipient.
- **Article normalization.** Both no-hit arms emitted `photograph the archive
  seals`; the frozen target is `photograph archive seals`. This is the same
  narrow article-removal weakness previously seen after v7.
- **Subject composition.** When recall supplies a project name, the
  canonicalizer can preserve the project slot but does not always combine the
  event's domain noun phrase into the subject or remove the now-redundant item
  slot.

These defects should be fixed with deterministic source-grounded normalization,
not by broadening OpenMemory authority or adding more prompt instructions.

## Integrity and scope

The full source commit was frozen before execution. One preliminary invocation
was rejected before SDK/model work because the supplied full Git SHA was wrong.
A second invocation performed a partial SDK operation but zero model calls and
failed at loopback connection because it ran outside the server's sandbox
network namespace; its records were cleaned and it produced no output. The one
complete matrix then ran in the correct namespace and produced exactly 12 model
calls, with no retries, repair, cache, or alternative output.

The raw run contains no API key, provider memory identifier, or absolute local
path and is published with the result. OpenMemory remained retrospective and
non-authoritative throughout: its IDs and scores never became prospective state,
triggers, executions, or action evidence.

## Next experiment

The next architecture revision should be deterministic and narrow:

1. remove a `room`/`address` slot when it only duplicates a sourced recipient;
2. remove simple English articles from imperative subjects without touching
   proper nouns;
3. compose the current event's domain noun phrase into generic recalled-project
   subjects and drop a redundant `item` slot.

That change requires new prospective cases. V8 must not be rerun or tuned on
these six cases, and this diagnostic does not justify opening sealed data.
