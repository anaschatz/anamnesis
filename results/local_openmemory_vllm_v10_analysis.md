# Real OpenMemory SDK v10 analysis

The larger frozen v10 gate passed. Across 12 fresh paired cases, OpenMemory
retrieval and scoped cleanup were both 12/12, all 24 structured calls were
accepted, exact baseline accuracy was 4/12, and exact recall accuracy was 10/12.
Recall produced five of six preregistered helpful gains and zero safety
regressions.

## What generalized

- Two recipient/address cases became exact only after recall supplied the named
  office and street address. The inherited address-slot normalizer corrected
  `room` to `address` in both.
- Two shipment cases became exact only after recall supplied the identifier.
  Subject/item normalization remained stable on new nouns.
- Salt-marsh beetle recall supplied `Bronze Estuary`; canonicalizer v2 composed
  `upload salt-marsh beetle inspection brief` and removed the redundant item.
- The current-gallery control respected the current recipient over stale recall;
  an inherited redundancy rule removed a duplicated shipment slot.
- Both injection/neutral observation controls remained no-action in both arms.
- The article rule activated live for `photograph the restoration labels` and
  produced the canonical `photograph restoration labels` in both arms. The kiln
  case was already emitted without an article and required no transformation.

These outcomes extend the positive v9 result to twice as many fresh cases and
more paraphrase diversity. They remain development evidence rather than a
hypothesis test.

## Two residual failures

1. Glacier algae recall retrieved the correct project `Ice Meridian`, but the
   model emitted generic subject `upload status note` and omitted the event noun
   phrase from the `item` slot. Canonicalizer v2 deliberately requires that
   source-grounded item before composing the subject, so it failed closed. This
   is a coverage gap between safe composition and sparse model output.
2. In the Helios Studio control, both arms preserved current-event authority but
   emitted `room: not the former hall`. That negated contrast phrase is not a
   real room and is not removed by the exact-duplicate rule. The failure is
   payload canonicalization, not stale-memory override or recall regression.

The failures identify a precise next architecture revision: represent explicit
negative contrast clauses as forbidden slot values, and allow event-derived
upload composition when a recalled project is exact even if the model omits the
redundant item. Both changes require new adversarial unit tests and a new dataset;
v10 must not be rerun or tuned on these 12 cases.

Usage was 11,179 input and 2,558 output tokens at `$0.00` provider API cost.
Synthetic embedding operations expose no token/cost accounting; electricity,
hardware, and human effort were unmeasured. Latency was not a headline metric.
