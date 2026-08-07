# Local smoke diagnostic analysis

This analysis refers only to the 10-scenario local diagnostic in
[`local_smoke.md`](local_smoke.md). It is not a hypothesis test and it does not
justify opening the 35-scenario development set or a sealed set.

## Decision

Do not expand the current `ollama/qwen3:4b-instruct` compiler/prompt
configuration. All four systems missed all 8 expected actions. Anamnesis also
increased input tokens by 23.5% relative to full-context and produced more
false reminders.

## Headline findings

- All systems: 0 true positives, 8 false negatives, F1 0.
- No memory, full-context, and vector RAG: 1 false reminder each.
- Anamnesis: 7 false reminders across 6 checkpoints and 2 invalid compiler
  outputs.
- Anamnesis input: 81,617 tokens, split into 41,920 decision tokens and 39,697
  compiler tokens.
- Full-context input: 66,078 tokens. The Anamnesis reduction target was missed
  by 53.5 percentage points: the required reduction was at least 30%, while the
  observed value was -23.5%.
- Provider API cost was exactly USD 0 for every system. Electricity and hardware
  cost were not measured.

## Observed failure modes

The strict scorer requires the correct root `action_key`, canonical payload,
time window, and an accepted minimal causal evidence set.

- Full-context recognized the assignment reminder in `s01`, but emitted a
  revised-event ID as the action key rather than the stable creating-event ID.
- Vector RAG recognized the flight-delay reminder in `s08`, but paraphrased the
  canonical subject and flight payload, so it did not satisfy the action
  contract.
- No memory emitted an unrelated tax-letter reminder in the address-update
  scenario `s04`.
- Anamnesis kept the correct root key for the assignment reminder but
  paraphrased its subject and omitted required causal evidence.
- Anamnesis fired an immediate passport reminder at creation time in `s04`, then
  emitted an unrelated tax-letter action with the wrong root key at `s04-e04`,
  before the visa reminder's due checkpoint.
- In recurring scenario `s05`, Anamnesis emitted four reminders rather than the
  two required occurrences and omitted the occurrence date from the payload.
- The threshold scenario `s08` contained both invalid Anamnesis compiler
  outputs; no correct action was emitted.

These failures show two different bottlenecks: the 4B writer does not reliably
compile canonical memory records, and the shared 4B decision model does not
reliably preserve the benchmark action contract even when relevant context is
available.

## Next experiments, smoke data only

1. Run the deterministic/oracle compiler ceiling on the same 10 cases. This is
   diagnostic only; it separates memory-store/trigger errors from writer errors
   and can never be reported as headline Anamnesis.
2. Add a writer-contract ablation that forbids paraphrasing canonical action
   slots and makes stable root-key copying explicit. Freeze it as a new prompt
   version before rerunning the smoke set.
3. Add a shared decision-contract ablation for all four systems, not just
   Anamnesis, to test whether canonical payload/key failures are caused by the
   common action policy rather than memory.
4. If the 4B model still produces invalid deltas or F1 0, reject it and test one
   stronger local model that fits the same hardware. Pin its exact blobs and
   create a new preflight/manifest; do not reuse these results.
5. Proceed to the 35 development scenarios only after a smoke configuration
   produces non-zero recall and passes the strict reporter without repair calls.

## Latency limitation

The Ollama server accumulated an 8+ GiB warm prompt cache during the sequential
baseline tasks and was restarted before Anamnesis to avoid macOS swap and disk
failure. Consequently, the recorded latency remains useful for operational
diagnosis within each run but is not a fair cross-system latency comparison.
Token, cost, accuracy, prompt/schema, and raw-call accounting remain fully
validated.
