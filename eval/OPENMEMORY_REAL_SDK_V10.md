# Real OpenMemory SDK v10 generalization diagnostic

V10 is a larger prospective development diagnostic over 12 fresh cases and 24
paired model calls. Six cases require retrieved parameters; six are controls
covering current-event authority, article normalization, prompt injection, and
neutral observations. No `omsdk8_*` or `omsdk9_*` case is reused.

The architecture is frozen to the v9 model, decision prompt/schema, official
OpenMemory SDK v1.3.0, synthetic local embeddings, and additive deterministic
canonicalizer v2. OpenMemory is retrospective recall only and cannot contribute
action evidence or mutate authoritative temporal state.

Exactly one complete matrix is authorized. The gate requires retrieval and
scoped cleanup 12/12, accepted structured calls 24/24, at least five of six
helpful recall gains, zero recall-induced control regression, and recall exact
accuracy above baseline exact accuracy. No retry, repair, cache, case selection,
or second v10 run is allowed. This is not a hypothesis test or latency claim.
