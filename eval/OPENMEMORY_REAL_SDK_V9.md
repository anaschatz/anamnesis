# Real OpenMemory SDK v9 diagnostic

This is a development-only, non-hypothesis paired diagnostic of deterministic
immediate-action canonicalizer v2. It uses six fresh `omsdk9_*` cases that were
not present in v8. The upstream OpenMemory Python SDK remains a local,
non-authoritative retrospective-recall index; it cannot create action evidence
or mutate Anamnesis temporal state.

The only intended architecture change from v8 is
`immediate-action-canonicalizer.v2`, with three source-grounded rules:

1. remove a `room` or `address` only when it exactly duplicates an event-grounded
   recipient;
2. remove a leading simple English article from an event-quoted imperative
   direct object;
3. compose a generic upload subject with the event's domain noun phrase only
   when a named project is present in retrieved recall, then remove the
   redundant `item` slot.

The model, decision prompt/schema, OpenMemory SDK revision, synthetic embedding
provider, seed, temperature, retry/cache policy, top-k, and six-case paired
baseline/recall design remain fixed. The run is exactly one complete matrix of
12 model calls. No selection, retry, repair, or second v9 run is allowed.

The gate requires 6/6 retrieval and cleanup checks, 12/12 accepted structured
calls, at least two helpful recall gains, no recall-induced safety regression,
and higher recall exact accuracy than baseline exact accuracy. Provider API cost
is zero; electricity, hardware, and human effort remain unmeasured.
