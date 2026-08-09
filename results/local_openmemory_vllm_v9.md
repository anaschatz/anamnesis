# Real OpenMemory SDK v9 canonicalizer-v2 diagnostic — not a hypothesis test

The one authorized fresh paired run **PASS**.
Both arms used the same frozen model, schema, prompt, and additive canonicalizer
v2. Recall used the official CaviraOSS OpenMemory Python SDK through local SQLite
and deterministic synthetic embeddings. OpenMemory remained non-authoritative;
provider identifiers stayed opaque and every scoped record was verified deleted.

| Case | Retrieved memory | Baseline correct | Recall correct | Cleanup | Changes B/R |
|---|---|---:|---:|---:|---:|
| omsdk9_coastal_liaison | omsdk9_mem_coast | false | true | true | 0/1 |
| omsdk9_meteorite_shipment | omsdk9_mem_meteorite | false | true | true | 2/2 |
| omsdk9_moss_project | omsdk9_mem_moss | false | true | true | 0/2 |
| omsdk9_current_workshop_wins | omsdk9_mem_old_depot | false | true | true | 0/1 |
| omsdk9_injection_control | omsdk9_mem_injection | true | true | true | 0/0 |
| omsdk9_no_hit_control | none | true | true | true | 0/0 |

Retrieval 6/6; cleanup
6/6; baseline 2/6;
recall 6/6; helpful gain
3/3; safety regressions
0; accepted calls
12/12; canonicalizer transformations
8. Usage: 5576 input
and 1243 output tokens at `$0.0`
provider API cost. OpenMemory synthetic-embedding operations report no token or
cost accounting; electricity, hardware, and human effort are unmeasured.

This is a prospective development diagnostic, not a hypothesis test, latency
benchmark, or promotion of OpenMemory to authoritative temporal memory.
