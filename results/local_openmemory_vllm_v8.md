# Real OpenMemory SDK v8 prospective diagnostic — not a hypothesis test

The one authorized fresh paired run **PASS**.
Both arms used the same frozen model, schema, prompt, and canonicalizer. Recall
was populated by the official CaviraOSS OpenMemory Python SDK through local
SQLite and deterministic synthetic embeddings. Provider identifiers remained
opaque and every case's scoped records were verified deleted.

| Case | Retrieved memory | Baseline correct | Recall correct | Cleanup | Changes B/R |
|---|---|---:|---:|---:|---:|
| omsdk8_harbor_coordinator | omsdk8_mem_harbor | false | true | true | 0/1 |
| omsdk8_glaze_shipment | omsdk8_mem_glaze | false | true | true | 3/2 |
| omsdk8_lichen_project | omsdk8_mem_lichen | false | false | true | 0/0 |
| omsdk8_current_lab_wins | omsdk8_mem_old_lab | false | false | true | 0/0 |
| omsdk8_injection_control | omsdk8_mem_injection | true | true | true | 0/0 |
| omsdk8_no_hit_control | none | false | false | true | 0/0 |

Retrieval 6/6; cleanup
6/6; baseline
1/6; recall 3/6;
helpful gain 2; safety regressions
0; accepted calls
12/12; canonicalizer transformations
6. Usage: 5575 input
and 1260 output tokens at `$0.0`
provider API cost. OpenMemory synthetic-embedding operations report no token or
cost accounting; electricity, hardware, and human effort are unmeasured.

This is a prospective development diagnostic, not a hypothesis test, latency
benchmark, or promotion of OpenMemory to authoritative temporal memory.
