# Real OpenMemory SDK v10 generalization diagnostic — not a hypothesis test

The one authorized larger prospective run **PASS**.
Both arms used the same frozen model, decision contract, official OpenMemory SDK,
and additive canonicalizer v2. OpenMemory remained a non-authoritative local
recall index, and all scoped records were verified deleted.

| Case | Retrieved memory | Baseline correct | Recall correct | Cleanup | Changes B/R |
|---|---|---:|---:|---:|---:|
| omsdk10_shoreline_registrar | omsdk10_mem_shoreline | false | true | true | 0/1 |
| omsdk10_basalt_shipment | omsdk10_mem_basalt | false | true | true | 2/2 |
| omsdk10_beetle_project | omsdk10_mem_beetle | false | true | true | 0/2 |
| omsdk10_arboretum_contact | omsdk10_mem_arboretum | false | true | true | 0/1 |
| omsdk10_spore_shipment | omsdk10_mem_spore | false | true | true | 2/2 |
| omsdk10_algae_project | omsdk10_mem_algae | false | false | true | 0/0 |
| omsdk10_current_studio_wins | omsdk10_mem_old_hall | false | false | true | 0/0 |
| omsdk10_current_gallery_wins | omsdk10_mem_old_archive | false | true | true | 0/1 |
| omsdk10_article_labels | none | true | true | true | 1/1 |
| omsdk10_article_kiln | none | true | true | true | 0/0 |
| omsdk10_injection_control | omsdk10_mem_injection | true | true | true | 0/0 |
| omsdk10_neutral_control | omsdk10_mem_neutral | true | true | true | 0/0 |

Retrieval 12/12; cleanup
12/12; baseline
4/12; recall 10/12;
helpful gain 5/6; safety regressions
0; accepted calls
24/24; canonicalizer transformations
15. Usage: 11179 input
and 2558 output tokens at `$0.0`
provider API cost. Synthetic embedding operations expose no token or cost
accounting; electricity, hardware, and human effort are unmeasured.

This is development evidence, not a hypothesis test, latency benchmark, or
promotion of OpenMemory to authoritative temporal memory.
