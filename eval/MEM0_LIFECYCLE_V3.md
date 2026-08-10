# Mem0 + deterministic lifecycle filter v3

This development-only paired cell asks whether Anamnesis can safely consume
Mem0 recall without treating stale corrections or cancelled obligations as
active state.

The Mem0 side remains official `v2.0.17` automatic extraction and vector
retrieval. The additive Anamnesis layer receives only authored causal lifecycle
directives keyed by observable source-event IDs. It never trusts provider IDs,
never creates action evidence, and never infers a correction or cancellation
from retrieved prose.

The cell contains six fresh events and four queries. Raw Mem0 recall must expose
two stale-hit opportunities; filtered recall must exactly retain the corrected
preference, suppress the cancelled obligation, retain the unrelated project
fact, and preserve cross-user isolation.

This measures deterministic filtering **conditional on correct directives**.
It does not measure automatic lifecycle-directive extraction. Exactly one run is
allowed, with no retry, repair, cache, filter tuning, or prompt tuning.
