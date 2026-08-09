# Mem0 automatic-memory diagnostic v1 — infrastructure failure

The one authorized `infer=true` Mem0 run is **not an interpretable memory-quality
result**. Official Mem0 `v2.0.17` did execute all seven planned local extraction
calls, and all seven returned `done_reason=stop`, but the runner failed while
constructing the final Pydantic result object. No provider API was used and no
second attempt was made.

## What failed

The runner was invoked with `python -m`, so its audit class existed under the
`__main__` module. Mem0's provider factory independently imported the same file
through `anamnesis.mem0_inference_diagnostic`. The seven resulting
`LlmCallAudit` objects had identical fields but a different Python class
identity from the type expected by `Mem0InferenceResult`. Pydantic rejected all
seven at the final serialization boundary.

This is an Anamnesis runner defect, not evidence that Mem0 passed or failed
extraction, deduplication, correction, cancellation, or scope isolation. The
event-level states existed only in the terminated process and are not
reconstructed from partial logs.

## Additional transport finding

The Ollama server was configured and resident at an 8192-token context, but its
terminal recorded a 4098-token input limit and explicit prompt truncations of
8374→4098, 8419→4098, and 8220→4098 tokens. Consequently even a successfully
serialized result from this attempt would have required a context-fidelity
caveat. Mem0's additive prompt grew beyond the effective input budget as prior
memories accumulated.

## Disposition

The serializer now crosses the factory boundary as JSON-compatible data and a
regression test covers the exact dual-module failure. The v1 stopping rule is
still binding: this run is published as an integrity failure, its semantic
metrics are not interpreted, and the same seven cases will not be rerun. A new
measured attempt requires a separately frozen protocol with fresh events and an
explicit context-fit preflight.

The machine-readable record is
[`mem0_inference_v1_failure.json`](mem0_inference_v1_failure.json).
