# Mem0 automatic-memory diagnostic v2

This is the one permitted successor to the v1 infrastructure failure. It uses
seven fresh events and the same official Mem0 `v2.0.17` additive extraction
pipeline, default Mem0 prompts, Qwen3.5 9B artifact, seed, sampling, storage,
embedding, and semantic gates. It is development-only and not a hypothesis
test.

The only experimental-runtime changes are the corrected JSON-data audit
boundary and context length 32768. The larger context is fixed before any v2
model call because v1 contemporaneous server logs showed prompts as large as
8419 tokens being truncated to 4098. The v2 operator must observe the complete
Ollama server log; any `truncating input prompt` warning invalidates the run.

Exactly one attempt and seven model calls are allowed. There is no retry,
repair, cache, prompt tuning, or selection among outputs. An integrity failure
is published without semantic interpretation. A valid semantic failure is the
result. Any later prompt change requires fresh events.
