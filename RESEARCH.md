# Anamnesis

## Research question
Can an explicit temporal and provenance-aware memory improve an
LLM agent's ability to execute future intentions compared with
full-context prompting and vector RAG?

## Hypothesis
Separating facts, events and future intentions will improve execution
accuracy and reduce obsolete-memory errors, using fewer input tokens.

## Initial scope
- Text only
- One simulated user
- Seven simulated days
- 50 scenarios
- No UI
- No model training initially

## Baselines
1. No persistent memory
2. Full conversation history
3. Basic vector RAG

## Metrics
- Precision: how many triggered actions were correct
- Recall: how many required actions were executed
- False-alarm rate
- Obsolete-memory errors
- Input tokens and cost

## Definition of done for v0
Run all three baselines on the same 50 scenarios and produce one
reproducible results table.
