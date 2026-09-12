# Decoding Parameters (Ollama)

This repo standardizes a few key generation parameters so results are comparable across prompt families and models.

## Parameters

### `num_ctx` (context length)
- **What it is:** The maximum number of tokens the model keeps in its working window (prompt + response).
- **Why it matters:** Too small → your prompt or context truncates; too large (relative to the model) can slow generation and still truncate long inputs.
- **Typical values:** 2048–8192 for 7–8B SLMs. We commonly use **4096**.

### `num_predict` (max new tokens)
- **What it is:** The cap on tokens the model can generate in its response.
- **Why it matters:** Too small → the model may get cut off before producing “Answer: \<scalar\>”. Too large → extra latency and higher chance of rambling.
- **Typical values:** **256–512** for QA with short reasoning. We commonly use **512**.

### `temperature`
- **What it is:** Controls randomness. Lower = more deterministic; higher = more diverse.
- **Why it matters:** For baseline comparisons, use a low temperature (e.g., **0.0–0.2**). For **Self-Consistency**, use moderate temperature (e.g., **0.3–0.6**) and sample k times.
- **Tip:** Keep temperature and k fixed when comparing prompts, so differences come from **prompt structure**, not randomness.

### `seed`
- **What it is:** PRNG seed for sampling operations.
- **Why it matters:** Fixing the seed improves reproducibility (especially at low temperature). With Self-Consistency, you can vary seeds across samples or hold them constant for A/B tests.

## Recommended Baselines

- **Deterministic baseline:** `temperature=0.0–0.2`, `seed=1234`
- **Self-Consistency profile (k=5):** `temperature=0.4–0.5`, run 5 independent samples and majority-vote the final `Answer:` token

## Practical Notes

- Always enforce a **single final line**: `Answer: <scalar>`, and add stop sequences to avoid trailing commentary.
- Log **latency** and **token counts** with each run to understand accuracy vs. budget trade-offs.
- Keep the same decoding profile across prompt families when doing head-to-head comparisons.
