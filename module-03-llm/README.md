# LLM Behavior Lab

Hands-on exploration of empirical LLM behavior — tokenization, sampling, hallucination, context position, and cost — using the Google Gemini API.

## What I Was Asked To Do

Per the module's hands-on/coding exercises and mini-project:
1. Compare token counts across languages and prose vs. code.
2. Test determinism at `temperature=0` vs. variation at `temperature=1.0`.
3. Deliberately induce hallucinations and observe model behavior.
4. Compare answer quality when a key fact sits at the start / middle / end of a long context ("lost in the middle").
5. Trigger and fix a truncated-JSON failure from a too-low `max_tokens`.
6. Estimate token cost growth across a multi-turn conversation.
7. Compare `top_p` settings at fixed temperature.
8. Build a small "Model Behavior Lab" (cost calculator, lost-in-the-middle test, hallucination suite, sampling comparison).
9. Write a one-page, non-technical "LLM Behavior Primer".

## What I Did

**`coding-excercises/excercise.py`** — Core utility functions: `token_estimate` (tiktoken), `manage_history` (trims old turns to a token budget), `cost_estimation`, `check_json` (detects truncated vs. malformed JSON), `budget_allocation` (splits tokens across system/history/retrieval), `token_bound_segment`, `check_hallucination`, plus scripts to compare temperature outputs, latency vs. `max_tokens`, and evaluate models on latency/cost/output.

**`hands-on-excercise/experiments.py`** — Runnable snippets (commented in/out) for: multilingual token counts, temperature determinism, hallucination prompts, "lost in the middle" using `text_start.txt` / `text_middle.txt` / `text_end.txt` (a synthetic archive with one hidden secret code `OBSIDIAN-7429-XR` placed at different positions), and `top_p` comparisons.

**`mini-project/notebooks/model_behaviour.ipynb`** — Notebook version of the mini-project: a `cost_count` function tested against sample token counts, a hallucination test suite (5 prompts) written to `hallutination_result.txt`, and a sampling-parameter sweep (5 `top_p` × 6 `temperature` combos) written to `sampling_result.txt`.

**`logs/experiment.log`** — Raw run history: multilingual token counts, repeated temperature=0 vs. 1.0 outputs, AI-generation prompts, hallucination probes (HR policy, gold price, fake library `fastvisionx`), a multi-turn conversation with growing token counts, and `top_p` story generations.

**`assignment/llm_behaviour_primer.txt`** — One-page, plain-language explainer for a non-AI teammate covering: what a token is, why context windows matter beyond "running out of room" (latency and cost), why hallucination happens, and how grounding + "permission to say I don't know" reduces it.

## Key Results

- **Determinism**: `temperature=0` gave near-identical outputs across 3 runs; `temperature=1.0` varied more (see log lines from 2026-08-03 13:25).
- **Hallucination suite**: model correctly refused/deflected on all 5 tricky prompts (future event, private diary, nonexistent object, private employee ID, missing document) instead of fabricating — see `hallutination_result.txt`.
- **Cost**: `gemini-3.5-flash-lite` real pricing (100 in / 200 out tokens) ≈ $0.00195 vs. demo `gemini-4` pricing ≈ $0.09.
- **Sampling**: higher `top_p` at fixed temperature produced more varied/creative story outputs (see log lines from 15:46).

## Folder Map
```
assignment/        one-page primer
coding-excercises/ reusable utility functions
hands-on-excercise/ exercise scripts + lost-in-the-middle test texts
logs/               raw experiment.log
mini-project/       notebook + saved results (hallucination, sampling)
```