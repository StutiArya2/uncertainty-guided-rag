# CLAUDE.md

This file provides context for Claude (or Claude Code) when working in this repository.

## Project Overview

This is a research project building an **Uncertainty-Guided Reversible Evidence Compression
RAG system**. The core idea: compress retrieved evidence aggressively to save tokens, but do
so *reversibly* and *uncertainty-aware*, so evidence can be restored if the compressed version
turns out insufficient to support a claim.

The baseline pipeline (stages 1–7) is implemented and measured. Everything runs locally —
no API keys, no per-query cost.

## Pipeline Stages (in order)

1. `retrieval.py` — retrieves candidate documents from the KB for a user query
2. `evidence_mapping.py` — maps retrieved evidence to individual claims (claim-wise mapping)
3. Initial evidence set — full, high-token evidence per claim (output of stage 2, not its own module)
4. `compression.py` — uncertainty-guided compression of evidence (reduces tokens);
   uncertainty itself is estimated in `uncertainty.py`
5. `evaluation.py` — claim support evaluation; decides if compressed evidence is "enough"
6. If insufficient:
   - `restoration.py` — restores fuller evidence (reverses compression)
   - Re-run `evaluation.py` — re-check sufficiency
7. `generation.py` — final output generation once evidence is sufficient, OR abstain/clarify/retrieve-more if still insufficient after restoration

`pipeline.py` orchestrates all of it. `claims.py` (stage 2a) decomposes the query into
claims before mapping.

`server.py` + `web/index.html` are a local web interface over the same pipeline
(`python -m src.server`), and `ingest.py` handles adding papers. The UI is built on
stdlib `http.server` with no web framework — keep it that way; the dependency rule below
applies to Flask/FastAPI/Streamlit/Gradio too. It is a thin layer: it must not reimplement
pipeline logic, only present what `PipelineTrace` already records.

## Load-Bearing Design Decisions

Change these only deliberately — each is the answer to a problem that was measured, and the
measurements are recorded in the relevant module docstring.

- **Spans, not strings.** Evidence is `(doc_id, start, end)` offsets into a source document.
  This is what makes compression reversible. `chunk.text` must always equal
  `corpus[doc_id].text[start:end]`; `tests/test_kb.py` enforces it.
- **Support scoring is relevance, not entailment — because claims come from the query.**
  Query-derived claims are topics, not assertions, and NLI scores them near zero even when
  the KB plainly answers them ("dense retrieval differ from lexical retrieval" → 0.009).
  If claim decomposition ever switches to draft-answer decomposition, switch
  `evaluation.scorer` to `nli` — those claims *are* assertions.
- **Evidence is title-prefixed before scoring.** Chunks are anaphoric; without the document
  title the scorer cannot resolve the subject (0.004 → 0.997 on the same chunk).
- **Units are scored individually, aggregated with max.** Concatenating a claim's evidence
  dilutes the signal (0.997 → 0.510).
- **`identity` compression mode is not dead code.** It is the no-compression baseline arm
  that `scripts/run_eval.py` measures reduction against.

## Conventions

- Keep each pipeline stage an independent, testable module in `src/`
- Uncertainty scores are logged/returned alongside compression decisions for reproducibility
  (see `trace.py` — every run records uncertainty, keep ratio, token counts, and branch)
- Explicit config (`config/default.yaml`) over hardcoded thresholds — there should be no
  bare numeric threshold anywhere in `src/`
- Greedy decoding by default so runs are reproducible

## Testing

```bash
pytest tests/ -v                     # 106 tests
pytest tests/ -m "not integration"   # skip tests needing model weights
```

`tests/test_compression.py::TestRoundTrip` is the gate for the whole project: compression is
only legitimate if it can be undone exactly. If it fails, nothing downstream is trustworthy.
`scripts/run_eval.py` re-verifies the same property on real retrieved evidence.

## Current Priorities

- [x] Fix mid-August deadline deliverable / define concrete outcome
- [x] Get baseline pipeline (stages 1–7) working end-to-end
- [x] Measure against a no-compression baseline (43.3% token reduction, keyword recall
      56.7% -> 60.0%, 0% false abstain, reversibility PASS)
- [ ] Expand the evaluation set — 16 questions is a smoke test, not a benchmark
- [ ] Plan Fall '26–'27 extension for conference/journal submission

## When helping in this repo

- Dependencies are deliberately minimal: `torch`, `transformers`, `numpy`, `pyyaml`,
  `pytest`. Ask before introducing anything else (LangChain, LlamaIndex, FAISS,
  sentence-transformers) — FAISS and sentence-transformers were evaluated and dropped
  on purpose, not overlooked
- Flag anywhere token-count assumptions (high-token vs reduced-token) aren't made explicit
  in code. `tokens.py` reports `is_exact`; a reduction measured with estimated counts must
  say so
- Preserve reversibility of compression — don't suggest lossy compression without a
  restoration path
- When changing a threshold or model, re-run `scripts/run_eval.py` and update the results
  table in `README.md`. Numbers in docstrings are measurements, not decoration
