# CLAUDE.md

This file provides context for Claude (or Claude Code) when working in this repository.

## Project Overview

This is a research project building an **Uncertainty-Guided Reversible Evidence Compression
RAG system**. The core idea: compress retrieved evidence aggressively to save tokens, but do
so *reversibly* and *uncertainty-aware*, so evidence can be restored if the compressed version
turns out insufficient to support a claim.

## Pipeline Stages (in order)

1. `retrieval.py` — retrieves candidate documents from the KB for a user query
2. `evidence_mapping.py` — maps retrieved evidence to individual claims (claim-wise mapping)
3. Initial evidence set — full, high-token evidence per claim (output of stage 2/3, not necessarily its own module)
4. `compression.py` — uncertainty-guided compression of evidence (reduces tokens)
5. `evaluation.py` — claim support evaluation; decides if compressed evidence is "enough"
6. If insufficient:
   - `restoration.py` — restores fuller evidence (reverses compression)
   - Re-run `evaluation.py` — re-check sufficiency
7. `generation.py` — final output generation once evidence is sufficient, OR abstain/clarify/retrieve-more if still insufficient after restoration

## Conventions

- Keep each pipeline stage as an independent, testable module in `src/`
- Uncertainty scores should be logged/returned alongside compression decisions for reproducibility
- Prefer explicit config (yaml/json) over hardcoded thresholds for "enough evidence" checks

## Current Priorities

- [ ] Fix mid-August deadline deliverable / define concrete outcome
- [ ] Get baseline pipeline (stages 1–7) working end-to-end
- [ ] Plan Fall '26–'27 extension for conference/journal submission

## When helping in this repo

- Assume Python, no fixed framework locked in yet — ask before introducing a new dependency (e.g. LangChain, LlamaIndex, FAISS) if one isn't already in `requirements.txt`
- Flag anywhere token-count assumptions (high-token vs reduced-token) aren't made explicit in code
- Preserve reversibility of compression — don't suggest lossy compression without a restoration path