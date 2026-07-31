# Uncertainty-Guided Reversible Evidence Compression RAG System

A Retrieval-Augmented Generation (RAG) pipeline that compresses retrieved evidence in an
uncertainty-aware, *reversible* manner — reducing token load while preserving the ability
to restore evidence when claim support is insufficient.

Runs fully locally: no API keys, no per-query cost.

## Pipeline Overview

1. **User Query** → submitted by the user
2. **Retrieval** → pulls candidate documents from the Knowledge Base (KB)
3. **Evidence Mapping** → maps retrieved evidence to individual claims (claim-wise)
4. **Initial Evidence Set** → full, high-token evidence set per claim
5. **Uncertainty-Guided Compression** → compresses evidence based on uncertainty estimates, reducing token count
6. **Claim Support Evaluation** → checks whether compressed evidence is sufficient to support each claim
   - **Sufficient** → proceed to Output Generation
   - **Insufficient** →
     6.1. **Evidence Restoration** → reverses compression to recover fuller evidence
     6.2. **Re-evaluation** → re-checks claim support with restored evidence
       - **Sufficient** → Output Generation
       - **Still insufficient** → **Abstain / Clarify / Retrieve More**
7. **Output Generation** → final answer generation once evidence is sufficient

Restoration is applied **per claim**: claims whose compressed evidence was already
sufficient keep their savings, and only the claims that actually failed pay their full
evidence cost back.

## Current Results

Measured on the seeded 24-document KB over a 16-question eval set
(`python scripts/run_eval.py`):

| metric | no compression | uncertainty-guided |
|---|---|---|
| tokens sent to model | 2496 | 1414 |
| **token reduction** | — | **43.3%** |
| keyword recall (answer quality) | 56.7% | 60.0% |
| restoration rate | 0% | 6.2% |
| false abstain (answerable questions) | 0% | 0% |
| correct abstain (unanswerable questions) | 100% | 100% |
| reversibility on real evidence | PASS | PASS |

Compression removes 43% of evidence tokens **without costing answer quality** — keyword
recall is marginally higher with compression than without, consistent with the known
"lost in the middle" effect where trimming weak evidence helps as much as it hurts.
Abstention behaviour is identical to the uncompressed baseline. Restoration fires on 6.2%
of questions — the cases where compression removed something that turned out to be
needed, and recovered it.

Caveats worth stating: 16 questions is a smoke test, not a benchmark, and keyword recall
is a coarse quality proxy (it detects evidence destruction, it is not accuracy). Both
arms are otherwise identical, so the comparison between them is fair.

## How It Works

### Reversibility

Evidence is never a free-floating string. Every unit is a **`Span`** — `(doc_id, start,
end)` character offsets into a source document. Compression drops spans from the
*prompt*; it never destroys them. Restoration re-resolves dropped spans **from the
corpus**, so reversibility is a verifiable property of the span index rather than an
artifact of keeping a backup copy in memory.

`tests/test_compression.py` asserts that both recovery routes — reinstating retained
copies, and re-reading text from spans — produce byte-identical evidence, at every
compression level. `scripts/run_eval.py` re-checks the same property on real retrieved
evidence rather than fixtures.

**What is actually compressed:** the saving is in tokens *sent to the generator*, not in
RAM. Dropped units stay in memory so restoration is immediate.

### Uncertainty

Uncertainty is derived from retrieval scores — no extra model calls, fully deterministic:

```
u_claim    = w_score * (1 - s_top) + w_margin * (1 - margin / margin_scale)
keep_ratio = min_keep + u_claim * (max_keep - min_keep)
```

Two independent failure signals are combined. A **low top score** means nothing in the KB
matches the claim well. A **narrow margin** means the top hit is barely ahead of the next,
so the ranking is unstable and dropping the rest is risky precisely there.

Low uncertainty → compress hard. High uncertainty → keep more. `UncertaintyEstimator` is
an interface; token-logprob and self-consistency estimators are planned as ablations.

### Claim support

Claims are decomposed from the **query**, which avoids the circularity of verifying text
the model just invented. That choice determines how support must be scored.

Query-derived claims are *topics*, not assertions, so NLI entailment is the wrong tool for
them — measured with DeBERTa-v3-large-mnli, "dense retrieval differ from lexical
retrieval" scored **0.009** while the bare noun "dense retrieval" scored **0.877**. The
model was answering its own question correctly, just not ours, and it produced a 14.3%
false-abstain rate on questions the KB plainly answers.

A **relevance cross-encoder** asks what we actually mean — "does this evidence address
this claim?" — and separates cleanly on the same set (lowest answerable claim 0.205,
highest unanswerable 0.001), taking false abstentions to **0%**.

Both scorers ship behind one interface (`evaluation.scorer` in config). NLI becomes the
correct choice if claims are ever decomposed from a draft answer instead, since those
*are* assertions.

Evidence is scored **title-prefixed**. Sentence chunks are often anaphoric — "The function
has two tunable parameters" never names BM25 — and without the document title the scorer
cannot resolve the subject (measured 0.004 → 0.997 for the same chunk).

## Tech Stack

- **Python 3.14**, PyTorch (MPS/CUDA/CPU auto-selected)
- **Retrieval** — mean-pooled `all-MiniLM-L6-v2` embeddings, exact brute-force cosine
  over numpy. No FAISS: for a KB this size exact search is fast, always returns the true
  nearest neighbours, and avoids approximate-index tuning as a silent source of recall loss.
- **Claim support** — `cross-encoder/ms-marco-MiniLM-L-6-v2` (relevance) or
  `DeBERTa-v3-large-mnli-fever-anli-ling-wanli` (entailment)
- **Generation** — `Qwen2.5-0.5B-Instruct`, greedy decoding for reproducibility.
  `Qwen2.5-1.5B-Instruct` is a drop-in upgrade; the default is the smaller model so the
  shipped config matches the numbers reported above.
- Dependencies: `torch`, `transformers`, `numpy`, `pyyaml`, `pytest` — that is all

## Repository Structure

```
uncertainty-guided-rag/
├── README.md
├── CLAUDE.md
├── requirements.txt
├── config/
│   └── default.yaml          # every threshold lives here, none in code
├── src/
│   ├── types.py              # Span / EvidenceUnit / CompressedEvidence
│   ├── config.py             # config loading + validation
│   ├── tokens.py             # measured token counts
│   ├── trace.py              # per-query instrumentation
│   ├── kb.py                 # corpus loading + offset-preserving chunking
│   ├── retrieval.py          # stage 1
│   ├── claims.py             # stage 2a — query -> claims
│   ├── evidence_mapping.py   # stages 2-3
│   ├── uncertainty.py        # stage 4a
│   ├── compression.py        # stage 4
│   ├── evaluation.py         # stages 5 and 6.2
│   ├── restoration.py        # stage 6.1
│   ├── generation.py         # stage 7
│   ├── pipeline.py           # orchestration
│   ├── ingest.py             # adding papers (text, .txt, optional PDF)
│   └── server.py             # local web interface (stdlib only)
├── web/
│   └── index.html            # the reading room, self-contained
├── data/
│   ├── kb/                   # 24 seeded documents
│   └── eval/questions.yaml   # eval set
├── scripts/
│   ├── seed_kb.py            # regenerate the KB
│   └── run_eval.py           # baseline comparison — the headline result
├── notebooks/
└── tests/                    # 106 tests
```

## Setup

```bash
git clone https://github.com/StutiArya2/uncertainty-guided-rag.git
cd uncertainty-guided-rag

# --system-site-packages reuses an already-installed torch instead of
# re-downloading ~2.5 GB. Drop it for a fully isolated environment.
python3 -m venv --system-site-packages venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

python scripts/seed_kb.py         # write the starter knowledge base
```

Model weights download automatically on first run (~4 GB total).

## The reading room (web interface)

```bash
python -m src.server        # opens http://127.0.0.1:8000
```

A local, non-technical way to use the system: add papers, ask questions, and see the
exact sentences each answer was built from.

It is built on the standard library's `http.server` — no Flask, FastAPI, Streamlit, or
Gradio — because a GUI is not a good reason to add a web framework to a five-package
dependency list. Everything is served from `web/index.html`.

What it shows, and why:

- **The marked-up passage.** Sentences the system used are highlighted; sentences it set
  aside stay visible but dimmed. Compression stops being a statistic and becomes
  something you can see and check.
- **Plain language, not scores.** `uncertainty 0.496` becomes "Reasonable match";
  `support 0.72` becomes "Evidence: Strong". Raw numbers stay one click away under
  **Technical detail**.
- **Held back is a first-class outcome.** When the papers don't answer the question, the
  interface says so in the answer's place and suggests what to do — it never fills the
  gap with a guess.
- **Restoration is visible.** If compression removed something that turned out to be
  needed, those sentences are tagged `set aside · brought back`.

Papers are stored as plain `.txt` in `data/kb/`, identical to what `scripts/seed_kb.py`
writes, so anything added through the browser works with every other tool in the repo.
Adding a paper re-embeds the index; the scorer and generator stay loaded.

PDF import needs `pypdf`, which is deliberately **not** in `requirements.txt` (see the
dependency note in CLAUDE.md). Without it, pasting text and `.txt` upload still work and
the interface says exactly how to enable PDFs.

### What the interface is good at showing you

Asking a freshly added paper "What are global tokens in sparse attention?" produced a
**confidently wrong answer** — it described sliding-window attention instead, because
compression had set aside the one sentence that defined global tokens, and the support
scorer still rated the remaining evidence 0.72, so restoration never fired.

The interface made that visible: the dropped sentence is right there on screen, greyed
out, directly above an answer that contradicts it. That is the honest argument for
showing evidence rather than only answers — and it is a real limitation of the current
support scorer, recorded under Planned extensions below.

## Command line

```bash
# Single query, with the stage-by-stage trace
python -m src.pipeline --query "What are the two tunable parameters of BM25?" --trace

# Machine-readable trace
python -m src.pipeline --query "What is calibration?" --json

# No-compression baseline, for comparison
python -m src.pipeline --query "What is calibration?" --mode identity --trace

# The headline result: compressed vs uncompressed
python scripts/run_eval.py
python scripts/run_eval.py --no-generate       # token/branch metrics only, fast
```

## Tests

```bash
pytest tests/ -v                     # everything
pytest tests/ -m "not integration"   # skip tests needing model weights
pytest tests/test_compression.py -v  # the reversibility gate
```

`tests/test_compression.py::TestRoundTrip` is the gate for the project: compression is
only legitimate if it can be undone exactly. If it fails, every other result is
untrustworthy.

## Project Status

- Target milestone: Mid-August — baseline pipeline (stages 1–7) working end-to-end **and
  measured against a no-compression baseline** ✅
- Longer-term goal: Fall '26–'27 mini-project extension, targeting conference/journal
  submission (indexed publication)

### Planned extensions

- **Topical support is not answer support.** The relevance scorer rates evidence that is
  *about* the claim highly, even when the specific sentence that answers it has been
  dropped — measured live in the web interface: support 0.72 on evidence that produced a
  wrong answer, so restoration never triggered. Detecting "on-topic but not answering"
  is the most valuable open problem in the pipeline, and the strongest argument for
  decomposing claims from a draft answer instead of the query.
- **Graded restoration** — restore in tiers and re-check between them, instead of jumping
  straight back to full evidence, preserving more of the saving on the restore path
- **Alternative uncertainty estimators** — token logprobs, self-consistency sampling; the
  ablation comparing them is the natural headline table for a paper
- **Draft-answer claim decomposition** — verifies exactly what will be shown to the user,
  and makes NLI entailment the correct support scorer
- **Abstractive compression** — higher ratios, but not invertible from its output alone,
  so it needs a different reversibility mechanism
- **Larger evaluation** — the current 16-question set is a smoke test, not a benchmark

## Collaborators

- Stuti Arya — [@StutiArya2](https://github.com/StutiArya2)
