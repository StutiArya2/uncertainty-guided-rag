# Paper outline

Working title:

> **Reversible Evidence Compression for RAG: selection errors become lost savings, and
> retrieval confidence does not help you avoid them**

Every number below traces to a run in `artifacts/`. Numbers still marked *(pending)* are
awaiting the definitive runs; do not quote them until they are filled in from the archived
JSON.

---

## The claim we can defend

Three results, in decreasing order of strength:

1. **Ranked extractive selection is worth a large, significant saving at equal answer
   quality.** At a binding budget, ranked selection realises ~53% evidence-token reduction
   against ~34% for random, with no distinguishable difference in answer F1. The mechanism
   is measured, not asserted: random selection drops needed evidence more often, triggers
   restoration more often, and restoration hands the savings back.

2. **Uncertainty-guided allocation adds nothing over a uniform budget.** Bounded null at
   two budget regimes, with the sign flipping between them. The founding premise —
   *confident retrieval implies redundant evidence* — is false here, and we can say why:
   retrieval confidence carries almost no information about whether the evidence answers
   the question (top-ranked unit contains the gold answer 22% of the time against a 12%
   chance baseline; correlation +0.045).

3. **The restoration trigger misses most of the losses it exists to catch.** Against
   human-marked evidence, the absolute trigger fails to fire on 88.9% of questions where
   compression removed needed evidence, and on 6 of 6 where it removed all of it. A
   relative trigger roughly halves this. Reported as a finding and an open problem.

## What we must not claim

- **Not** "uncertainty-guided compression works." It does not, and we tested it properly.
- **Not** "the system knows when it doesn't know." 11 unanswerable questions cannot
  support that, and the evidence-level results argue against it.
- **Not** "reversibility guarantees safety." Reversibility guarantees *recoverability*.
  Detection is separate and is where the system is weakest.
- **Not** open-corpus RAG. See scope below.

---

## Structure

### 1. Introduction

Adaptive evidence compression assumes a system can tell which evidence it can afford to
drop. We test that assumption directly and it fails — but the *reversible* framing that
was supposed to make adaptivity safe turns out to be the interesting part, for a reason we
did not anticipate: it changes where selection errors land.

### 2. Scope — state this early and plainly

**This is known-document long-document QA, not open-corpus RAG.** QASPER questions are
asked about one specific paper and are not self-identifying ("which datasets did *they*
experiment with?"). Retrieval is restricted to the named paper (`restrict_to`,
`src/pipeline.py`). A deployed open-corpus system must first *find* the document, and we
do not evaluate that step.

This is not a detail. Run without the restriction, the baseline scored ~1% F1 and abstained
on 83% of answerable questions — retrieval confidently returned evidence from the wrong
paper. Document selection is a hard, separate problem and our numbers assume it solved.

### 3. Method

- Span-based extractive compression: evidence is `(doc_id, start, end)`, never a string.
  Compression drops spans from the prompt, never from the index.
- Restoration re-resolves dropped spans **from the corpus**, verified against the retained
  copies. Round-trip integrity checked on all 287 questions, both paths agreeing.
- Claim decomposition from the query; support scored by relevance, not entailment
  (query-derived claims are topics, not assertions — NLI scores them near zero).

### 4. Experimental design

| arm | selection | ablates |
|---|---|---|
| `identity` | keep everything | baseline |
| `uncertainty_guided` | budget scales with retrieval uncertainty | — (the proposal) |
| `fixed_ratio` | constant fraction, top-ranked | adaptivity |
| `random` × 20 seeds | constant fraction, random | adaptivity **and** ranking |
| `oracle` | units overlapping human-marked evidence | headroom ceiling |

Design decisions worth defending in text:

- **Budget-matched, not setting-matched.** Ablation arms are calibrated to the budget the
  uncertainty arm actually spent. Comparing at equal settings would let an arm that simply
  keeps more evidence look better on quality and worse on cost.
- **Paper-level inference.** Questions cluster within papers; a paired test over questions
  is pseudo-replication. CIs come from a bootstrap resampling **papers**, p-values from a
  paper-level sign-flip permutation test.
- **Held-out split primary.** The support threshold is calibrated on dev papers only.
- **Random is a distribution.** Over seeds its reduction ranged 34.8–58.0% on a smoke test;
  a single draw would have been an anecdote in either direction.
- **A minimum effect declared in advance** (0.02 F1, ~12% relative) so a null can be
  reported as a bounded null rather than as "inconclusive".

### 5. Results

#### 5.1 Cost — report both numbers

Evidence-token reduction and **end-to-end prompt-token reduction**. The latter is lower:
the preamble, per-claim labels, title prefixes, question and chat template do not compress.
Only the second is a cost claim.

#### 5.2 The ablation

Uncertainty vs fixed at matched budget, both regimes, clustered CIs. *(pending)*

#### 5.3 Ranking, and the mechanism

Ranked vs random: reduction, restoration rate, F1. The causal chain from dropped evidence
→ restoration → forfeited savings. *(pending)*

#### 5.4 Evidence-level safety — the new contribution

Against human-marked evidence: gold-evidence recall before restoration, restoration
detector recall, and the **dangerous false-negative rate**. Absolute vs relative trigger.
*(pending)*

#### 5.5 Oracle ceiling

How much of the achievable saving ranked selection captures. Without this, "53%" has no
scale. *(pending)*

### 6. Limitations — write these, do not let a reviewer find them

1. **One generator (Qwen2.5-0.5B), one retriever (MiniLM), one dataset (QASPER).** A
   robustness run at 1.5B addresses the "generator too weak to reward better evidence"
   objection; a second retriever and dataset are not attempted.
2. **Abstention is not evaluated.** 11 unanswerable questions. We report what we have and
   claim nothing. The clean fix is *more real QASPER papers*, not invented negatives —
   authoring our own negatives is precisely the failure mode that produced a +0.99 result
   from a corpus we wrote ourselves.
3. **12% of marked evidence is table or figure captions**, absent from our text-only
   corpus. Those questions are unanswerable from our KB through no fault of the
   compressor, and they inflate the false-abstain rate. Reported separately.
4. **Evidence alignment succeeds on ~84% of marked passages.** Evidence-level metrics are
   computed over that subset and say so.
5. **Answer F1 is a weak proxy for grounding.** One case scored F1 0.33 having lost 100%
   of the marked evidence — the generator answered from other text or from parametric
   knowledge. This is the reason §5.4 exists.
6. **Restoration and abstention were entangled** through one threshold; separating them is
   new and only the absolute policy is used by the historical results.

### 7. What would change our mind

Stated so the negative result is falsifiable rather than merely asserted:

- A different uncertainty estimator. We tested retrieval-score uncertainty only; logprob
  and self-consistency estimators drop into the same interface and are untested. **This
  should be pre-declared before running**, not chosen after seeing results.
- A budget harsh enough that allocation dominates. We tested keep ≈ 0.65 and ≈ 0.29 and
  saw no adaptivity effect at either, but the space is not exhausted.
- A stronger support signal. Every failure in this paper traces back to the same place:
  the scorer cannot tell answer-bearing evidence from on-topic evidence. Fix that and both
  the allocation policy and the restoration trigger have something real to run on.

---

## Reproducibility

Every result JSON in `artifacts/` embeds its commit, resolved config, dataset hashes,
resolved model revisions and environment. `scripts/make_bundle.py --check` refuses to
certify a run made from a dirty tree.

```bash
python scripts/make_bundle.py artifacts/qasper-main --check
```
