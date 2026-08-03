"""End-to-end orchestration, including the restoration loop and the abstain branch.

    query
      -> decompose into claims                     (claims.py)
      -> retrieve + map evidence claim-wise        (evidence_mapping.py)
      -> uncertainty-guided compression            (compression.py)
      -> claim support evaluation                  (evaluation.py)
           sufficient   -> generate                (generation.py)
           insufficient -> restore                 (restoration.py)
                           -> re-evaluate          (evaluation.py, same instance)
                                sufficient   -> generate
                                insufficient -> abstain / clarify / retrieve more

Restoration is applied per claim, not globally: a claim whose compressed evidence was
already sufficient keeps its savings, and only the claims that actually failed pay the
cost of their full evidence set back.
"""

from __future__ import annotations

import argparse
import logging
import sys

from .claims import decompose
from .compression import compress, to_claim_evidence
from .config import Config, default_config
from .evaluation import SupportEvaluator
from .evidence_mapping import map_evidence
from .generation import Generator, abstain, build_prompt, generate_answer
from .restoration import restore
from .retrieval import Retriever
from .tokens import TokenCounter, shared_counter
from .trace import ClaimTrace, EvidenceRecord, PipelineTrace
from .types import Answer, Branch, ClaimEvidence, EvidenceUnit

logger = logging.getLogger(__name__)


def _record(units: list[EvidenceUnit]) -> list[EvidenceRecord]:
    return [
        EvidenceRecord(
            doc_id=u.span.doc_id,
            text=u.text,
            score=round(u.retrieval_score, 4),
            start=u.span.start,
            end=u.span.end,
        )
        for u in units
    ]


class Pipeline:
    """Holds the loaded models so they are paid for once, not per query."""

    def __init__(
        self,
        cfg: Config | None = None,
        retriever: Retriever | None = None,
        evaluator: SupportEvaluator | None = None,
        generator: Generator | None = None,
        counter: TokenCounter | None = None,
    ) -> None:
        self.cfg = cfg or default_config()
        self.retriever = retriever or Retriever(cfg=self.cfg)
        self.evaluator = evaluator or SupportEvaluator(cfg=self.cfg)
        self.generator = generator or Generator(cfg=self.cfg)
        self.counter = counter or shared_counter(self.cfg)

    def _choose_abstain_branch(
        self, evidence: list[ClaimEvidence], unsupported: list[str]
    ) -> tuple[Branch, str]:
        """Decide between 'retrieve more' and 'clarify'.

        If nothing retrieved for the failing claims scored well, the knowledge base
        probably lacks the material. If retrieval looked fine but support still failed
        across several claims, the question itself is likely underspecified.
        """
        low_score = float(self.cfg.get_path("abstain.low_retrieval_score", 0.25))
        ambiguous_count = int(self.cfg.get_path("abstain.ambiguous_claim_count", 3))

        failing = [e for e in evidence if e.claim in set(unsupported)]
        best_scores = [
            max((u.retrieval_score for u in e.units), default=0.0) for e in failing
        ]

        if best_scores and max(best_scores) < low_score:
            return "abstain_retrieve", (
                f"Retrieved evidence scored below {low_score:.2f}, so the knowledge "
                "base likely does not cover this."
            )
        if len(evidence) >= ambiguous_count and len(unsupported) > 1:
            return "abstain_clarify", (
                "Evidence was retrieved but did not support several parts of the "
                "question."
            )
        return "abstain_retrieve", (
            "Evidence was retrieved but did not support the question."
        )

    def _should_restore(self, comp, original: ClaimEvidence, verdict) -> bool:
        """Decide whether compression removed something this claim needed.

        `absolute` asks whether the compressed evidence clears the abstain threshold.
        That conflates two questions — "is this claim supportable at all?" and "did
        compression break it?" — and answers the second badly: the support signal is
        topical, so removing the one sentence carrying the answer usually leaves enough
        on-topic text to keep the score high. Measured on QASPER, it fired on 0 of 6
        cases where compression removed every trace of the marked answer evidence.

        `relative` compares against the full evidence set's own score, asking whether
        compression *changed* the support rather than whether the result clears some
        absolute bar. Self-calibrating, and it needs no per-corpus constant.
        """
        if not comp.dropped:
            # Nothing was removed, so there is nothing to put back regardless of policy.
            return False

        policy = self.cfg.get_path("restoration.policy", "absolute")
        if policy == "absolute":
            return not verdict.is_sufficient

        if policy != "relative":
            raise ValueError(
                f"unknown restoration.policy {policy!r}; expected 'absolute' or 'relative'"
            )

        full = self.evaluator.evaluate(comp.claim, original.units)
        if full.support_score <= 0.0:
            # The full set supports nothing either, so compression cannot be the cause.
            # Fall back to the absolute test rather than restoring on a meaningless ratio.
            return not verdict.is_sufficient

        retain = float(self.cfg.get_path("restoration.retain_fraction", 0.9))
        return verdict.support_score < retain * full.support_score

    def _record_prompt_cost(
        self,
        trace: PipelineTrace,
        query: str,
        final_evidence: list[ClaimEvidence],
        full_evidence: list[ClaimEvidence],
    ) -> None:
        """Measure the real prompt, and the prompt the baseline would have sent.

        Best-effort: a stub generator in tests has no tokenizer, and a provenance-style
        measurement must never be the reason an experiment dies. Zeros mean "not
        measured", and `prompt_reduction` returns 0.0 rather than inventing a ratio.
        """
        counter = getattr(self.generator, "count_prompt_tokens", None)
        if counter is None:
            return
        style = getattr(self.generator, "style", {})
        instruction = style.get("instruction") if isinstance(style, dict) else None
        try:
            trace.prompt_tokens = counter(
                build_prompt(query, final_evidence, instruction=instruction)
            )
            trace.prompt_tokens_uncompressed = counter(
                build_prompt(query, full_evidence, instruction=instruction)
            )
        except Exception:  # noqa: BLE001 - measurement must not break the run
            logger.debug("prompt token measurement failed", exc_info=True)

    def run(
        self,
        query: str,
        compression_mode: str | None = None,
        restrict_to: str | None = None,
        oracle_spans: set | None = None,
        seed: int | None = None,
    ) -> tuple[Answer, PipelineTrace]:
        """Run one query end to end, returning the answer and its trace.

        `restrict_to` confines retrieval to a single document, for single-paper
        benchmarks where the question is only meaningful relative to a given paper.

        `oracle_spans` and `seed` exist for the evaluation harness only. `oracle_spans`
        carries human-marked answer evidence and is rejected by every mode except
        `oracle`, so it cannot leak into a measured system by accident; `seed` lets the
        random arm be run many times instead of once.
        """
        trace = PipelineTrace(query=query, token_counts_exact=self.counter.is_exact)

        # Stages 1-3: claims, retrieval, claim-wise evidence.
        claims = decompose(query, cfg=self.cfg)
        evidence = map_evidence(
            claims, self.retriever, cfg=self.cfg, counter=self.counter,
            restrict_to=restrict_to,
        )

        # Stage 4: compression (uncertainty is estimated and stored inside).
        compressed = compress(
            evidence,
            cfg=self.cfg,
            mode=compression_mode,
            oracle_spans=oracle_spans,
            seed=seed,
        )

        # Stages 5-6: evaluate, and restore only the claims that failed.
        final_evidence: list[ClaimEvidence] = []
        verdicts = []
        restoration_triggered = False

        for original, comp in zip(evidence, compressed):
            claim_trace = ClaimTrace(
                claim=comp.claim,
                n_candidates=len(original.units),
                uncertainty=comp.uncertainty,
                keep_ratio=comp.keep_ratio,
                n_kept=len(comp.kept),
                n_dropped=len(comp.dropped),
                baseline_tokens=comp.original_token_count,
                compressed_tokens=comp.compressed_token_count,
                kept=_record(comp.kept),
                dropped=_record(comp.dropped),
            )

            current = to_claim_evidence(comp)
            verdict = self.evaluator.evaluate(comp.claim, current.units)
            claim_trace.support_score = verdict.support_score
            claim_trace.is_sufficient = verdict.is_sufficient

            if self._should_restore(comp, original, verdict):
                # Stage 6.1 + 6.2 — restore this claim and re-check.
                restoration_triggered = True
                current = restore(comp)
                verdict = self.evaluator.evaluate(comp.claim, current.units)
                claim_trace.restored_support_score = verdict.support_score
                claim_trace.restored_sufficient = verdict.is_sufficient

            final_evidence.append(current)
            verdicts.append(verdict)
            trace.claims.append(claim_trace)

        trace.restoration_triggered = restoration_triggered
        unsupported = [v.claim for v in verdicts if not v.is_sufficient]

        # End-to-end prompt cost, measured against the same prompt built from the *full*
        # evidence. Both are needed: the difference between them is the reduction actually
        # paid for, as opposed to the reduction in evidence text, which ignores every
        # incompressible part of the prompt.
        self._record_prompt_cost(trace, query, final_evidence, evidence)

        # Stage 7: generate, or abstain.
        if unsupported:
            branch, reason = self._choose_abstain_branch(final_evidence, unsupported)
            answer = abstain(query, final_evidence, verdicts, branch, reason)
        else:
            branch = "restored" if restoration_triggered else "direct"
            answer = generate_answer(
                query,
                final_evidence,
                verdicts,
                branch,
                generator=self.generator,
                cfg=self.cfg,
            )

        trace.branch = answer.branch
        trace.abstained = answer.abstained
        trace.answer = answer.text
        return answer, trace


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one query through the pipeline.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--trace", action="store_true", help="print the stage trace")
    parser.add_argument("--json", action="store_true", help="print the trace as JSON")
    parser.add_argument(
        "--mode",
        choices=["uncertainty_guided", "identity"],
        default="uncertainty_guided",
        help="identity = no-compression baseline",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s"
    )

    answer, trace = Pipeline().run(args.query, compression_mode=args.mode)

    if args.json:
        print(trace.to_json())
        return 0

    if args.trace:
        print(trace.summary())
        print()
    print(answer.text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
