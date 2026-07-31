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
from .generation import Generator, abstain, generate_answer
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

    def run(self, query: str, compression_mode: str | None = None) -> tuple[Answer, PipelineTrace]:
        """Run one query end to end, returning the answer and its trace."""
        trace = PipelineTrace(query=query, token_counts_exact=self.counter.is_exact)

        # Stages 1-3: claims, retrieval, claim-wise evidence.
        claims = decompose(query, cfg=self.cfg)
        evidence = map_evidence(
            claims, self.retriever, cfg=self.cfg, counter=self.counter
        )

        # Stage 4: compression (uncertainty is estimated and stored inside).
        compressed = compress(evidence, cfg=self.cfg, mode=compression_mode)

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

            if not verdict.is_sufficient and comp.dropped:
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
