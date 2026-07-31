"""Measure the compressed pipeline against the no-compression baseline.

This script produces the project's headline result. The claim under test is "compressing
evidence saves tokens without losing the ability to answer", so it reports cost and
quality side by side — a token reduction reported alone would be meaningless, since
dropping all evidence would score 100%.

Two arms, identical in every other respect:

    identity            every retrieved unit goes into the prompt (baseline)
    uncertainty_guided  keep budget scales with retrieval uncertainty

Reported per arm:

    tokens              baseline -> actually sent to the generator
    reduction           the headline number
    restoration rate    how often compression removed something it needed back
    abstain rate        split into answerable questions (bad) and unanswerable (good)
    keyword recall      coarse answer-quality proxy on answerable questions

Usage:
    python scripts/run_eval.py                    # both arms, with generation
    python scripts/run_eval.py --no-generate      # token/branch metrics only, fast
    python scripts/run_eval.py --limit 5          # first 5 questions
    python scripts/run_eval.py --json out.json    # machine-readable results
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.compression import compress  # noqa: E402
from src.config import load_config  # noqa: E402
from src.evidence_mapping import map_evidence  # noqa: E402
from src.claims import decompose  # noqa: E402
from src.pipeline import Pipeline  # noqa: E402
from src.restoration import restore, restore_from_corpus  # noqa: E402

DEFAULT_QUESTIONS = REPO_ROOT / "data" / "eval" / "questions.yaml"

MODES = ["identity", "uncertainty_guided"]


class NullGenerator:
    """Stands in for the generator when --no-generate is used."""

    def complete(self, prompt, max_new_tokens=None):
        return "[generation skipped]"


@dataclass
class ArmResult:
    mode: str
    baseline_tokens: int = 0
    final_tokens: int = 0
    n_questions: int = 0
    n_answerable: int = 0
    restorations: int = 0
    abstained_answerable: int = 0
    abstained_unanswerable: int = 0
    keyword_hits: int = 0
    keyword_total: int = 0
    elapsed: float = 0.0
    rows: list[dict] = field(default_factory=list)

    @property
    def reduction(self) -> float:
        if self.baseline_tokens == 0:
            return 0.0
        return (self.baseline_tokens - self.final_tokens) / self.baseline_tokens

    @property
    def restoration_rate(self) -> float:
        return self.restorations / self.n_questions if self.n_questions else 0.0

    @property
    def false_abstain_rate(self) -> float:
        """Abstaining on a question the KB *can* answer. Lower is better."""
        return (
            self.abstained_answerable / self.n_answerable if self.n_answerable else 0.0
        )

    @property
    def correct_abstain_rate(self) -> float:
        """Abstaining on a question the KB cannot answer. Higher is better."""
        n_unanswerable = self.n_questions - self.n_answerable
        return self.abstained_unanswerable / n_unanswerable if n_unanswerable else 0.0

    @property
    def keyword_recall(self) -> float:
        return self.keyword_hits / self.keyword_total if self.keyword_total else 0.0

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "questions": self.n_questions,
            "baseline_tokens": self.baseline_tokens,
            "final_tokens": self.final_tokens,
            "reduction": round(self.reduction, 4),
            "restoration_rate": round(self.restoration_rate, 4),
            "false_abstain_rate": round(self.false_abstain_rate, 4),
            "correct_abstain_rate": round(self.correct_abstain_rate, 4),
            "keyword_recall": round(self.keyword_recall, 4),
            "elapsed_seconds": round(self.elapsed, 1),
            "rows": self.rows,
        }


def check_round_trip(pipeline: Pipeline, queries: list[str], cfg) -> tuple[int, int]:
    """Re-verify reversibility on real retrieved evidence, not just fixtures.

    The unit tests prove the property on synthetic data; this proves it on whatever the
    retriever actually returns, which is the case that matters for a reported result.
    """
    checked = failures = 0
    for query in queries:
        evidence = map_evidence(decompose(query, cfg=cfg), pipeline.retriever, cfg=cfg)
        for comp in compress(evidence, cfg=cfg):
            fast = [(u.span.doc_id, u.span.start, u.span.end, u.text) for u in restore(comp).units]
            from_corpus = [
                (u.span.doc_id, u.span.start, u.span.end, u.text)
                for u in restore_from_corpus(comp, pipeline.retriever.corpus).units
            ]
            checked += 1
            if sorted(fast) != sorted(from_corpus):
                failures += 1
    return checked, failures


def run_arm(pipeline: Pipeline, questions: list[dict], mode: str) -> ArmResult:
    result = ArmResult(mode=mode)
    start = time.time()

    for item in questions:
        query = item["query"]
        answerable = item.get("answerable", True)
        expected = [k.lower() for k in item.get("expect", [])]

        answer, trace = pipeline.run(query, compression_mode=mode)

        result.n_questions += 1
        result.baseline_tokens += trace.baseline_tokens
        result.final_tokens += trace.final_tokens
        if trace.restoration_triggered:
            result.restorations += 1

        if answerable:
            result.n_answerable += 1
            if answer.abstained:
                result.abstained_answerable += 1
            elif expected:
                text = answer.text.lower()
                result.keyword_hits += sum(1 for k in expected if k in text)
                result.keyword_total += len(expected)
        elif answer.abstained:
            result.abstained_unanswerable += 1

        result.rows.append(
            {
                "query": query,
                "branch": answer.branch,
                "abstained": answer.abstained,
                "answerable": answerable,
                "baseline_tokens": trace.baseline_tokens,
                "final_tokens": trace.final_tokens,
                "reduction": round(trace.reduction, 4),
                "uncertainty": [round(c.uncertainty, 3) for c in trace.claims],
                "support": [round(c.support_score, 3) for c in trace.claims],
            }
        )

    result.elapsed = time.time() - start
    return result


def print_report(
    arms: list[ArmResult],
    round_trip: tuple[int, int],
    exact: bool,
    generated: bool = True,
) -> None:
    baseline = next((a for a in arms if a.mode == "identity"), None)

    print("\n" + "=" * 74)
    print("UNCERTAINTY-GUIDED COMPRESSION — BASELINE COMPARISON")
    print("=" * 74)

    checked, failures = round_trip
    status = "PASS" if failures == 0 else f"FAIL ({failures}/{checked})"
    print(f"\nReversibility on real retrieved evidence: {status}  [{checked} claims checked]")
    if not exact:
        print("WARNING: token counts are ESTIMATED, not tokenizer-exact.")

    print(f"\n{'metric':<26}" + "".join(f"{a.mode:>23}" for a in arms))
    print("-" * 74)

    def row(label, fn, fmt="{:.1%}"):
        print(f"{label:<26}" + "".join(f"{fmt.format(fn(a)):>23}" for a in arms))

    row("questions", lambda a: a.n_questions, "{:d}")
    row("baseline tokens", lambda a: a.baseline_tokens, "{:d}")
    row("tokens sent to model", lambda a: a.final_tokens, "{:d}")
    row("token reduction", lambda a: a.reduction)
    row("restoration rate", lambda a: a.restoration_rate)
    if generated:
        row("keyword recall", lambda a: a.keyword_recall)
    else:
        # Without generation there is no answer text to search, so a 0% here would read
        # as a quality failure rather than "not measured".
        print(f"{'keyword recall':<26}" + "".join(f"{'n/a (--no-generate)':>23}" for _ in arms))
    row("false abstain (answerable)", lambda a: a.false_abstain_rate)
    row("correct abstain (unansw.)", lambda a: a.correct_abstain_rate)
    row("elapsed (s)", lambda a: a.elapsed, "{:.1f}")

    compressed = next((a for a in arms if a.mode == "uncertainty_guided"), None)
    if baseline and compressed:
        saved = baseline.final_tokens - compressed.final_tokens
        print("\n" + "-" * 74)
        quality = (
            f"keyword recall {compressed.keyword_recall - baseline.keyword_recall:+.1%} "
            "vs baseline"
            if generated
            else "answer quality not measured (--no-generate)"
        )
        print(
            f"RESULT: {saved} tokens saved ({compressed.reduction:.1%} reduction), "
            f"{quality}."
        )
        if compressed.false_abstain_rate > baseline.false_abstain_rate:
            print(
                "NOTE: compression increased false abstentions "
                f"({baseline.false_abstain_rate:.1%} -> "
                f"{compressed.false_abstain_rate:.1%}) — evidence needed to answer is "
                "being dropped and not always recovered."
            )
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--no-generate",
        action="store_true",
        help="skip answer generation; token and branch metrics only",
    )
    parser.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    parser.add_argument(
        "--generation-model",
        default=None,
        help="override models.generation (e.g. a smaller model for a quicker run)",
    )
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    questions = yaml.safe_load(args.questions.read_text(encoding="utf-8"))
    if args.limit:
        questions = questions[: args.limit]

    cfg = load_config()
    if args.generation_model:
        cfg["models"] = dict(cfg["models"], generation=args.generation_model)
    pipeline = Pipeline(cfg=cfg)
    if args.no_generate:
        pipeline.generator = NullGenerator()

    print(f"Loaded {len(questions)} questions. Building index...")
    print(f"Generation model: {cfg.require('models.generation')}")
    pipeline.retriever.build()

    round_trip = check_round_trip(pipeline, [q["query"] for q in questions], cfg)

    arms = []
    for mode in args.modes:
        print(f"Running arm: {mode} ...")
        arms.append(run_arm(pipeline, questions, mode))

    print_report(
        arms, round_trip, pipeline.counter.is_exact, generated=not args.no_generate
    )

    if args.json:
        payload = {
            "round_trip_checked": round_trip[0],
            "round_trip_failures": round_trip[1],
            "token_counts_exact": pipeline.counter.is_exact,
            "generation_skipped": args.no_generate,
            "arms": [a.to_dict() for a in arms],
        }
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")

    return 1 if round_trip[1] else 0


if __name__ == "__main__":
    sys.exit(main())
