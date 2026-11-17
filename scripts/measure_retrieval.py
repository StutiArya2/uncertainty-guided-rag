"""Measure first-stage retrieval against human-marked evidence, one mode at a time.

WHY THIS IS ITS OWN SCRIPT

`run_eval.py` reports retrieval recall as a by-product of a full pipeline run, which needs
the generator and the support scorer loaded and takes minutes per arm. Retrieval can be
judged without either: the only question is whether the marked passages are among the
candidates. So this runs retrieval alone, and the `bm25` mode needs no model at all.

Everything downstream is capped by this number — compression and reranking can only choose
among what retrieval returned — which is why it deserves a direct measurement.

WHAT IS MEASURED

For each question with aligned evidence, the candidates are the union of `top_k` units per
claim, restricted to the question's paper: exactly what `Pipeline.run` retrieves. Then:

    found     any marked character is among the candidates
    recall    fraction of marked characters among the candidates (0 when not found)

`run_eval.py`'s `retrieval_evidence_recall` averages recall over *found* questions only.
Both are printed here, because the conditional figure hides how often retrieval finds
nothing: 55.1% conditional is 31.8% over all questions.

Usage:
    # lexical only, no model weights needed
    python scripts/measure_retrieval.py --questions data/qasper/questions.yaml \\
        --kb data/qasper/kb --modes bm25 \\
        --compare-run artifacts/qasper-main/tight_relative.json \\
        --json artifacts/retrieval-recall/bm25_vs_dense.json

    # all three, needs the embedding model
    python scripts/measure_retrieval.py --questions data/qasper/questions.yaml \\
        --kb data/qasper/kb --modes dense bm25 hybrid \\
        --json artifacts/retrieval-recall/all_modes.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_eval import (  # noqa: E402
    align_all,
    apply_override,
    cluster_bootstrap,
    cluster_permutation_p,
)
from src import provenance  # noqa: E402
from src.claims import decompose  # noqa: E402
from src.config import Config, load_config  # noqa: E402
from src.gold_evidence import recall as evidence_recall  # noqa: E402
from src.kb import load_corpus  # noqa: E402
from src.retrieval import MODES, Retriever  # noqa: E402


def measure_mode(mode: str, questions: list[dict], spans: dict[int, list], corpus, cfg) -> dict:
    """Per-question marked-evidence recall for one retrieval mode."""
    mode_cfg = Config(cfg)
    mode_cfg["retrieval"] = dict(cfg["retrieval"], mode=mode)
    retriever = Retriever(corpus=corpus, cfg=mode_cfg)

    rows = []
    for index in sorted(spans):
        item = questions[index]
        paper = item.get("paper", "")
        seen: dict[tuple[int, int], object] = {}
        for claim in decompose(item["query"], cfg=mode_cfg):
            for unit in retriever.retrieve(claim, restrict_to=paper):
                seen[(unit.span.start, unit.span.end)] = unit
        value = evidence_recall(spans[index], list(seen.values()), paper)
        rows.append({"index": index, "paper": paper, "recall": value, "n_candidates": len(seen)})
    return {"mode": mode, "rows": rows, **summarise(rows)}


def summarise(rows: list[dict]) -> dict:
    n = len(rows)
    recalls = [r["recall"] for r in rows]
    found = [r for r in recalls if r > 0]
    return {
        "questions": n,
        "found": len(found),
        "found_rate": len(found) / n if n else 0.0,
        "recall": sum(recalls) / n if n else 0.0,
        # The figure run_eval.py calls retrieval_evidence_recall.
        "recall_given_found": sum(found) / len(found) if found else 0.0,
    }


def rows_from_run(path: Path, questions: list[dict], spans: dict[int, list]) -> dict:
    """Recover per-question retrieval recall from an archived `run_eval.py` JSON.

    Retrieval does not depend on the compression arm, so the `identity` arm's rows carry
    it. This is what lets a mode that needs no model be compared against the dense
    retriever without re-running the dense retriever.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    arm = next((a for a in data["arms"] if a["mode"] == "identity"), data["arms"][0])
    archived = arm["rows"]
    if len(archived) != len(questions) or any(
        a["query"] != q["query"] for a, q in zip(archived, questions)
    ):
        raise SystemExit(f"{path} was not run on this question file; rows do not line up")

    rows = []
    for index in sorted(spans):
        outcome = archived[index].get("evidence_outcome")
        if outcome is None:
            raise SystemExit(
                f"{path}: question {index} has aligned evidence here but none in the "
                "archived run; the corpus or the alignment has changed since"
            )
        rows.append({
            "index": index,
            "paper": questions[index].get("paper", ""),
            "recall": float(outcome.get("retrieval_recall", 0.0)),
        })
    git = data.get("provenance", {}).get("git", {})
    return {
        "mode": f"dense (archived: {path.name})",
        "source": {"path": str(path), "commit": git.get("commit"),
                   "top_k": data.get("provenance", {}).get("config", {}).get("retrieval", {}).get("top_k")},
        "rows": rows,
        **summarise(rows),
    }


def compare(a: dict, b: dict) -> dict:
    """Paired comparison of two modes over the same questions, clustered by paper."""
    xs = [r["recall"] for r in a["rows"]]
    ys = [r["recall"] for r in b["rows"]]
    papers = [r["paper"] for r in a["rows"]]
    mean, lo, hi = cluster_bootstrap(xs, ys, papers)
    fx, fy = [float(x > 0) for x in xs], [float(y > 0) for y in ys]
    found_mean, found_lo, found_hi = cluster_bootstrap(fx, fy, papers)
    both = sum(1 for x, y in zip(xs, ys) if x > 0 and y > 0)
    only_a = sum(1 for x, y in zip(xs, ys) if x > 0 and y == 0)
    only_b = sum(1 for x, y in zip(xs, ys) if x == 0 and y > 0)
    return {
        "a": a["mode"], "b": b["mode"],
        "recall_diff": mean, "recall_diff_ci": [lo, hi],
        "recall_diff_p": cluster_permutation_p(xs, ys, papers),
        "found_rate_diff": found_mean, "found_rate_diff_ci": [found_lo, found_hi],
        "found_rate_diff_p": cluster_permutation_p(fx, fy, papers),
        "found_by_both": both, "found_only_by_a": only_a, "found_only_by_b": only_b,
        "found_by_neither": len(xs) - both - only_a - only_b,
        # What a retriever returning BOTH candidate sets would reach. It retrieves up to
        # twice as many units, so it is a ceiling for fusion at the same top_k, not an
        # estimate of it.
        "found_by_either_rate": (both + only_a + only_b) / len(xs) if xs else 0.0,
    }


def print_report(results: list[dict], comparisons: list[dict], stats: dict, top_k: int) -> None:
    print(f"\nmarked evidence: {stats['aligned']}/{stats['evidence']} strings aligned, "
          f"{stats['questions_usable']} usable questions, top_k={top_k} per claim\n")
    print(f"{'mode':<44} {'found':>7} {'recall':>8} {'recall | found':>15}")
    print("-" * 78)
    for r in results:
        print(f"{r['mode']:<44} {r['found_rate']:>7.1%} {r['recall']:>8.1%} "
              f"{r['recall_given_found']:>15.1%}")
    for c in comparisons:
        print(f"\n{c['a']}  minus  {c['b']}   (clustered by paper)")
        print(f"  recall      {c['recall_diff']:+.3f}  CI [{c['recall_diff_ci'][0]:+.3f}, "
              f"{c['recall_diff_ci'][1]:+.3f}]  p={c['recall_diff_p']:.4f}")
        print(f"  found rate  {c['found_rate_diff']:+.3f}  CI [{c['found_rate_diff_ci'][0]:+.3f}, "
              f"{c['found_rate_diff_ci'][1]:+.3f}]  p={c['found_rate_diff_p']:.4f}")
        print(f"  found by both {c['found_by_both']}, only {c['a']} {c['found_only_by_a']}, "
              f"only {c['b']} {c['found_only_by_b']}, neither {c['found_by_neither']}")
        print(f"  found by either: {c['found_by_either_rate']:.1%} "
              "(ceiling for fusion; it uses up to twice the candidates)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--kb", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=["bm25"])
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="KEY=VALUE", help="override a config key, as in run_eval.py")
    parser.add_argument("--compare-run", type=Path, default=None,
                        help="archived run_eval.py JSON to take dense retrieval recall from")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    cfg = load_config()
    for assignment in args.overrides:
        apply_override(cfg, assignment)
    if args.top_k is not None:
        cfg["retrieval"] = dict(cfg["retrieval"], top_k=args.top_k)
    top_k = int(cfg.require("retrieval.top_k"))

    questions = yaml.safe_load(args.questions.read_text(encoding="utf-8"))
    corpus = load_corpus(args.kb, cfg)
    spans, stats = align_all(questions, corpus)
    if not spans:
        raise SystemExit("no question has alignable marked evidence; nothing to measure")

    results = []
    if args.compare_run is not None:
        archived = rows_from_run(args.compare_run, questions, spans)
        if archived["source"]["top_k"] not in (None, top_k):
            raise SystemExit(
                f"{args.compare_run} used top_k={archived['source']['top_k']}, this run "
                f"uses top_k={top_k}; the comparison would not be at equal budget"
            )
        results.append(archived)
    for mode in args.modes:
        results.append(measure_mode(mode, questions, spans, corpus, cfg))

    comparisons = [compare(later, results[0]) for later in results[1:]]
    print_report(results, comparisons, stats, top_k)

    if args.json is not None:
        record = {
            "provenance": provenance.collect(
                cfg, extra={"dataset": provenance.dataset_record(args.questions, args.kb)}
            ),
            "top_k": top_k,
            "alignment": stats,
            "modes": results,
            "comparisons": comparisons,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
