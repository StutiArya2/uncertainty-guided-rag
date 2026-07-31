"""Pipeline stage 6.1 — evidence restoration.

Reverses compression when the compressed evidence turned out insufficient. Because
compression is extractive (units are dropped, never rewritten) restoration is exact
rather than approximate.

Two recovery routes exist, and they must agree:

* `restore` reinstates the retained `dropped` units — the fast path used at runtime.
* `restore_from_corpus` ignores those retained copies entirely and re-reads each unit's
  text from its `Span`. This is the route that proves reversibility is a real property
  of the span index rather than a side effect of keeping a backup in memory.

tests/test_compression.py asserts both produce identical evidence.
"""

from __future__ import annotations

from .types import ClaimEvidence, CompressedEvidence, Corpus, EvidenceUnit


def _ordered(units: list[EvidenceUnit]) -> list[EvidenceUnit]:
    """Stable ordering by descending retrieval score, then by span position.

    Restoration must be deterministic: the same compressed input always yields the same
    restored evidence, regardless of the order units happened to be dropped in.
    """
    return sorted(
        units,
        key=lambda u: (-u.retrieval_score, u.span.doc_id, u.span.start),
    )


def restore(compressed: CompressedEvidence) -> ClaimEvidence:
    """Reinstate dropped evidence, returning the claim's full evidence set."""
    units = _ordered(list(compressed.kept) + list(compressed.dropped))
    return ClaimEvidence(
        claim=compressed.claim,
        units=units,
        token_count=sum(u.token_count for u in units),
        uncertainty=compressed.uncertainty,
    )


def restore_from_corpus(
    compressed: CompressedEvidence, corpus: Corpus
) -> ClaimEvidence:
    """Rebuild evidence text from spans alone, ignoring any retained copies.

    If this diverges from `restore`, the span index is corrupt and the reversibility
    guarantee no longer holds — which is why the round-trip test checks it.
    """
    rebuilt: list[EvidenceUnit] = []
    for unit in list(compressed.kept) + list(compressed.dropped):
        rebuilt.append(
            EvidenceUnit(
                span=unit.span,
                text=unit.span.resolve(corpus),
                retrieval_score=unit.retrieval_score,
                uncertainty=unit.uncertainty,
                token_count=unit.token_count,
            )
        )

    units = _ordered(rebuilt)
    return ClaimEvidence(
        claim=compressed.claim,
        units=units,
        token_count=sum(u.token_count for u in units),
        uncertainty=compressed.uncertainty,
    )


def restore_all(compressed: list[CompressedEvidence]) -> list[ClaimEvidence]:
    return [restore(c) for c in compressed]
