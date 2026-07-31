"""Pipeline stage 4 — uncertainty-guided reversible evidence compression.

WHAT IS ACTUALLY COMPRESSED (stated explicitly, per CLAUDE.md):
the saving is in tokens *sent to the generator*, not in memory. Dropped units are
retained in `CompressedEvidence.dropped` so restoration is immediate, and each one keeps
its `Span`, so the original text remains recoverable from the corpus even if those
retained copies were discarded. Both recovery routes are asserted equivalent by the
round-trip test in tests/test_compression.py.

Compression is extractive — whole evidence units are kept or dropped, never rewritten.
That is what makes restoration exact rather than approximate. Abstractive compression
would reach higher ratios but could not be inverted from its output alone, which would
break the core guarantee of the project.

Two modes:
  identity           — no-op. Not dead code: this is the no-compression baseline arm
                       that scripts/run_eval.py measures the reduction against.
  uncertainty_guided — keep budget scales with claim uncertainty.
"""

from __future__ import annotations

from .config import Config, default_config
from .types import ClaimEvidence, CompressedEvidence
from .uncertainty import estimator_from_config


def _identity(item: ClaimEvidence) -> CompressedEvidence:
    """Keep everything. The baseline arm."""
    return CompressedEvidence(
        claim=item.claim,
        kept=list(item.units),
        dropped=[],
        uncertainty=item.uncertainty,
        keep_ratio=1.0,
        original_token_count=item.token_count,
        compressed_token_count=item.token_count,
    )


def _uncertainty_guided(item: ClaimEvidence, cfg: Config) -> CompressedEvidence:
    """Drop the weakest evidence, keeping more of it when uncertainty is high.

        keep_ratio = min_keep + u * (max_keep - min_keep)

    Low uncertainty means retrieval was decisive — one clear winner well above the rest —
    so most units are redundant and can go. High uncertainty means the ranking is
    unstable, so more is retained to avoid discarding the unit that actually mattered.
    """
    min_keep = float(cfg.require("compression.min_keep"))
    max_keep = float(cfg.require("compression.max_keep"))
    floor_units = int(cfg.get_path("compression.floor_units", 1))

    units = sorted(item.units, key=lambda u: u.retrieval_score, reverse=True)
    if not units:
        return _identity(item)

    keep_ratio = min_keep + item.uncertainty * (max_keep - min_keep)
    n_keep = max(floor_units, round(keep_ratio * len(units)))
    n_keep = min(n_keep, len(units))

    kept, dropped = units[:n_keep], units[n_keep:]
    return CompressedEvidence(
        claim=item.claim,
        kept=kept,
        dropped=dropped,
        uncertainty=item.uncertainty,
        keep_ratio=keep_ratio,
        original_token_count=item.token_count,
        compressed_token_count=sum(u.token_count for u in kept),
    )


def compress(
    evidence: list[ClaimEvidence],
    cfg: Config | None = None,
    mode: str | None = None,
) -> list[CompressedEvidence]:
    """Compress each claim's evidence set.

    Uncertainty is estimated here and stored on the result, so every compression
    decision is returned alongside the score that drove it (CLAUDE.md reproducibility
    requirement) rather than being recomputed or lost.
    """
    cfg = cfg or default_config()
    mode = mode or "uncertainty_guided"
    estimator = estimator_from_config(cfg)

    out: list[CompressedEvidence] = []
    for item in evidence:
        # Uncertainty is estimated in both modes, even though identity ignores it, so the
        # baseline arm's traces carry the same scores and the two arms stay comparable
        # when analysing a run.
        item.uncertainty = estimator.estimate(item.units)
        out.append(
            _identity(item) if mode == "identity" else _uncertainty_guided(item, cfg)
        )
    return out


def to_claim_evidence(compressed: CompressedEvidence) -> ClaimEvidence:
    """View a compressed set as a ClaimEvidence, for prompt assembly and evaluation."""
    return ClaimEvidence(
        claim=compressed.claim,
        units=list(compressed.kept),
        token_count=compressed.compressed_token_count,
        uncertainty=compressed.uncertainty,
    )
