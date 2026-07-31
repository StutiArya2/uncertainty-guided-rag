"""Uncertainty estimator tests.

Properties matter more than exact values here: the estimator only has to order claims
correctly for compression to behave sensibly, and pinning precise numbers would make
every future weight change look like a regression.
"""

from __future__ import annotations

import pytest

from src.types import EvidenceUnit, Span
from src.uncertainty import RetrievalScoreEstimator, estimator_from_config


def make_units(scores: list[float]) -> list[EvidenceUnit]:
    return [
        EvidenceUnit(span=Span("d", i, i + 1), text="x", retrieval_score=s)
        for i, s in enumerate(scores)
    ]


class TestRetrievalScoreEstimator:
    def test_output_is_bounded(self, cfg):
        est = RetrievalScoreEstimator(cfg=cfg)
        for scores in ([0.9, 0.1], [0.0, 0.0], [1.0, 1.0], [-0.5, -0.9], [0.5]):
            assert 0.0 <= est.estimate(make_units(scores)) <= 1.0

    def test_empty_evidence_is_maximally_uncertain(self, cfg):
        assert RetrievalScoreEstimator(cfg=cfg).estimate([]) == 1.0

    def test_high_score_and_wide_margin_is_confident(self, cfg):
        est = RetrievalScoreEstimator(cfg=cfg)
        confident = est.estimate(make_units([0.95, 0.30, 0.20]))
        assert confident < 0.3

    def test_low_score_and_narrow_margin_is_uncertain(self, cfg):
        est = RetrievalScoreEstimator(cfg=cfg)
        unsure = est.estimate(make_units([0.20, 0.19, 0.18]))
        assert unsure > 0.7

    def test_lower_top_score_raises_uncertainty(self, cfg):
        """Holding margin fixed, weaker absolute evidence must be more uncertain."""
        est = RetrievalScoreEstimator(cfg=cfg)
        strong = est.estimate(make_units([0.90, 0.60]))
        weak = est.estimate(make_units([0.40, 0.10]))
        assert weak > strong

    def test_narrower_margin_raises_uncertainty(self, cfg):
        """Holding the top score fixed, an unstable ranking must be more uncertain."""
        est = RetrievalScoreEstimator(cfg=cfg)
        decisive = est.estimate(make_units([0.80, 0.40]))
        contested = est.estimate(make_units([0.80, 0.79]))
        assert contested > decisive

    def test_single_candidate_is_treated_as_unverified(self, cfg):
        """With nothing to compare against, the ranking cannot be trusted."""
        est = RetrievalScoreEstimator(cfg=cfg)
        assert est.estimate(make_units([0.80])) > est.estimate(make_units([0.80, 0.40]))

    def test_is_deterministic(self, cfg):
        est = RetrievalScoreEstimator(cfg=cfg)
        units = make_units([0.7, 0.5, 0.3])
        assert est.estimate(units) == est.estimate(units)

    def test_order_of_units_does_not_matter(self, cfg):
        est = RetrievalScoreEstimator(cfg=cfg)
        assert est.estimate(make_units([0.3, 0.9, 0.6])) == pytest.approx(
            est.estimate(make_units([0.9, 0.6, 0.3]))
        )


class TestRegistry:
    def test_builds_configured_estimator(self, cfg):
        assert isinstance(estimator_from_config(cfg), RetrievalScoreEstimator)

    def test_unknown_estimator_raises(self, cfg):
        broken = type(cfg)(dict(cfg))
        broken["uncertainty"] = dict(cfg["uncertainty"], estimator="does_not_exist")
        with pytest.raises(ValueError, match="unknown uncertainty estimator"):
            estimator_from_config(broken)
