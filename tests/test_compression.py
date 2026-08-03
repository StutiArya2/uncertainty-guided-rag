"""Compression and restoration tests.

`TestRoundTrip` is the gate for the whole project. Compression is only legitimate if it
can be undone exactly; if these fail, every downstream result is untrustworthy because
restored evidence would silently differ from what was originally retrieved.
"""

from __future__ import annotations

import pytest

from src.compression import compress, to_claim_evidence
from src.restoration import restore, restore_all, restore_from_corpus
from src.types import ClaimEvidence, EvidenceUnit


def evidence_from_units(units, claim="the boiling point of water"):
    for i, unit in enumerate(units):
        unit.retrieval_score = 0.9 - 0.1 * i
        unit.token_count = len(unit.text.split())
    return ClaimEvidence(
        claim=claim, units=units, token_count=sum(u.token_count for u in units)
    )


def signature(units: list[EvidenceUnit]):
    """Identity of an evidence set, independent of ordering."""
    return sorted((u.span.doc_id, u.span.start, u.span.end, u.text) for u in units)


@pytest.fixture
def claim_evidence(units):
    return evidence_from_units(list(units))


class TestRoundTrip:
    """THE gate: compress -> restore must return exactly the original evidence."""

    def test_restore_recovers_every_unit(self, claim_evidence, cfg):
        original = signature(claim_evidence.units)
        compressed = compress([claim_evidence], cfg=cfg)[0]
        restored = restore(compressed)
        assert signature(restored.units) == original

    def test_restore_from_corpus_matches_retained_copies(
        self, claim_evidence, corpus, cfg
    ):
        """The two recovery routes must agree.

        `restore` reinstates retained copies; `restore_from_corpus` ignores them and
        re-reads each span from the source. Agreement is what proves reversibility comes
        from the span index rather than from keeping a backup.
        """
        compressed = compress([claim_evidence], cfg=cfg)[0]
        assert signature(restore(compressed).units) == signature(
            restore_from_corpus(compressed, corpus).units
        )

    def test_restored_text_is_byte_identical(self, claim_evidence, corpus, cfg):
        compressed = compress([claim_evidence], cfg=cfg)[0]
        for unit in restore_from_corpus(compressed, corpus).units:
            assert unit.text == unit.span.resolve(corpus)

    def test_round_trip_holds_at_every_uncertainty_level(self, claim_evidence, cfg):
        """Reversibility must not depend on how aggressive compression happened to be."""
        original = signature(claim_evidence.units)
        for uncertainty in (0.0, 0.25, 0.5, 0.75, 1.0):
            item = evidence_from_units(list(claim_evidence.units))
            item.uncertainty = uncertainty
            from src.compression import _uncertainty_guided

            compressed = _uncertainty_guided(item, cfg)
            assert signature(restore(compressed).units) == original, (
                f"round trip broken at uncertainty={uncertainty}"
            )

    def test_token_accounting_is_conserved(self, claim_evidence, cfg):
        compressed = compress([claim_evidence], cfg=cfg)[0]
        kept = sum(u.token_count for u in compressed.kept)
        dropped = sum(u.token_count for u in compressed.dropped)
        assert kept + dropped == compressed.original_token_count
        assert compressed.compressed_token_count == kept

    def test_restore_is_deterministic(self, claim_evidence, cfg):
        compressed = compress([claim_evidence], cfg=cfg)[0]
        first = [u.text for u in restore(compressed).units]
        assert first == [u.text for u in restore(compressed).units]

    def test_restore_all_handles_multiple_claims(self, claim_evidence, cfg):
        compressed = compress([claim_evidence, claim_evidence], cfg=cfg)
        assert len(restore_all(compressed)) == 2


class TestCompressionBehaviour:
    def test_nothing_is_destroyed(self, claim_evidence, cfg):
        compressed = compress([claim_evidence], cfg=cfg)[0]
        assert len(compressed.kept) + len(compressed.dropped) == len(
            claim_evidence.units
        )

    def test_kept_units_are_the_highest_scoring(self, claim_evidence, cfg):
        compressed = compress([claim_evidence], cfg=cfg)[0]
        if compressed.dropped:
            assert min(u.retrieval_score for u in compressed.kept) >= max(
                u.retrieval_score for u in compressed.dropped
            )

    def test_higher_uncertainty_keeps_at_least_as_much(self, claim_evidence, cfg):
        """The core guarantee of 'uncertainty-guided': unsure means keep more."""
        from src.compression import _uncertainty_guided

        counts = []
        for uncertainty in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
            item = evidence_from_units(list(claim_evidence.units))
            item.uncertainty = uncertainty
            counts.append(len(_uncertainty_guided(item, cfg).kept))
        assert counts == sorted(counts), f"keep count not monotonic: {counts}"

    def test_maximum_uncertainty_keeps_everything(self, claim_evidence, cfg):
        from src.compression import _uncertainty_guided

        item = evidence_from_units(list(claim_evidence.units))
        item.uncertainty = 1.0
        compressed = _uncertainty_guided(item, cfg)
        assert compressed.dropped == []

    def test_floor_prevents_empty_evidence(self, claim_evidence, cfg):
        """Even total confidence must leave something to generate from."""
        from src.compression import _uncertainty_guided

        item = evidence_from_units(list(claim_evidence.units))
        item.uncertainty = 0.0
        compressed = _uncertainty_guided(item, cfg)
        assert len(compressed.kept) >= int(cfg.get_path("compression.floor_units", 1))

    def test_compression_reduces_tokens(self, claim_evidence, cfg):
        from src.compression import _uncertainty_guided

        item = evidence_from_units(list(claim_evidence.units))
        item.uncertainty = 0.0
        compressed = _uncertainty_guided(item, cfg)
        assert compressed.compressed_token_count < compressed.original_token_count
        assert compressed.reduction > 0

    def test_uncertainty_is_recorded_with_the_decision(self, claim_evidence, cfg):
        """CLAUDE.md: scores must be returned alongside compression decisions."""
        compressed = compress([claim_evidence], cfg=cfg)[0]
        assert 0.0 <= compressed.uncertainty <= 1.0
        assert 0.0 <= compressed.keep_ratio <= 1.0

    def test_empty_evidence_does_not_crash(self, cfg):
        empty = ClaimEvidence(claim="nothing", units=[], token_count=0)
        compressed = compress([empty], cfg=cfg)[0]
        assert compressed.kept == [] and compressed.dropped == []


class TestIdentityMode:
    """The no-compression baseline arm used by scripts/run_eval.py."""

    def test_keeps_everything(self, claim_evidence, cfg):
        compressed = compress([claim_evidence], cfg=cfg, mode="identity")[0]
        assert len(compressed.kept) == len(claim_evidence.units)
        assert compressed.dropped == []

    def test_reports_zero_reduction(self, claim_evidence, cfg):
        compressed = compress([claim_evidence], cfg=cfg, mode="identity")[0]
        assert compressed.reduction == 0.0
        assert compressed.tokens_saved == 0

    def test_round_trip_is_trivially_exact(self, claim_evidence, cfg):
        compressed = compress([claim_evidence], cfg=cfg, mode="identity")[0]
        assert signature(restore(compressed).units) == signature(claim_evidence.units)


class TestAblationArms:
    """The arms that make the central claim falsifiable.

    Both keep a constant fraction regardless of uncertainty. If they matched the
    uncertainty arm at equal budget, the proposal would add nothing.
    """

    def test_fixed_ratio_ignores_uncertainty(self, claim_evidence, cfg):
        from src.compression import _fixed_ratio

        counts = []
        for uncertainty in (0.0, 0.5, 1.0):
            item = evidence_from_units(list(claim_evidence.units))
            item.uncertainty = uncertainty
            counts.append(len(_fixed_ratio(item, cfg).kept))
        assert len(set(counts)) == 1, f"fixed arm varied with uncertainty: {counts}"

    def test_fixed_ratio_keeps_highest_scoring(self, claim_evidence, cfg):
        compressed = compress([claim_evidence], cfg=cfg, mode="fixed_ratio")[0]
        if compressed.dropped:
            assert min(u.retrieval_score for u in compressed.kept) >= max(
                u.retrieval_score for u in compressed.dropped
            )

    def test_random_arm_is_reproducible(self, claim_evidence, cfg):
        """A seeded arm, or the ablation could not be rerun."""
        first = compress([claim_evidence], cfg=cfg, mode="random")[0]
        second = compress([claim_evidence], cfg=cfg, mode="random")[0]
        assert [u.text for u in first.kept] == [u.text for u in second.kept]

    def test_random_arm_ignores_score_order(self, cfg):
        """It must actually be random, or it is just a second fixed_ratio arm."""
        from src.types import EvidenceUnit, Span

        units = [
            EvidenceUnit(span=Span("d", i, i + 1), text=f"unit {i}", token_count=1)
            for i in range(40)
        ]
        for i, u in enumerate(units):
            u.retrieval_score = 1.0 - i * 0.01
        item = ClaimEvidence(claim="c", units=units, token_count=len(units))

        compressed = compress([item], cfg=cfg, mode="random")[0]
        ranked_top = {u.text for u in sorted(units, key=lambda x: -x.retrieval_score)[
            : len(compressed.kept)
        ]}
        assert {u.text for u in compressed.kept} != ranked_top

    @pytest.mark.parametrize("mode", ["fixed_ratio", "random"])
    def test_ablation_arms_are_reversible(self, claim_evidence, corpus, cfg, mode):
        """Reversibility is a property of the design, not of one mode."""
        original = signature(claim_evidence.units)
        compressed = compress([claim_evidence], cfg=cfg, mode=mode)[0]
        assert signature(restore(compressed).units) == original
        assert signature(restore_from_corpus(compressed, corpus).units) == original

    @pytest.mark.parametrize("mode", ["fixed_ratio", "random"])
    def test_ablation_arms_conserve_units(self, claim_evidence, cfg, mode):
        compressed = compress([claim_evidence], cfg=cfg, mode=mode)[0]
        assert len(compressed.kept) + len(compressed.dropped) == len(
            claim_evidence.units
        )

    def test_unknown_mode_raises(self, claim_evidence, cfg):
        with pytest.raises(ValueError, match="unknown compression mode"):
            compress([claim_evidence], cfg=cfg, mode="nonsense")


class TestClaimEvidenceView:
    def test_view_exposes_only_kept_units(self, claim_evidence, cfg):
        compressed = compress([claim_evidence], cfg=cfg)[0]
        view = to_claim_evidence(compressed)
        assert view.units == compressed.kept
        assert view.token_count == compressed.compressed_token_count
        assert view.claim == compressed.claim
