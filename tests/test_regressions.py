"""Regression tests for bugs found by review (2026-11). Each failed before its fix."""

from __future__ import annotations

import numpy as np
import pytest

from src.compression import compress
from src.pipeline import Pipeline
from src.restoration import restore, restore_from_corpus
from src.types import ClaimEvidence, ClaimVerdict, EvidenceUnit

from tests.test_pipeline import StubCounter, StubGenerator, StubRetriever


def _claim_evidence(units):
    out = []
    for i, u in enumerate(units[:6]):
        out.append(EvidenceUnit(span=u.span, text=u.text, retrieval_score=0.9 - 0.1 * i,
                                token_count=len(u.text.split())))
    return ClaimEvidence(claim="c", units=out, token_count=sum(u.token_count for u in out))


class TestRoundTripAfterReranking:
    """`restore_from_corpus` rebuilt units without their rerank score, so the two recovery
    routes ordered evidence differently whenever `rerank.enabled` was on."""

    def test_both_recovery_routes_agree_when_reranked(self, units, corpus, cfg):
        item = _claim_evidence(units)
        # A reranker that reverses the retrieval order.
        for i, unit in enumerate(item.units):
            unit.rerank_score = 0.1 + 0.1 * i
        comp = compress([item], cfg=cfg, mode="fixed_ratio")[0]

        fast, rebuilt = restore(comp), restore_from_corpus(comp, corpus)
        assert [u.span for u in fast.units] == [u.span for u in rebuilt.units]
        assert [u.text for u in fast.units] == [u.text for u in rebuilt.units]
        assert [u.rerank_score for u in fast.units] == [u.rerank_score for u in rebuilt.units]


class _PerClaimEvaluator:
    """Support depends on the claim and on how much evidence it is shown."""

    threshold = 0.15
    contextualize = False

    def __init__(self, fragile_claim_index=0):
        self.claims: list[str] = []
        self.fragile = fragile_claim_index

    def evaluate(self, claim, units):
        if claim not in self.claims:
            self.claims.append(claim)
        fragile = self.claims.index(claim) == self.fragile
        # The fragile claim loses support when compressed; the other does not.
        score = 0.9 if (not fragile or len(units) >= 6) else 0.3
        return ClaimVerdict(claim=claim, is_sufficient=score >= self.threshold,
                            support_score=score, best_unit_index=0)


class _NothingEntailed:
    def score(self, premises, claim):
        return [0.0] * len(premises)


class TestAnswerAwareTokenAccounting:
    """Under `answer_aware`, every claim was marked restored once any claim was, so claims
    that kept their compression were still charged their full baseline cost."""

    def _run(self, cfg, units):
        cfg["restoration"] = dict(cfg["restoration"], policy="answer_aware")
        pipeline = Pipeline(cfg=cfg, retriever=StubRetriever(units), evaluator=_PerClaimEvaluator(),
                            generator=StubGenerator(), counter=StubCounter())
        pipeline._verifier = _NothingEntailed()
        return pipeline.run("What is the boiling point of water? How deep is the Pacific ocean?")

    def test_only_the_restored_claim_pays_its_tokens_back(self, cfg, units):
        _, trace = self._run(cfg, units)
        assert len(trace.claims) == 2 and trace.restoration_triggered
        fragile, stable = trace.claims
        assert fragile.n_dropped > 0 and stable.n_dropped > 0

        assert fragile.n_restored == fragile.n_dropped
        assert fragile.restored_support_score == pytest.approx(0.9)
        assert stable.n_restored == 0
        assert stable.restored_support_score is None
        assert trace.final_tokens == fragile.baseline_tokens + stable.compressed_tokens
        assert trace.reduction > 0

    def test_final_tokens_match_the_evidence_actually_sent(self, cfg, units):
        _, trace = self._run(cfg, units)
        # 6 units for the restored claim, the kept ones for the other.
        sent = sum(len(u.text.split()) for u in units[:6]) + trace.claims[1].compressed_tokens
        assert trace.final_tokens == sent


class TestSharedCounterRespectsTheModel:
    """`shared_counter` returned the first counter ever built, whatever model was asked for."""

    def test_a_different_generation_model_gets_its_own_counter(self, cfg):
        from src import tokens

        other = type(cfg)(cfg)
        other["models"] = dict(cfg["models"], generation="some/other-model")
        first, second = tokens.shared_counter(cfg), tokens.shared_counter(other)
        assert first.model_id == cfg["models"]["generation"]
        assert second.model_id == "some/other-model"
        assert tokens.shared_counter(cfg) is first


class TestEmbeddingCacheKey:
    """The cache was keyed by model and corpus text only, so re-chunking the same corpus into
    the same *number* of units silently reused embeddings of different text."""

    def test_changing_the_chunking_changes_the_cache_file(self, corpus, cfg, tmp_path):
        from src.retrieval import Retriever

        cfg["retrieval"] = dict(cfg["retrieval"], cache_dir=str(tmp_path))
        a = Retriever(corpus=corpus, cfg=cfg)
        rechunked = type(cfg)(cfg)
        rechunked["kb"] = dict(cfg["kb"], min_chunk_chars=5, max_chunk_chars=45)
        b = Retriever(corpus=corpus, cfg=rechunked)
        assert [u.span for u in a.units] != [u.span for u in b.units]
        assert a._cache_path() != b._cache_path()
        assert a._cache_path() == Retriever(corpus=corpus, cfg=cfg)._cache_path()


class TestVacuousDraftRestoresWhatItReports:
    """A draft with nothing to verify ("Yes") reported restoration without restoring: the
    trace said `restored`, charged full cost, and the generator still got compressed evidence."""

    def test_flagged_claim_is_actually_restored(self, cfg, units):
        class YesGenerator(StubGenerator):
            def complete(self, prompt, max_new_tokens=None):
                self.prompts.append(prompt)
                return "Yes"

        cfg["restoration"] = dict(cfg["restoration"], policy="answer_aware")
        generator = YesGenerator()
        pipeline = Pipeline(cfg=cfg, retriever=StubRetriever(units), evaluator=_PerClaimEvaluator(),
                            generator=generator, counter=StubCounter())
        pipeline._verifier = _NothingEntailed()
        answer, trace = pipeline.run("What is the boiling point of water? How deep is the Pacific ocean?")

        fragile, stable = trace.claims
        assert trace.grounding["vacuous"] and answer.branch == "restored"
        assert fragile.n_restored == fragile.n_dropped and stable.n_restored == 0
        final_prompt = generator.prompts[-1]
        assert all(u.text in final_prompt for u in units[:6])
        assert trace.final_tokens == fragile.baseline_tokens + stable.compressed_tokens
