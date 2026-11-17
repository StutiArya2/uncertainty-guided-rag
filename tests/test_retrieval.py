"""Retrieval tests, including single-document restriction.

`restrict_to` exists because of a real measurement error. QASPER questions are asked
about one specific paper and are not self-identifying — "which datasets did they
experiment with?" means nothing without knowing which paper "they" is. Run against a
110-paper corpus, retrieval returned confident evidence from the wrong papers, and the
whole evaluation reported ~1% answer F1 and 83% false abstention *even with no
compression at all*. The bug was in the harness, not the pipeline, and these tests pin
the fix so it cannot regress silently.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.retrieval import Retriever
from src.types import Document


@pytest.fixture
def multi_doc_corpus():
    return {
        "paper_a": Document(
            doc_id="paper_a",
            text=(
                "We evaluate on the Europarl corpus. "
                "Our model uses a transformer encoder with eight layers. "
                "Training took three days on four GPUs."
            ),
        ),
        "paper_b": Document(
            doc_id="paper_b",
            text=(
                "We evaluate on the SQuAD dataset. "
                "Our model uses a recurrent encoder with two layers. "
                "Training took six hours on one GPU."
            ),
        ),
    }


class FakeEmbedder:
    """Deterministic bag-of-words vectors — keeps the test offline and exact."""

    model_id = "fake"

    def __init__(self, vocab):
        self.vocab = vocab

    def encode(self, texts, batch_size=32):
        out = np.zeros((len(texts), len(self.vocab)), dtype=np.float32)
        for i, text in enumerate(texts):
            words = set(text.lower().replace(".", "").split())
            for j, term in enumerate(self.vocab):
                out[i, j] = 1.0 if term in words else 0.0
            norm = np.linalg.norm(out[i])
            if norm:
                out[i] /= norm
        return out


@pytest.fixture
def retriever(multi_doc_corpus, cfg):
    r = Retriever(corpus=multi_doc_corpus, cfg=cfg)
    r.embedder = FakeEmbedder(
        ["europarl", "squad", "transformer", "recurrent", "gpus", "gpu", "dataset", "corpus"]
    )
    r._matrix = r.embedder.encode([u.text for u in r.units])
    return r


class TestRestrictTo:
    def test_unrestricted_search_spans_the_corpus(self, retriever):
        docs = {u.span.doc_id for u in retriever.retrieve("encoder layers", top_k=6)}
        assert docs == {"paper_a", "paper_b"}

    def test_restriction_confines_results_to_one_paper(self, retriever):
        results = retriever.retrieve("encoder layers", top_k=6, restrict_to="paper_b")
        assert results
        assert {u.span.doc_id for u in results} == {"paper_b"}

    def test_restriction_returns_that_paper_s_own_answer(self, retriever):
        """The point of the fix: the right paper's evidence, not a confident wrong one."""
        results = retriever.retrieve("which dataset", top_k=3, restrict_to="paper_a")
        assert any("Europarl" in u.text for u in results)
        assert not any("SQuAD" in u.text for u in results)

    def test_unknown_document_returns_nothing(self, retriever):
        assert retriever.retrieve("anything", restrict_to="paper_zzz") == []

    def test_spans_still_resolve_after_restriction(self, retriever, multi_doc_corpus):
        """Index remapping must not corrupt the span invariant."""
        for unit in retriever.retrieve("encoder", top_k=3, restrict_to="paper_b"):
            assert unit.verify_against(multi_doc_corpus)

    def test_scores_match_the_unrestricted_run(self, retriever):
        """Restriction filters candidates; it must not change how they are scored."""
        restricted = retriever.retrieve("encoder layers", top_k=6, restrict_to="paper_a")
        everything = retriever.retrieve("encoder layers", top_k=99)
        expected = {u.text: u.retrieval_score for u in everything}
        for unit in restricted:
            assert unit.retrieval_score == pytest.approx(expected[unit.text])

    def test_results_stay_sorted(self, retriever):
        scores = [
            u.retrieval_score
            for u in retriever.retrieve("encoder", top_k=5, restrict_to="paper_a")
        ]
        assert scores == sorted(scores, reverse=True)


def _in_mode(retriever, cfg, mode):
    """The same retriever switched to another first-stage mode."""
    retriever.mode = mode
    return retriever


class TestBM25Index:
    def test_matches_the_okapi_formula_by_hand(self):
        from src.retrieval import BM25Index

        index = BM25Index(["alpha beta beta", "alpha gamma", "delta"], k1=1.2, b=0.75)
        # "beta": df=1 of N=3, tf=2 in doc 0, |d|=3, avgdl=2.
        idf = np.log(1.0 + (3 - 1 + 0.5) / (1 + 0.5))
        expected = idf * 2 * 2.2 / (2 + 1.2 * (1 - 0.75 + 0.75 * 3 / 2))
        scores = index.scores("beta")
        assert scores[0] == pytest.approx(expected)
        assert scores[1] == 0.0 and scores[2] == 0.0

    def test_rarer_terms_weigh_more_and_idf_is_never_negative(self):
        from src.retrieval import BM25Index

        index = BM25Index(["a b", "a c", "a d"], k1=1.2, b=0.75)
        assert index.idf("b") > index.idf("a") > 0.0
        assert index.idf("missing") > index.idf("b")

    def test_repeated_query_terms_count_once(self):
        from src.retrieval import BM25Index

        index = BM25Index(["alpha beta", "gamma"], k1=1.2, b=0.75)
        assert np.allclose(index.scores("alpha alpha alpha"), index.scores("alpha"))

    def test_no_score_exceeds_the_ceiling(self):
        from src.retrieval import BM25Index

        index = BM25Index(["alpha " * 50, "alpha beta", "gamma"], k1=1.2, b=0.75)
        assert index.scores("alpha beta").max() <= index.max_score("alpha beta")

    def test_empty_index_and_unknown_terms_are_safe(self):
        from src.retrieval import BM25Index

        assert BM25Index([], k1=1.2, b=0.75).scores("anything").shape == (0,)
        assert not BM25Index(["alpha"], k1=1.2, b=0.75).scores("zzz ???").any()


class TestModes:
    def test_unknown_mode_is_rejected(self, multi_doc_corpus, cfg):
        cfg["retrieval"] = dict(cfg["retrieval"], mode="sparse")
        with pytest.raises(ValueError, match="retrieval.mode"):
            Retriever(corpus=multi_doc_corpus, cfg=cfg)

    def test_bm25_finds_the_lexical_match_without_the_embedder(self, retriever, cfg):
        r = _in_mode(retriever, cfg, "bm25")
        r._matrix, r.embedder = None, None  # any dense call would now fail
        top = r.retrieve("recurrent encoder", top_k=1)[0]
        assert top.span.doc_id == "paper_b" and "recurrent" in top.text
        assert 0.0 < top.retrieval_score <= 1.0
        assert top.fusion_score is None

    def test_bm25_respects_restriction(self, retriever, cfg):
        r = _in_mode(retriever, cfg, "bm25")
        results = r.retrieve("encoder layers", restrict_to="paper_a")
        assert results and {u.span.doc_id for u in results} == {"paper_a"}

    def test_hybrid_keeps_cosine_and_orders_by_fusion(self, retriever, cfg):
        query = "which dataset squad"
        dense = {u.span: u.retrieval_score for u in retriever.retrieve(query, top_k=6)}
        hybrid = _in_mode(retriever, cfg, "hybrid").retrieve(query, top_k=6)
        for unit in hybrid:
            assert unit.retrieval_score == pytest.approx(dense[unit.span])
            assert unit.rank_score == unit.fusion_score
        fused = [u.fusion_score for u in hybrid]
        assert fused == sorted(fused, reverse=True)

    def test_a_unit_ranked_first_by_both_scores_one(self, retriever, cfg):
        top = _in_mode(retriever, cfg, "hybrid").retrieve("squad dataset", top_k=1)[0]
        assert "SQuAD" in top.text
        assert top.fusion_score == pytest.approx(1.0)

    def test_hybrid_surfaces_a_match_dense_cannot_see(self, retriever, cfg):
        # "hours" is outside the fake embedder's vocabulary: dense scores every unit zero.
        query = "hours"
        assert all(u.retrieval_score == 0.0 for u in retriever.retrieve(query, top_k=6))
        top = _in_mode(retriever, cfg, "hybrid").retrieve(query, top_k=1)[0]
        assert "six hours" in top.text

    def test_results_are_deterministic(self, retriever, cfg):
        r = _in_mode(retriever, cfg, "hybrid")
        a = [(u.span, u.fusion_score) for u in r.retrieve("encoder layers", top_k=6)]
        b = [(u.span, u.fusion_score) for u in r.retrieve("encoder layers", top_k=6)]
        assert a == b

    def test_tied_scores_share_a_rank(self):
        from src.retrieval import _ranks

        assert _ranks(np.array([0.5, 0.9, 0.5, 0.1])).tolist() == [2, 1, 2, 4]
        assert _ranks(np.zeros(3)).tolist() == [1, 1, 1]

    def test_units_without_a_lexical_match_get_no_lexical_credit(self, retriever, cfg):
        # Nothing matches lexically, so fusion must reduce to the dense order exactly.
        query = "europarl"
        dense = [u.span for u in retriever.retrieve(query, top_k=6)]
        r = _in_mode(retriever, cfg, "hybrid")
        r._bm25 = type("NoMatch", (), {"scores": lambda self, q: np.zeros(len(r.units))})()
        assert [u.span for u in r.retrieve(query, top_k=6)] == dense
