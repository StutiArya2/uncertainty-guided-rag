"""Claim decomposition tests.

The key property is that a claim must not read as a question: the NLI evaluator scores
claims as hypotheses, and interrogatives score near zero regardless of evidence quality.
"""

from __future__ import annotations

import pytest

from src.claims import decompose, to_topic_phrase


class TestTopicPhrase:
    @pytest.mark.parametrize(
        "question,expected",
        [
            ("What are the two tunable parameters of BM25?", "the two tunable parameters of BM25"),
            ("What is calibration?", "calibration"),
            ("Explain prompt compression.", "prompt compression"),
            ("Describe chunking", "chunking"),
            ("How many parameters does BM25 have?", "BM25 have parameters"),
            ("Tell me about reranking", "reranking"),
        ],
    )
    def test_strips_interrogative_frame(self, question, expected):
        assert to_topic_phrase(question) == expected

    def test_subject_is_preserved(self):
        """Losing the subject would make relevant and irrelevant evidence score alike."""
        assert "BM25" in to_topic_phrase("What are the tunable parameters of BM25?")

    def test_no_trailing_question_mark(self):
        assert not to_topic_phrase("What is chunking?").endswith("?")

    def test_no_leading_wh_word(self):
        for q in ["What is X in Y?", "Why does X happen?", "When did X occur?"]:
            first = to_topic_phrase(q).split()[0].lower()
            assert first not in {"what", "why", "when", "how", "which", "who"}

    def test_empty_input(self):
        assert to_topic_phrase("") == ""
        assert to_topic_phrase("?") == ""


class TestDecompose:
    def test_simple_query_yields_one_claim(self):
        assert len(decompose("What is calibration?")) == 1

    def test_compound_query_splits(self):
        claims = decompose("What is chunking and also why do offsets matter?")
        assert len(claims) == 2
        assert "chunking" in claims

    def test_semicolon_splits(self):
        assert len(decompose("Why do models hallucinate; what reduces it?")) == 2

    def test_conjunction_inside_noun_phrase_is_not_split(self):
        """"k1 and b" must stay one claim — splitting on bare 'and' would shred it."""
        assert len(decompose("What are k1 and b in BM25?")) == 1

    def test_single_word_topics_survive(self):
        """"chunking" is a legitimate claim; an over-eager length filter dropped it."""
        assert decompose("What is chunking?") == ["chunking"]

    def test_never_returns_empty(self):
        for query in ["?", "what", "How?", ""]:
            assert decompose(query), f"empty claim set for {query!r}"

    def test_respects_max_claims(self, cfg):
        query = "; ".join(f"what is topic{i}" for i in range(20))
        assert len(decompose(query, cfg=cfg)) <= int(cfg.require("claims.max_claims"))

    def test_deduplicates(self):
        assert len(decompose("What is chunking; what is chunking?")) == 1

    def test_is_deterministic(self):
        query = "What is chunking and also why do offsets matter?"
        assert decompose(query) == decompose(query)
