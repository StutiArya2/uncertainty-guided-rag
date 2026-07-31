"""Paper ingestion tests.

Text cleaning runs *before* a paper is stored, never after, because chunk offsets are
computed against the stored file. A rewrite applied later would leave every span
pointing at the wrong characters and silently break restoration — so the ordering is
worth pinning down.
"""

from __future__ import annotations

import pytest

from src.config import Config, load_config
from src.ingest import (
    IngestError,
    add_document,
    clean_text,
    delete_document,
    list_documents,
    slugify,
)

LONG_ENOUGH = (
    "Sparse attention reduces the quadratic cost of self-attention. "
    "Sliding window attention lets a token attend to its neighbours. "
    "Global tokens preserve long-range information flow across the sequence."
)


@pytest.fixture
def temp_cfg(tmp_path):
    """Config pointed at a throwaway KB, so tests never touch the real corpus."""
    cfg = Config(dict(load_config()))
    cfg["kb"] = dict(cfg["kb"], path=str(tmp_path / "kb"))
    return cfg


class TestSlugify:
    @pytest.mark.parametrize(
        "title,expected",
        [
            ("Attention Is All You Need", "attention_is_all_you_need"),
            ("BM25: a ranking function", "bm25_a_ranking_function"),
            ("  spaced  out  ", "spaced_out"),
            ("Café Naïve", "cafe_naive"),
        ],
    )
    def test_makes_safe_ids(self, title, expected):
        assert slugify(title) == expected

    def test_never_empty(self):
        assert slugify("!!!") == "paper"

    def test_bounded_length(self):
        assert len(slugify("word " * 100)) <= 60


class TestCleanText:
    def test_expands_ligatures(self):
        assert clean_text("the eﬀect of ﬁne-tuning") == "the effect of fine-tuning"

    def test_rejoins_hyphenated_line_breaks(self):
        assert "compression" in clean_text("compres-\nsion works")

    def test_single_newlines_become_spaces(self):
        assert clean_text("one\ntwo") == "one two"

    def test_paragraph_breaks_survive(self):
        assert "\n\n" in clean_text("para one\n\npara two")

    def test_normalises_quotes_and_dashes(self):
        assert clean_text("“quoted” — dashed") == '"quoted" - dashed'


class TestAddDocument:
    def test_stores_and_returns_id(self, temp_cfg):
        doc_id = add_document("Sparse Attention", LONG_ENOUGH, cfg=temp_cfg)
        assert doc_id == "sparse_attention"
        assert len(list_documents(temp_cfg)) == 1

    def test_stored_text_is_already_clean(self, temp_cfg):
        """Cleaning must happen before storage, or spans would point at stale text."""
        add_document("Ligature", "The eﬀect is clear. " + LONG_ENOUGH, cfg=temp_cfg)
        stored = list_documents(temp_cfg)[0]["preview"]
        assert "ﬀ" not in stored
        assert "effect" in stored

    def test_rejects_empty_title(self, temp_cfg):
        with pytest.raises(IngestError, match="title"):
            add_document("", LONG_ENOUGH, cfg=temp_cfg)

    def test_rejects_too_short_text(self, temp_cfg):
        with pytest.raises(IngestError, match="too short"):
            add_document("Stub", "Not much.", cfg=temp_cfg)

    def test_duplicate_titles_do_not_overwrite(self, temp_cfg):
        first = add_document("Same Name", LONG_ENOUGH, cfg=temp_cfg)
        second = add_document("Same Name", LONG_ENOUGH, cfg=temp_cfg)
        assert first != second
        assert len(list_documents(temp_cfg)) == 2

    def test_added_paper_is_chunkable(self, temp_cfg):
        """A paper added through the UI must behave like a seeded one."""
        from src.kb import chunk_corpus, load_corpus

        add_document("Sparse Attention", LONG_ENOUGH, cfg=temp_cfg)
        corpus = load_corpus(cfg=temp_cfg)
        units = chunk_corpus(corpus, cfg=temp_cfg)
        assert units
        # The span invariant the whole reversibility guarantee rests on.
        assert all(u.verify_against(corpus) for u in units)


class TestDeleteDocument:
    def test_removes(self, temp_cfg):
        doc_id = add_document("Removable", LONG_ENOUGH, cfg=temp_cfg)
        assert delete_document(doc_id, cfg=temp_cfg) is True
        assert list_documents(temp_cfg) == []

    def test_missing_returns_false(self, temp_cfg):
        assert delete_document("never_existed", cfg=temp_cfg) is False

    @pytest.mark.parametrize("bad", ["../../etc/passwd", "a/b", "", "x.txt"])
    def test_rejects_path_traversal(self, temp_cfg, bad):
        """doc_id comes from an HTTP request, so it must not escape the KB directory."""
        with pytest.raises(IngestError):
            delete_document(bad, cfg=temp_cfg)


class TestListDocuments:
    def test_empty_kb(self, temp_cfg):
        assert list_documents(temp_cfg) == []

    def test_reports_counts(self, temp_cfg):
        add_document("Counted", LONG_ENOUGH, cfg=temp_cfg)
        entry = list_documents(temp_cfg)[0]
        assert entry["words"] > 10
        assert entry["characters"] > 100
        assert entry["title"] == "counted"
