"""Chunking and span-resolution tests.

The offset invariant verified here is the foundation of the whole reversibility claim:
if a span does not resolve to exactly the text it was cut from, restoration returns the
wrong evidence and every downstream guarantee is void.
"""

from __future__ import annotations

import pytest

from src.kb import chunk_corpus, chunk_document, corpus_fingerprint, load_corpus
from src.types import Document, Span


class TestSpanResolution:
    def test_span_resolves_to_exact_text(self, corpus):
        span = Span("doc_a", 0, 51)
        assert span.resolve(corpus) == "Water boils at one hundred degrees Celsius at sea level."[:51]

    def test_unknown_document_raises(self, corpus):
        with pytest.raises(KeyError):
            Span("nope", 0, 5).resolve(corpus)

    def test_out_of_bounds_raises(self, corpus):
        with pytest.raises(ValueError):
            Span("doc_a", 0, 10_000).resolve(corpus)

    def test_inverted_span_raises(self, corpus):
        with pytest.raises(ValueError):
            Span("doc_a", 20, 10).resolve(corpus)


class TestChunkingInvariant:
    """THE Phase 1 gate: chunk.text must equal what chunk.span points at."""

    def test_fixture_corpus_invariant(self, corpus, units):
        assert units, "expected non-empty chunk list"
        for unit in units:
            assert unit.span.resolve(corpus) == unit.text
            assert unit.verify_against(corpus)

    def test_real_kb_invariant(self, cfg):
        """Same invariant over the actual seeded knowledge base."""
        real_corpus = load_corpus(cfg=cfg)
        real_units = chunk_corpus(real_corpus, cfg=cfg)
        assert len(real_units) > 50, "seeded KB should produce a substantial chunk set"
        violations = [u for u in real_units if not u.verify_against(real_corpus)]
        assert violations == [], f"{len(violations)} chunks broke the offset invariant"

    def test_chunks_are_ordered_and_non_overlapping(self, corpus, cfg):
        for doc_id, doc in corpus.items():
            doc_units = chunk_document(doc, cfg=cfg)
            spans = [(u.span.start, u.span.end) for u in doc_units]
            assert spans == sorted(spans)
            for (_, prev_end), (next_start, _) in zip(spans, spans[1:]):
                assert next_start >= prev_end, f"overlapping chunks in {doc_id}"

    def test_no_content_is_lost(self, corpus, cfg):
        """Every non-whitespace character survives into some chunk."""
        for doc in corpus.values():
            doc_units = chunk_document(doc, cfg=cfg)
            covered = "".join(u.text for u in doc_units)
            assert "".join(covered.split()) == "".join(doc.text.split())

    def test_no_empty_or_whitespace_chunks(self, units):
        assert all(u.text.strip() for u in units)


class TestChunkSizing:
    def test_respects_max_chunk_chars(self, cfg):
        long_doc = Document(doc_id="long", text="word " * 800)
        max_chars = int(cfg.require("kb.max_chunk_chars"))
        for unit in chunk_document(long_doc, cfg=cfg):
            assert len(unit.text) <= max_chars

    def test_long_document_still_holds_invariant(self, cfg):
        long_doc = Document(doc_id="long", text="alpha beta gamma delta. " * 200)
        corpus = {"long": long_doc}
        for unit in chunk_document(long_doc, cfg=cfg):
            assert unit.span.resolve(corpus) == unit.text

    def test_abbreviations_do_not_produce_fragments(self, cfg):
        doc = Document(
            doc_id="abbr",
            text=(
                "Dr. Smith joined the lab in 2019 and now leads the retrieval group. "
                "She previously worked at a search company for six years."
            ),
        )
        chunks = chunk_document(doc, cfg=cfg)
        # "Dr." must not survive as its own evidence unit.
        assert all(len(c.text) >= 10 for c in chunks)
        assert not any(c.text.strip() == "Dr." for c in chunks)


class TestCorpusLoading:
    def test_fingerprint_is_stable_and_content_sensitive(self, corpus):
        first = corpus_fingerprint(corpus)
        assert first == corpus_fingerprint(corpus)

        mutated = dict(corpus)
        mutated["doc_a"] = Document(doc_id="doc_a", text=corpus["doc_a"].text + " More.")
        assert corpus_fingerprint(mutated) != first

    def test_missing_directory_raises(self):
        with pytest.raises(FileNotFoundError):
            load_corpus(path="/nonexistent/kb/path")

    def test_real_corpus_loads(self, cfg):
        real = load_corpus(cfg=cfg)
        assert len(real) >= 20
        assert all(doc.text.strip() for doc in real.values())
