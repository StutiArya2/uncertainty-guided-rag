"""Knowledge base loading and offset-preserving chunking.

Everything downstream depends on one invariant:

    corpus[chunk.span.doc_id].text[chunk.span.start:chunk.span.end] == chunk.text

If that ever breaks, restoration silently returns the wrong text and the reversibility
guarantee is void. It is enforced structurally here — chunk text is *always* sliced out of
the document rather than accumulated by string concatenation — and verified independently
in tests/test_kb.py.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from .config import Config, default_config
from .types import Corpus, Document, EvidenceUnit, Span

# Split after ., ! or ? followed by whitespace. Deliberately simple: no nltk/spacy
# dependency (CLAUDE.md asks before adding those). Over-splitting on abbreviations like
# "Dr. Smith" is repaired by the min_chunk_chars merge pass below.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
_WHITESPACE = re.compile(r"\s+")


def load_corpus(path: str | Path | None = None, cfg: Config | None = None) -> Corpus:
    """Load every .txt file under the KB directory. doc_id is the filename stem."""
    cfg = cfg or default_config()
    kb_path = Path(path) if path is not None else Path(cfg.require("kb.path"))
    if not kb_path.is_absolute():
        from .config import REPO_ROOT

        kb_path = REPO_ROOT / kb_path

    if not kb_path.exists():
        raise FileNotFoundError(f"knowledge base directory not found: {kb_path}")

    corpus: Corpus = {}
    for file in sorted(kb_path.glob("*.txt")):
        text = file.read_text(encoding="utf-8")
        if not text.strip():
            continue
        corpus[file.stem] = Document(
            doc_id=file.stem, text=text, meta={"source": str(file)}
        )

    if not corpus:
        raise ValueError(f"no non-empty .txt documents found in {kb_path}")
    return corpus


def _sentence_offsets(text: str) -> list[tuple[int, int]]:
    """Character offsets of each sentence. Inter-sentence whitespace is excluded."""
    offsets: list[tuple[int, int]] = []
    start = 0
    for match in _SENTENCE_BOUNDARY.finditer(text):
        end = match.start()
        if end > start:
            offsets.append((start, end))
        start = match.end()
    if start < len(text):
        offsets.append((start, len(text)))
    return offsets


def _merge_short(
    offsets: list[tuple[int, int]], min_chars: int
) -> list[tuple[int, int]]:
    """Merge undersized fragments into the previous chunk.

    Repairs abbreviation over-splits ("Dr." + "Smith was...") and stops stray bullets
    becoming standalone evidence units. Merging simply widens the span, so the slicing
    invariant is untouched.
    """
    if not offsets:
        return []
    merged = [offsets[0]]
    for start, end in offsets[1:]:
        prev_start, prev_end = merged[-1]
        if (prev_end - prev_start) < min_chars or (end - start) < min_chars:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))
    return merged


def _split_long(
    text: str, offsets: list[tuple[int, int]], max_chars: int
) -> list[tuple[int, int]]:
    """Break oversized chunks at whitespace so no single unit dominates the prompt."""
    result: list[tuple[int, int]] = []
    for start, end in offsets:
        if (end - start) <= max_chars:
            result.append((start, end))
            continue
        cursor = start
        while (end - cursor) > max_chars:
            window_end = cursor + max_chars
            # Prefer the last whitespace inside the window; fall back to a hard cut.
            breaks = [m.start() for m in _WHITESPACE.finditer(text, cursor, window_end)]
            split_at = breaks[-1] if breaks else window_end
            if split_at <= cursor:
                split_at = window_end
            result.append((cursor, split_at))
            cursor = split_at
            while cursor < end and text[cursor].isspace():
                cursor += 1
        if cursor < end:
            result.append((cursor, end))
    return result


def chunk_document(doc: Document, cfg: Config | None = None) -> list[EvidenceUnit]:
    """Split one document into evidence units with exact character offsets."""
    cfg = cfg or default_config()
    min_chars = int(cfg.require("kb.min_chunk_chars"))
    max_chars = int(cfg.require("kb.max_chunk_chars"))

    offsets = _sentence_offsets(doc.text)
    offsets = _merge_short(offsets, min_chars)
    offsets = _split_long(doc.text, offsets, max_chars)

    units: list[EvidenceUnit] = []
    for start, end in offsets:
        # Text is sliced from the document, never rebuilt — this is what makes the
        # span/text invariant true by construction.
        chunk_text = doc.text[start:end]
        if not chunk_text.strip():
            continue
        units.append(
            EvidenceUnit(span=Span(doc.doc_id, start, end), text=chunk_text)
        )
    return units


def chunk_corpus(corpus: Corpus, cfg: Config | None = None) -> list[EvidenceUnit]:
    """Chunk every document in the corpus into a flat, retrievable unit list."""
    units: list[EvidenceUnit] = []
    for doc_id in sorted(corpus):
        units.extend(chunk_document(corpus[doc_id], cfg=cfg))
    return units


def corpus_fingerprint(corpus: Corpus) -> str:
    """Stable hash of corpus content, used to invalidate the embedding cache."""
    digest = hashlib.sha256()
    for doc_id in sorted(corpus):
        digest.update(doc_id.encode("utf-8"))
        digest.update(corpus[doc_id].text.encode("utf-8"))
    return digest.hexdigest()[:16]
