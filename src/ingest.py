"""Adding and removing papers in the knowledge base.

Papers are stored as plain .txt in `data/kb/` — the same format `scripts/seed_kb.py`
writes, so anything added through the web UI is indistinguishable from the seeded corpus
and works with every other tool in the repo.

PDF support is optional. `pypdf` is not in requirements.txt (the dependency set is
deliberately minimal, see CLAUDE.md), so PDF extraction degrades to a clear message
telling the reader how to enable it rather than failing obscurely.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

from .config import REPO_ROOT, Config, default_config

# Ligatures and typographic characters that PDF extraction leaves behind. Left in place,
# they split words and corrupt the character offsets that restoration depends on.
_PDF_SUBSTITUTIONS = {
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl",
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "−": "-", " ": " ",
}


class IngestError(Exception):
    """Raised with a message intended to be shown directly to the reader."""


def slugify(title: str) -> str:
    """Turn a paper title into a safe document id."""
    normalised = unicodedata.normalize("NFKD", title)
    ascii_only = normalised.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", ascii_only).strip("_").lower()
    slug = re.sub(r"_+", "_", slug)
    return slug[:60] or "paper"


def kb_dir(cfg: Config | None = None) -> Path:
    cfg = cfg or default_config()
    path = Path(cfg.require("kb.path"))
    if not path.is_absolute():
        path = REPO_ROOT / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def clean_text(text: str) -> str:
    """Normalise extracted text.

    Runs before the text is stored, never after — chunk offsets are computed against the
    stored file, so any rewriting has to happen first or spans would point at the wrong
    characters.
    """
    for bad, good in _PDF_SUBSTITUTIONS.items():
        text = text.replace(bad, good)

    # Rejoin words split across a line break by hyphenation ("compres-\nsion").
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # Single newlines inside a paragraph become spaces; blank lines stay as breaks.
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf_text(data: bytes) -> str:
    """Extract text from a PDF. Requires pypdf, which is an optional extra."""
    try:
        from pypdf import PdfReader
    except ImportError:
        raise IngestError(
            "PDF reading needs the pypdf package. Install it with "
            "'pip install pypdf', then try again. You can also paste the "
            "paper's text directly."
        ) from None

    import io

    try:
        reader = PdfReader(io.BytesIO(data))
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception as exc:  # noqa: BLE001 - surfaced to the reader as a message
        raise IngestError(f"That PDF could not be read: {exc}") from exc

    text = clean_text("\n\n".join(pages))
    if len(text) < 200:
        raise IngestError(
            "That PDF has almost no extractable text — it is probably a scan. "
            "Try a text-based PDF, or paste the text directly."
        )
    return text


def add_document(
    title: str, text: str, cfg: Config | None = None, overwrite: bool = False
) -> str:
    """Store a paper in the KB and return its document id."""
    title = (title or "").strip()
    if not title:
        raise IngestError("Give the paper a title so you can recognise it later.")

    text = clean_text(text or "")
    if len(text) < 100:
        raise IngestError(
            "That is too short to answer questions from — paste at least a "
            "few paragraphs."
        )

    directory = kb_dir(cfg)
    doc_id = slugify(title)
    path = directory / f"{doc_id}.txt"

    if path.exists() and not overwrite:
        suffix = 2
        while (directory / f"{doc_id}_{suffix}.txt").exists():
            suffix += 1
        doc_id = f"{doc_id}_{suffix}"
        path = directory / f"{doc_id}.txt"

    path.write_text(text + "\n", encoding="utf-8")
    return doc_id


def delete_document(doc_id: str, cfg: Config | None = None) -> bool:
    """Remove a paper. Returns False if it was not there."""
    # Guard against a doc_id escaping the KB directory via path traversal.
    if not re.fullmatch(r"[A-Za-z0-9_\-]+", doc_id or ""):
        raise IngestError("That paper name is not valid.")

    path = kb_dir(cfg) / f"{doc_id}.txt"
    if not path.exists():
        return False
    path.unlink()
    return True


def list_documents(cfg: Config | None = None) -> list[dict]:
    """Every paper in the KB, with a short preview for display."""
    out = []
    for path in sorted(kb_dir(cfg).glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        out.append(
            {
                "doc_id": path.stem,
                "title": path.stem.replace("_", " "),
                "characters": len(text),
                "words": len(text.split()),
                "preview": text[:220].strip(),
            }
        )
    return out
