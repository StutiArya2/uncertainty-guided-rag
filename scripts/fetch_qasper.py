"""Build a real-paper evaluation set from QASPER.

QASPER (Dasigi et al., 2021) is question answering over NLP research papers: the
questions were written by domain experts who had read only the title and abstract, and
answered by other experts with access to the full text. That matters here for three
reasons:

* The papers are real, so the corpus is not something we wrote to suit our own system.
* The questions are not ours either. Authoring both the benchmark and the system that
  answers it invites bias that no amount of care fully removes.
* Roughly a fifth of the questions are marked *unanswerable from this paper* by human
  annotators. Those are real labels for the abstain path, which until now had only
  questions we invented.

Fetched over the HuggingFace datasets-server REST API — no `datasets` dependency, in
keeping with the project's minimal requirements.

Writes, under data/qasper/:
    kb/<paper_id>.txt   one document per paper (title, abstract, body)
    questions.yaml      the eval set, in the same format as data/eval/questions.yaml

Usage:
    python scripts/fetch_qasper.py --papers 80
    python scripts/fetch_qasper.py --papers 80 --force
"""

from __future__ import annotations

import argparse
import json
import re
import ssl
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.ingest import clean_text, slugify  # noqa: E402

OUT_DIR = REPO_ROOT / "data" / "qasper"
KB_DIR = OUT_DIR / "kb"
QUESTIONS = OUT_DIR / "questions.yaml"
CACHE = OUT_DIR / ".raw_cache.json"

API = (
    "https://datasets-server.huggingface.co/rows"
    "?dataset=allenai%2Fqasper&config=qasper&split={split}&offset={offset}&length={length}"
)
PAGE = 50

# QASPER inlines citation and float markers into answer text. As expected answer content
# they are noise — no generator could produce "BIBREF19" from the prose it was given.
_MARKERS = re.compile(r"\b(?:BIBREF|TABREF|FIGREF|SECREF|FLOAT|UNKREF)\d*\b")


def fetch_page(split: str, offset: int, length: int, retries: int = 4) -> list[dict]:
    url = API.format(split=split, offset=offset, length=length)
    request = urllib.request.Request(url, headers={"User-Agent": "uncertainty-guided-rag"})
    ctx = ssl.create_default_context()
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=90, context=ctx) as response:
                return json.loads(response.read().decode())["rows"]
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            wait = 2 ** attempt
            print(f"  page {offset} failed ({type(exc).__name__}), retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"could not fetch rows at offset {offset} after {retries} tries")


def paper_text(row: dict) -> str:
    """Flatten a QASPER record into one plain-text document."""
    parts = [row["title"], "", row["abstract"]]
    full = row.get("full_text") or {}
    for name, paragraphs in zip(full.get("section_name") or [], full.get("paragraphs") or []):
        body = "\n\n".join(p for p in paragraphs if p and p.strip())
        if not body.strip():
            continue
        parts.append("")
        if name:
            parts.append(str(name))
        parts.append(body)
    return clean_text("\n".join(parts))


def gold_answer(answer: dict) -> str | None:
    """The reference answer as text, or None if the annotator marked it unanswerable."""
    if answer.get("unanswerable"):
        return None
    if answer.get("yes_no") is not None:
        return "yes" if answer["yes_no"] else "no"

    free_form = (answer.get("free_form_answer") or "").strip()
    if free_form:
        return _MARKERS.sub("", free_form).strip()

    spans = [s for s in (answer.get("extractive_spans") or []) if s and s.strip()]
    cleaned = [_MARKERS.sub("", s).strip() for s in spans]
    cleaned = [s for s in cleaned if len(s) > 2]
    return "; ".join(cleaned) if cleaned else None


def collect(rows: list[dict], seen: set[str]) -> tuple[list[dict], list[dict]]:
    """Turn raw records into KB documents and eval questions."""
    documents, questions = [], []

    for entry in rows:
        row = entry["row"]
        title = (row.get("title") or "").strip()
        text = paper_text(row)
        if len(text) < 1500:
            continue

        doc_id = slugify(title)
        if not doc_id or doc_id in seen:
            continue
        seen.add(doc_id)
        documents.append({"doc_id": doc_id, "text": text})

        qas = row.get("qas") or {}
        for question, answer_set in zip(qas.get("question") or [], qas.get("answers") or []):
            question = (question or "").strip()
            if len(question) < 10:
                continue

            annotations = answer_set.get("answer") or []
            golds = [gold_answer(a) for a in annotations]
            answerable = [g for g in golds if g]

            # Trust the annotators' majority view on answerability.
            if annotations and len(answerable) * 2 < len(annotations):
                questions.append(
                    {"query": question, "paper": doc_id, "answerable": False, "gold": []}
                )
            elif answerable:
                questions.append(
                    {
                        "query": question,
                        "paper": doc_id,
                        "answerable": True,
                        "gold": answerable[:3],
                    }
                )

    return documents, questions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--papers", type=int, default=80)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--force", action="store_true", help="refetch and overwrite")
    args = parser.parse_args(argv)

    KB_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    if CACHE.exists() and not args.force:
        rows = json.loads(CACHE.read_text(encoding="utf-8"))
        print(f"reusing {len(rows)} cached records (use --force to refetch)")

    offset = len(rows)
    while len(rows) < args.papers:
        want = min(PAGE, args.papers - len(rows))
        print(f"fetching papers {offset}..{offset + want}")
        page = fetch_page(args.split, offset, want)
        if not page:
            print("no more records available")
            break
        rows.extend(page)
        offset += len(page)
        CACHE.write_text(json.dumps(rows), encoding="utf-8")

    documents, questions = collect(rows, set())

    for doc in documents:
        (KB_DIR / f"{doc['doc_id']}.txt").write_text(doc["text"] + "\n", encoding="utf-8")

    # Written by hand rather than via yaml.dump: the file is meant to be read and edited,
    # and dump would mangle the long gold answers into unreadable flow scalars.
    lines = [
        "# Generated by scripts/fetch_qasper.py — do not edit by hand.",
        "#",
        "# Real NLP papers, questions written by domain experts, answers by other experts.",
        "# `answerable: false` entries are human-labelled as unanswerable from the paper,",
        "# so they test the abstain path against real labels rather than invented ones.",
        "",
    ]
    for q in questions:
        lines.append(f"- query: {json.dumps(q['query'])}")
        lines.append(f"  paper: {q['paper']}")
        lines.append(f"  answerable: {str(q['answerable']).lower()}")
        if q["gold"]:
            lines.append("  gold:")
            lines.extend(f"    - {json.dumps(g)}" for g in q["gold"])
        else:
            lines.append("  gold: []")
    QUESTIONS.write_text("\n".join(lines) + "\n", encoding="utf-8")

    answerable = sum(1 for q in questions if q["answerable"])
    print(f"\n{len(documents)} papers -> {KB_DIR}")
    print(f"{len(questions)} questions -> {QUESTIONS}")
    print(f"  answerable:   {answerable}")
    print(f"  unanswerable: {len(questions) - answerable}  (real labels for the abstain path)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
