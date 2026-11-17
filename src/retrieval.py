"""Pipeline stage 1 — retrieve candidate evidence from the knowledge base.

Deliberately dependency-light: embeddings come from `transformers.AutoModel` with mean
pooling (no sentence-transformers), and search is exact brute-force cosine over numpy (no
FAISS). For a KB of this size exact search is fast, always returns the true nearest
neighbours, and removes an approximate-index tuning burden that would otherwise be a
silent source of recall loss.

THREE MODES (`retrieval.mode`)

  dense  — bi-encoder cosine. The default, and what every archived result used.
  bm25   — lexical BM25 over the same chunks. No model, so it runs without torch.
  hybrid — both rankings fused by reciprocal rank (RRF).

Why the lexical arm exists: everything after this stage is bounded by first-stage recall,
and reranking cannot recover evidence retrieval never returned. Measured on QASPER at
top_k=8 against human-marked evidence (255 questions, `artifacts/retrieval-recall/`):

                         finds any marked evidence    marked characters retrieved
    dense  (archived)            57.6%                        31.8%
    bm25                         52.9%                        24.4%

BM25 alone is worse than dense. It is kept because the two fail on different questions:
BM25 finds marked evidence on 30 of the 108 questions where dense finds none. `hybrid` is
the arm that could use that, and it has NOT been measured — it needs the embedding model.
Run `scripts/measure_retrieval.py --modes dense bm25 hybrid` before quoting it.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np

from .config import REPO_ROOT, Config, default_config, resolve_device
from .kb import chunk_corpus, corpus_fingerprint, load_corpus
from .types import Corpus, EvidenceUnit

logger = logging.getLogger(__name__)


def _mean_pool(last_hidden_state, attention_mask):
    """Mean-pool token embeddings, ignoring padding positions."""
    import torch

    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


class Embedder:
    """Mean-pooled transformer encoder producing L2-normalised vectors.

    Vectors are unit-normalised at creation, so cosine similarity is a plain dot product
    and scores land in [-1, 1] — which the uncertainty estimator relies on.
    """

    def __init__(self, model_id: str | None = None, cfg: Config | None = None) -> None:
        cfg = cfg or default_config()
        self.model_id = model_id or cfg.require("models.embedding")
        self.device = resolve_device(cfg.get_path("models.device", "auto"))
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is None:
            import torch
            from transformers import AutoModel, AutoTokenizer

            logger.info("loading embedding model %s on %s", self.model_id, self.device)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self._model = AutoModel.from_pretrained(self.model_id).to(self.device)
            self._model.eval()
            self._torch = torch
        return self._model, self._tokenizer

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Encode texts to a (n, dim) float32 array of unit vectors."""
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)

        model, tokenizer = self._load()
        torch = self._torch
        out: list[np.ndarray] = []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                encoded = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(self.device)
                hidden = model(**encoded).last_hidden_state
                pooled = _mean_pool(hidden, encoded["attention_mask"])
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                out.append(pooled.cpu().numpy().astype(np.float32))

        return np.vstack(out)


MODES = ("dense", "bm25", "hybrid")

# Lowercased alphanumeric runs. No stemming and no stop list: IDF already discounts
# function words, and a stemmer would be a new dependency for a small expected gain.
_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25Index:
    """Okapi BM25 over a fixed list of texts, with an inverted index in numpy arrays.

        score(q, d) = sum over query terms t of
            idf(t) * tf(t, d) * (k1 + 1) / (tf(t, d) + k1 * (1 - b + b * |d| / avgdl))
        idf(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))        (never negative)

    Statistics are corpus-wide even when a query is restricted to one document. Measured
    on QASPER, computing them inside the single paper instead changed marked-evidence
    recall by under one point in either direction (24.4% vs 24.5% at top_k=8), so the
    simpler choice is kept.
    """

    def __init__(self, texts: list[str], k1: float, b: float) -> None:
        self.k1, self.b = float(k1), float(b)
        self.n = len(texts)
        lengths = np.zeros(self.n, dtype=np.float64)
        postings: dict[str, tuple[list[int], list[int]]] = {}
        for i, text in enumerate(texts):
            counts = Counter(tokenize(text))
            lengths[i] = sum(counts.values())
            for term, tf in counts.items():
                docs, tfs = postings.setdefault(term, ([], []))
                docs.append(i)
                tfs.append(tf)
        self.lengths = lengths
        self.avg_length = float(lengths.mean()) if self.n else 0.0
        self._postings = {
            term: (np.asarray(docs, dtype=np.int64), np.asarray(tfs, dtype=np.float64))
            for term, (docs, tfs) in postings.items()
        }

    def idf(self, term: str) -> float:
        entry = self._postings.get(term)
        df = 0 if entry is None else len(entry[0])
        return math.log(1.0 + (self.n - df + 0.5) / (df + 0.5))

    def scores(self, query: str) -> np.ndarray:
        """BM25 score of every indexed text for `query`. Zero where no term matches."""
        out = np.zeros(self.n, dtype=np.float64)
        if not self.n or self.avg_length == 0.0:
            return out
        for term in dict.fromkeys(tokenize(query)):
            entry = self._postings.get(term)
            if entry is None:
                continue
            docs, tf = entry
            norm = self.k1 * (1.0 - self.b + self.b * self.lengths[docs] / self.avg_length)
            out[docs] += self.idf(term) * tf * (self.k1 + 1.0) / (tf + norm)
        return out

    def max_score(self, query: str) -> float:
        """Upper bound on any text's score for `query`: every term at saturated frequency.

        Used to map BM25 onto [0, 1] in `bm25` mode. BM25 is unbounded and its scale
        depends on query length, and `retrieval_score` feeds thresholds that assume a
        bounded score.
        """
        return sum(self.idf(t) * (self.k1 + 1.0) for t in dict.fromkeys(tokenize(query)))


def _ranks(scores: np.ndarray) -> np.ndarray:
    """1-based rank of each entry, best first. Equal scores share the better rank.

    Shared ranks matter for fusion: handing tied entries distinct ranks by position would
    let reciprocal rank fusion prefer whichever chunk comes first in the document for no
    reason the scores support.
    """
    order = np.argsort(-scores, kind="stable")
    ranked = scores[order]
    # Position of the first entry of each run of equal scores, carried across the run.
    first = np.concatenate(([True], ranked[1:] != ranked[:-1]))
    shared = np.maximum.accumulate(np.where(first, np.arange(len(scores)), 0)) + 1
    ranks = np.empty(len(scores), dtype=np.int64)
    ranks[order] = shared
    return ranks


class Retriever:
    """Retriever over chunked KB evidence: dense cosine, lexical BM25, or both fused."""

    def __init__(self, corpus: Corpus | None = None, cfg: Config | None = None) -> None:
        self.cfg = cfg or default_config()
        self.corpus = corpus if corpus is not None else load_corpus(cfg=self.cfg)
        self.units: list[EvidenceUnit] = chunk_corpus(self.corpus, cfg=self.cfg)
        self.embedder = Embedder(cfg=self.cfg)
        self._matrix: np.ndarray | None = None
        self._bm25: BM25Index | None = None
        self.mode = self.cfg.get_path("retrieval.mode", "dense")
        if self.mode not in MODES:
            raise ValueError(f"unknown retrieval.mode {self.mode!r}; available: {list(MODES)}")

    def _cache_path(self) -> Path:
        cache_dir = Path(self.cfg.get_path("retrieval.cache_dir", ".cache/embeddings"))
        if not cache_dir.is_absolute():
            cache_dir = REPO_ROOT / cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        # Keyed by model AND corpus content: changing either misses rather than
        # returning a stale vector for text that no longer exists.
        # ...and by the chunk boundaries. The same corpus re-chunked into the same number
        # of units would otherwise pass the size check below and be served vectors that
        # belong to different text.
        model_key = self.embedder.model_id.replace("/", "__")
        return cache_dir / (
            f"{model_key}__{corpus_fingerprint(self.corpus)}_{self._chunk_fingerprint()}.npy"
        )

    def _chunk_fingerprint(self) -> str:
        digest = hashlib.sha256()
        for unit in self.units:
            digest.update(f"{unit.span.doc_id}:{unit.span.start}:{unit.span.end};".encode("utf-8"))
        return digest.hexdigest()[:8]

    def build(self, use_cache: bool = True) -> np.ndarray:
        """Embed every chunk, loading from disk cache when valid."""
        if self._matrix is not None:
            return self._matrix

        cache_path = self._cache_path()
        if use_cache and cache_path.exists():
            matrix = np.load(cache_path)
            if matrix.shape[0] == len(self.units):
                logger.info("loaded %d cached embeddings", matrix.shape[0])
                self._matrix = matrix
                return matrix
            logger.warning("cache size mismatch, re-embedding")

        logger.info("embedding %d chunks", len(self.units))
        matrix = self.embedder.encode([u.text for u in self.units])
        if use_cache:
            np.save(cache_path, matrix)
        self._matrix = matrix
        return matrix

    def bm25(self) -> BM25Index:
        """The lexical index, built on first use. Cheap enough not to need a disk cache."""
        if self._bm25 is None:
            self._bm25 = BM25Index(
                [u.text for u in self.units],
                k1=float(self.cfg.get_path("retrieval.bm25.k1", 1.2)),
                b=float(self.cfg.get_path("retrieval.bm25.b", 0.75)),
            )
        return self._bm25

    def _dense_scores(self, query: str, allowed: np.ndarray | None) -> np.ndarray:
        matrix = self.build()
        if allowed is not None:
            matrix = matrix[allowed]
        query_vec = self.embedder.encode([query])[0]
        # Unit vectors, so the dot product is cosine similarity.
        # errstate: numpy 2.x on Apple Accelerate leaks spurious divide/overflow/invalid
        # flags out of its vectorised matmul kernel. Verified harmless here — inputs
        # contain no NaN or inf, all rows are unit-norm, and results match a float64
        # computation to ~1e-8. Suppressed narrowly so real numerical faults elsewhere
        # still surface.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            return matrix @ query_vec

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        restrict_to: str | None = None,
    ) -> list[EvidenceUnit]:
        """Return the top-k evidence units for a query.

        `restrict_to` limits candidates to a single document. This is needed for
        single-document benchmarks such as QASPER, where questions are asked *about a
        given paper* and are not self-identifying — "which datasets did they experiment
        with?" has no answer without knowing which paper "they" refers to. Searching the
        whole corpus for such a question retrieves confident, well-scoring evidence from
        entirely the wrong paper.

        What `retrieval_score` holds depends on the mode, and the thresholds that read it
        (`uncertainty.*`, `abstain.low_retrieval_score`, `server.retrieval_floor`) were
        calibrated on cosine:

          dense  — cosine. Unchanged.
          hybrid — still cosine, so those thresholds keep their meaning. The fused order
                   goes in `fusion_score`, which `rank_score` prefers.
          bm25   — BM25 divided by the query's maximum possible score, in [0, 1]. Not the
                   cosine scale: recalibrate every threshold above before trusting them.

        Fresh EvidenceUnit copies are returned so per-query scores never mutate the
        retriever's shared chunk list.
        """
        k = top_k if top_k is not None else int(self.cfg.require("retrieval.top_k"))
        if not self.units or k <= 0:
            return []

        allowed = None
        if restrict_to is not None:
            allowed = np.array(
                [i for i, u in enumerate(self.units) if u.span.doc_id == restrict_to]
            )
            if allowed.size == 0:
                return []

        dense = lexical = fused = None
        if self.mode in ("dense", "hybrid"):
            dense = self._dense_scores(query, allowed)
        if self.mode in ("bm25", "hybrid"):
            lexical = self.bm25().scores(query)
            if allowed is not None:
                lexical = lexical[allowed]

        if self.mode == "dense":
            order_by = dense
        elif self.mode == "bm25":
            ceiling = self.bm25().max_score(query)
            lexical = lexical / ceiling if ceiling > 0 else lexical
            order_by = lexical
        else:
            # Reciprocal rank fusion: ranks, not scores, because cosine and BM25 live on
            # unrelated scales and any weighted sum would need a per-corpus weight.
            # Scaled so a unit ranked first by both scores 1.0.
            # A unit sharing no term with the query is not in the lexical ranking at all:
            # it gets nothing from that arm, as in RRF over two top-N lists. Ranking the
            # zero scores anyway would pay every non-matching chunk for its position.
            c = float(self.cfg.get_path("retrieval.rrf_k", 60))
            lexical_part = np.where(lexical > 0, 1.0 / (c + _ranks(lexical)), 0.0)
            fused = (1.0 / (c + _ranks(dense)) + lexical_part) * (c + 1.0) / 2.0
            order_by = fused

        k = min(k, len(order_by))
        # Stable sort, so ties resolve to document order and results are deterministic.
        top_idx = np.argsort(-order_by, kind="stable")[:k]

        results: list[EvidenceUnit] = []
        for idx in top_idx:
            # Map back to corpus positions when the candidate set was narrowed.
            unit = self.units[allowed[idx] if allowed is not None else idx]
            results.append(
                EvidenceUnit(
                    span=unit.span,
                    text=unit.text,
                    retrieval_score=float(lexical[idx] if self.mode == "bm25" else dense[idx]),
                    fusion_score=None if fused is None else float(fused[idx]),
                )
            )
        return results
