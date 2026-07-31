"""Per-query instrumentation.

Two reasons this exists rather than scattered print statements:

1. CLAUDE.md requires uncertainty scores be "logged/returned alongside compression
   decisions for reproducibility".
2. The mid-August deliverable is a *measurement* — token reduction vs. a no-compression
   baseline. That table is built from these traces, so the numbers are collected from the
   first working run rather than bolted on at the end.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class EvidenceRecord:
    """One evidence unit as the pipeline saw it, kept or set aside.

    Carries the source span so a reader can be pointed at the exact characters the
    answer came from — the same provenance that makes restoration exact.
    """

    doc_id: str
    text: str
    score: float
    start: int
    end: int


@dataclass
class ClaimTrace:
    """What happened to one claim as it moved through the pipeline."""

    claim: str
    n_candidates: int = 0
    uncertainty: float = 0.0
    keep_ratio: float = 0.0
    n_kept: int = 0
    n_dropped: int = 0
    baseline_tokens: int = 0
    compressed_tokens: int = 0
    support_score: float = 0.0
    is_sufficient: bool = False
    # Populated only when the restoration branch ran.
    restored_support_score: float | None = None
    restored_sufficient: bool | None = None
    # The evidence itself, so the record shows *which* sentences were used rather than
    # only how many. Consumed by the web UI to mark up the source passage.
    kept: list[EvidenceRecord] = field(default_factory=list)
    dropped: list[EvidenceRecord] = field(default_factory=list)

    @property
    def tokens_saved(self) -> int:
        return self.baseline_tokens - self.compressed_tokens

    @property
    def reduction(self) -> float:
        if self.baseline_tokens == 0:
            return 0.0
        return self.tokens_saved / self.baseline_tokens


@dataclass
class PipelineTrace:
    """Full record of a single query run."""

    query: str
    claims: list[ClaimTrace] = field(default_factory=list)
    branch: str = ""
    abstained: bool = False
    restoration_triggered: bool = False
    # False when token counts are character-ratio estimates rather than real tokenization.
    token_counts_exact: bool = True
    answer: str = ""

    @property
    def baseline_tokens(self) -> int:
        return sum(c.baseline_tokens for c in self.claims)

    @property
    def final_tokens(self) -> int:
        """Tokens actually sent to the generator.

        Restored claims cost their full baseline again, so a run that restores saves
        less than the compression stage alone suggests. Counting it here keeps the
        headline number honest.
        """
        total = 0
        for c in self.claims:
            restored = c.restored_support_score is not None
            total += c.baseline_tokens if restored else c.compressed_tokens
        return total

    @property
    def reduction(self) -> float:
        if self.baseline_tokens == 0:
            return 0.0
        return (self.baseline_tokens - self.final_tokens) / self.baseline_tokens

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["baseline_tokens"] = self.baseline_tokens
        data["final_tokens"] = self.final_tokens
        data["reduction"] = round(self.reduction, 4)
        for claim_data, claim in zip(data["claims"], self.claims):
            claim_data["tokens_saved"] = claim.tokens_saved
            claim_data["reduction"] = round(claim.reduction, 4)
        return data

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def summary(self) -> str:
        """Human-readable one-screen summary for CLI runs."""
        exact = "exact" if self.token_counts_exact else "ESTIMATED"
        lines = [
            f"query:  {self.query}",
            f"branch: {self.branch}" + ("  (ABSTAINED)" if self.abstained else ""),
            # Spelled out rather than signed: a bare "+29.3%" on a line about token
            # counts reads as an increase, which is the opposite of what it means.
            f"tokens: {self.baseline_tokens} -> {self.final_tokens} "
            f"({self.reduction:.1%} reduction, counts {exact})",
            "",
            f"{'claim':<44} {'unc':>5} {'keep':>5} {'tok':>13} {'supp':>6}  ok",
            "-" * 84,
        ]
        for c in self.claims:
            claim = (c.claim[:41] + "...") if len(c.claim) > 44 else c.claim
            support = c.support_score
            ok = "yes" if c.is_sufficient else "no"
            if c.restored_support_score is not None:
                support = c.restored_support_score
                ok = ("yes" if c.restored_sufficient else "no") + " (restored)"
            lines.append(
                f"{claim:<44} {c.uncertainty:>5.2f} {c.keep_ratio:>5.2f} "
                f"{c.baseline_tokens:>5}->{c.compressed_tokens:<6} {support:>6.2f}  {ok}"
            )
        return "\n".join(lines)
