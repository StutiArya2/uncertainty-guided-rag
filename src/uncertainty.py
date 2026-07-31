"""Pipeline stage 4a — uncertainty estimation.

The `UncertaintyEstimator` interface exists even though only one implementation ships
today. Swapping estimators is exactly the ablation a reviewer will ask for ("is the gain
from *uncertainty* guidance, or would any budget schedule do?"), and retrofitting an
interface after the fact is far more disruptive than defining it now. Token-logprob and
self-consistency estimators are the planned additions (see README, Fall extension).

The shipped estimator derives uncertainty from retrieval scores alone: no extra model
calls, fully deterministic, and cheap enough to run on every claim.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .config import Config, default_config
from .types import EvidenceUnit


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


class UncertaintyEstimator(ABC):
    """Maps a claim's candidate evidence to a scalar uncertainty in [0, 1].

    0 means "retrieval is confident" (compress hard), 1 means "retrieval is unsure"
    (keep more).
    """

    name: str = "base"

    @abstractmethod
    def estimate(self, units: list[EvidenceUnit]) -> float:
        ...


class RetrievalScoreEstimator(UncertaintyEstimator):
    """Uncertainty from the top score and the margin to the runner-up.

        u = w_score * (1 - s_top) + w_margin * (1 - margin_norm)

    Two independent failure signals are combined:

    * **Low top score** — nothing in the KB matches the claim well, so the evidence is
      weak in absolute terms.
    * **Narrow margin** — the top hit is barely ahead of the next one, so the ranking is
      unstable and the "best" unit may not really be best. Dropping the rest is risky
      precisely here.

    A claim can be confident on one and shaky on the other, which is why both terms are
    kept rather than collapsed into a single score.

    Margins are small in absolute terms (cosine gaps of ~0.1 are decisive in practice),
    so the raw margin is divided by `margin_scale` before use rather than compared
    against the [0, 1] range it never reaches.
    """

    name = "retrieval_score"

    def __init__(self, cfg: Config | None = None) -> None:
        cfg = cfg or default_config()
        self.w_score = float(cfg.require("uncertainty.w_score"))
        self.w_margin = float(cfg.require("uncertainty.w_margin"))
        self.margin_scale = float(cfg.get_path("uncertainty.margin_scale", 0.15))

    def estimate(self, units: list[EvidenceUnit]) -> float:
        if not units:
            # No evidence at all is maximum uncertainty.
            return 1.0

        scores = sorted((u.retrieval_score for u in units), reverse=True)
        top = _clamp(scores[0])
        score_term = 1.0 - top

        if len(scores) < 2:
            # A single candidate offers no comparison, so the ranking is unverified.
            margin_term = 1.0
        else:
            margin = max(0.0, scores[0] - scores[1])
            margin_term = 1.0 - _clamp(margin / self.margin_scale)

        return _clamp(self.w_score * score_term + self.w_margin * margin_term)


_REGISTRY: dict[str, type[UncertaintyEstimator]] = {
    RetrievalScoreEstimator.name: RetrievalScoreEstimator,
}


def estimator_from_config(cfg: Config | None = None) -> UncertaintyEstimator:
    """Build the estimator named by `uncertainty.estimator`."""
    cfg = cfg or default_config()
    name = cfg.get_path("uncertainty.estimator", RetrievalScoreEstimator.name)
    if name not in _REGISTRY:
        raise ValueError(
            f"unknown uncertainty estimator {name!r}; available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name](cfg=cfg)
