"""Config loading. All pipeline thresholds come from here, never from literals in code.

Per CLAUDE.md: "Prefer explicit config (yaml/json) over hardcoded thresholds for
'enough evidence' checks." Anything tunable that would appear in an ablation table
belongs in config/default.yaml.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "default.yaml"


class Config(dict):
    """Dict with dotted-path access, so callers read `cfg.get_path("compression.min_keep")`."""

    def get_path(self, path: str, default: Any = None) -> Any:
        node: Any = self
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def require(self, path: str) -> Any:
        sentinel = object()
        value = self.get_path(path, sentinel)
        if value is sentinel:
            raise KeyError(f"missing required config key: {path}")
        return value


def _validate(cfg: Config) -> None:
    """Fail loudly at load time rather than producing silently wrong scores later."""
    w_score = float(cfg.require("uncertainty.w_score"))
    w_margin = float(cfg.require("uncertainty.w_margin"))
    if abs((w_score + w_margin) - 1.0) > 1e-6:
        raise ValueError(
            f"uncertainty.w_score + uncertainty.w_margin must sum to 1.0, "
            f"got {w_score + w_margin}"
        )

    min_keep = float(cfg.require("compression.min_keep"))
    max_keep = float(cfg.require("compression.max_keep"))
    if not 0.0 < min_keep <= max_keep <= 1.0:
        raise ValueError(
            f"require 0 < compression.min_keep <= compression.max_keep <= 1, "
            f"got min_keep={min_keep} max_keep={max_keep}"
        )

    scorer = cfg.require("evaluation.scorer")
    settings = cfg.get_path(f"evaluation.scorers.{scorer}")
    if not isinstance(settings, dict):
        raise KeyError(
            f"evaluation.scorer is {scorer!r} but evaluation.scorers.{scorer} is missing"
        )
    for key in ("model", "threshold"):
        if key not in settings:
            raise KeyError(f"evaluation.scorers.{scorer} is missing {key!r}")

    threshold = float(settings["threshold"])
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            f"evaluation.scorers.{scorer}.threshold must be in [0, 1], got {threshold}"
        )


def load_config(path: str | Path | None = None) -> Config:
    """Load and validate pipeline config."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(config_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    cfg = Config(raw)
    _validate(cfg)
    return cfg


@functools.lru_cache(maxsize=1)
def default_config() -> Config:
    """Cached default config, for callers that don't need a custom path."""
    return load_config()


def resolve_device(spec: str = "auto") -> str:
    """Pick a torch device. 'auto' prefers mps on arm64 Macs, then cuda, then cpu."""
    if spec != "auto":
        return spec
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard dependency at runtime
        return "cpu"
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
