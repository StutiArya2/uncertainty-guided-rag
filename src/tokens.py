"""Token counting.

The entire premise of this project is "high-token evidence -> reduced-token evidence", so
token counts must be *measured*, never assumed. Per CLAUDE.md, anywhere a token-count
assumption is not explicit is a bug.

Counts come from the real generation tokenizer whenever it is available. If the model
cannot be loaded (no network on first run, for example) we fall back to a character-ratio
estimate — but `is_exact` goes False and that flag is carried into the pipeline trace, so
a reported compression ratio is never silently based on a guess.
"""

from __future__ import annotations

import logging

from .config import Config, default_config

logger = logging.getLogger(__name__)

# Rough English average when no tokenizer is available. Only used in fallback mode.
_CHARS_PER_TOKEN = 4


class TokenCounter:
    """Counts tokens with the generation tokenizer, falling back to an estimate."""

    def __init__(self, model_id: str | None = None, cfg: Config | None = None) -> None:
        cfg = cfg or default_config()
        self.model_id = model_id or cfg.require("models.generation")
        self._tokenizer = None
        self._load_attempted = False

    def _tokenizer_or_none(self):
        if not self._load_attempted:
            self._load_attempted = True
            try:
                from transformers import AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            except Exception as exc:  # noqa: BLE001 - any load failure means fallback
                logger.warning(
                    "could not load tokenizer %s (%s); token counts will be ESTIMATED "
                    "at ~%d chars/token, not exact",
                    self.model_id,
                    exc,
                    _CHARS_PER_TOKEN,
                )
        return self._tokenizer

    @property
    def is_exact(self) -> bool:
        """False if counts are character-ratio estimates rather than real tokenization."""
        return self._tokenizer_or_none() is not None

    def count(self, text: str) -> int:
        if not text:
            return 0
        tokenizer = self._tokenizer_or_none()
        if tokenizer is None:
            return max(1, len(text) // _CHARS_PER_TOKEN)
        return len(tokenizer.encode(text, add_special_tokens=False))

    def count_many(self, texts: list[str]) -> list[int]:
        return [self.count(t) for t in texts]


_shared: TokenCounter | None = None


def shared_counter(cfg: Config | None = None) -> TokenCounter:
    """Process-wide counter, so the tokenizer is loaded at most once."""
    global _shared
    if _shared is None:
        _shared = TokenCounter(cfg=cfg)
    return _shared
