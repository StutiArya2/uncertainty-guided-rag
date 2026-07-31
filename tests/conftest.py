"""Shared fixtures.

Most tests run against a tiny hand-built corpus rather than the real KB: they stay fast,
stay offline, and their expected outputs are obvious by inspection. Tests that genuinely
need model weights are marked `integration` and skipped unless the models are present.
"""

from __future__ import annotations

import pytest

from src.config import load_config
from src.types import Document

DOC_A = (
    "Water boils at one hundred degrees Celsius at sea level. "
    "The boiling point falls as altitude increases. "
    "At the summit of Everest water boils near seventy degrees Celsius. "
    "This happens because atmospheric pressure decreases with height."
)

DOC_B = (
    "The Pacific is the largest ocean on Earth. "
    "It covers about one hundred and sixty five million square kilometres. "
    "Its deepest point is the Challenger Deep in the Mariana Trench. "
    "That trench reaches nearly eleven thousand metres below sea level."
)


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def corpus():
    return {
        "doc_a": Document(doc_id="doc_a", text=DOC_A),
        "doc_b": Document(doc_id="doc_b", text=DOC_B),
    }


@pytest.fixture
def units(corpus, cfg):
    from src.kb import chunk_corpus

    return chunk_corpus(corpus, cfg=cfg)


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: requires downloaded model weights"
    )
