"""Optimizer backend selection for MaGRIP."""

from __future__ import annotations


def apollo_available() -> bool:
    """Return whether APOLLO integration is available.

    APOLLO support is planned for M6 and is not wired yet.
    """

    return False

