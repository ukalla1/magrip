"""Small helpers for working with nested PyTorch modules."""

from __future__ import annotations

from collections.abc import Iterable


def get_module_by_path(root: object, path: str) -> object:
    """Resolve a dotted module path such as ``transformer.h.0.mlp.c_fc``."""

    current = root
    if not path:
        return current
    for part in path.split("."):
        if part.isdigit() and _supports_indexing(current):
            current = current[int(part)]
        else:
            current = getattr(current, part)
    return current


def has_path(root: object, path: str) -> bool:
    """Return whether ``path`` resolves from ``root``."""

    try:
        get_module_by_path(root, path)
    except (AttributeError, IndexError, TypeError):
        return False
    return True


def join_path(parts: Iterable[str | int]) -> str:
    """Join path parts into the dotted format used by ``named_modules``."""

    return ".".join(str(part) for part in parts)


def _supports_indexing(value: object) -> bool:
    return hasattr(value, "__getitem__") and not isinstance(value, (str, bytes, bytearray))
