"""Logging helpers for MaGRIP."""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path
from typing import Any


def format_ratio(value: float) -> str:
    """Format a ratio as a percentage string."""

    return f"{value * 100:.2f}%"


class RunLogger:
    """Write structured run events and summaries to disk."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.run_dir / "events.jsonl"
        self.summary_path = self.run_dir / "summary.json"

    def log(self, event: str, **payload: Any) -> None:
        """Append one event to ``events.jsonl``."""

        record = {
            "time": time.time(),
            "time_readable": time.strftime("%Y-%m-%d %H:%M:%S"),
            "event": event,
            **to_jsonable(payload),
        }
        with self.events_path.open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def write_summary(self, summary: dict[str, Any]) -> None:
        """Write final run summary JSON."""

        text = json.dumps(to_jsonable(summary), indent=2, sort_keys=True) + "\n"
        self.summary_path.write_text(text)


def system_info() -> dict[str, Any]:
    """Return useful environment metadata for reproducibility."""

    info: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_device_count"] = torch.cuda.device_count()
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
    except Exception as exc:
        info["torch_error"] = repr(exc)

    try:
        import transformers

        info["transformers"] = transformers.__version__
    except Exception as exc:
        info["transformers_error"] = repr(exc)

    return info


def cuda_memory_snapshot() -> dict[str, Any]:
    """Return CUDA memory counters when available."""

    try:
        import torch

        if not torch.cuda.is_available():
            return {"cuda_available": False}
        return {
            "cuda_available": True,
            "allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(),
            "max_allocated_bytes": torch.cuda.max_memory_allocated(),
            "max_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
    except Exception as exc:
        return {"cuda_error": repr(exc)}


def tensor_stats(value: object) -> dict[str, Any]:
    """Summarize a tensor-like object for logging."""

    tensor = value.detach().float().cpu()
    return {
        "shape": list(tensor.shape),
        "numel": int(tensor.numel()),
        "min": float(tensor.min().item()) if tensor.numel() else None,
        "max": float(tensor.max().item()) if tensor.numel() else None,
        "mean": float(tensor.mean().item()) if tensor.numel() else None,
        "median": float(tensor.median().item()) if tensor.numel() else None,
        "std": float(tensor.std(unbiased=False).item()) if tensor.numel() else None,
        "sum": float(tensor.sum().item()) if tensor.numel() else None,
    }


def to_jsonable(value: Any) -> Any:
    """Convert common Python/PyTorch values into JSON-serializable values."""

    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "detach"):
        return tensor_stats(value)
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value
