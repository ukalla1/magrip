"""Calibration data helpers for MaGRIP smoke tests and baselines."""

from __future__ import annotations

from collections.abc import Sequence


DEFAULT_CALIBRATION_TEXT = (
    "Magnitude and gradient informed pruning estimates which feed-forward "
    "channels can be removed while preserving language model behavior."
)


def load_text_calibration_batches(
    tokenizer: object,
    dataset_name: str = "wikitext",
    dataset_config: str | None = "wikitext-2-raw-v1",
    dataset_split: str = "validation",
    text_column: str = "text",
    num_samples: int = 8,
    max_length: int = 128,
    batch_size: int = 1,
    device: object | None = None,
) -> list[object]:
    """Load a small text dataset and return fixed-length token batches.

    The default dataset is WikiText-2 validation. It is intentionally small and stable
    enough for smoke tests while still providing more meaningful saliency estimates than
    a single handcrafted sentence.
    """

    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")
    if max_length <= 1:
        raise ValueError("max_length must be greater than 1.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    from datasets import load_dataset

    token_ids: list[int] = []
    dataset = load_dataset(dataset_name, dataset_config, split=dataset_split)
    for example in dataset:
        text = str(example.get(text_column, "")).strip()
        if not text:
            continue
        token_ids.extend(tokenizer(text, add_special_tokens=False)["input_ids"])
        if len(token_ids) >= num_samples * max_length:
            break

    if not token_ids:
        raise RuntimeError("No calibration tokens were available.")

    return _tokens_to_batches(
        token_ids=token_ids,
        tokenizer=tokenizer,
        num_samples=num_samples,
        max_length=max_length,
        batch_size=batch_size,
        device=device,
    )


def load_inline_text_batches(
    tokenizer: object,
    text: str = DEFAULT_CALIBRATION_TEXT,
    num_samples: int = 1,
    max_length: int = 128,
    batch_size: int = 1,
    device: object | None = None,
) -> list[object]:
    """Return calibration batches from inline text."""

    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if not token_ids:
        raise RuntimeError("No calibration tokens were available from inline text.")
    return _tokens_to_batches(
        token_ids=token_ids,
        tokenizer=tokenizer,
        num_samples=num_samples,
        max_length=max_length,
        batch_size=batch_size,
        device=device,
    )


def batches_token_count(batches: Sequence[object]) -> int:
    """Return total token count for tensor-like batches."""

    return int(sum(batch.numel() for batch in batches))


def _tokens_to_batches(
    token_ids: Sequence[int],
    tokenizer: object,
    num_samples: int,
    max_length: int,
    batch_size: int,
    device: object | None,
) -> list[object]:
    import torch

    pad_token_id = _pad_token_id(tokenizer)
    windows: list[list[int]] = []
    for sample_index in range(num_samples):
        start = sample_index * max_length
        window = token_ids[start : start + max_length]
        if not window:
            break
        if len(window) < max_length:
            window = window + [pad_token_id] * (max_length - len(window))
        windows.append(window)

    if not windows:
        raise RuntimeError("No calibration windows were produced.")

    tensors = [torch.tensor(window, dtype=torch.long, device=device) for window in windows]
    batches = []
    for start in range(0, len(tensors), batch_size):
        batches.append(torch.stack(tensors[start : start + batch_size], dim=0))
    return batches


def _pad_token_id(tokenizer: object) -> int:
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is not None:
        return int(pad_token_id)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        return int(eos_token_id)
    return 0
