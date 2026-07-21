"""Tokenizer asset preservation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def save_tokenizer_assets(
    *,
    tokenizer: object,
    output_dir: Path,
    tokenizer_source: str,
    trust_remote_code: bool = False,
    require_sentencepiece_assets: bool = False,
    token: str | None = None,
) -> dict[str, Any]:
    """Save tokenizer files and preserve SentencePiece assets when needed."""

    saved_files = [str(path) for path in tokenizer.save_pretrained(output_dir)]
    info: dict[str, Any] = {
        "source": tokenizer_source,
        "saved_files": saved_files,
        "sentencepiece_required": False,
        "sentencepiece_model": None,
        "fallbacks": [],
    }
    sentencepiece_path = output_dir / "tokenizer.model"
    if sentencepiece_path.exists():
        info["sentencepiece_model"] = str(sentencepiece_path)
        return info
    if not require_sentencepiece_assets and not expects_sentencepiece_model(
        tokenizer,
        tokenizer_source,
    ):
        return info

    info["sentencepiece_required"] = True
    slow_info = _try_save_slow_tokenizer(
        output_dir=output_dir,
        tokenizer_source=tokenizer_source,
        trust_remote_code=trust_remote_code,
        token=token,
    )
    info["fallbacks"].append(slow_info)
    if sentencepiece_path.exists():
        info["sentencepiece_model"] = str(sentencepiece_path)
        return info

    copy_info = _try_copy_sentencepiece_model(
        tokenizer_source=tokenizer_source,
        output_dir=output_dir,
        token=token,
    )
    info["fallbacks"].append(copy_info)
    if sentencepiece_path.exists():
        info["sentencepiece_model"] = str(sentencepiece_path)
        return info

    raise FileNotFoundError(
        "SentencePiece tokenizer.model was required but could not be recovered from "
        f"{tokenizer_source!r}. Install sentencepiece, authenticate with Hugging Face if "
        "the model is gated, or provide a local tokenizer source containing tokenizer.model."
    )


def expects_sentencepiece_model(tokenizer: object, tokenizer_source: str) -> bool:
    tokenizer_class = type(tokenizer).__name__.lower()
    source = str(tokenizer_source).lower()
    markers = (
        "gemma",
        "llama",
        "mistral",
        "mixtral",
        "sentencepiece",
        "spm",
        "t5",
    )
    return any(marker in tokenizer_class or marker in source for marker in markers)


def _try_save_slow_tokenizer(
    *,
    output_dir: Path,
    tokenizer_source: str,
    trust_remote_code: bool,
    token: str | None,
) -> dict[str, Any]:
    try:
        from transformers import AutoTokenizer

        kwargs: dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
            "use_fast": False,
        }
        if token:
            kwargs["token"] = token
        slow_tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, **kwargs)
        saved_files = [str(path) for path in slow_tokenizer.save_pretrained(output_dir)]
        return {
            "method": "slow_tokenizer_save",
            "ok": True,
            "saved_files": saved_files,
        }
    except Exception as exc:
        return {
            "method": "slow_tokenizer_save",
            "ok": False,
            "error": repr(exc),
        }


def _try_copy_sentencepiece_model(
    *,
    tokenizer_source: str,
    output_dir: Path,
    token: str | None,
) -> dict[str, Any]:
    destination = output_dir / "tokenizer.model"
    local_source = Path(tokenizer_source)
    if local_source.exists():
        candidate = local_source / "tokenizer.model"
        if candidate.exists():
            return _copy_tokenizer_model(candidate, destination, method="local_tokenizer_source")

    try:
        from huggingface_hub import hf_hub_download

        kwargs: dict[str, Any] = {
            "repo_id": tokenizer_source,
            "filename": "tokenizer.model",
        }
        if token:
            kwargs["token"] = token
        downloaded = Path(hf_hub_download(**kwargs))
        return _copy_tokenizer_model(downloaded, destination, method="hf_hub_download")
    except Exception as exc:
        return {
            "method": "hf_hub_download",
            "ok": False,
            "error": repr(exc),
        }


def _copy_tokenizer_model(source: Path, destination: Path, *, method: str) -> dict[str, Any]:
    import shutil

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return {
        "method": method,
        "ok": True,
        "source": str(source),
        "destination": str(destination),
    }
