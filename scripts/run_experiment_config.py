"""Run a MaGRIP experiment recipe from configs/experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Path to an experiment JSON file.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command without executing it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text())
    command = _command_from_config(config)
    print(" ".join(command))
    if not args.dry_run:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def _command_from_config(config: dict[str, Any]) -> list[str]:
    script = config.get("script")
    if not script:
        raise ValueError("Experiment config is missing `script`.")
    command = [sys.executable, str(PROJECT_ROOT / script)]
    for key, value in (config.get("args") or {}).items():
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                command.append(flag)
            continue
        if isinstance(value, list):
            for item in value:
                command.extend([flag, str(item)])
            continue
        command.extend([flag, str(value)])
    return command


if __name__ == "__main__":
    main()
