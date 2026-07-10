"""Run the generic MaGRIP smoke test.

This is the preferred entrypoint. ``run_gpt2_smoke.py`` is kept as a backwards-compatible
name from the first dense-FFN smoke test.
"""

from __future__ import annotations

from run_gpt2_smoke import main


if __name__ == "__main__":
    main()
