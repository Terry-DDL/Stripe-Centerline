"""Windows entry point for one warm-up plus ten measured benchmark runs."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.windows_benchmark_runner import main


if __name__ == "__main__":
    raise SystemExit(main(["--run-count", "11", *sys.argv[1:]]))
