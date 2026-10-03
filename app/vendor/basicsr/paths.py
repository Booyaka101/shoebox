"""Locate the shoebox project root from a vendored module.

`vendor_dir` is app/vendor (this file sits at app/vendor/basicsr/paths.py);
the project root is two levels above it. Model weights are cached in
<project root>/weights/ so they survive re-installs and are visible next to
run.bat.
"""

from pathlib import Path

VENDOR_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = VENDOR_DIR.parents[1]
WEIGHTS_ROOT = PROJECT_ROOT / 'weights'


def weights_root() -> Path:
    WEIGHTS_ROOT.mkdir(parents=True, exist_ok=True)
    return WEIGHTS_ROOT
