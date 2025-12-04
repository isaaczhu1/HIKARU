"""Global pytest configuration for path setup."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for extra in (ROOT, ROOT / "new_stuff" / "NEWKARU"):
    path_str = str(extra)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
