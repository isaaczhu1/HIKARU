"""Ensure NEWKARU package modules can be imported no matter where pytest runs."""

from __future__ import annotations

import sys
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parent
if str(PKG_ROOT) not in sys.path:  # pragma: no branch - keeps relative imports working
    sys.path.insert(0, str(PKG_ROOT))
