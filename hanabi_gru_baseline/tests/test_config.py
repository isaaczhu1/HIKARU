from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:  # pragma: no branch
    sys.path.insert(0, str(ROOT))

from hanabi_gru_baseline.config import CFG


def test_default_seq_len_is_one():
    assert CFG.ppo.seq_len == 1
