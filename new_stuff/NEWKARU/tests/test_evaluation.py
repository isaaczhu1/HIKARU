from __future__ import annotations

# add ./new_stuff/NEWKARU to sys.path for imports
import os
import sys
path_string = "/Users/isaaczhu/MIT/25-26/HIKARU/new_stuff/NEWKARU"
sys.path.append(os.path.abspath(path_string))

from blueprints.heuristic_blueprint import HeuristicBlueprint
from search.evaluation import run_self_play


def test_run_self_play_returns_scores() -> None:
    episodes = 3
    result = run_self_play(lambda: HeuristicBlueprint(), num_episodes=episodes, seed=123)
    assert len(result.scores) == episodes
    assert len(result.blueprint_stats) == 2
    for score in result.scores:
        assert 0 <= score <= 25
    assert result.mean_score >= 0
    assert result.stddev_score >= 0
