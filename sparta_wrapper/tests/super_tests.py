from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hanabi_learning_environment import pyhanabi

from sparta_wrapper.hanabi_utils import advance_state, fabricate
from sparta_wrapper.sparta_config import HANABI_GAME_CONFIG


def human_readable_card(card):
    return "RYGBW"[card[0]] + str(card[1]+1)

def _initial_state(seed: int) -> pyhanabi.HanabiState:
    cfg = dict(HANABI_GAME_CONFIG)
    cfg["seed"] = seed
    game = pyhanabi.HanabiGame(cfg)
    state = game.new_initial_state()
    # Deal until a player must act so we have hands to inspect.
    state.deal_specific_card(0, 2, 4)
    state.deal_specific_card(0, 4, 0)
    state.deal_specific_card(0, 4, 1)
    state.deal_specific_card(0, 4, 0)
    state.deal_specific_card(0, 2, 3)

    state.deal_specific_card(1, 1, 3)
    state.deal_specific_card(1, 1, 1)
    state.deal_specific_card(1, 0, 0)
    state.deal_specific_card(1, 1, 0)
    state.deal_specific_card(1, 2, 1)
    return state


def test_fabrication_roundtrip_simple():
    """Fabrication should let us replay a state with a swapped hand."""
    seed = 1234
    state = _initial_state(seed)
    target = 0

    original_hands = [
        [(c.color(), c.rank()) for c in hand] for hand in state.player_hands()
    ]
    for hand in original_hands:
        print(list(map(human_readable_card,hand)))

    print(state.legal_moves())
    advance_state(state, [state.legal_moves()[-1]])

    print(state.legal_moves())
    advance_state(state, [state.legal_moves()[10]])

    print(state.legal_moves())
    advance_state(state, [state.legal_moves()[2]])
    advance_state(state, [(0, 3, 4)])

    '''
    print(state.legal_moves())
    advance_state(state, [state.legal_moves()[0]])
    advance_state(state, [(1, 2, 0)])
    '''

    print(state)

    fabricated_hand = [(2, 3), (1, 1), (4, 4), (2, 4), (3, 4)]
    print(list(map(human_readable_card,fabricated_hand)))

    fabricated_history = fabricate(state, target, fabricated_hand)
    print(list(map(human_readable_card,fabricated_history)))

    # Replay fabricated history into a fresh state.
    replay_game = pyhanabi.HanabiGame(dict(HANABI_GAME_CONFIG, seed=seed))
    replay_state = replay_game.new_initial_state()
    advance_state(replay_state, fabricated_history)

    replay_hands = [
        [(c.color(), c.rank()) for c in hand] for hand in replay_state.player_hands()
    ]

    assert replay_hands[target] == fabricated_hand
    assert replay_hands[1 - target] == original_hands[1 - target]

if __name__ == "__main__":
    # Allow running as a standalone script.
    # import pytest
    test_fabrication_roundtrip_simple()

    # sys.exit(pytest.main([__file__]))
