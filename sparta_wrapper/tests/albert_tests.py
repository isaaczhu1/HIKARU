import os
import sys

from hanabi_learning_environment import pyhanabi

# Ensure repo root is on sys.path so `sparta_wrapper` is importable when running directly.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import random

rng = random.Random(42)

from sparta_wrapper.belief_models import sample_world_state, _iter_all_hands, _compute_remaining_deck, _predict_partner, _hand_multiplicity, _sample_hand, _iter_all_hands
from sparta_wrapper.hanabi_utils import build_observation, HanabiLookback1
from sparta_wrapper.hanabi_utils import HanabiObservation, _advance_chance_events, build_observation

from sparta_wrapper.sparta_config import HANABI_GAME_CONFIG

class SignalBlueprint:
    """Signal by looking at the next player's first card."""

    def act(self, observation: HanabiObservation) -> pyhanabi.HanabiMove:
        partner = (observation.current_player_offset + 1) % len(observation.observed_hands)
        partner_hand = observation.observed_hands[partner]
        if not partner_hand:
            return observation.legal_moves[0]
        partner_rank = partner_hand[0]["rank"]

        if partner_rank == 0:
            return self._pick_move(observation, pyhanabi.HanabiMoveType.PLAY)
        if partner_rank > 0:
            return self._pick_move(observation, pyhanabi.HanabiMoveType.DISCARD)
        return observation.legal_moves[0]

    def _pick_move(self, observation: HanabiObservation, move_type: pyhanabi.HanabiMoveType) -> pyhanabi.HanabiMove:
        for move in observation.legal_moves:
            if move.type() == move_type and move.card_index() == 0:
                return move
        return observation.legal_moves[0]


def signal_blueprint_factory() -> SignalBlueprint:
    return SignalBlueprint()


def test_rem_deck():

    from sparta_wrapper.belief_models import _compute_remaining_deck
    game = pyhanabi.HanabiGame(HANABI_GAME_CONFIG)
    state = game.new_initial_state()

    while state.cur_player() == pyhanabi.CHANCE_PLAYER_ID:
        _advance_chance_events(state)
        continue

    obs = build_observation(state, 0)

    print(obs.__dict__)

    print()
    print()
    print()

    print(_compute_remaining_deck(obs))




def squid_game(seed):

    lookback1 = HanabiLookback1(HANABI_GAME_CONFIG, seed)

    obs = build_observation(lookback1.cur_state, 0)

    print(obs)
    move = obs.legal_moves[3] # idiotic red clue
    print("!!!!!!!!!", move)

    # I LIKE TO MOVE IT MOVE IT
    lookback1.apply_move(move)

    obs = build_observation(lookback1.cur_state, 1)
    move = obs.legal_moves[-1] # urinate on player one
    print("!!!!!!!!!", move)
    
    # I LIKE TO MOVE IT MOVE IT
    lookback1.apply_move(move)

    obs = build_observation(lookback1.cur_state, 0)
    print(obs)
    print("====")
    print(build_observation(lookback1.cur_state, 1))

    print(obs.legal_moves)

    print("====")
    act = signal_blueprint_factory().act(obs)

    print("Player 0 acts:", act)

    # I LIKE TO MOVE IT MOVE IT
    lookback1.apply_move(act)

    print("surviving")

    print("SUP", lookback1.cur_state.cur_player())

    obs1 = build_observation(lookback1.cur_state, 1)
    print(obs1)

    print("SHIT 1")
    print(lookback1.cur_state)
    print("SHIT 2")
    print(lookback1.prev_state)
    print("SHIT 3")
    print(obs1)
    print("END SHIT")

    player_1_obs_deck = _compute_remaining_deck(obs1)
    print(player_1_obs_deck)
    print(obs1.raw_observation.card_knowledge()[1])
    print("SUPER SHIT $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    for hand in _iter_all_hands(player_1_obs_deck, obs1.raw_observation.card_knowledge()[0], HANABI_GAME_CONFIG["hand_size"]):
        print(hand)
    print("STOP ppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppp")
    belief = (tuple(x) for x in _sample_hand(player_1_obs_deck, obs1.raw_observation.card_knowledge()[0], HANABI_GAME_CONFIG["hand_size"], rng=rng, takes=50))
    frequencies = dict()
    for b in belief:
        if b not in frequencies:
            frequencies[b] = 0
        frequencies[b] = frequencies[b] + 1
    for f, k in frequencies.items():
        print(f, k)
    print("sampled stuff")
    belief = (tuple(x) for x in sample_world_state(
        lagging_state=lookback1.prev_state,
        obs=obs1,
        rng=rng,
        blueprint_factory=signal_blueprint_factory,
        takes=50,
        upstream_factor=5,
        max_attempts=3,
    ))

    frequencies = dict()
    for b in belief:
        if b not in frequencies:
            frequencies[b] = 0
        frequencies[b] = frequencies[b] + 1
    for f, k in frequencies.items():
        print(f, k)


if __name__ == "__main__":
    squid_game(67)
