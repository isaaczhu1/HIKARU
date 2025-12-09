import os
import sys

from hanabi_learning_environment import pyhanabi

# Ensure repo root is on sys.path so `sparta_wrapper` is importable when running directly.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from sparta_wrapper.belief_models import sample_world_state, _iter_all_hands, _compute_remaining_deck, _predict_partner
from sparta_wrapper.hanabi_utils import build_observation
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

def squid_game():
    game = pyhanabi.HanabiGame(HANABI_GAME_CONFIG)
    state = game.new_initial_state()

    while state.cur_player() == pyhanabi.CHANCE_PLAYER_ID:
        print("peepee chance event")
        _advance_chance_events(state)
        continue

    obs = build_observation(state, 0)

    print(obs.legal_moves)

    # transition with first legal move
    move = obs.legal_moves[-1]
    print("applying move:", move)

    state.apply_move(move)

    while state.cur_player() == pyhanabi.CHANCE_PLAYER_ID:
        print("peepee chance event")
        _advance_chance_events(state)
        continue

    print("SUP", state.cur_player())

    obs1 = build_observation(state, 1)

    player_1_obs_deck = _compute_remaining_deck(obs1)
    for hand in _iter_all_hands(player_1_obs_deck, obs1.raw_observation.card_knowledge()[0]):
        print("possible hand for player 1:", hand)
        _predict_partner(signal_blueprint_factory, state, 1, 0, hand)

if __name__ == "__main__":
    squid_game()
