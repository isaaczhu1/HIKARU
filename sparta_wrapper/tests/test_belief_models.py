from __future__ import annotations

from dataclasses import dataclass
import random
from typing import List, Tuple

from hanabi_learning_environment import pyhanabi

from sparta_wrapper.belief_models import apply_sampled_hands, sample_world_state
from sparta_wrapper.hanabi_utils import HanabiObservation, _advance_chance_events, build_observation


class SignalBlueprint:
    """Signal by looking at the next player's first card."""

    def act(self, observation: HanabiObservation, rng: random.Random) -> pyhanabi.HanabiMove:
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


@dataclass
class TrajectoryStep:
    player: int
    observation: HanabiObservation
    move: pyhanabi.HanabiMove
    state: pyhanabi.HanabiState
    next_state: pyhanabi.HanabiState


def rollout_signal_trajectory(turns: int = 8) -> Tuple[pyhanabi.HanabiState, List[TrajectoryStep]]:
    """Play a short game where players follow the SignalBlueprint."""
    config = {
        "players": 2,
        "colors": 1,
        "ranks": 5,
        "hand_size": 1,
        "max_information_tokens": 10,
        "max_life_tokens": 10,
        "seed": 0,
    }
    game = pyhanabi.HanabiGame(config)
    state = game.new_initial_state()
    # Keep game alive alongside returned state to avoid C++ object destruction.
    state._game_ref = game

    _advance_chance_events(state)
    _burn_clues(state, 8)

    steps: List[TrajectoryStep] = []
    blueprints = [signal_blueprint_factory() for _ in range(game.num_players())]

    for _ in range(turns):
        if state.is_terminal():
            break
        if state.cur_player() == pyhanabi.CHANCE_PLAYER_ID:
            _advance_chance_events(state)
            continue
        player = state.cur_player()
        obs = build_observation(state, player)
        move = blueprints[player].act(obs)
        before = state.copy()
        before._game_ref = game

        state.apply_move(move)
        _advance_chance_events(state)
        after = state.copy()
        after._game_ref = game
        steps.append(TrajectoryStep(player=player, observation=obs, move=move, state=before, next_state=after))

    return state, steps


def _burn_clues(state: pyhanabi.HanabiState, burn_count: int = 0) -> None:
    """Spend a couple of information tokens so discards become legal; hints do not affect the signal."""
    for _ in range(burn_count):
        if state.is_terminal() or state.information_tokens() == 0:
            break
        if state.cur_player() == pyhanabi.CHANCE_PLAYER_ID:
            _advance_chance_events(state)
            continue
        obs = build_observation(state, state.cur_player())
        hint_moves = [
            m for m in obs.legal_moves if m.type() in (pyhanabi.HanabiMoveType.REVEAL_COLOR, pyhanabi.HanabiMoveType.REVEAL_RANK)
        ]
        if not hint_moves:
            break
        move = hint_moves[0]
        state.apply_move(move)
        _advance_chance_events(state)


def test_sample_world_state_respects_signal_blueprint():
    rng = random.Random(0)
    _, steps = rollout_signal_trajectory(turns=12)

    saw_play = False
    saw_discard = False

    for step in steps:
        if step.next_state.is_terminal():
            continue
        move_type = step.move.type()
        if move_type not in (pyhanabi.HanabiMoveType.PLAY, pyhanabi.HanabiMoveType.DISCARD):
            continue
        if step.move.card_index() != 0:
            continue

        observer = 1 - step.player
        obs = build_observation(step.next_state, observer)
        if not any(m.get("player") not in (observer, pyhanabi.CHANCE_PLAYER_ID) for m in obs.last_moves):
            continue
        matched = False
        for _ in range(10):
            sampled = sample_world_state(step.next_state, obs, rng, blueprint_factory=signal_blueprint_factory)
            # Reconstruct what the partner would do under this sample.
            sim_state = step.next_state.copy()
            sim_state._game_ref = getattr(step.next_state, "_game_ref", None)
            apply_sampled_hands(sim_state, sampled)
            partner_obs = build_observation(sim_state, step.player)
            if not partner_obs.legal_moves:
                partner_obs.legal_moves = (step.move,)
                partner_obs.legal_moves_dict = [obs.last_moves[-1]["move"]] if obs.last_moves else []
            try:
                predicted = signal_blueprint_factory().act(partner_obs)
            except Exception:
                predicted = None
            if predicted == step.move:
                matched = True
                break

        if move_type == pyhanabi.HanabiMoveType.PLAY:
            saw_play = True
        else:
            saw_discard = True
        assert matched, "Sampled belief was incompatible with the signaling blueprint."

    assert saw_play, "Expected to observe a partner PLAY on slot 0 in the rollout."
    assert saw_discard, "Expected to observe a partner DISCARD on slot 0 in the rollout."
