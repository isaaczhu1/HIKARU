from hanabi_learning_environment import pyhanabi

def make_game(players=2, seed=0):
    return pyhanabi.HanabiGame({
        "players": players,
        "colors": 5,
        "ranks": 5,
        "hand_size": 5,
        "max_information_tokens": 8,
        "max_life_tokens": 3,
        "random_start_player": False,
        "seed": seed,
        "observation_type": pyhanabi.AgentObservationType.CARD_KNOWLEDGE.value,
    })

def new_empty_state(game):
    state = game.new_initial_state()
    assert state.cur_player() == pyhanabi.CHANCE_PLAYER_ID
    return state

def deal_initial_hands(game, state, hand0, hand1):
    """
    Deterministically deal the initial hands without deal_random_card.
    PlayerToDeal fills player 0 to 5, then player 1 to 5.
    hand0/hand1: list of (color_char, rank_idx) length == game.hand_size()
    """
    assert len(hand0) == game.hand_size()
    assert len(hand1) == game.hand_size()

    i0 = i1 = 0
    while state.cur_player() == pyhanabi.CHANCE_PLAYER_ID:
        hands = state.player_hands()
        pid = next(i for i,h in enumerate(hands) if len(h) < game.hand_size())
        if pid == 0:
            c, r = hand0[i0]; i0 += 1
        else:
            c, r = hand1[i1]; i1 += 1
        state.deal_specific_card(pid, pyhanabi.color_char_to_idx(c), r)

    assert i0 == game.hand_size() and i1 == game.hand_size()
    assert state.cur_player() == 0  # with random_start_player=False

def deal_missing_specific(game, state, color_char, rank_idx):
    """Resolve chance steps deterministically by dealing a chosen card to whoever is short."""
    color = pyhanabi.color_char_to_idx(color_char)
    while state.cur_player() == pyhanabi.CHANCE_PLAYER_ID:
        hands = state.player_hands()
        pid = next(i for i,h in enumerate(hands) if len(h) < game.hand_size())
        state.deal_specific_card(pid, color, rank_idx)

def canonical_card_knowledge_start(game):
    """Compute start offset of CardKnowledge section in canonical encoding."""
    num_players = game.num_players()
    num_colors = game.num_colors()
    num_ranks = game.num_ranks()
    hand_size = game.hand_size()
    bits_per_card = num_colors * num_ranks

    max_deck_size = sum(game.num_cards(c, r) for c in range(num_colors) for r in range(num_ranks))
    max_info = game.max_information_tokens()
    max_life = game.max_life_tokens()

    hands_len = (num_players - 1) * hand_size * bits_per_card + num_players
    board_len = (max_deck_size - num_players * hand_size) + (num_colors * num_ranks) + max_info + max_life
    discard_len = max_deck_size
    last_action_len = (
        num_players + 4 + num_players + num_colors + num_ranks +
        hand_size + hand_size + bits_per_card + 2
    )
    return hands_len + board_len + discard_len + last_action_len

def assert_banned_in_p0_hand(game, obs0, color_char, rank_idx):
    """
    Assert that in player-0’s observation, for every slot in player 0’s hand,
    the (color_char, rank_idx) bit in the *plausible-card* subvector is 0.
    """
    enc = pyhanabi.ObservationEncoder(game).encode(obs0)

    num_players = game.num_players()
    num_colors = game.num_colors()
    num_ranks = game.num_ranks()
    hand_size = game.hand_size()
    bits_per_card = num_colors * num_ranks
    stride = bits_per_card + num_colors + num_ranks

    ck0 = obs0.card_knowledge()[0]
    my_slots = len(ck0)

    start = canonical_card_knowledge_start(game)
    c = pyhanabi.color_char_to_idx(color_char)
    pair_bit = c * num_ranks + rank_idx

    bad = []
    for i in range(my_slots):
        base = start + (0 * hand_size + i) * stride  # player 0, slot i
        slot_enc = enc[base:base+stride]
        print(f"Slot {i} encoding: {slot_enc}")
        bit = enc[base + pair_bit]
        if bit != 0:
            bad.append(i)

    assert not bad, f"{color_char}{rank_idx+1} still plausible in slots {bad}"


def run_test1():
    game = make_game(seed=10)
    state = new_empty_state(game)

    # Deal initial hands deterministically.
    hand0 = [("R",0), ("Y",0), ("B",0), ("W",0), ("R",1)]             # no G5
    hand1 = [("G",4), ("Y",1), ("B",1), ("W",1), ("R",2)]             # partner has G5
    deal_initial_hands(game, state, hand0, hand1)

    obs0 = state.observation(0)
    partner = obs0.observed_hands()[1]
    assert any(str(c) == "G5" for c in partner), [str(c) for c in partner]

    # New exact-pair masking check:
    assert_banned_in_p0_hand(game, obs0, "G", 4)
    print("OK: G5 visible in partner, and G5 is banned from all p0 slots.")


def run_test2():
    game = make_game(seed=20)
    state = new_empty_state(game)

    # Give p0 a B2 (to discard) and p1 a B2 (to keep visible).
    hand0 = [("B",1), ("Y",0), ("R",0), ("G",0), ("W",0)]             # p0 has B2 at slot 0
    hand1 = [("B",1), ("R",1), ("Y",1), ("G",1), ("W",1)]             # p1 has B2
    deal_initial_hands(game, state, hand0, hand1)

    # Need info tokens < max to discard; do two hints to return turn to p0.
    state.apply_move(pyhanabi.HanabiMove.get_reveal_color_move(1, pyhanabi.color_char_to_idx("R")))  # p0->p1
    state.apply_move(pyhanabi.HanabiMove.get_reveal_color_move(1, pyhanabi.color_char_to_idx("Y")))  # p1->p0

    assert state.cur_player() == 0
    assert state.information_tokens() < game.max_information_tokens()

    # Discard B2 (slot 0)
    state.apply_move(pyhanabi.HanabiMove.get_discard_move(0))
    deal_missing_specific(game, state, "W", 2)  # deterministic refill (W3), NOT B2

    # Verify required visibility
    obs0 = state.observation(0)
    partner = obs0.observed_hands()[1]
    discards = state.discard_pile()
    assert any(str(c) == "B2" for c in partner), [str(c) for c in partner]
    assert any(str(c) == "B2" for c in discards), [str(c) for c in discards]

    # New exact-pair masking check:
    assert_banned_in_p0_hand(game, obs0, "B", 1)
    print("OK: B2 in partner + discard, and B2 is banned from all p0 slots.")



def run_test3():
    game = make_game(seed=30)
    state = new_empty_state(game)

    # Use all 3 copies of R1: two in p0 (play + discard), one in p1 (visible).
    hand0 = [("R",0), ("R",0), ("Y",0), ("B",0), ("W",0)]
    hand1 = [("R",0), ("Y",1), ("G",1), ("B",2), ("W",2)]
    deal_initial_hands(game, state, hand0, hand1)

    # p0 plays R1 (slot 0)
    state.apply_move(pyhanabi.HanabiMove.get_play_move(0))
    deal_missing_specific(game, state, "G", 2)  # deterministic refill (G3), NOT R1

    # p1 spends a token by hinting p0 (so p0 can discard)
    assert state.cur_player() == 1
    state.apply_move(pyhanabi.HanabiMove.get_reveal_color_move(1, pyhanabi.color_char_to_idx("Y")))  # p1->p0

    # p0 discards the remaining R1 (it should now be at slot 0)
    assert state.cur_player() == 0
    assert state.information_tokens() < game.max_information_tokens()
    state.apply_move(pyhanabi.HanabiMove.get_discard_move(0))
    deal_missing_specific(game, state, "W", 3)  # deterministic refill (W4), NOT R1

    # Verify required conditions
    fireworks = state.fireworks()
    assert fireworks[pyhanabi.color_char_to_idx("R")] == 1, fireworks

    obs0 = state.observation(0)
    partner = obs0.observed_hands()[1]
    discards = state.discard_pile()
    assert any(str(c) == "R1" for c in partner), [str(c) for c in partner]
    assert any(str(c) == "R1" for c in discards), [str(c) for c in discards]

    # New exact-pair masking check:
    assert_banned_in_p0_hand(game, obs0, "R", 0)
    print("OK: R1 played + in partner + in discard, and R1 is banned from all p0 slots.")


if __name__ == "__main__":
    run_test1()
    run_test2()
    run_test3()