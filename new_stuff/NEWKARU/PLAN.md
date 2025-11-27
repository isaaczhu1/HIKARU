# Hanabi SPARTA-Lite Project Plan

This document is a technical blueprint for starting your Hanabi + SPARTA-style project, aimed at being clearly connected to 6.S890. It assumes:

- You’ll eventually have **two tracks**:
  - A **Tiny-Hanabi / toy game** track for theory-ish analysis (idea #1).
  - A **full 2-player Hanabi** track with a learned or heuristic blueprint and decision-time search (idea #3).

Below is an ordered plan with concrete steps, implementation details, and relevant references.

---

## 0. Key References & Concepts

You’ll be drawing on:

- **Hanabi Learning Environment (HLE)** – DeepMind’s standard environment.
- **SPARTA** – “Improving Policies via Search in Cooperative Partially Observable Games”.
- **SAD** – Simplified Action Decoder, a strong learned blueprint.
- **Tiny Hanabi Suite / CAPI** – small common-payoff games and approximate policy iteration in public belief space.

You don’t need to *implement* SAD or CAPI to start, but you should understand at a high level what they do (see reference section at the end).

---

## 1. Repo / Environment Setup

### 1.1. Create a project structure

Suggested layout:

```text
hanabi-sparta/
  envs/
    full_hanabi_env.py      # thin wrapper around DeepMind HLE
    tiny_hanabi_env.py      # custom tiny game or tiny-hanabi wrapper
  blueprints/
    heuristic_blueprint.py  # hand-coded policy
    gru_blueprint.py        # RL policy (later)
  search/
    belief_models.py        # public & private belief representations
    sparta_single.py        # single-agent search wrapper
    evaluation.py           # evaluation utilities
  experiments/
    baseline_full.py        # baseline full Hanabi experiments
    baseline_tiny.py        # baseline tiny game experiments
    search_full.py          # SPARTA-lite full Hanabi
    search_tiny.py          # SPARTA & policy iteration in tiny game
  utils/
    logging.py
    plot_results.ipynb
  README.md


1.2. Python & dependencies

Use a Conda env:

conda create -n hanabi python=3.10
conda activate hanabi
pip install hanabi_learning_environment numpy torch matplotlib


2. Full 2-Player Hanabi: Baseline & Heuristic Blueprint
2.1. Wrap the Hanabi Learning Environment

In envs/full_hanabi_env.py:

Use hanabi_learning_environment to create a 2-player game.


2.2. Hand-coded heuristic blueprint

In blueprints/heuristic_blueprint.py:

Define a policy HeuristicBlueprint that maps from observation to an action:

High-level logic for 2-player Hanabi:

Identify guaranteed playable cards in your hand:

Based on hints and fireworks piles, you may know a card’s color and rank and see it’s exactly the next needed rank for that color.

Play if safe:

If any card is guaranteed playable, play one of them (pick first or prefer lower index).

Hint if info tokens available:

If no guaranteed play, and there are >= 1 info tokens:

Look at partner’s hand.

Try to give a hint (color or rank) that:

Reveals at least one new piece of information.

Ideally marks a currently playable card in partner’s hand.

Discard otherwise:

If no safe play and no info tokens (or hints are unhelpful):

Discard a card that seems least useful:

E.g., a card with lowest probability of being part of a future stack.

As a simple heuristic: discard the card with lowest rank or least hinted.

Note: You don’t need the blueprint to be strong; it just needs to be coherent and not random.


2.3. Baseline evaluation

Create experiments/baseline_full.py:

Self-play: both players use HeuristicBlueprint.

Run N episodes (e.g., N = 500–1000).

Log:

Average final score,

Standard deviation,

Distribution of scores (maybe a histogram).

This becomes your baseline to compare against SPARTA-lite.


3. Belief Modeling for Full 2-Player Hanabi

For SPARTA-like search, you need a way to sample hidden worlds consistent with public info and hints.

3.1. Representing a hidden world

Define a WorldState class in search/belief_models.py containing:

Public:

Fireworks piles,

Discard pile,

Information tokens,

Life tokens,

Turn index,

Whose turn it is.

Private:

Each player’s hand as a list of card identities (color, rank),

Remaining deck as a multiset/list of card identities (in some order).

You don’t need to store & update the entire deck order; you can treat the deck as a multiset from which you randomly draw cards.

3.2. Sampling a world from approximate beliefs

Given:

Public log (piles, discards, tokens),

Your own hand (from environment state),

Hints each player has received,

you’ll implement:

def sample_world_from_belief(public_state, my_id):
    """
    Returns a sampled WorldState consistent with:
    - card counts,
    - discards & fireworks,
    - hints (roughly),
    - my own observed cards for my hand.
    """


Simple approach:

Construct the full deck multiset for standard Hanabi (e.g., 5 colors, ranks 1–5, known multiplicities).

Subtract all cards already visible in:

Discard pile,

Fireworks piles,

Your own hand (in the simulated world),

Any known cards in partner’s hand (if hints fully identify a card).

For each unknown card in partner’s hand:

Sample a card from the remaining multiset conditional on hints:

If a card is hinted “red”, only sample reds.

If a card is hinted “2”, only sample rank-2 cards.

If both, ensure color and rank match.

The leftover multiset is the deck.

This is an approximate belief: it ignores some correlations between cards, but that’s okay for a first pass.

Optionally: you can treat your own hand as known in simulation, even though in real Hanabi you don’t see your own cards. That’s a modeling choice: for SPARTA’s internal rollouts, you might assume card identities once sampled, and only enforce belief consistency from the acting player’s perspective when making decisions.

4. SPARTA-Lite Single-Agent Search Wrapper (Full Hanabi)

This is the core of your initial project: a single-agent SPARTA-like wrapper around your heuristic blueprint.

In search/sparta_single.py implement:

4.1. Interface
class SpartaSingleAgentWrapper:
    def __init__(self, blueprint, belief_sampler, num_rollouts=64, epsilon=0.2):
        self.blueprint = blueprint
        self.belief_sampler = belief_sampler
        self.num_rollouts = num_rollouts
        self.epsilon = epsilon  # min gain over blueprint to deviate

    def act(self, env, player_id, obs):
        # env: a copyable Hanabi env (or a lightweight representation)
        # obs: observation for player_id at decision time
        # returns chosen action id

4.2. Action evaluation

Given obs and player_id:

Get legal_actions from env.

For each action a in legal_actions:

Initialize array of returns R[a] = [].

Repeat num_rollouts times:

Sample a WorldState using belief_sampler (conditioned on player_id’s perspective).

Create a simulated environment from that WorldState (you can write a small helper that reconstructs an HLE game from a WorldState).

In the simulated env:

Apply action a for player_id.

For all subsequent steps until terminal:

At each step t, for current p:

Use blueprint.act(obs_t^p) to select an action.

At terminal, record final score G.

Append G to R[a].

Estimate Q_hat[a] = mean(R[a]).

Find blueprint action a_bp = blueprint.act(obs) and its estimated value Q_hat[a_bp].

Find best action a_star = argmax_a Q_hat[a].

Decision rule:

If Q_hat[a_star] >= Q_hat[a_bp] + epsilon:

play a_star.

Else:

play a_bp.

This is “SPARTA-lite”: a 1-ply Monte Carlo improvement on the blueprint, with a conservative deviation threshold.

4.3. Use only one searcher at first

For simplicity:

Let Player 1 use SPARTA-lite wrapper.

Let Player 2 use the plain heuristic blueprint.

This is enough to:

Demonstrate that SPARTA improves performance for the searcher,

Avoid multi-agent belief-consistency headaches initially.

5. Evaluation on Full Hanabi

In experiments/search_full.py:

Setup experiments:

Baseline:

P1 = heuristic blueprint,

P2 = heuristic blueprint.

Single-agent SPARTA:

P1 = SPARTA(heuristic),

P2 = heuristic.

Run N episodes (e.g. N = 500–1000) for each setting.

Metrics:

Mean score and standard error.

Score histogram (0–25).

Possibly breakdown by:

Number of hint tokens used,

Number of disasters.

Expectations:

Some improvement in P1’s average score with SPARTA-lite vs baseline.

Improvement magnitude depends on blueprint and belief quality; even +1–3 points is a good sign.

This gives you a fully-working “SPARTA-on-top-of-a-blueprint” story for the full game.

6. Tiny-Hanabi / Toy Game Track (For Theory / Idea #1)

To connect strongly to 6.S890’s equilibrium & public belief content, you want a small common-payoff imperfect-information game where you can:

Enumerate states,

Compute optimal joint policy,

Explicitly represent public belief states.

6.1. Use Tiny Hanabi Suite or roll your own

Option A: Use the Tiny Hanabi Suite:

It’s a collection of very small two-player common-payoff games inspired by Hanabi.

They provide tabular algorithms to compute joint policies and public belief states.

You can either:

Use it directly as a Python dependency, or

Study its design and reimplement a small variant in your own tiny_hanabi_env.py.

Option B: Design your own Tiny-Hanabi:

2 players,

2 colors, ranks 1–3,

Hand size 2,

Short deck,

Horizon 3–4 steps.

6.2. Explicit public belief MDP

For the tiny game, define:

Public state: everything common knowledge (discarded cards, plays, hints, whose turn).

Private info: the hidden cards in hands / deck.

You can:

Enumerate all possible public states (there will be a manageable number).

For each public state, enumerate all possible hidden card configurations consistent with it.

Public belief at state s_pub is a distribution over these card configurations.

6.3. Toy SPARTA operator in public belief space

Define a tabular blueprint π for the tiny game (e.g., simple heuristic or random policy).

Then define a SPARTA-like policy improvement operator I(π):

At each public belief state b:

Generate all possible joint actions (or a subset).

For each joint action a:

Use rollouts where:

Start from belief b,

Sample a hidden configuration from b,

Apply a,

From then on, follow π until terminal,

Average returns → Q^π(b, a).

Let I(π)(b) be the greedy prescription (or joint action distribution) at b w.r.t. Q^π.

You now have:

A policy-improvement operator in the public belief MDP.

This is the tiny-game analogue of SPARTA.

6.4. Compare to optimal joint policy and/or CAPI

For the tiny game, you can:

Compute the optimal joint policy (e.g., via dynamic programming over the public belief MDP).

Optionally, implement a simple CAPI-style approximate policy iteration, if time permits.

Then compare:

Value(π),

Value(I(π)),

Value(optimal_policy).

You can empirically check:

Does one application of I(π) (tiny-SPARTA) strictly improve π’s value?

If you iterate π_{k+1} = I(π_k) for k = 0,1,2,…, does it converge to the optimal policy?

Are there any cycles or instability?

This is your idea #1: SPARTA as a policy-improvement operator in a small common-payoff game, strongly connected to public belief and CAPI.

7. Later: GRU/LSTM Blueprint (Idea #3)

Once the basic stack works on heuristic blueprint, you can upgrade the blueprint to a learned recurrent policy.

7.1. GRU blueprint training

In blueprints/gru_blueprint.py:

Use a GRU or LSTM with:

Input: flattened Hanabi observation vector,

Hidden state h_t,

Output: either:

Q-values over actions (DQN-style), or

logits over actions (policy gradient / A2C-style).

Training setup (simplified):

Self-play with 2 players, both using the same network.

For DQN-style:

Maintain a replay buffer,

Use ε-greedy over Q-values.

For policy-grad:

Use on-policy rollouts and REINFORCE / A2C.

You don’t need to reach SAD performance; a blueprint that’s meaningfully better than random is enough.

7.2. Plug GRU into SPARTA

Replace HeuristicBlueprint with GRUBlueprint in your SpartaSingleAgentWrapper:

In rollouts, both agents use the GRU blueprint.

At decision time, SPARTA uses GRU as the rollout policy.

Evaluate:

GRU alone vs GRU+SPARTA-lite on full Hanabi.