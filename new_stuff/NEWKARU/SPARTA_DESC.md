Here’s the picture: SPARTA takes any fixed joint policy (“blueprint”) for a cooperative partially observable game and wraps it in a Monte Carlo 1-ply search procedure that (a) uses explicit Bayesian beliefs and (b) is proven to be at least as good as the blueprint in expectation, up to an $O(1/\sqrt{K})$ term from finite rollouts. 
ar5iv
+1

Below I’ll first summarize the paper, then give you the concrete technical recipe to implement the single-agent SPARTA search wrapper (the part you actually need). I’ll also sketch how they implement beliefs in Hanabi.

1. High-level summary of the paper

Setting. They work in a cooperative Dec-POMDP: multiple agents, shared team reward, partial observability. At each time $t$:

State: $s_t$ (full Markov state, hidden).

Agent $i$ gets observation $o^i_t$ and takes action $a^i_t$.

Joint action $a_t = (a^1_t,\dots,a^N_t)$ leads to new state and reward $r_t$. 
ar5iv

Each agent’s action–observation history (AOH) is
$h^i_t = (o^i_1,a^i_1,\dots,o^i_t)$.

They assume:

A deterministic joint “blueprint” policy $\pi^* = (\pi^{*1},\dots,\pi^{*N})$ is common knowledge and specifies what every agent would do for every possible AOH. 
ar5iv

During play, an agent can optionally deviate from $\pi^$ by running SPARTA search; everyone knows the search algorithm and that $\pi^$ is the fallback whenever search is disabled.

The main contributions:

Single-agent search. Exactly one agent $i$ is allowed to search. Everyone else always follows $\pi^{-i}$. From $i$’s perspective, this is just a POMDP with known environment dynamics (including partner policies). She maintains a belief over trajectories and does 1-ply Monte Carlo policy improvement with rollouts that follow $\pi^$ after the first move. 
ar5iv

Multi-agent search. All agents can search, but they must simulate each other’s search to keep beliefs consistent. They do this via range-search (search over all possible AOHs consistent with public info) and retrospective belief updates plus a max range (MR) cutoff so they can skip search when the belief space is too big. 
ar5iv

Theoretical guarantee. For any Dec-POMDP with bounded rewards and horizon $T$, if you do SPARTA search with $K$ MC rollouts per decision, the expected value of the search policy is never worse than the blueprint policy, up to $O(R_{\max} T \sqrt{\log |A| / K})$. So as $K\to\infty$, search cannot hurt. 
ar5iv
+1

Hanabi results. Wrapping strong blueprints (SAD, SmartBot, DQN, etc.) with SPARTA bumps 2-player Hanabi from 24.08 → 24.61 / 25, with ~75.5% perfect games for SAD+SPARTA in 2-player self-play. 
ar5iv
+1

2. Core technical objects you need
2.1 Trajectories and beliefs

They define a trajectory up to time $t$ as
$\tau_t = (s_0,a_0,r_0,\dots,s_t)$ – complete world history. 
ar5iv

The private belief of agent $i$ at time $t$ is:

𝐵
𝑡
𝑖
(
𝜏
𝑡
)
=
Pr
⁡
(
𝜏
𝑡
∣
ℎ
𝑡
𝑖
)
,
B
t
i
	​

(τ
t
	​

)=Pr(τ
t
	​

∣h
t
i
	​

),

stored over the trajectory range
$\beta^i_t = {\tau_t : B^i_t(\tau_t) > 0}$. 
ar5iv

The common-knowledge belief is $\hat B_t(\tau_t) = \Pr(\tau_t \mid \text{CK}_t)$ over $\hat\beta_t$.

You don’t literally have to store full trajectories; you just need some way to:

Maintain a distribution over “hidden worlds” consistent with what you’ve seen and the known partner policies; and

Sample from it when you do rollouts.

2.2 Blueprint policy

You need a joint blueprint policy $\pi^*$:

Each agent $i$ has $\pi^{*i}(a^i_t \mid h^i_t)$, deterministic or stochastic. 
ar5iv

During search:

For the rollouts, you assume all agents (including the searcher) follow $\pi^*$ after the first step.

At the real decision point, the searcher may choose a different action $a^\text{search}$ instead of $\pi^{*i}(h^i_t)$.

3. Single-agent SPARTA search wrapper (what you implement first)

Assume:

You have some environment API (e.g., DeepMind HLE for Hanabi) that you can clone/reset.

You have a blueprint policy object that can act given an observation/AOH.

You have (or can hack together) a belief sampler over hidden states/worlds.

We’re in the single-agent setting: only agent $i$ (say, Player 1) runs search and may deviate; everyone else always plays the blueprint.

3.1 Maintain beliefs between moves

On each step, the searcher maintains $B^i_t(\cdot)$ over the trajectory/worlds consistent with her AOH $h^i_t$ and the assumption that others follow $\pi^{*-i}$.

When some event happens (partner’s action, new observation), you update the belief via two components: 
ar5iv
+1

Policy-based part. If partner $j$ is known to follow $\pi^{*j}$ and you observe $a^j_t$, you reweight worlds where $\pi^{*j}$ would have chosen $a^j_t$ more heavily, and set probability to zero for worlds where $\pi^{*j}$ would have chosen a different action.

Dynamics-based part. Given the joint action, propagate each world forward by your transition model and update probabilities based on observation likelihoods.

Formally they write this as a Bayesian update using equations (5)–(6): beliefs over trajectories are updated using the known policy and transition function. 
ar5iv

In practice (e.g., Hanabi) they don’t explicitly track full trajectories; they track a distribution over hands + deck counts and update it by: 
ar5iv
+1

removing revealed/discarded cards from counts,

zeroing out hand configurations inconsistent with hints,

using the known blueprint policy to treat partner actions as information.

For a first implementation, you can get away with a simpler approximate belief sampler as long as it respects card counts and hints.

3.2 When it’s your turn: Monte Carlo action evaluation

At a decision point for agent $i$ with AOH $h^i_t$ and belief $B^i_t$:

Let $A_\text{legal}$ be all legal actions for agent $i$.

For each $a \in A_\text{legal}$, estimate $Q^{\pi^*}(h^i_t, a)$ by MC rollouts:

Sample hidden world $w$ from $B^i_t$ (or from an approximate belief).

Clone the environment and implant this world.

Apply action $a$ for agent $i$.

From the next step until terminal:

All agents, including $i$, follow the blueprint $\pi^*$.

Record the total return $G$ from this rollout.

Repeat $K$ times; let $\hat Q(a)$ be the sample mean. 
ar5iv
+1

They call this 1-ply search because you optimize only the next action; the rest of the game follows the blueprint.

3.2.1 UCB-style rollout pruning

To save computation, they add a simple UCB-like pruning mechanism: 
ar5iv
+1

Maintain for each $a$: current mean $\hat Q(a)$ and sample standard deviation $\hat\sigma(a)$.

Force a minimum of, say, 100 rollouts per action.

Afterwards, after each new rollout:

Let $a_\text{best}$ be the action with highest current mean.

If for some action $a$,

𝑄
^
(
𝑎
)
+
2
𝜎
^
(
𝑎
)
<
𝑄
^
(
𝑎
best
)
,
Q
^
	​

(a)+2
σ
^
(a)<
Q
^
	​

(a
best
	​

),

then stop doing rollouts for $a$. Intuition: with high confidence, $a$ can’t catch up.

So you keep sampling only actions that might still be best; this often cuts rollout cost by ~10× with no performance drop. 
ar5iv

3.2.2 Deviation threshold from blueprint

They also use a deviation threshold: search will only override the blueprint action if it is clearly better.

Let $a_\text{bp} = \pi^{*i}(h^i_t)$ be the blueprint action.

Let $a_\text{best} = \arg\max_a \hat Q(a)$.

Let $\Delta = \hat Q(a_\text{best}) - \hat Q(a_\text{bp})$.

If $\Delta < \delta$ (small positive constant), stick with the blueprint; otherwise, play $a_\text{best}$.

They use a threshold around 0.05 points in Hanabi and find that this both reduces harmful deviations and slightly improves score even for large numbers of rollouts. 
ar5iv
+1

This is also philosophically nice: deviating from the blueprint “corrupts” your partner’s beliefs (in naive multi-agent search), so you only deviate when it really matters.

3.3 Theoretical guarantee

The core theorem (Theorem 1/2) is:

Assume rewards are bounded, $|r_t| \le R_{\max}$, horizon $T$, action set size $|A|$.

At each decision, for each action you collect $K$ independent bounded rollout returns to estimate $Q^{\pi^*}(h,a)$, and choose the empirical best. 
ar5iv

Then:

𝑉
search
  
≥
  
𝑉
blueprint
−
𝑂
 ⁣
(
𝑅
max
⁡
 
𝑇
 
log
⁡
∣
𝐴
∣
𝐾
)
V
search
	​

≥V
blueprint
	​

−O(R
max
	​

T
K
log∣A∣
	​

	​

)

i.e. in expectation the search policy cannot be worse than the blueprint by more than a term that shrinks as $1/\sqrt{K}$. 
ar5iv
+1

Practically: if you use a reasonable $K$ (tens to hundreds of rollouts per action) and the environment reward scale isn’t huge, you should see improved or at least equal performance.

4. Hanabi-specific implementation details (what they actually do)

For Hanabi, they exploit structure: it’s a factorized observation game, where the only hidden information is the identities of players’ cards and the deck. 
ar5iv

4.1 Belief representation in Hanabi

They represent beliefs as distributions over possible hands consistent with public information:

Start with exact card counts (how many of each color/rank exist).

Public info includes:

which cards have been played to fireworks,

which cards have been discarded,

whose turn it is, info / life tokens, hints given. 
ar5iv

Then:

A public belief over hands assumes independence between players’ hands, just constrained by card counts and hints. For 2-player Hanabi they can factor the public belief into independent distributions over each player’s hand (conditional on CK). 
ar5iv

A private belief for player $i$ over partner’s hand is obtained by taking the public belief and conditioning on $i$’s privately known hand (so adjusting card counts). This “public → private” conversion is their ConditionOnAOH() helper in the appendix. 
ar5iv
+1

Belief updates on observations:

When a card is revealed (played/discarded/drawn), update the card counts.

When a hint is given, set probabilities of hand configurations inconsistent with that hint to 0 and renormalize. 
ar5iv
+1

For SPARTA search:

To sample a world for a rollout, they:

Sample a concrete assignment of cards to each hand according to the belief.

Sample a deck order consistent with remaining card counts.

Then they run a Hanabi simulator forward using that hidden state.

4.2 Rollout details in Hanabi

Rollouts are full episodes (to end of game), not truncated with value bootstrapping.

They parallelize rollouts across many CPU cores (40-core machines); a single 2-player game with single-agent SPARTA on SmartBot costs ~2 core-hours; multi-agent SPARTA with MR=10k can cost ~90 core-hours per game. 
ar5iv
+1

For action evaluation they use:

Min 100 rollouts per action,

UCB pruning,

Deviation threshold ≈ 0.05. 
ar5iv

You don’t need exactly those numbers, but they’re a good starting point.

5. How to implement the SPARTA wrapper in your own code

Putting it all together, here’s what you actually need to code for a single-agent SPARTA wrapper around a blueprint policy:

5.1 Required components

Environment interface (e.g., Hanabi HLE wrapper):

env.clone() or a way to reconstruct env from a WorldState.

env.step(player_id, action) → (next_obs, reward, done, info).

Access to full hidden state for simulation (hands, deck, piles).

Blueprint policy:

blueprint.act(player_id, obs, internal_state) -> (action, new_internal_state)
(internal_state could be RNN hidden state, or you keep AOH manually).

Belief model:

Data structure representing belief over hidden worlds given acting player’s AOH (or public info + their hand).

sample_world_from_belief(belief) -> WorldState.

For a first pass, this can be approximate as long as it respects card counts and hints.

Search wrapper:

sparta.act(env, player_id, obs) -> action.