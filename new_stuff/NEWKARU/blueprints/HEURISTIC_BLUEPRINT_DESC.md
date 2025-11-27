# Simple Heuristic Blueprint for 2-Player Hanabi (Minimal Version)

This is a **very simple** rule-based policy for 2-player Hanabi. It’s designed to be:

- Easy to implement.
- Reasonably intelligent.
- A good starting blueprint for adding SPARTA-style search on top.

We describe it purely as **rules**, not code.

---

## High-Level Idea

On your turn, you:

1. **Play** a card when you’re sure it’s playable.
2. **Tell your partner** about their playable cards when possible.
3. **Discard** cards you know are useless.
4. Otherwise, **give a simple hint** or **discard an unhinted card**.

That’s it. Only **five** simple rules, in a fixed order.

---

## Concepts the Agent Uses

The agent keeps only **very lightweight** knowledge:

- For each card in **its own hand**:
  - Whether it has been **hinted** (at all).
- From the board:
  - What rank is needed next for each color on the **fireworks piles**.
  - Which cards are in the **discard pile**.
  - How many **information tokens** and **life tokens** remain.
- It sees the **partner’s hand** (true colors and ranks).

We assume that **multiple hints** on the same card can tell the agent its exact identity (color and rank), but we don’t formalize all the details here—you can implement this as simply as you like.

---

## The Rules (In Priority Order)

Assume it is **my turn**. I run through these rules from top to bottom and do the first one that applies.

### Rule 1 — Play a card I *know* is playable

**When:**  
If I know the exact color and rank of a card in my hand, and that card is **exactly what is needed** on its color pile.

More concretely:

- For some card in my hand:
  - I know its color `C` and rank `R`.
  - On the board, the firework pile for color `C` currently shows rank `R - 1` (or is empty and `R = 1`).
  - Then this card is definitely playable.

**Action:**

- Play that card (if multiple, pick any—e.g., the leftmost or rightmost).

---

### Rule 2 — Hint partner about a card they can play **now**

**When:**  

- I have at least **1 information token**.
- In my partner’s hand, I see at least one card that is playable **immediately**:
  - For some card in their hand:
    - Its color is `C`, its rank is `R`.
    - The firework pile for `C` shows `R - 1` (or empty and `R = 1`).

**Action:**

- Pick one such playable card in partner’s hand.
- Give **either**:
  - a color hint that includes this card, or  
  - a rank hint that includes this card.
- Prefer a hint that highlights as few extra cards as possible (so it’s obvious which one is meant).

Interpretation: if I point at your playable card with a hint, I’m saying “play this soon”.

---

### Rule 3 — Discard a card I *know* is useless

**When:**  

If I can identify a card in my hand that can **never** help us.

Simple criteria for “useless” (you can implement any or all):

- Its color and rank are known, and:
  - That rank in that color is already **fully burned**:
    - Either that card’s rank is already on the firework pile and we don’t need another copy, **and**
    - All copies of it are visible in discards/piles; **or**
  - It’s a low rank that is no longer needed because the pile has advanced past it and there are no mechanics requiring additional copies.

You can keep this check very rudimentary at first (e.g., “this card’s rank is already on the pile and we’ve seen it many times in discards”).

**Action:**

- Discard one such useless card (e.g., the oldest one in hand that’s definitely useless).

---

### Rule 4 — Give a simple information hint (if tokens are available)

**When:**

- I have at least **1 information token**.
- None of Rules 1–3 apply (no certain play, no immediate playable in partner’s hand, no clearly useless card to discard).

**Action:**

- Choose a simple hint (color or rank) that:
  - Reveals something **new** about at least one card in partner’s hand.
  - Prefer hints that mark **low-rank** or **likely-useful** cards if possible.
- For example:
  - Hint a color that covers several unknown cards, or
  - Hint a rank that will be needed soon (like rank 2 or 3 in colors we’re building).

We don’t try to be super clever here—just avoid obviously pointless hints.

---

### Rule 5 — Discard the oldest unhinted card

**When:**

- None of Rules 1–4 apply:
  - No sure play.
  - No immediate playable in partner’s hand (or no info tokens).
  - No clearly useless card.
  - No reasonably good hint (e.g., no info tokens, or we choose not to use one).

**Action:**

- Look at my hand. Among all cards that have **never been hinted**, discard the **oldest** one.
- If all cards have been hinted at least once, just discard the oldest card in hand.

Interpretation:  
“Cards you’ve never hinted me about are probably safe to discard; if you cared about them, you would have told me.”

---

## 4. Why This Blueprint Is Good Enough (For Now)

This blueprint:

- Has **only five simple rules**.
- Is very straightforward to implement: no deep reasoning about exact distributions or complex conventions.
- Still captures the essential cooperative behavior:
  - Play when safe (Rule 1),
  - Make partner’s plays possible (Rule 2),
  - Throw away junk first (Rule 3),
  - Use hints to gradually label partner’s hand (Rule 4),
  - Use unhinted-oldest as a default discard heuristic (Rule 5).

This is a perfect candidate to serve as a **first blueprint** for your SPARTA-style search:

- You’ll get plenty of suboptimal decisions for search to improve.
- The policy is deterministic and interpretable, which helps with debugging beliefs and search.

Later, you can refine this (more sophisticated hint logic, better uselessness detection), or replace it with a trained GRU/LSTM policy—but this is a very easy starting point.
