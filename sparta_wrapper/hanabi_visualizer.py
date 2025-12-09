"""Run a GRU blueprint game and expose a simple localhost GUI to step through moves.

The server plays one full game with two GRU agents, captures an omniscient log
plus per-player observations, then serves a minimal HTML/JS viewer on
http://localhost:PORT (default 8000).
"""

from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
from typing import Callable, Dict, List

import torch
from hanabi_learning_environment import pyhanabi, rl_env

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # pragma: no branch
    sys.path.insert(0, str(ROOT))

from sparta_wrapper.hanabi_utils import (
    _card_to_dict,
    _hand_to_dict,
    _fireworks_to_dict,
    _move_to_action_dict,
    build_observation,
    HanabiObservation,
)
from sparta_wrapper.naive_gru_blueprint import GRU_CFG, HanabiGRUBlueprint

DEFAULT_RANK_MULT = [3, 2, 2, 2, 1]


# ----------------------------- Serialization -------------------------------- #
def _state_to_dict(state: pyhanabi.HanabiState) -> Dict:
    return {
        "current_player": state.cur_player(),
        "score": state.score(),
        "deck_size": state.deck_size(),
        "information_tokens": state.information_tokens(),
        "life_tokens": state.life_tokens(),
        "fireworks": _fireworks_to_dict(state.fireworks()),
        "discard_pile": [_card_to_dict(c) for c in state.discard_pile()],
        "hands_full": [[_card_to_dict(c) for c in hand] for hand in state.player_hands()],
    }


def _obs_to_dict(obs: HanabiObservation) -> Dict:
    # Compute remaining card counts (seen hands + discard + fireworks) to prune plausibility
    num_colors = len(obs.fireworks)
    num_ranks = max(len(DEFAULT_RANK_MULT), 5)
    counts = [[DEFAULT_RANK_MULT[r] if r < len(DEFAULT_RANK_MULT) else DEFAULT_RANK_MULT[-1] for r in range(num_ranks)] for _ in range(num_colors)]

    def _color_idx(c):
        if c is None:
            return None
        if isinstance(c, int):
            return c
        try:
            return pyhanabi.color_char_to_idx(c)
        except Exception:
            return None

    def _dec(card: Dict):
        ci = _color_idx(card.get("color"))
        r = card.get("rank")
        if ci is None or r is None:
            return
        if 0 <= ci < num_colors and 0 <= r < num_ranks and counts[ci][r] > 0:
            counts[ci][r] -= 1

    for hand in obs.observed_hands:
        for card in hand:
            _dec(card)
    for card in obs.discard_pile:
        _dec(card)
    for color_char, level in obs.fireworks.items():
        ci = _color_idx(color_char)
        for r in range(min(level, num_ranks)):
            _dec({"color": ci, "rank": r})

    pruned_knowledge = []
    for k in obs.card_knowledge:
        row = []
        for card_info in k:
            raw_mask = card_info.get("mask", [])
            mask_pruned = [
                [
                    bool(raw_mask[c][r] and counts[c][r] > 0) if c < len(raw_mask) and r < len(raw_mask[c]) else False
                    for r in range(num_ranks)
                ]
                for c in range(num_colors)
            ]
            card_info = dict(card_info)
            card_info["mask_pruned"] = mask_pruned
            row.append(card_info)
        pruned_knowledge.append(row)

    # Build a player-indexed view of hands/knowledge; pyhanabi orders them starting at observer.
    num_players = len(pruned_knowledge)
    player_ordered_hands: List[List[Dict[str, Any]]] = [None for _ in range(num_players)]  # type: ignore
    player_ordered_knowledge: List[List[Dict[str, Any]]] = [None for _ in range(num_players)]  # type: ignore
    for i, hand in enumerate(obs.observed_hands):
        pid = (obs.player_id + i) % num_players
        player_ordered_hands[pid] = hand
        player_ordered_knowledge[pid] = pruned_knowledge[i]

    return {
        "player_id": obs.player_id,
        "current_player": obs.current_player,
        "current_player_offset": obs.current_player_offset,
        "observed_hands": player_ordered_hands,
        "card_knowledge": player_ordered_knowledge,
        "discard_pile": obs.discard_pile,
        "fireworks": obs.fireworks,
        "deck_size": obs.deck_size,
        "information_tokens": obs.information_tokens,
        "life_tokens": obs.life_tokens,
        "legal_moves": obs.legal_moves_dict,
        "last_moves": obs.last_moves,
    }


def _play_one_game(blueprint_factory: Callable[[], HanabiGRUBlueprint]) -> Dict:
    env = rl_env.HanabiEnv({"players": 2})
    env.reset()
    state = env.state

    blueprints = [blueprint_factory(), blueprint_factory()]

    trajectory: List[Dict] = []
    # Initial snapshot before any move
    trajectory.append(
        {
            "turn": 0,
            "actor": state.cur_player(),
            "action": None,
            "state": _state_to_dict(state),
            "views": [_obs_to_dict(build_observation(state, pid)) for pid in range(state.num_players())],
        }
    )

    turn = 0
    while not state.is_terminal():
        pid = state.cur_player()
        obs = build_observation(state, pid)
        action_move = blueprints[pid].act(obs)
        action_dict = _move_to_action_dict(action_move)
        # Track effects
        fireworks_before = list(state.fireworks())
        discard_before = list(state.discard_pile())
        played_card = None
        if action_move.type() in (pyhanabi.HanabiMoveType.PLAY, pyhanabi.HanabiMoveType.DISCARD):
            try:
                played_card = state.player_hands()[pid][action_move.card_index()]
            except Exception:
                played_card = None
        _, _, done, info = env.step(action_dict)
        state = env.state
        turn += 1
        fireworks_after = list(state.fireworks())
        discard_after = list(state.discard_pile())
        fire_highlight = []
        for i, (b, a) in enumerate(zip(fireworks_before, fireworks_after)):
            if a > b:
                fire_highlight.append(pyhanabi.COLOR_CHAR[i])
        discard_highlight = []
        if len(discard_after) > len(discard_before):
            new_cards = discard_after[len(discard_before):]
            discard_highlight = [_card_to_dict(c) for c in new_cards]
        elif played_card is not None and action_move.type() == pyhanabi.HanabiMoveType.DISCARD:
            discard_highlight = [_card_to_dict(played_card)]
        effects = {
            "fireworks": fire_highlight,
            "discard": discard_highlight,
            "action_type": action_dict.get("action_type"),
            "actor": pid,
        }
        if action_move.type() in (pyhanabi.HanabiMoveType.PLAY, pyhanabi.HanabiMoveType.DISCARD) and played_card is not None:
            effects["played_card"] = _card_to_dict(played_card)
        if action_move.type() in (pyhanabi.HanabiMoveType.REVEAL_COLOR, pyhanabi.HanabiMoveType.REVEAL_RANK):
            effects["hint_target"] = (pid + action_move.target_offset()) % state.num_players()
            if action_move.type() == pyhanabi.HanabiMoveType.REVEAL_COLOR:
                effects["hint_color"] = pyhanabi.COLOR_CHAR[action_move.color()]
            if action_move.type() == pyhanabi.HanabiMoveType.REVEAL_RANK:
                effects["hint_rank"] = action_move.rank()

        trajectory.append(
            {
                "turn": turn,
                "actor": pid,
                "action": action_dict,
                "state": _state_to_dict(state),
                "views": [_obs_to_dict(build_observation(state, p)) for p in range(state.num_players())],
                "score": float(info.get("score", state.score())),
                "effects": effects,
            }
        )
        if done:
            break

    return {
        "players": state.num_players(),
        "trajectory": trajectory,
        "final_score": trajectory[-1].get("score", state.score()),
    }


# ------------------------------- Web server -------------------------------- #
HTML_PAGE = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Hanabi GRU Visualizer</title>
  <style>
    body { font-family: sans-serif; margin: 1rem; line-height: 1.4; background:#0d1117; color:#e6edf3; }
    .row { display: flex; gap: 1rem; flex-wrap: wrap; }
    .card { padding: 0.4rem 0.6rem; border: 1px solid #30363d; border-radius: 4px; margin: 0.1rem; background:#161b22; }
    .section { border: 1px solid #30363d; padding: 0.75rem; border-radius: 6px; margin-bottom: 1rem; background:#0f1620; }
    .label { font-weight: bold; color:#f0f6fc; }
    button { padding: 0.4rem 0.8rem; margin-right: 0.4rem; background:#238636; color:#fff; border:1px solid #2ea043; border-radius:4px; cursor:pointer; }
    button:hover { background:#2ea043; }
    input, select { padding: 0.25rem; background:#161b22; color:#e6edf3; border:1px solid #30363d; border-radius:4px; }
    .c-R { background:#3d0d0d; border-color:#a54545; color:#ffb3b3; }
    .c-Y { background:#3d350d; border-color:#a58c45; color:#ffe9a6; }
    .c-G { background:#0d3d1a; border-color:#45a55b; color:#b3ffd1; }
    .c-B { background:#0d233d; border-color:#457aa5; color:#b3d4ff; }
    .c-W { background:#1e1e1e; border-color:#5a5a5a; color:#f0f6fc; }
    .pill { padding:0.25rem 0.45rem; border-radius:4px; border:1px solid #30363d; margin:0.1rem; display:inline-block; background:#161b22; }
    .mask-table { border-collapse: collapse; margin-top:6px; }
    .mask-table td { border:1px solid #30363d; width:20px; height:20px; text-align:center; font-size:11px; color:#6e7681; background:#0f1620; }
    .mask-table td.on { color:#fff; }
    .mask-table td.c-R { background:#3d0d0d; border-color:#a54545; color:#ffb3b3; }
    .mask-table td.c-Y { background:#3d350d; border-color:#a58c45; color:#ffe9a6; }
    .mask-table td.c-G { background:#0d3d1a; border-color:#45a55b; color:#b3ffd1; }
    .mask-table td.c-B { background:#0d233d; border-color:#457aa5; color:#b3d4ff; }
    .mask-table td.c-W { background:#1e1e1e; border-color:#5a5a5a; color:#f0f6fc; }
    .fireworks-strip { display:flex; gap:10px; align-items:center; justify-content:flex-start; }
    .firework-pill { min-width:70px; text-align:center; font-size:24px; padding:10px 14px; border-radius:10px; border:2px solid #30363d; background:#161b22; }
    .highlight { box-shadow: 0 0 10px 2px #ffd700; border-color:#ffd700 !important; }
  </style>
</head>
<body>
  <h2>Hanabi GRU Visualizer</h2>
  <div class="section">
    <button id="prevBtn">&larr; Prev</button>
    <button id="nextBtn">Next &rarr;</button>
    <input id="stepSlider" type="range" min="0" max="0" value="0" style="width:300px;">
    <span id="stepLabel">Step 0</span>
    <label style="margin-left:1rem;">
      <input type="checkbox" id="omniscientToggle"> Show actual hands
    </label>
  </div>
  <div class="section" id="fireworksStrip"></div>
  <div class="section" id="summary"></div>
  <div class="section" id="hands"></div>
  <div class="section" id="board"></div>
  <div class="section" id="lastMoves"></div>
  <script>
    let data = null;
    let step = 0;
    const stepSlider = document.getElementById('stepSlider');
    const stepLabel = document.getElementById('stepLabel');
    const omniscientToggle = document.getElementById('omniscientToggle');

    function render() {
      if (!data) return;
      const traj = data.trajectory;
      const entry = traj[step];
      stepLabel.textContent = `Step ${step} / ${traj.length - 1} (actor P${entry.actor}${entry.action ? '' : ' [initial]'})`;

      // Summary
      const s = entry.state;
      const effects = entry.effects || {};
      document.getElementById('fireworksStrip').innerHTML = renderFireworksStrip(s.fireworks, effects.fireworks || []);
      document.getElementById('summary').innerHTML = `
        <div><span class="label">Score:</span> ${s.score} | <span class="label">Deck:</span> ${s.deck_size} | <span class="label">Info:</span> ${s.information_tokens} | <span class="label">Lives:</span> ${s.life_tokens}</div>
        <div><span class="label">Fireworks:</span> ${Object.entries(s.fireworks).map(([c,v]) => c+':'+v).join(' ')}</div>
        <div><span class="label">Discard:</span> ${renderDiscard(s.discard_pile, effects.discard || [])}</div>
        <div><span class="label">Action:</span> ${renderAction(entry.action, entry.actor, effects)}</div>
      `;

      // Per-player perspectives side-by-side
      const views = entry.views;
      const handsDiv = document.getElementById('hands');
      let html = '<div class="label">Per-player perspectives</div><div class="row">';
      views.forEach((v, idx) => {
        html += renderPerspective(v, idx, s, omniscientToggle.checked, effects);
      });
      html += '</div>';
      handsDiv.innerHTML = html;

      // Shared board info and per-player legal moves
      let boardHtml = `
        <div><span class="label">Deck:</span> ${s.deck_size} | <span class="label">Info:</span> ${s.information_tokens} | <span class="label">Lives:</span> ${s.life_tokens}</div>
        <div><span class="label">Fireworks:</span> ${Object.entries(s.fireworks).map(([c,v]) => `<span class="pill c-${c}">${c}:${v}</span>`).join(' ')}</div>
        <div><span class="label">Discard:</span> ${renderDiscard(s.discard_pile)}</div>
      `;
      boardHtml += '<div class="row">';
      views.forEach((v, idx) => {
        boardHtml += `<div class="section"><div class="label">Legal moves for P${idx} (current player: P${v.current_player})</div>`;
        boardHtml += v.legal_moves.map(m => m.action_type + (m.card_index !== undefined ? ' idx=' + m.card_index : '') + (m.color ? ' color=' + m.color : '') + (m.rank !== undefined ? ' rank=' + m.rank : '')).join(' | ');
        boardHtml += '</div>';
      });
      boardHtml += '</div>';
      document.getElementById('board').innerHTML = boardHtml;

      // Last moves (shared)
      const lm = (views[0] && views[0].last_moves) || [];
      document.getElementById('lastMoves').innerHTML = '<div class="label">Last moves (most recent first):</div>' + lm.map(m => `<div>P${m.player}: ${JSON.stringify(m.move)}</div>`).join('');
    }

    function renderCard(c, highlighted=false, obfuscate=false) {
      let color = c.color;
      let rank = c.rank;
      if (obfuscate) {
        // Never reveal own-hand contents; show placeholders
        color = '?';
        rank = '?';
      } else {
        color = color || '?';
        rank = rank !== null && rank !== undefined ? rank + 1 : '?';
      }
      const cls = color ? 'c-' + color : '';
      const hl = highlighted ? ' highlight' : '';
      return `<span class="card ${cls}${hl}">${color}${rank}</span>`;
    }

    function renderMask(mask, colorLabels, rankLabels) {
      if (!mask || !mask.length) return '';
      colorLabels = colorLabels || [];
      rankLabels = rankLabels || [];
      let html = '<table class="mask-table">';
      for (let r = 0; r < mask.length; r++) {
        const rowLabel = colorLabels[r] || '';
        html += `<tr>`;
        for (let c = 0; c < mask[r].length; c++) {
          const on = mask[r][c];
          const rankVal = rankLabels[c] || (c + 1);
          const color = colorLabels[r] || '';
          const cls = `c-${color} ${on ? 'on' : ''}`;
          const style = on ? '' : 'opacity:0.35;';
          html += `<td class="${cls}" style="${style}" title="Color ${color || r}, Rank ${rankVal}">${rankVal}</td>`;
        }
        html += '</tr>';
      }
      html += '</table>';
      return html;
    }

    function renderDiscard(discard, highlights) {
      if (!discard || !discard.length) return '-';
      const order = ['R','Y','G','B','W'];
      const buckets = {};
      discard.forEach(c => {
        const col = c.color || '?';
        if (!buckets[col]) buckets[col] = [];
        buckets[col].push(c.rank);
      });
      // Sort ranks and stack duplicates
      order.forEach(col => { if (buckets[col]) buckets[col].sort((a,b)=>a-b); });
      const others = Object.keys(buckets).filter(k => !order.includes(k));
      let html = '';
      const highlightMap = {};
      (highlights || []).forEach(c => {
        const key = `${c.color||'?'}:${c.rank}`;
        highlightMap[key] = (highlightMap[key] || 0) + 1;
      });
      const addCol = col => {
        const ranks = buckets[col] || [];
        if (!ranks.length) return;
        const counts = {};
        ranks.forEach(r => { counts[r] = (counts[r] || 0) + 1; });
        Object.entries(counts).forEach(([rank, count]) => {
          const key = `${col}:${parseInt(rank,10)}`;
          const hlCount = Math.min(count, highlightMap[key] || 0);
          const cls = hlCount > 0 ? 'card c-' + col + ' highlight' : 'card c-' + col;
          html += `<span class="${cls}" title="${col}${parseInt(rank,10)+1} x${count}">${col}${parseInt(rank,10)+1}`;
          if (count > 1) html += `×${count}`;
          html += `</span>`;
          if (hlCount > 0) highlightMap[key] -= hlCount;
        });
      };
      order.forEach(addCol);
      others.forEach(addCol);
      return html || '-';
    }

    function renderPerspective(view, seatIdx, state, showOmniscient, effects) {
      let html = `<div class="section" style="min-width:280px;"><div class="label">P${seatIdx} perspective (current player: P${view.current_player})</div>`;
      html += '<div class="label">Observed hands</div><div class="row">';
      // Use omniscient hand info for highlighting; hide own values in renderCard via obfuscate flag.
      const handsForView = view.observed_hands.map((hand, idx) => state.hands_full[idx] || hand);
      handsForView.forEach((hand, idx) => {
        const hintMarks = computeHintMarks(hand, idx, effects);
        html += `<div class="section"><div class="label">P${idx}${effects && effects.hint_target === idx ? ' (hinted)' : ''}</div>`;
        html += hand.map((c, i) => renderCard(c, hintMarks[i], idx === view.player_id)).join('');
        html += '</div>';
      });
      html += '</div>';
      html += '<div class="label">Knowledge about own hand</div><div class="row">';
      view.card_knowledge[seatIdx].forEach((k, idx) => {
        const colors = Object.keys(view.fireworks);
        const ranks = k.mask && k.mask[0] ? k.mask[0].map((_, i) => i + 1) : [];
        html += `<div class="section"><div class="label">Card ${idx}</div>`;
        html += renderCard({color:k.color, rank:k.rank});
        const mask = k.mask_pruned || k.mask;
        html += renderMask(mask, colors, ranks);
        html += '</div>';
      });
      html += '</div>';
      if (showOmniscient) {
        html += '<div class="label">Actual hands</div><div class="row">';
        state.hands_full.forEach((hand, idx) => {
          const hintMarks = computeHintMarks(hand, idx, effects);
          html += `<div class="section"><div class="label">P${idx}</div>`;
          html += hand.map((c, i) => renderCard(c, hintMarks[i])).join('');
          html += '</div>';
        });
        html += '</div>';
      }
      html += '</div>';
      return html;
    }

    function renderFireworksStrip(fireworks, highlights) {
      const order = ['R','Y','G','B','W'];
      let html = '<div class="fireworks-strip">';
      order.forEach(c => {
        if (fireworks[c] === undefined) return;
        const cls = (highlights || []).includes(c) ? `firework-pill c-${c} highlight` : `firework-pill c-${c}`;
        html += `<div class="${cls}">${c}:${fireworks[c]}</div>`;
      });
      html += '</div>';
      return html;
    }

    function renderAction(action, actor, effects) {
      if (!action) return 'Initial state';
      const at = action.action_type;
      if (at === 'PLAY' || at === 'DISCARD') {
        const cardHtml = effects && effects.played_card ? renderCard(effects.played_card, true, false) : `idx=${action.card_index}`;
        return `P${actor} ${at.toLowerCase()}s ${cardHtml}`;
      }
      if (at === 'REVEAL_COLOR') {
        const target = effects && effects.hint_target !== undefined ? ` -> P${effects.hint_target}` : '';
        const badge = `<span class="card c-${action.color}">color ${action.color}</span>`;
        return `P${actor} hints ${badge}${target}`;
      }
      if (at === 'REVEAL_RANK') {
        const target = effects && effects.hint_target !== undefined ? ` -> P${effects.hint_target}` : '';
        const rankDisp = (action.rank + 1);
        const badge = `<span class="card">rank ${rankDisp}</span>`;
        return `P${actor} hints ${badge}${target}`;
      }
      return `P${actor} ${JSON.stringify(action)}`;
    }

    function computeHintMarks(hand, handIdx, effects) {
      if (!effects || effects.hint_target === undefined || effects.hint_target !== handIdx) {
        return Array(hand.length).fill(false);
      }
      return hand.map(card => {
        if (effects.hint_color !== undefined && card.color === effects.hint_color) return true;
        if (effects.hint_rank !== undefined && card.rank === effects.hint_rank) return true;
        return false;
      });
    }

    function setStep(v) {
      step = Math.max(0, Math.min(v, data.trajectory.length - 1));
      stepSlider.value = step;
      render();
    }

    document.getElementById('prevBtn').onclick = () => setStep(step - 1);
    document.getElementById('nextBtn').onclick = () => setStep(step + 1);
    stepSlider.oninput = e => setStep(parseInt(e.target.value, 10));
    omniscientToggle.onchange = render;

    fetch('/data').then(r => r.json()).then(d => {
      data = d;
      stepSlider.max = d.trajectory.length - 1;
      render();
    }).catch(err => {
      document.body.innerHTML = '<pre>Failed to load data: ' + err + '</pre>';
    });
  </script>
</body>
</html>
"""


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # pragma: no cover - quiet logging
        return

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/index"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
            return
        if self.path.startswith("/data"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(self.server.payload).encode("utf-8"))
            return
        self.send_response(404)
        self.end_headers()


def _start_server(host: str, port: int, payload: Dict) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), _Handler)
    server.payload = payload  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def main() -> None:
    ap = argparse.ArgumentParser(description="Serve a simple Hanabi GRU GUI on localhost.")
    ap.add_argument("--ckpt", type=str, default="gru_checkpoints/ckpt_020000.pt", help="Checkpoint path.")
    ap.add_argument("--device", type=str, default="cpu", help="Device for GRU blueprint (cpu|cuda).")
    ap.add_argument("--port", type=int, default=8000, help="Port for the local web server.")
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt).resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    blueprint_factory = lambda: HanabiGRUBlueprint(GRU_CFG(), ckpt_path, device=args.device)
    payload = _play_one_game(blueprint_factory)

    server = _start_server("127.0.0.1", args.port, payload)
    print(f"Serving Hanabi GUI on http://127.0.0.1:{args.port}")
    print("Press Ctrl+C to stop.")
    try:
        while True:
            threading.Event().wait(1.0)
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
