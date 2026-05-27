// src/posts/ChessAgentPlayer.jsx
// Shared interactive widget for the Algorithms series:
// you play white, an algorithm plays black. The `agent` prop is a function
//   (chess, depth) -> { move, nodes }
// so the same UI can drop in Minimax / Alpha-Beta / MCTS in each post.
import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Chess } from 'chess.js';
import './ChessAgentPlayer.css';

/* ============================================================
   Built-in agents
   ============================================================ */

// Material-only evaluation, white-positive. Scan the FEN string instead of
// allocating an 8x8 array via chess.board() — fastest available eval.
const MATERIAL_VAL = { p: 1, n: 3, b: 3.1, r: 5, q: 9, k: 0 };
export function materialEval(chess) {
  const board = chess.fen();
  let score = 0;
  for (let i = 0; i < board.length; i++) {
    const ch = board[i];
    if (ch === ' ') break;            // end of board section in FEN
    if (ch === '/' || (ch >= '0' && ch <= '9')) continue;
    const lower = ch.toLowerCase();
    const v = MATERIAL_VAL[lower];
    if (v === undefined) continue;
    score += (ch === lower ? -v : v);
  }
  return score;
}

// Vanilla minimax (no pruning). Returns { move, nodes }.
// Performance notes:
//   - Use verbose move objects so chess.move() skips SAN parsing.
//   - Skip chess.isGameOver() (slow); detect via moves.length === 0 instead.
const MATE_SCORE = 1e6;
export function minimaxAgent(chess, depth) {
  let nodes = 0;
  const search = (c, d, maximizing) => {
    nodes++;
    if (d === 0) return materialEval(c);
    const moves = c.moves({ verbose: true });
    if (moves.length === 0) {
      // checkmate or stalemate
      if (c.inCheck()) return maximizing ? -MATE_SCORE : MATE_SCORE;
      return 0;
    }
    let best = maximizing ? -Infinity : Infinity;
    for (let i = 0; i < moves.length; i++) {
      c.move(moves[i]);
      const s = search(c, d - 1, !maximizing);
      c.undo();
      if (maximizing) { if (s > best) best = s; }
      else            { if (s < best) best = s; }
    }
    return best;
  };

  const isMax = chess.turn() === 'w';
  const moves = chess.moves({ verbose: true });
  // Shuffle for tie-breaking variety
  for (let i = moves.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [moves[i], moves[j]] = [moves[j], moves[i]];
  }
  let bestMove = null;
  let bestScore = isMax ? -Infinity : Infinity;
  for (let i = 0; i < moves.length; i++) {
    const m = moves[i];
    chess.move(m);
    const s = search(chess, depth - 1, !isMax);
    chess.undo();
    if (isMax ? s > bestScore : s < bestScore) {
      bestScore = s;
      bestMove = m;
    }
  }
  return { move: bestMove, nodes };
}

// Alpha-beta pruning — same answer as minimax, far fewer nodes evaluated
// in practice. Same signature so it drops into ChessAgentPlayer in place
// of `minimaxAgent`.
export function alphaBetaAgent(chess, depth) {
  let nodes = 0;
  const search = (c, d, alpha, beta, maximizing) => {
    nodes++;
    if (d === 0) return materialEval(c);
    const moves = c.moves({ verbose: true });
    if (moves.length === 0) {
      if (c.inCheck()) return maximizing ? -MATE_SCORE : MATE_SCORE;
      return 0;
    }
    if (maximizing) {
      let best = -Infinity;
      for (let i = 0; i < moves.length; i++) {
        c.move(moves[i]);
        const s = search(c, d - 1, alpha, beta, false);
        c.undo();
        if (s > best) best = s;
        if (best > alpha) alpha = best;
        if (alpha >= beta) break; // beta cutoff
      }
      return best;
    } else {
      let best = Infinity;
      for (let i = 0; i < moves.length; i++) {
        c.move(moves[i]);
        const s = search(c, d - 1, alpha, beta, true);
        c.undo();
        if (s < best) best = s;
        if (best < beta) beta = best;
        if (alpha >= beta) break; // alpha cutoff
      }
      return best;
    }
  };

  const isMax = chess.turn() === 'w';
  const moves = chess.moves({ verbose: true });
  for (let i = moves.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [moves[i], moves[j]] = [moves[j], moves[i]];
  }
  let bestMove = null;
  let bestScore = isMax ? -Infinity : Infinity;
  let alpha = -Infinity, beta = Infinity;
  for (let i = 0; i < moves.length; i++) {
    const m = moves[i];
    chess.move(m);
    const s = search(chess, depth - 1, alpha, beta, !isMax);
    chess.undo();
    if (isMax ? s > bestScore : s < bestScore) {
      bestScore = s;
      bestMove = m;
    }
    if (isMax) { if (bestScore > alpha) alpha = bestScore; }
    else       { if (bestScore < beta)  beta  = bestScore; }
  }
  return { move: bestMove, nodes };
}

// Monte Carlo Tree Search with random rollouts. Iteration count comes from
// `iterations` (positional 2nd arg, kept compatible with the (chess, depth)
// signature). UCB1 with c = sqrt(2). Rollouts cap at ROLLOUT_LIMIT plies and
// score by terminal result + material eval as a soft signal.
const UCB_C = Math.SQRT2;
const ROLLOUT_LIMIT = 10; // plies per rollout; trades rollout fidelity for UI responsiveness

export function mctsAgent(chess, iterations = 500) {
  let nodesVisited = 0;
  const rootColor = chess.turn();

  const makeNode = (parent, move) => ({
    parent, move,
    children: [],
    visits: 0,
    wins: 0,
    untried: null, // lazily populated
  });
  const root = makeNode(null, null);

  const isFullyExpanded = (node) => node.untried !== null && node.untried.length === 0;
  const ucb1 = (child, parentVisits) =>
    child.visits === 0 ? Infinity :
      (child.wins / child.visits) + UCB_C * Math.sqrt(Math.log(parentVisits) / child.visits);

  for (let it = 0; it < iterations; it++) {
    let node = root;
    const path = [node];
    let movesPlayed = 0;

    // SELECTION
    while (isFullyExpanded(node) && node.children.length > 0) {
      let best = null, bestScore = -Infinity;
      for (const c of node.children) {
        const s = ucb1(c, node.visits);
        if (s > bestScore) { bestScore = s; best = c; }
      }
      node = best;
      chess.move(node.move);
      movesPlayed++;
      path.push(node);
      nodesVisited++;
    }

    // EXPANSION
    if (node.untried === null) node.untried = chess.moves({ verbose: true });
    if (node.untried.length > 0) {
      const idx = Math.floor(Math.random() * node.untried.length);
      const move = node.untried.splice(idx, 1)[0];
      chess.move(move);
      movesPlayed++;
      const child = makeNode(node, move);
      node.children.push(child);
      node = child;
      path.push(node);
      nodesVisited++;
    }

    // SIMULATION (random rollout)
    let rolloutPlies = 0;
    let terminalNoMoves = false;
    while (rolloutPlies < ROLLOUT_LIMIT) {
      const moves = chess.moves({ verbose: true });
      if (moves.length === 0) { terminalNoMoves = true; break; }
      chess.move(moves[Math.floor(Math.random() * moves.length)]);
      rolloutPlies++;
    }

    // RESULT (from rootColor's perspective: 1 = win, 0 = loss, 0.5 = draw)
    let result;
    if (terminalNoMoves) {
      // No legal moves → checkmate (if in check) or stalemate
      result = chess.inCheck() ? (chess.turn() === rootColor ? 0 : 1) : 0.5;
    } else {
      // Depth-limited cutoff: soft eval from material
      const mat = materialEval(chess);
      const persp = rootColor === 'w' ? mat : -mat;
      if (persp > 0.5) result = 1;
      else if (persp < -0.5) result = 0;
      else result = 0.5;
    }

    // Undo rollout
    for (let i = 0; i < rolloutPlies; i++) chess.undo();

    // BACKPROPAGATION (from rootColor's perspective; flip for opposing nodes)
    for (let i = path.length - 1; i >= 0; i--) {
      path[i].visits++;
      // Each node represents the *position after* its move; the player to move at the
      // parent picked this move. So a "win for rootColor" is also a win at every node
      // in the line — we just attribute the same `result` everywhere. Simple and
      // sufficient for this educational MCTS.
      path[i].wins += result;
    }

    // Undo selection/expansion moves
    for (let i = 0; i < movesPlayed; i++) chess.undo();
  }

  // Choose the most-visited root child (robust against unlucky single rollouts)
  let bestChild = null, bestVisits = -1;
  for (const c of root.children) {
    if (c.visits > bestVisits) { bestVisits = c.visits; bestChild = c; }
  }
  return { move: bestChild ? bestChild.move : null, nodes: nodesVisited };
}

/* ============================================================
   Inline chess board (SVG, unicode pieces, click-to-move)
   ============================================================ */
const PIECE_GLYPH = {
  K: '♚', Q: '♛', R: '♜', B: '♝', N: '♞', P: '♟',
  k: '♚', q: '♛', r: '♜', b: '♝', n: '♞', p: '♟',
};
function squareName(col, row) {
  return `${'abcdefgh'[col]}${8 - row}`;
}
function PlayableBoard({ chess, onSquareClick, selected, legalDests, lastMove, hintMove, size = 360, flipped = false }) {
  const sq = size / 8;
  const board = chess.board(); // 8x8, row 0 = rank 8
  const view = (r, c) => flipped ? [7 - r, 7 - c] : [r, c];
  return (
    <svg
      viewBox={`0 0 ${size} ${size}`}
      width="100%" height={size}
      className="cap-board"
      preserveAspectRatio="xMidYMid meet"
    >
      {Array.from({ length: 64 }).map((_, idx) => {
        const r = Math.floor(idx / 8);
        const c = idx % 8;
        const [br, bc] = view(r, c);
        const piece = board[br][bc];
        const sqId = squareName(bc, br);
        const isLight = (r + c) % 2 === 0;
        const isSel = selected === sqId;
        const isDest = legalDests && legalDests.includes(sqId);
        const isLast = lastMove && (lastMove.from === sqId || lastMove.to === sqId);
        const isHintFrom = hintMove && hintMove.from === sqId;
        const isHintTo   = hintMove && hintMove.to   === sqId;
        const isWhite = piece && piece.color === 'w';
        return (
          <g key={idx} onClick={() => onSquareClick(sqId)} style={{ cursor: 'pointer' }}>
            <rect
              x={c * sq} y={r * sq} width={sq} height={sq}
              fill={isLight ? '#eed9b6' : '#b58863'}
            />
            {isLast && (
              <rect x={c * sq} y={r * sq} width={sq} height={sq}
                fill="hsl(48, 95%, 60%)" fillOpacity="0.32" />
            )}
            {(isHintFrom || isHintTo) && (
              <rect x={c * sq} y={r * sq} width={sq} height={sq}
                fill="hsl(150, 75%, 45%)" fillOpacity={isHintTo ? 0.45 : 0.32} />
            )}
            {isSel && (
              <rect x={c * sq} y={r * sq} width={sq} height={sq}
                fill="hsl(218, 80%, 52%)" fillOpacity="0.34" />
            )}
            {isDest && !piece && (
              <circle cx={c * sq + sq / 2} cy={r * sq + sq / 2} r={sq * 0.16}
                fill="hsl(218, 80%, 52%)" opacity="0.42" />
            )}
            {isDest && piece && (
              <circle cx={c * sq + sq / 2} cy={r * sq + sq / 2} r={sq * 0.46}
                fill="none" stroke="hsl(218, 80%, 52%)" strokeWidth="3" opacity="0.7" />
            )}
            {piece && (
              <text
                x={c * sq + sq / 2}
                y={r * sq + sq * 0.74}
                fontSize={sq * 0.78}
                textAnchor="middle"
                fill={isWhite ? '#fafafa' : '#101418'}
                stroke={isWhite ? '#101418' : 'none'}
                strokeWidth={isWhite ? 0.8 : 0}
                style={{ paintOrder: 'stroke', pointerEvents: 'none' }}
              >
                {PIECE_GLYPH[piece.type.toUpperCase()]}
              </text>
            )}
          </g>
        );
      })}
      {/* Border */}
      <rect x="0" y="0" width={size} height={size} fill="none" stroke="#0f172a" strokeOpacity="0.18" />
    </svg>
  );
}

/* ============================================================
   ChessAgentPlayer — main exported component
   ============================================================ */
export default function ChessAgentPlayer({
  agent = minimaxAgent,
  algorithmName = 'minimax',
  initialDepth = 2,
  minDepth = 1,
  maxDepth = 3,
  depthStep = 1,
  depthLabel = 'Agent depth',
  depthFormatter = (d) => d,
  intro,
}) {
  const chessRef = useRef(null);
  if (!chessRef.current) chessRef.current = new Chess();
  const chess = chessRef.current;

  const [fen, setFen] = useState(chess.fen());
  const [selected, setSelected] = useState(null);
  const [legalDests, setLegalDests] = useState([]);
  const [history, setHistory] = useState([]); // {san, color, by}
  const [thinking, setThinking] = useState(false);
  const [depth, setDepth] = useState(initialDepth);
  const [lastNodes, setLastNodes] = useState(null);
  const [lastMs, setLastMs] = useState(null);
  const [hintMove, setHintMove] = useState(null);
  const [simulating, setSimulating] = useState(false);
  const simTimerRef = useRef(null);

  // Re-render bump
  const refresh = useCallback(() => {
    setFen(chess.fen());
  }, [chess]);

  // Apply a move (object {from, to, promotion} or san string), record history
  const applyMove = useCallback((mv, by) => {
    let made = null;
    try {
      made = chess.move(mv);
    } catch (err) {
      // chess.js v1 throws on illegal moves; treat as a no-op
      return null;
    }
    if (!made) return null;
    setHistory((h) => [...h, { san: made.san, color: made.color, by }]);
    refresh();
    return made;
  }, [chess, refresh]);

  // Agent turn — runs after the user's move.
  // IMPORTANT: do NOT include `thinking` in the deps array. setThinking(true)
  // would re-run the effect, the cleanup would clearTimeout the queued agent
  // invocation, and the agent would never execute.
  useEffect(() => {
    if (simulating) return; // simulate loop drives both colors instead
    if (chess.turn() !== 'b' || chess.isGameOver()) return;
    setThinking(true);
    // setTimeout so the UI paints "thinking…" before the search blocks the thread
    const t = setTimeout(() => {
      try {
        const t0 = performance.now();
        const { move, nodes } = agent(chess, depth);
        const ms = performance.now() - t0;
        if (move) applyMove(move, 'agent');
        setLastNodes(nodes);
        setLastMs(ms);
      } catch (err) {
        // Surface to console so the post doesn't silently freeze on a bug
        // eslint-disable-next-line no-console
        console.error('Agent error:', err);
      } finally {
        setThinking(false);
      }
    }, 60);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fen, depth]);

  const isGameOver = chess.isGameOver();
  const turn = chess.turn();
  const inCheck = chess.inCheck();

  const handleSquareClick = (sqId) => {
    if (thinking || simulating || isGameOver || turn !== 'w') return;
    if (selected && legalDests.includes(sqId)) {
      // Try the move (auto-queen on promotion)
      const moved = applyMove({ from: selected, to: sqId, promotion: 'q' }, 'human');
      if (moved) {
        setSelected(null);
        setLegalDests([]);
        setHintMove(null);
        return;
      }
    }
    // Toggle / re-select
    const piece = chess.get(sqId);
    if (piece && piece.color === 'w') {
      setSelected(sqId);
      const moves = chess.moves({ square: sqId, verbose: true });
      setLegalDests(moves.map((m) => m.to));
    } else {
      setSelected(null);
      setLegalDests([]);
    }
  };

  // Engine hint — compute the engine's pick for whoever is to move
  const showHint = useCallback(() => {
    if (thinking || simulating || isGameOver) return;
    setThinking(true);
    setTimeout(() => {
      try {
        const t0 = performance.now();
        const { move, nodes } = agent(chess, depth);
        const ms = performance.now() - t0;
        if (move) setHintMove({ from: move.from, to: move.to });
        setLastNodes(nodes);
        setLastMs(ms);
      } catch (err) {
        // eslint-disable-next-line no-console
        console.error('Hint error:', err);
      } finally {
        setThinking(false);
      }
    }, 30);
  }, [agent, chess, depth, thinking, simulating, isGameOver]);

  // Engine-vs-engine simulation
  const stopSimulate = useCallback(() => {
    if (simTimerRef.current) {
      clearTimeout(simTimerRef.current);
      simTimerRef.current = null;
    }
    setSimulating(false);
    setThinking(false);
  }, []);

  const startSimulate = useCallback(() => {
    if (thinking || simulating || isGameOver) return;
    setHintMove(null);
    setSelected(null);
    setLegalDests([]);
    setSimulating(true);

    const SIM_DELAY = 250;
    const PLY_CAP = 200;

    const step = () => {
      if (chess.isGameOver() || chess.history().length >= PLY_CAP) {
        setSimulating(false);
        setThinking(false);
        return;
      }
      setThinking(true);
      // setTimeout(..., 0) so React paints "thinking" before the search blocks
      simTimerRef.current = setTimeout(() => {
        try {
          const t0 = performance.now();
          const { move, nodes } = agent(chess, depth);
          const ms = performance.now() - t0;
          if (move) applyMove(move, 'agent');
          setLastNodes(nodes);
          setLastMs(ms);
        } catch (err) {
          // eslint-disable-next-line no-console
          console.error('Simulate error:', err);
        } finally {
          setThinking(false);
          simTimerRef.current = setTimeout(step, SIM_DELAY);
        }
      }, 30);
    };
    step();
  }, [agent, applyMove, chess, depth, thinking, simulating, isGameOver]);

  // Clean up sim timer on unmount
  useEffect(() => () => { if (simTimerRef.current) clearTimeout(simTimerRef.current); }, []);

  const newGame = () => {
    stopSimulate();
    chess.reset();
    setHistory([]);
    setSelected(null);
    setLegalDests([]);
    setLastNodes(null);
    setLastMs(null);
    setHintMove(null);
    refresh();
  };
  const undo = () => {
    if (thinking || simulating) return;
    // Undo agent move + your move so it's your turn again
    chess.undo(); chess.undo();
    setHistory((h) => h.slice(0, -2));
    setSelected(null);
    setLegalDests([]);
    setHintMove(null);
    refresh();
  };

  const lastMove = useMemo(() => {
    const arr = chess.history({ verbose: true });
    return arr.length ? arr[arr.length - 1] : null;
  }, [fen, chess]);

  const status = useMemo(() => {
    if (chess.isCheckmate()) return turn === 'w' ? '✗ Checkmate — you lose' : '✓ Checkmate — you win';
    if (chess.isStalemate()) return '½ Stalemate — draw';
    if (chess.isInsufficientMaterial()) return '½ Insufficient material — draw';
    if (chess.isThreefoldRepetition()) return '½ Threefold repetition — draw';
    if (chess.isDraw()) return '½ Draw';
    if (simulating) return thinking ? 'Simulating · engine thinking…' : 'Simulating · engine vs engine';
    if (thinking) return 'Agent is thinking…';
    if (inCheck) return turn === 'w' ? '⚠ You are in check' : '⚠ Agent is in check';
    return turn === 'w' ? 'Your move (white)' : 'Agent to move (black)';
  }, [chess, turn, inCheck, thinking, simulating, fen]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="viz-panel cap-root">
      {intro && <p className="cap-intro">{intro}</p>}
      <div className="cap-grid">
        <div className="cap-board-wrap">
          <PlayableBoard
            chess={chess}
            onSquareClick={handleSquareClick}
            selected={selected}
            legalDests={legalDests}
            lastMove={lastMove}
            hintMove={hintMove}
            size={360}
          />
        </div>
        <div className="cap-side">
          <div className={`cap-status${inCheck ? ' is-warn' : ''}${isGameOver ? ' is-over' : ''}`}>
            {status}
          </div>

          <label className="cap-slider">
            <span>{depthLabel}</span>
            <input
              type="range" min={minDepth} max={maxDepth} step={depthStep}
              value={depth}
              onChange={(e) => setDepth(+e.target.value)}
              disabled={thinking}
            />
            <span className="cap-val">{depthFormatter(depth)}</span>
          </label>

          <div className="cap-stats">
            <div className="cap-stat">
              <div className="cap-stat-label">Algorithm</div>
              <div className="cap-stat-value">{algorithmName}</div>
            </div>
            <div className="cap-stat">
              <div className="cap-stat-label">Last search · nodes</div>
              <div className="cap-stat-value cap-num">
                {lastNodes !== null ? lastNodes.toLocaleString('en-US') : '—'}
              </div>
            </div>
            <div className="cap-stat">
              <div className="cap-stat-label">Last search · time</div>
              <div className="cap-stat-value cap-num">
                {lastMs !== null ? `${lastMs.toFixed(0)} ms` : '—'}
              </div>
            </div>
          </div>

          <div className="cap-actions">
            <button type="button" className="cap-btn cap-btn-primary" onClick={newGame}>↻ New game</button>
            <button type="button" className="cap-btn" onClick={undo}
              disabled={thinking || simulating || history.length < 2}>↶ Undo</button>
          </div>
          <div className="cap-actions">
            {hintMove ? (
              <button type="button" className="cap-btn" onClick={() => setHintMove(null)}>✕ Clear hint</button>
            ) : (
              <button type="button" className="cap-btn"
                onClick={showHint}
                disabled={thinking || simulating || isGameOver}>★ Engine's pick</button>
            )}
            {simulating ? (
              <button type="button" className="cap-btn" onClick={stopSimulate}>■ Stop</button>
            ) : (
              <button type="button" className="cap-btn"
                onClick={startSimulate}
                disabled={thinking || isGameOver}>▶ Simulate</button>
            )}
          </div>

          <div className="cap-history" aria-label="Move history">
            {history.length === 0 ? (
              <span className="cap-history-empty">No moves yet — click a white piece to start.</span>
            ) : (
              history.reduce((rows, mv, i) => {
                if (i % 2 === 0) rows.push([mv, null]);
                else rows[rows.length - 1][1] = mv;
                return rows;
              }, []).map(([w, b], i) => (
                <div className="cap-history-row" key={i}>
                  <span className="cap-history-num">{i + 1}.</span>
                  <span className="cap-history-w">{w?.san}</span>
                  <span className="cap-history-b">{b?.san ?? ''}</span>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
      <p className="viz-caption">
        You play white; the agent plays black. Each agent reply runs <em>{algorithmName}</em>{' '}
        from scratch at the chosen depth — slide the slider to feel how the search work
        scales. <strong>Depth 3</strong> already looks ahead three plies (your move →
        agent's reply → your reply).
      </p>
    </div>
  );
}
