// src/posts/MinimaxSearch.jsx
// Algorithms · Part 1 of 3 — Minimax Search.
// Ported from the original 2023 markdown post; prose preserved, visuals reimagined
// as interactive widgets, design tokens aligned with the redesigned site.
import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import usePageMeta from '../usePageMeta';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneLight } from 'react-syntax-highlighter/dist/esm/styles/prism';
import ChessAgentPlayer, { minimaxAgent } from './ChessAgentPlayer';
import './PostChrome.css';
import './MinimaxSearch.css';

/* ============================================================
   Tiny inline chess board (SVG, unicode pieces)
   board: 8 strings of 8 chars (rank 8 → rank 1).
   Uppercase = white, lowercase = black, '.' = empty.
   ============================================================ */
// Use the *filled* black-piece glyphs (U+265A–U+265F) for both colors and
// recolor via SVG `fill`. The trailing U+FE0E variation selector forces
// text presentation so iOS Safari doesn't substitute color-emoji glyphs
// (which would ignore our fill and always render dark).
const VS = '\uFE0E';
const PIECE_GLYPH = {
  K: '♚' + VS, Q: '♛' + VS, R: '♜' + VS, B: '♝' + VS, N: '♞' + VS, P: '♟' + VS,
  k: '♚' + VS, q: '♛' + VS, r: '♜' + VS, b: '♝' + VS, n: '♞' + VS, p: '♟' + VS,
};
function ChessBoardSvg({ board, size = 224, highlight = null }) {
  const sq = size / 8;
  return (
    <svg
      viewBox={`0 0 ${size} ${size}`}
      width={size} height={size}
      className="mm-chess"
      aria-hidden="true"
    >
      {Array.from({ length: 64 }).map((_, idx) => {
        const r = Math.floor(idx / 8);
        const c = idx % 8;
        const isLight = (r + c) % 2 === 0;
        const piece = board[r][c];
        const isWhite = piece !== '.' && piece === piece.toUpperCase();
        const isHi = highlight && highlight.includes(`${'abcdefgh'[c]}${8 - r}`);
        return (
          <g key={idx}>
            <rect
              x={c * sq} y={r * sq} width={sq} height={sq}
              fill={isLight ? '#eed9b6' : '#b58863'}
            />
            {isHi && (
              <rect
                x={c * sq} y={r * sq} width={sq} height={sq}
                fill="hsl(218, 80%, 52%)" fillOpacity="0.22"
              />
            )}
            {piece !== '.' && (
              <text
                x={c * sq + sq / 2}
                y={r * sq + sq * 0.74}
                fontSize={sq * 0.82}
                textAnchor="middle"
                fill={isWhite ? '#f5f5f5' : '#101418'}
                stroke={isWhite ? '#101418' : 'none'}
                strokeWidth={isWhite ? sq * 0.025 : 0}
                style={{
                  paintOrder: 'stroke',
                  fontFamily: '"DejaVu Sans", "Arial Unicode MS", system-ui, sans-serif',
                }}
              >
                {PIECE_GLYPH[piece]}
              </text>
            )}
          </g>
        );
      })}
      <rect x="0" y="0" width={size} height={size}
        fill="none" stroke="#0f172a" strokeOpacity="0.18" />
    </svg>
  );
}

/* ============================================================
   Widget 1 — Define a game tree, interactively
   Three abstract nodes (Current / d4 / Nxe5). Hover any to
   reveal its chess position in the adjacent panel.
   ============================================================ */
const POSITIONS = {
  root: {
    label: 'Current state',
    move: 'White to move',
    highlight: [],
    board: [
      'r.bqkb.r',
      'pppp.ppp',
      '.....n..',
      '....p...',
      '....P...',
      '.....N..',
      'PPPP.PPP',
      'R.BQKB.R',
    ],
  },
  d4: {
    label: 'After d4',
    move: 'pawn d2 → d4',
    highlight: ['d2', 'd4'],
    board: [
      'r.bqkb.r',
      'pppp.ppp',
      '.....n..',
      '....p...',
      '...PP...',
      '.....N..',
      'PPP..PPP',
      'R.BQKB.R',
    ],
  },
  nxe5: {
    label: 'After Nxe5',
    move: 'knight f3 × e5',
    highlight: ['f3', 'e5'],
    board: [
      'r.bqkb.r',
      'pppp.ppp',
      '.....n..',
      '....N...',
      '....P...',
      '........',
      'PPPP.PPP',
      'R.BQKB.R',
    ],
  },
};

function GameTreeInteractive() {
  const [hovered, setHovered] = useState('root');
  const active = POSITIONS[hovered];

  // SVG layout for the abstract tree
  const W = 360, H = 220;
  const rootX = W / 2, rootY = 40;
  const leftX = 80, rightX = W - 80, childY = H - 40;
  const r = 28;

  const TreeNode = ({ id, x, y, label }) => (
    <g
      className={`mm-gti-node${hovered === id ? ' is-active' : ''}`}
      onMouseEnter={() => setHovered(id)}
      onFocus={() => setHovered(id)}
      tabIndex={0}
      role="button"
      aria-label={`Show ${POSITIONS[id].label}`}
    >
      <circle cx={x} cy={y} r={r} />
      <text x={x} y={y + 5} className="mm-gti-node-label">{label}</text>
    </g>
  );

  return (
    <div className="viz-panel mm-gti">
      <div className="mm-gti-grid">
        <div className="mm-gti-svg-wrap">
          <svg viewBox={`0 0 ${W} ${H}`} width="100%" preserveAspectRatio="xMidYMid meet">
            <line x1={rootX} y1={rootY + r} x2={leftX} y2={childY - r}
              className={`mm-gti-edge${hovered === 'd4' ? ' is-active' : ''}`} />
            <line x1={rootX} y1={rootY + r} x2={rightX} y2={childY - r}
              className={`mm-gti-edge${hovered === 'nxe5' ? ' is-active' : ''}`} />
            <text x={(rootX + leftX) / 2 - 24} y={(rootY + childY) / 2 - 4} className="mm-gti-edge-label">d4</text>
            <text x={(rootX + rightX) / 2 + 6} y={(rootY + childY) / 2 - 4} className="mm-gti-edge-label">Nxe5</text>
            <TreeNode id="root" x={rootX} y={rootY} label="s" />
            <TreeNode id="d4"   x={leftX}  y={childY} label="s'" />
            <TreeNode id="nxe5" x={rightX} y={childY} label="s''" />
          </svg>
          <p className="viz-caption">A node is a game state; an edge is a legal move. Hover any node to see the position.</p>
        </div>
        <div className="mm-gti-board">
          <div className="mm-gti-board-head">
            <div className="mm-gti-board-title">{active.label}</div>
            <div className="mm-gti-board-move">{active.move}</div>
          </div>
          <ChessBoardSvg board={active.board} highlight={active.highlight} size={224} />
        </div>
      </div>
    </div>
  );
}

/* ============================================================
   Widget 2 — Depth budget meter (b^d explodes)
   ============================================================ */
function DepthBudget() {
  const [b, setB] = useState(2);
  const [d, setD] = useState(3);
  const nodes = useMemo(() => {
    let total = 0;
    for (let i = 0; i <= d; i++) total += Math.pow(b, i);
    return total;
  }, [b, d]);
  const overflow = nodes > 1e6;
  const pretty = (n) => n.toLocaleString('en-US');
  return (
    <div className="viz-panel mm-budget">
      <div className="mm-budget-controls">
        <label className="mm-slider">
          <span>Branching factor <strong>b</strong></span>
          <input type="range" min={2} max={20} step={1} value={b} onChange={(e) => setB(+e.target.value)} />
          <span className="mm-val">{b}</span>
        </label>
        <label className="mm-slider">
          <span>Depth <strong>d</strong></span>
          <input type="range" min={1} max={12} step={1} value={d} onChange={(e) => setD(+e.target.value)} />
          <span className="mm-val">{d}</span>
        </label>
      </div>
      <div className="mm-budget-readout">
        <div className="mm-budget-formula">nodes ≈ b<sup>0</sup> + b<sup>1</sup> + … + b<sup>{d}</sup></div>
        <div className={`mm-budget-count${overflow ? ' overflow' : ''}`}>
          {overflow ? '> 1,000,000' : pretty(nodes)}
          <span className="mm-budget-unit">nodes</span>
        </div>
      </div>
      <p className="viz-caption">
        Chess has roughly <em>b ≈ 35</em>. Push <em>d</em> to 4 and you're already past a
        million states — the reason we can't search to the end of the game.
      </p>
    </div>
  );
}

/* ============================================================
   Widget 3 — Minimax tree (frozen or interactive)
   Steps:
     0 = blank
     1 = evaluate leaves          (Eval row painted)
     2 = max over depth-2 children (Max row painted)
     3 = min over depth-1 children (Min row painted)
     4 = max at root, pick path   (root + chosen path highlighted)
   ============================================================ */
const DEFAULT_LEAVES = [4, 5, 2, 8, 1, 6, 6, 7]; // matches original step PNGs
const PLAY_INTERVAL_MS = 1100;

function computeTree(leaves) {
  const d2 = [];
  for (let i = 0; i < 8; i += 2) d2.push(Math.max(leaves[i], leaves[i + 1]));
  const d1 = [Math.min(d2[0], d2[1]), Math.min(d2[2], d2[3])];
  const root = Math.max(d1[0], d1[1]);
  const rootChild = d1[0] >= d1[1] ? 0 : 1;
  const d1Child = d2[rootChild * 2] <= d2[rootChild * 2 + 1] ? 0 : 1;
  const d2NodeIdx = rootChild * 2 + d1Child;
  const d2Child = leaves[d2NodeIdx * 2] >= leaves[d2NodeIdx * 2 + 1] ? 0 : 1;
  const path = {
    root,
    d1Idx: rootChild,
    d2Idx: d2NodeIdx,
    leafIdx: d2NodeIdx * 2 + d2Child,
  };
  return { d2, d1, root, path };
}

function MinimaxTree({ frozenStep = null, interactive = true, initialStep = 0 }) {
  const [leaves, setLeaves] = useState(DEFAULT_LEAVES);
  const [stepState, setStepState] = useState(frozenStep ?? initialStep);
  const step = frozenStep ?? stepState;
  const [playing, setPlaying] = useState(false);
  const computed = useMemo(() => computeTree(leaves), [leaves]);
  const timerRef = useRef(null);

  useEffect(() => {
    if (frozenStep !== null) return;
    if (!playing) return;
    timerRef.current = setInterval(() => {
      setStepState((s) => {
        if (s >= 4) { setPlaying(false); return s; }
        return s + 1;
      });
    }, PLAY_INTERVAL_MS);
    return () => clearInterval(timerRef.current);
  }, [playing, frozenStep]);

  const reset = useCallback(() => { setStepState(0); setPlaying(false); }, []);
  const shuffle = useCallback(() => {
    const fresh = Array.from({ length: 8 }, () => Math.floor(Math.random() * 9) + 1);
    setLeaves(fresh); setStepState(0); setPlaying(false);
  }, []);

  const W = 760, H = 360;
  const margin = { x: 80, top: 30, bottom: 30 };
  const rows = [
    { y: margin.top + 0,                                        n: 1, label: 'Max',  color: 'green',  visible: step >= 4 },
    { y: margin.top + (H - margin.top - margin.bottom) / 3,     n: 2, label: 'Min',  color: 'red',    visible: step >= 3 },
    { y: margin.top + 2 * (H - margin.top - margin.bottom) / 3, n: 4, label: 'Max',  color: 'green',  visible: step >= 2 },
    { y: H - margin.bottom,                                     n: 8, label: 'Eval', color: 'yellow', visible: step >= 1 },
  ];
  const colXs = (n) => Array.from({ length: n }, (_, i) => margin.x + ((W - 2 * margin.x) * (i + 0.5)) / n);

  const path = computed.path;
  const isHighlightedEdge = (fromRow, fromIdx, toIdx) => {
    if (step < 4) return false;
    if (fromRow === 'root' && toIdx === path.d1Idx) return true;
    if (fromRow === 'd1'   && fromIdx === path.d1Idx && (fromIdx * 2 + toIdx) === path.d2Idx) return true;
    if (fromRow === 'd2'   && fromIdx === path.d2Idx && (fromIdx * 2 + toIdx) === path.leafIdx) return true;
    return false;
  };

  const captions = [
    'Start with the tree empty. The maximizer (us) sits at the root; the minimizer (opponent) is one level down.',
    'Evaluated every leaf with the heuristic. The numbers are placeholders for a chess evaluator.',
    'Maximizer at depth 2 — each parent takes the larger of its two children.',
    'Minimizer at depth 1 — each parent takes the smaller of its two children.',
    'Back at the root, the maximizer picks the larger of the two minimizer values. That child is the move to play.',
  ];

  return (
    <div className="viz-panel mm-tree">
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" preserveAspectRatio="xMidYMid meet">
        {colXs(4).map((px, pi) => colXs(8).slice(pi * 2, pi * 2 + 2).map((cx, ci) => (
          <line key={`e-l-${pi}-${ci}`} x1={px} y1={rows[2].y} x2={cx} y2={rows[3].y}
            className={`mm-edge${isHighlightedEdge('d2', pi, ci) ? ' is-active' : ''}`} />
        )))}
        {colXs(2).map((px, pi) => colXs(4).slice(pi * 2, pi * 2 + 2).map((cx, ci) => (
          <line key={`e-d1-${pi}-${ci}`} x1={px} y1={rows[1].y} x2={cx} y2={rows[2].y}
            className={`mm-edge${isHighlightedEdge('d1', pi, ci) ? ' is-active' : ''}`} />
        )))}
        {colXs(2).map((cx, ci) => (
          <line key={`e-root-${ci}`} x1={colXs(1)[0]} y1={rows[0].y} x2={cx} y2={rows[1].y}
            className={`mm-edge${isHighlightedEdge('root', 0, ci) ? ' is-active' : ''}`} />
        ))}

        {/* Row band labels — pulled left and shrunk so they don't touch leaf 0 */}
        {rows.map((row, ri) => (
          <g key={`band-${ri}`} className={`mm-band mm-band-${row.color}${row.visible ? ' on' : ''}`}>
            <rect x={2} y={row.y - 16} width={48} height={32} rx={8} />
            <text x={26} y={row.y + 4} className="mm-band-label">{row.label}</text>
          </g>
        ))}

        {colXs(8).map((x, i) => (
          <g key={`leaf-${i}`} className={`mm-cell mm-cell-yellow${rows[3].visible ? ' on' : ''}${step >= 4 && path.leafIdx === i ? ' is-chosen' : ''}`}>
            <rect x={x - 20} y={rows[3].y - 18} width={40} height={36} rx={8} />
            <text x={x} y={rows[3].y + 5} className="mm-cell-label">
              {rows[3].visible ? leaves[i] : ''}
            </text>
          </g>
        ))}

        {colXs(4).map((x, i) => (
          <g key={`d2-${i}`} className={`mm-cell mm-cell-green${rows[2].visible ? ' on' : ''}${step >= 4 && path.d2Idx === i ? ' is-chosen' : ''}`}>
            <rect x={x - 20} y={rows[2].y - 18} width={40} height={36} rx={8} />
            <text x={x} y={rows[2].y + 5} className="mm-cell-label">
              {rows[2].visible ? computed.d2[i] : ''}
            </text>
          </g>
        ))}

        {colXs(2).map((x, i) => (
          <g key={`d1-${i}`} className={`mm-cell mm-cell-red${rows[1].visible ? ' on' : ''}${step >= 4 && path.d1Idx === i ? ' is-chosen' : ''}`}>
            <rect x={x - 20} y={rows[1].y - 18} width={40} height={36} rx={8} />
            <text x={x} y={rows[1].y + 5} className="mm-cell-label">
              {rows[1].visible ? computed.d1[i] : ''}
            </text>
          </g>
        ))}

        {(() => {
          const x = colXs(1)[0];
          return (
            <g className={`mm-cell mm-cell-green${rows[0].visible ? ' on' : ''} is-root`}>
              <rect x={x - 22} y={rows[0].y - 20} width={44} height={40} rx={10} />
              <text x={x} y={rows[0].y + 5} className="mm-cell-label">
                {rows[0].visible ? computed.root : ''}
              </text>
            </g>
          );
        })()}
      </svg>

      <div className="mm-caption" aria-live="polite">{captions[step]}</div>

      {interactive && (
        <div className="mm-controls">
          <div className="mm-steps">
            {[0, 1, 2, 3, 4].map((s) => (
              <button
                key={s}
                type="button"
                className={`mm-step-btn${step === s ? ' active' : ''}`}
                onClick={() => { setPlaying(false); setStepState(s); }}
                aria-label={s === 0 ? 'Reset' : `Go to step ${s}`}
              >
                {s === 0 ? 'Reset' : `Step ${s}`}
              </button>
            ))}
          </div>
          <div className="mm-actions">
            <button type="button" className="mm-action mm-play"
              onClick={() => { if (step === 4) setStepState(0); setPlaying((p) => !p); }}>
              {playing ? '❚❚ Pause' : (step === 4 ? '↻ Replay' : '▶ Play all')}
            </button>
            <button type="button" className="mm-action" onClick={shuffle}>↻ Shuffle leaves</button>
            <button type="button" className="mm-action" onClick={reset}>Clear</button>
          </div>
        </div>
      )}
    </div>
  );
}

/* ============================================================
   Code block (Python)
   ============================================================ */
const CODE = `import random
import math

# Counter to show-case the creation of Nodes
# Note this grows with O(2^depth) and we always have root node.
creation_counter = 1

# This is a simple Node class that represents a node in the game tree.
# Each node gets two children (two possible moves) and if evaluated gets a random value.
# There are no terminal nodes in this example.
class Node:
    def __init__(self):
        self._children = None
        self.value = None

    @property
    def children(self):
        # Two Children for each Node
        if self._children is None:
            global creation_counter
            creation_counter += 2
            self._children = [Node() for _ in range(2)]
        return self._children

    def evaluate(self):
        # Random Value for the Node if Evaluation is needed
        if self.value is None:
            self.value = random.randint(0, 100)
        return self.value

    def is_terminal(self):
        # In this example no Node is terminal
        return False

# Core Algorithm
# Recursively evaluates the game tree and returns the value of the best move.
def minimax(node, depth, maximizingPlayer):
    if depth == 0 or node.is_terminal():
        value = node.evaluate()
        node.value = value
        return value

    if maximizingPlayer:
        value = -math.inf
        for child in node.children:
            value = max(value, minimax(child, depth - 1, False))
        node.value = value
        return value
    else:
        value = math.inf
        for child in node.children:
            value = min(value, minimax(child, depth - 1, True))
        node.value = value
        return value

# Main Function
if __name__ == "__main__":
    start_node = Node()
    depth = 3
    minimax_value = minimax(start_node, depth=depth, maximizingPlayer=True)

    print(f'Found Best Action with Value: {minimax_value}')
    if start_node._children is not None:
        print(f'Child Left: {start_node.children[0].value}')
        print(f'Child Right: {start_node.children[1].value}')
    print(f'Number of Nodes Created: {creation_counter}')

    # Result - Depth 3
    # Found Best Action with Value: 66
    # Child Left: 44
    # Child Right: 66
    # Number of Nodes Created: 15
`;

function CodeBlock() {
  return (
    <details className="post-code-details">
      <summary>
        <span className="post-code-summary-icon" aria-hidden="true">›</span>
        <span>Show implementation in Python</span>
        <span className="post-code-summary-hint">click to expand</span>
      </summary>
      <SyntaxHighlighter
        style={oneLight}
        language="python"
        PreTag="div"
        customStyle={{
          borderRadius: 10, overflow: 'auto', padding: '16px 18px',
          background: '#f1f5f9', border: '1px solid #e2e8f0',
          fontSize: 13.5, lineHeight: 1.55, margin: '12px 0 0',
        }}
        codeTagProps={{ style: { background: 'transparent', textShadow: 'none' } }}
      >
        {CODE}
      </SyntaxHighlighter>
    </details>
  );
}

/* ============================================================
   Page
   ============================================================ */
export default function MinimaxSearch() {
  usePageMeta({
    title: 'Minimax Search',
    description: 'Simple decision-making algorithm for two-player games — walk every move sequence, assume the opponent plays optimally, pick the move that minimizes your worst case.',
    slug: 'mini-max',
    publishedDate: '2026-01-15',
    keywords: ['minimax', 'chess', 'search', 'algorithms', 'game theory'],
  });

  return (
    <article className="post-2026 mm-post">
      <div className="post-wrap">
        <div className="post-hero">
        <div className="post-series-tag"><span className="post-live-dot" aria-hidden="true" />Algorithms · Part 1 of 3</div>
        <h1>Minimax Search</h1>
        <p className="post-lede">
          Imagine a game of chess where you're playing against an opponent. You want to
          win, but you don't know what moves they'll make, and which moves will lead to a
          win. Minimax is the classic answer: assume the opponent plays their best move
          every turn, and pick the move that leaves you with the best worst case.
        </p>
        <div className="post-byline">
          By <strong>Bernhard Walser</strong> &amp;{' '}
          <a className="post-link" href="https://www.anthropic.com/claude" target="_blank" rel="noreferrer"><strong>Claude</strong></a>
          {' '}(Anthropic) · co-written and co-designed ·{' '}
          <a className="post-link" href="https://www.linkedin.com/in/bernhardwalser/" target="_blank" rel="noreferrer">LinkedIn</a>
          {' · '}
          <a className="post-link" href="https://github.com/berniwal" target="_blank" rel="noreferrer">GitHub</a>
        </div>
        </div>

        <h2 className="reveal">How can we win the game?</h2>
        <p>
          You maximize your score; the opponent minimizes it. The algorithm walks every
          move sequence up to some depth, then chooses the move that minimizes your
          worst case.
        </p>

        <h2 className="reveal">Let's define a game tree</h2>
        <p>
          A <strong>game tree</strong> is a way of laying out every possible
          continuation: each node is a state of the game, each edge is a legal move. The
          widget below shows a real chess position one ply deep — hover any node in the
          abstract tree to see the corresponding board.
        </p>
        <GameTreeInteractive />
        <p>
          From the starting position, white has two notable options: <em>d4</em> (the
          d-pawn advances and threatens the black pawn on e5) or <em>Nxe5</em> (the
          knight takes that pawn outright). In a real game tree the branching continues
          for many more layers — every legal move on every turn.
        </p>

        <h2 className="reveal">How do we decide which move to make?</h2>
        <p>
          Minimax evaluates every move sequence down to a chosen depth, then propagates
          values back up the tree. We pick the move that leads to the best outcome,
          assuming the opponent does the same on their turn.
        </p>

        <h2 className="reveal">Why do we need to restrict the depth?</h2>
        <p>
          We can't simulate every game to the end. The number of nodes grows as <em>b<sup>d</sup></em>,
          where <em>b</em> is the branching factor (how many moves are legal in a state)
          and <em>d</em> is the depth. Slide the controls below.
        </p>
        <DepthBudget />

        <h2 className="reveal">Step 1 — Evaluate the leaf nodes</h2>
        <p>
          Once the depth is fixed, we evaluate the leaves with a <strong>heuristic
          function</strong>. In chess this combines material count, piece position, and
          mobility. The widget below uses small integers as stand-ins.
        </p>
        <MinimaxTree frozenStep={1} interactive={false} />

        <h2 className="reveal">Step 2 — Propagate the values up the tree</h2>
        <p>
          We move up one level at a time, alternating between the maximizer (us) and the
          minimizer (the opponent). At each node we pick the child with the value that
          suits the current player — larger if it's our turn, smaller if it's theirs.
        </p>
        <p>
          Here the maximizer at depth 2 takes the larger of its two children (green
          band), then the minimizer at depth 1 takes the smaller of its two (red band).
        </p>
        <MinimaxTree frozenStep={3} interactive={false} />

        <h2 className="reveal">Step 3 — Choose the best move</h2>
        <p>
          Back at the root the maximizer picks the larger of the two minimizer values.
          The child it picks is the move to play — highlighted as the solid path below.
          Play the sequence end-to-end or shuffle the leaves to re-run with different
          numbers.
        </p>
        <MinimaxTree initialStep={4} interactive={true} />

        <h2 className="reveal">Code implementation</h2>
        <p>
          Here's a compact Python version. Each node has two children and a random
          heuristic value when evaluated. No node is terminal in this toy world; in a
          real implementation you'd check for checkmate, stalemate, or draws.
        </p>
        <CodeBlock />
        <p>
          Try different depths and watch the node count grow. Each extra level of
          look-ahead doubles the work in this binary toy — and roughly multiplies it
          by 35 in real chess.
        </p>

        <h2 className="reveal">Play against the algorithm</h2>
        <p>
          You play white; the agent below plays black with the minimax routine you just
          read. Slide the depth from 1 to 3 and notice how the node count balloons —
          depth 1 thinks one move ahead, depth 3 looks three plies into the future. Each
          extra ply roughly multiplies the search by the branching factor.
        </p>
        <ChessAgentPlayer
          agent={minimaxAgent}
          algorithmName="minimax"
          initialDepth={2}
          minDepth={1}
          maxDepth={3}
          intro="Click one of your pieces (white), then click a highlighted square to move. The agent replies once it finishes searching."
        />

        <h2 className="reveal">Can we improve?</h2>
        <p>
          Minimax is doing more work than it needs to. Once it sees that one branch is
          worse than something already found, it could stop searching that branch
          entirely. That's the idea behind{' '}
          <a className="post-link" href="/blog/alpha-beta">alpha-beta pruning</a> — same
          answer as minimax, far fewer nodes evaluated.
        </p>

        <h2 className="reveal">References</h2>
        <div className="post-refs">
          <div className="head ref-cite">Cite</div>
          <div className="head ref-title">Title</div>
          <div className="head ref-link">Link</div>

          <div className="ref-cite">Wikipedia</div>
          <div>
            <div className="ref-title">
              <a href="https://en.wikipedia.org/wiki/Minimax" target="_blank" rel="noreferrer">Minimax</a>
            </div>
            <div className="ref-note">Encyclopaedic overview of the algorithm and its history.</div>
          </div>
          <div className="ref-link"><a href="https://en.wikipedia.org/wiki/Minimax" target="_blank" rel="noreferrer">wikipedia.org</a></div>

          <div className="ref-cite">Chess Programming Wiki</div>
          <div>
            <div className="ref-title">
              <a href="https://www.chessprogramming.org/Minimax" target="_blank" rel="noreferrer">Minimax</a>
            </div>
            <div className="ref-note">Practical notes from engine authors.</div>
          </div>
          <div className="ref-link"><a href="https://www.chessprogramming.org/Minimax" target="_blank" rel="noreferrer">chessprogramming.org</a></div>
        </div>

        <footer className="post-footer">
          <p>
            <strong>Part 1 of Algorithms</strong> · Next:{' '}
            <a className="post-link" href="/blog/alpha-beta">Alpha-Beta Pruning</a>
            {' · '}Continues with:{' '}
            <a className="post-link" href="/blog/monte-carlo">Monte Carlo Tree Search</a>.
          </p>
          <p style={{ marginBottom: 0 }}>
            Bernhard Walser · ML Engineer, Digitec Galaxus · ETH Computer Science ·{' '}
            <a className="post-link" href="https://www.linkedin.com/in/bernhardwalser/" target="_blank" rel="noreferrer">LinkedIn</a>
            {' · '}
            <a className="post-link" href="https://github.com/berniwal" target="_blank" rel="noreferrer">GitHub</a>
          </p>
        </footer>
      </div>
    </article>
  );
}
