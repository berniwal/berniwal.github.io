// src/posts/AlphaBetaPruning.jsx
// Algorithms · Part 2 of 3 — Alpha-Beta Pruning.
// Ported from the original 2023 markdown post; prose preserved, visuals
// reimagined as interactive widgets, design tokens aligned with the
// redesigned site (post-2026 chrome).
import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneLight } from 'react-syntax-highlighter/dist/esm/styles/prism';
import ChessAgentPlayer, { alphaBetaAgent } from './ChessAgentPlayer';
import './PostChrome.css';
import './AlphaBetaPruning.css';

/* ============================================================
   Build the trace of an alpha-beta walk on a depth-3 binary tree.
   Returns: visitedLeaves (Set), prunedLeaves (Set), prunedSubtrees (Set
   of d2 indices), final values d2[4], d1[2], root, plus a chosen path.
   ============================================================ */
function abTrace(leaves) {
  const d2 = new Array(4).fill(null);
  const d1 = new Array(2).fill(null);
  let root = null;
  const visited = new Set();
  const pruned = new Set();
  const prunedD2 = new Set(); // d2 nodes pruned wholesale (alpha cutoff at their parent)
  // d2 values that get computed
  const cutoffs = []; // {kind: 'beta'|'alpha', at: 'd2'|'d1', idx, msg}

  // We'll execute alpha-beta with detailed tracing.
  let rootAlpha = -Infinity;
  for (let d1Idx = 0; d1Idx < 2; d1Idx++) {
    let d1Beta = Infinity;
    let d1Alpha = rootAlpha; // inherited
    let d1Value = Infinity;
    let d1Cut = false;
    for (let d2Idx = d1Idx * 2; d2Idx < d1Idx * 2 + 2; d2Idx++) {
      if (d1Cut) {
        prunedD2.add(d2Idx);
        for (let li = d2Idx * 2; li < d2Idx * 2 + 2; li++) pruned.add(li);
        continue;
      }
      let d2Alpha = d1Alpha;
      let d2Beta = d1Beta;
      let d2Value = -Infinity;
      let d2Cut = false;
      for (let li = d2Idx * 2; li < d2Idx * 2 + 2; li++) {
        if (d2Cut) { pruned.add(li); continue; }
        visited.add(li);
        const v = leaves[li];
        if (v > d2Value) d2Value = v;
        if (d2Value > d2Alpha) d2Alpha = d2Value;
        if (d2Alpha >= d2Beta) {
          d2Cut = true;
          cutoffs.push({ kind: 'beta', at: 'd2', idx: d2Idx, alpha: d2Alpha, beta: d2Beta });
        }
      }
      d2[d2Idx] = d2Value;
      if (d2Value < d1Value) d1Value = d2Value;
      if (d1Value < d1Beta) d1Beta = d1Value;
      if (d1Alpha >= d1Beta) {
        d1Cut = true;
        cutoffs.push({ kind: 'alpha', at: 'd1', idx: d1Idx, alpha: d1Alpha, beta: d1Beta });
      }
    }
    d1[d1Idx] = d1Value;
    if (d1Value > rootAlpha) rootAlpha = d1Value;
  }
  root = rootAlpha;

  // Chosen path (same as minimax)
  const rootChild = d1[0] >= d1[1] ? 0 : 1;
  const d2NodeIdx = rootChild * 2 + (d2[rootChild * 2] <= d2[rootChild * 2 + 1] ? 0 : 1);
  const leafChild = d2[d2NodeIdx];
  const chosenLeaf = (() => {
    const a = leaves[d2NodeIdx * 2];
    const b = leaves[d2NodeIdx * 2 + 1];
    return a >= b ? d2NodeIdx * 2 : d2NodeIdx * 2 + 1;
  })();

  const totalNodesAB =
    visited.size +
    (d2.filter((v) => v !== null).length) +
    (d1.filter((v) => v !== null).length) +
    1; // root

  return {
    d2, d1, root,
    visited, pruned, prunedD2,
    cutoffs,
    chosen: { rootChild, d2NodeIdx, leafIdx: chosenLeaf, leafChild },
    totalNodesAB,
  };
}

/* ============================================================
   Widget 1 — Alpha-Beta pruning tree
   Same depth-3 binary tree as the Minimax post. Click "Run" to
   animate the DFS walk; pruned leaves/subtrees fade out.
   ============================================================ */

// Hand-picked example: produces both a beta cutoff (leaf 3 pruned) and
// an alpha cutoff (d2[3] subtree pruned).
const AB_LEAVES = [5, 8, 9, 4, 3, 0, 6, 7];

function AlphaBetaTree() {
  const [leaves, setLeaves] = useState(AB_LEAVES);
  const trace = useMemo(() => abTrace(leaves), [leaves]);
  const [played, setPlayed] = useState(false); // whether the animation has run

  const W = 760, H = 400;
  // bottom margin made larger so the β-cut label sits inside the viewBox
  const margin = { x: 80, top: 30, bottom: 70 };
  const rows = [
    { y: margin.top + 0,                                        label: 'Max',  color: 'green' },
    { y: margin.top + (H - margin.top - margin.bottom) / 3,     label: 'Min',  color: 'red' },
    { y: margin.top + 2 * (H - margin.top - margin.bottom) / 3, label: 'Max',  color: 'green' },
    { y: H - margin.bottom,                                     label: 'Eval', color: 'yellow' },
  ];
  const colXs = (n) => Array.from({ length: n }, (_, i) => margin.x + ((W - 2 * margin.x) * (i + 0.5)) / n);

  // For animation we don't truly step; just show full result with fade.
  // Reset → unplayed → only leaves shown / no fades. Play → fades + values.
  const isPruned = (li) => trace.pruned.has(li);
  const isPrunedD2 = (di) => trace.prunedD2.has(di);

  const chosen = trace.chosen;
  const onPath = {
    d1: chosen.rootChild,
    d2: chosen.d2NodeIdx,
    leaf: chosen.leafIdx,
  };

  const reset = () => { setPlayed(false); };
  const play  = () => { setPlayed(true); };
  const shuffle = () => {
    const fresh = Array.from({ length: 8 }, () => Math.floor(Math.random() * 9) + 1);
    setLeaves(fresh);
    setPlayed(false);
  };

  // Minimax baseline for comparison
  const minimaxNodes = 15; // depth-3 binary: 1 + 2 + 4 + 8 = 15
  const abNodes = trace.totalNodesAB;
  const savings = (((minimaxNodes - abNodes) / minimaxNodes) * 100).toFixed(0);

  return (
    <div className="viz-panel ab-tree">
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" preserveAspectRatio="xMidYMid meet">
        {/* Edges leaf <- d2 */}
        {colXs(4).map((px, pi) => colXs(8).slice(pi * 2, pi * 2 + 2).map((cx, ci) => {
          const li = pi * 2 + ci;
          const pruneOpacity = played && (isPruned(li) || isPrunedD2(pi)) ? 0.18 : 1;
          const onPathEdge = played && onPath.d2 === pi && onPath.leaf === li;
          return (
            <line key={`e-l-${pi}-${ci}`}
              x1={px} y1={rows[2].y} x2={cx} y2={rows[3].y}
              className={`ab-edge${onPathEdge ? ' is-active' : ''}`}
              style={{ opacity: pruneOpacity }} />
          );
        }))}
        {/* Edges d2 <- d1 */}
        {colXs(2).map((px, pi) => colXs(4).slice(pi * 2, pi * 2 + 2).map((cx, ci) => {
          const di = pi * 2 + ci;
          const op = played && isPrunedD2(di) ? 0.18 : 1;
          const onPathEdge = played && onPath.d1 === pi && onPath.d2 === di;
          return (
            <line key={`e-d1-${pi}-${ci}`}
              x1={px} y1={rows[1].y} x2={cx} y2={rows[2].y}
              className={`ab-edge${onPathEdge ? ' is-active' : ''}`}
              style={{ opacity: op }} />
          );
        }))}
        {/* Edges d1 <- root */}
        {colXs(2).map((cx, ci) => {
          const onPathEdge = played && onPath.d1 === ci;
          return (
            <line key={`e-root-${ci}`}
              x1={colXs(1)[0]} y1={rows[0].y} x2={cx} y2={rows[1].y}
              className={`ab-edge${onPathEdge ? ' is-active' : ''}`} />
          );
        })}

        {/* Row band labels */}
        {rows.map((row, ri) => (
          <g key={`band-${ri}`} className={`ab-band ab-band-${row.color} on`}>
            <rect x={2} y={row.y - 16} width={48} height={32} rx={8} />
            <text x={26} y={row.y + 4} className="ab-band-label">{row.label}</text>
          </g>
        ))}

        {/* Leaves */}
        {colXs(8).map((x, i) => {
          const fadedOut = played && (isPruned(i));
          const chosenLeaf = played && onPath.leaf === i;
          return (
            <g key={`leaf-${i}`}
               className={`ab-cell ab-cell-yellow on${chosenLeaf ? ' is-chosen' : ''}`}
               style={{ opacity: fadedOut ? 0.22 : 1 }}>
              <rect x={x - 20} y={rows[3].y - 18} width={40} height={36} rx={8} />
              <text x={x} y={rows[3].y + 5} className="ab-cell-label">
                {fadedOut ? '?' : leaves[i]}
              </text>
            </g>
          );
        })}

        {/* Depth-2 (Max) */}
        {colXs(4).map((x, i) => {
          const fadedOut = played && isPrunedD2(i);
          const visible = played && !fadedOut;
          const chosenNode = played && onPath.d2 === i;
          return (
            <g key={`d2-${i}`}
               className={`ab-cell ab-cell-green${visible ? ' on' : ''}${chosenNode ? ' is-chosen' : ''}`}
               style={{ opacity: fadedOut ? 0.22 : 1 }}>
              <rect x={x - 20} y={rows[2].y - 18} width={40} height={36} rx={8} />
              <text x={x} y={rows[2].y + 5} className="ab-cell-label">
                {fadedOut ? '—' : (visible ? trace.d2[i] : '')}
              </text>
            </g>
          );
        })}

        {/* Depth-1 (Min) */}
        {colXs(2).map((x, i) => {
          const visible = played;
          const chosenNode = played && onPath.d1 === i;
          return (
            <g key={`d1-${i}`}
               className={`ab-cell ab-cell-red${visible ? ' on' : ''}${chosenNode ? ' is-chosen' : ''}`}>
              <rect x={x - 20} y={rows[1].y - 18} width={40} height={36} rx={8} />
              <text x={x} y={rows[1].y + 5} className="ab-cell-label">
                {visible ? trace.d1[i] : ''}
              </text>
            </g>
          );
        })}

        {/* Root (Max) */}
        {(() => {
          const x = colXs(1)[0];
          const visible = played;
          return (
            <g className={`ab-cell ab-cell-green${visible ? ' on' : ''} is-root`}>
              <rect x={x - 22} y={rows[0].y - 20} width={44} height={40} rx={10} />
              <text x={x} y={rows[0].y + 5} className="ab-cell-label">
                {visible ? trace.root : ''}
              </text>
            </g>
          );
        })()}

        {/* Cutoff labels — small badges hanging off pruned branches */}
        {played && trace.cutoffs.map((c, k) => {
          if (c.at === 'd2') {
            // Beta cutoff at d2 — show on the pruned sibling leaf below this d2 node
            const di = c.idx;
            const cutLeaf = di * 2 + 1; // second leaf is the pruned one
            const x = colXs(8)[cutLeaf];
            return (
              <text key={`cut-${k}`} x={x} y={rows[3].y + 34} className="ab-cut-label ab-cut-beta">β-cut</text>
            );
          } else {
            // Alpha cutoff at d1 — show on the pruned d2 sibling
            const di = c.idx;
            const cutD2 = di * 2 + 1;
            const x = colXs(4)[cutD2];
            return (
              <text key={`cut-${k}`} x={x} y={rows[2].y + 34} className="ab-cut-label ab-cut-alpha">α-cut</text>
            );
          }
        })}
      </svg>

      <div className="ab-stats">
        <div className="ab-stat">
          <div className="ab-stat-label">Minimax</div>
          <div className="ab-stat-value">{minimaxNodes}<span className="ab-stat-unit">nodes</span></div>
        </div>
        <div className="ab-stat ab-stat-primary">
          <div className="ab-stat-label">Alpha-Beta</div>
          <div className="ab-stat-value">{played ? abNodes : '—'}<span className="ab-stat-unit">nodes</span></div>
        </div>
        <div className="ab-stat">
          <div className="ab-stat-label">Saved</div>
          <div className="ab-stat-value">{played ? `${savings}%` : '—'}</div>
        </div>
      </div>

      <div className="ab-controls">
        <button type="button" className="ab-btn ab-btn-primary" onClick={played ? reset : play}>
          {played ? '↻ Reset' : '▶ Run alpha-beta'}
        </button>
        <button type="button" className="ab-btn" onClick={shuffle}>↻ Shuffle leaves</button>
      </div>

      <p className="viz-caption">
        Faded leaves were never evaluated — alpha-beta knew their values couldn't
        change the answer. Same chosen move as minimax, fewer nodes visited.
      </p>
    </div>
  );
}

/* ============================================================
   Code block (alpha-beta Python)
   ============================================================ */
const CODE = `import random
import math

# Counter to showcase node visits. Note we always visit the root.
visit_counter = 1

# Simple Node class representing a node in the game tree.
class Node:
    def __init__(self):
        self._children = None
        self.value = None

    @property
    def children(self):
        # Two Children for each Node
        if self._children is None:
            self._children = [Node() for _ in range(2)]
        return self._children

    def evaluate(self):
        # Assign a random value if the node is evaluated.
        if self.value is None:
            self.value = random.randint(0, 100)
        return self.value

    def is_terminal(self):
        # In this example no node is terminal.
        return False

# Alpha-Beta Pruning Algorithm
def alphabeta(node, depth, alpha, beta, maximizingPlayer):
    global visit_counter
    if depth == 0 or node.is_terminal():
        value = node.evaluate()
        node.value = value
        return value

    if maximizingPlayer:
        value = -math.inf
        for child in node.children:
            visit_counter += 1
            value = max(value, alphabeta(child, depth - 1, alpha, beta, False))
            alpha = max(alpha, value)
            if alpha >= beta:
                break  # beta cutoff
        node.value = value
        return value
    else:
        value = math.inf
        for child in node.children:
            visit_counter += 1
            value = min(value, alphabeta(child, depth - 1, alpha, beta, True))
            beta = min(beta, value)
            if beta <= alpha:
                break  # alpha cutoff
        node.value = value
        return value

# Main Function
if __name__ == "__main__":
    start_node = Node()
    depth = 10
    best_value = alphabeta(start_node, depth, -math.inf, math.inf, True)

    print(f'Found Best Action with Value: {best_value}')
    if start_node._children is not None:
        print(f'Child Left: {start_node.children[0].value}')
        print(f'Child Right: {start_node.children[1].value}')
    print(f'Number of Nodes Created: {visit_counter}')

    # For Depth 10, previous implementation would have created 2047 nodes.
    # Here with Alpha Beta Pruning, we get the following result:
    # Found Best Action with Value: 40
    # Child Left: 40
    # Child Right: 29
    # Number of Nodes Created: 681
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
export default function AlphaBetaPruning() {
  useEffect(() => {
    const prev = document.title;
    document.title = 'Alpha-Beta Pruning — Bernhard Walser';
    return () => { document.title = prev; };
  }, []);

  return (
    <article className="post-2026 ab-post">
      <div className="post-wrap">
        <div className="post-hero">
          <div className="post-series-tag">
            <span className="post-live-dot" aria-hidden="true" />Algorithms · Part 2 of 3
          </div>
          <h1>Alpha-Beta Pruning</h1>
          <p className="post-lede">
            In the previous post, we explored the <a className="post-link" href="#/blog/mini-max">minimax algorithm</a> and saw how it evaluates every branch of the game tree to determine the best move. However, the exponential growth in nodes evaluated can be a major performance issue. That's where alpha-beta pruning comes in.
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

        <p>
          Alpha-beta pruning is an enhancement to minimax that reduces the number of nodes processed in the search tree. The main idea is to stop evaluating a move when it becomes clear that it won't influence the final decision.
        </p>

        <h2 className="reveal">Which moves can be pruned?</h2>
        <p>
          Look at the tree below and trace it left-to-right. After evaluating the first
          subtree (leaves <strong>5</strong> and <strong>8</strong>), the maximizer
          picks <strong>8</strong>, which becomes the opponent's current best — the
          worst the maximizer can force on them is <em>8</em>. Now we move to the second
          subtree.
        </p>
        <p>
          The first leaf there is <strong>9</strong>. The maximizer would clearly prefer
          this branch (9 &gt; 8), so the opponent will <em>never</em> choose it — they
          already have a guaranteed <strong>8</strong> from the left. Whatever the
          remaining leaf <strong>4</strong> turns out to be makes no difference. We can
          prune it.
        </p>
        <p>
          This is what we call a <strong>beta cutoff</strong>. The beta value represents
          the best score (lowest) the minimizing player is assured of. Once we find a
          child value &ge; beta inside a maximizer node, no further children matter.
        </p>
        <p>
          Now we move to the right side of the root. The first subtree over there has
          leaves <strong>3</strong> and <strong>0</strong>, so the maximizer in that
          subtree will end up with at most <strong>3</strong>. But the root maximizer
          already has <strong>8</strong> from the left side — they will never choose a
          branch that gives them 3 (or less). So whatever sits in the last subtree on
          the right cannot change the answer, and we can prune the whole thing without
          looking at it.
        </p>
        <p>
          This is what we call an <strong>alpha cutoff</strong>. The alpha value
          represents the best score (highest) the maximizing player is assured of. Once
          a minimizer node finds a child value &le; alpha, no further children matter.
          Hit <em>Run alpha-beta</em> below to watch both cutoffs fire.
        </p>
        <AlphaBetaTree />

        <h2 className="reveal">Code implementation</h2>
        <p>
          The implementation is very similar to minimax. The only difference is that we keep track of two values, alpha and beta, which represent the best score the maximizing player is assured of and the best score the minimizing player is assured of, respectively.
        </p>
        <CodeBlock />
        <p>
          Compare this algorithm to the number of nodes created and the computation time of minimax — significant improvement. You can now run with a greater depth and still get a result in a reasonable amount of time. The result is identical to minimax, since we only remove nodes from the evaluation that wouldn't have influenced the final decision anyway.
        </p>

        <h2 className="reveal">Play against the algorithm</h2>
        <p>
          Here is the same play-the-algorithm widget from part 1, but with the agent swapped for alpha-beta. Crucially, the depth cap is now <strong>4</strong> — one ply deeper than vanilla minimax could afford. The <em>nodes</em> readout is the real measure of pruning at work.
        </p>
        <ChessAgentPlayer
          agent={alphaBetaAgent}
          algorithmName="alpha-beta"
          initialDepth={3}
          minDepth={1}
          maxDepth={4}
          intro="You play white; the agent plays black with alpha-beta pruning. Bump the depth slider — same engine code, dramatically fewer nodes per move than minimax."
        />

        <h2 className="reveal">How much did the performance improve?</h2>
        <p>
          In the worst case, alpha-beta still has to evaluate all nodes in the game tree — <em>O(b<sup>d</sup>)</em> nodes, where <em>b</em> is the branching factor and <em>d</em> the depth. In the best case it only evaluates <em>O(b<sup>d/2</sup>)</em>, or equivalently <em>O(√b<sup>d</sup>)</em> — essentially halving the depth of the tree. A mathematical analysis of this scenario is in the references. That's a significant improvement over the <em>O(b<sup>d</sup>)</em> that minimax would always need.
        </p>

        <h2 className="reveal">What's next?</h2>
        <p>
          In the next post, we will explore{' '}
          <a className="post-link" href="#/blog/monte-carlo">Monte Carlo Tree Search (MCTS)</a> — the algorithm that helped AlphaGo defeat the world champion in Go.
        </p>

        <h2 className="reveal">References</h2>
        <div className="post-refs">
          <div className="head ref-cite">Cite</div>
          <div className="head ref-title">Title</div>
          <div className="head ref-link">Link</div>

          <div className="ref-cite">Wikipedia</div>
          <div>
            <div className="ref-title">
              <a href="https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning" target="_blank" rel="noreferrer">Alpha-Beta Pruning</a>
            </div>
            <div className="ref-note">Encyclopaedic overview of the algorithm and its history.</div>
          </div>
          <div className="ref-link"><a href="https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning" target="_blank" rel="noreferrer">wikipedia.org</a></div>

          <div className="ref-cite">CTU Lecture Notes</div>
          <div>
            <div className="ref-title">
              <a href="https://cw.fel.cvut.cz/b222/_media/courses/be5b33kui/labs/weekly/a-b-analysis.pdf" target="_blank" rel="noreferrer">Best-Case Analysis of Alpha-Beta Pruning</a>
            </div>
            <div className="ref-note">Where the O(b^(d/2)) bound comes from.</div>
          </div>
          <div className="ref-link"><a href="https://cw.fel.cvut.cz/b222/_media/courses/be5b33kui/labs/weekly/a-b-analysis.pdf" target="_blank" rel="noreferrer">pdf</a></div>

          <div className="ref-cite">Wikipedia</div>
          <div>
            <div className="ref-title">
              <a href="https://en.wikipedia.org/wiki/Iterative_deepening_depth-first_search" target="_blank" rel="noreferrer">Iterative Deepening</a>
            </div>
            <div className="ref-note">Strategy to apply alpha-beta as deeply as time allows.</div>
          </div>
          <div className="ref-link"><a href="https://en.wikipedia.org/wiki/Iterative_deepening_depth-first_search" target="_blank" rel="noreferrer">wikipedia.org</a></div>
        </div>

        <footer className="post-footer">
          <p>
            <strong>Part 2 of Algorithms</strong> · Previous:{' '}
            <a className="post-link" href="#/blog/mini-max">Minimax Search</a>
            {' · '}Next:{' '}
            <a className="post-link" href="#/blog/monte-carlo">Monte Carlo Tree Search</a>.
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
