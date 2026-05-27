// src/posts/MonteCarloTreeSearch.jsx
// Algorithms · Part 3 of 3 — Monte Carlo Tree Search.
// Prose preserved from the original 2023 markdown post; visuals are
// interactive widgets, design tokens match the redesigned site.
import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneLight } from 'react-syntax-highlighter/dist/esm/styles/prism';
import ChessAgentPlayer, { mctsAgent } from './ChessAgentPlayer';
import './PostChrome.css';
import './MonteCarloTreeSearch.css';

/* ============================================================
   KaTeX wrapper — uses the CDN-loaded global window.katex
   (already loaded by public/index.html).
   ============================================================ */
function Katex({ tex, block = false, className = '' }) {
  const ref = useRef(null);
  useEffect(() => {
    let cancelled = false;
    const render = () => {
      if (cancelled || !ref.current) return;
      if (window.katex) {
        try {
          window.katex.render(tex, ref.current, {
            displayMode: block,
            throwOnError: false,
          });
        } catch {
          if (ref.current) ref.current.textContent = tex;
        }
      } else {
        setTimeout(render, 60);
      }
    };
    render();
    return () => { cancelled = true; };
  }, [tex, block]);
  return block
    ? <div ref={ref} className={`mcts-katex-block ${className}`} />
    : <span ref={ref} className={`mcts-katex-inline ${className}`} />;
}

/* ============================================================
   Widget 1 — MCTS iteration walkthrough
   A small tree with pre-existing stats. Step buttons advance:
     0: rest state
     1: Selection (highlight UCB1 path root → leaf)
     2: Expansion (new child appears at the leaf)
     3: Simulation (dashed rollout arrows, W result)
     4: Backpropagation (counts tick up along the path)
   ============================================================ */

/* Tree layout — fixed positions for predictability.
   Each node: { id, x, y, parent, beforeW, beforeN, afterW, afterN }.
   The chosen UCB1 path (root → A → A1) and the new child go through
   afterW/afterN to show backprop updates. */
const TREE = {
  root: { id: 'root', x: 380, y: 70,  parent: null, beforeW: 10, beforeN: 12, afterW: 11, afterN: 13 },
  A:    { id: 'A',    x: 180, y: 190, parent: 'root', beforeW: 6,  beforeN: 7,  afterW: 7,  afterN: 8  },
  B:    { id: 'B',    x: 380, y: 190, parent: 'root', beforeW: 2,  beforeN: 3,  afterW: 2,  afterN: 3  },
  C:    { id: 'C',    x: 580, y: 190, parent: 'root', beforeW: 2,  beforeN: 2,  afterW: 2,  afterN: 2  },
  A1:   { id: 'A1',   x: 110, y: 310, parent: 'A',    beforeW: 4,  beforeN: 5,  afterW: 5,  afterN: 6  },
  A2:   { id: 'A2',   x: 250, y: 310, parent: 'A',    beforeW: 2,  beforeN: 2,  afterW: 2,  afterN: 2  },
};
const PATH = ['root', 'A', 'A1'];          // UCB1-selected path
const NEW_NODE = { id: 'A1-new', x: 110, y: 430, parent: 'A1' };

function MctsIterationTree() {
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const timer = useRef(null);
  const STEP_MS = 1400;

  useEffect(() => {
    if (!playing) return;
    timer.current = setInterval(() => {
      setStep((s) => {
        if (s >= 4) { setPlaying(false); return s; }
        return s + 1;
      });
    }, STEP_MS);
    return () => clearInterval(timer.current);
  }, [playing]);

  const reset = () => { setStep(0); setPlaying(false); };
  const togglePlay = () => {
    if (step === 4) setStep(0);
    setPlaying((p) => !p);
  };

  const captions = [
    'A tree built from several previous iterations. Each node shows wins ÷ visits, always from the perspective of the player who is about to move.',
    'Selection — descend from the root using UCB1, which trades exploration off against exploitation. We end up at the leaf A1.',
    'Expansion — A1 isn\'t terminal, so we add one new child to extend the tree into unexplored territory.',
    'Simulation — from the new child, play random moves until a terminal state or a depth cap. This rollout ends in a win.',
    'Backpropagation — propagate the result back up. Every node on the path gets +1 visit; winning nodes also get +1 win.',
  ];

  // Path lookups
  const onPath = (id) => PATH.includes(id);
  // viewBox extended below H=430 to fit the rollout zigzag + W result label
  const W = 760, H = 580;

  const nodeData = (id) => TREE[id];
  const valFor = (id) => {
    const n = nodeData(id);
    if (!onPath(id)) return { w: n.beforeW, n: n.beforeN };
    if (step >= 4) return { w: n.afterW, n: n.afterN };
    return { w: n.beforeW, n: n.beforeN };
  };

  const edges = Object.values(TREE).filter((n) => n.parent).map((n) => ({
    from: TREE[n.parent], to: n,
  }));
  const newNodeVisible = step >= 2;
  const rolloutVisible = step >= 3;
  const newChildVal =
    step >= 4 ? { w: 1, n: 1 } :
    step >= 3 ? { w: '?', n: '?' } :
    { w: 0, n: 0 };

  return (
    <div className="viz-panel mcts-tree">
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" preserveAspectRatio="xMidYMid meet">
        {/* Existing edges */}
        {edges.map((e, i) => {
          const active = step >= 1 && onPath(e.from.id) && onPath(e.to.id);
          return (
            <line key={`edge-${i}`}
              x1={e.from.x} y1={e.from.y + 26}
              x2={e.to.x}   y2={e.to.y - 26}
              className={`mcts-edge${active ? ' is-active' : ''}`} />
          );
        })}
        {/* New child edge */}
        {newNodeVisible && (
          <line
            x1={TREE.A1.x} y1={TREE.A1.y + 26}
            x2={NEW_NODE.x} y2={NEW_NODE.y - 26}
            className={`mcts-edge mcts-edge-new${step >= 2 ? ' is-active' : ''}`} />
        )}

        {/* Rollout arrows — dashed extension below the new node */}
        {rolloutVisible && (
          <>
            <line x1={NEW_NODE.x} y1={NEW_NODE.y + 22} x2={NEW_NODE.x - 30} y2={NEW_NODE.y + 50}
              className="mcts-rollout" strokeDasharray="4 4" />
            <line x1={NEW_NODE.x - 30} y1={NEW_NODE.y + 50} x2={NEW_NODE.x + 10} y2={NEW_NODE.y + 90}
              className="mcts-rollout" strokeDasharray="4 4" />
            <text x={NEW_NODE.x + 10} y={NEW_NODE.y + 80} textAnchor="middle" className="mcts-rollout-arrow">↓</text>
            <text x={NEW_NODE.x + 10} y={NEW_NODE.y + 105} textAnchor="middle" className="mcts-rollout-result">W</text>
          </>
        )}

        {/* Existing nodes */}
        {Object.values(TREE).map((n) => {
          const val = valFor(n.id);
          const highlight = step >= 1 && onPath(n.id);
          const justUpdated = step >= 4 && onPath(n.id);
          return (
            <g key={n.id} className={`mcts-node${highlight ? ' is-on-path' : ''}${justUpdated ? ' is-updated' : ''}`}>
              <circle cx={n.x} cy={n.y} r={26} />
              <text x={n.x} y={n.y - 2} textAnchor="middle" className="mcts-node-label">
                {val.w}/{val.n}
              </text>
              <text x={n.x} y={n.y + 14} textAnchor="middle" className="mcts-node-id">{n.id}</text>
            </g>
          );
        })}
        {/* New child node */}
        {newNodeVisible && (
          <g className="mcts-node mcts-node-new is-on-path">
            <circle cx={NEW_NODE.x} cy={NEW_NODE.y} r={26} />
            <text x={NEW_NODE.x} y={NEW_NODE.y - 2} textAnchor="middle" className="mcts-node-label">
              {newChildVal.w}/{newChildVal.n}
            </text>
            <text x={NEW_NODE.x} y={NEW_NODE.y + 14} textAnchor="middle" className="mcts-node-id">new</text>
          </g>
        )}
      </svg>

      <div className="mcts-caption" aria-live="polite">{captions[step]}</div>

      <div className="mcts-controls">
        <div className="mcts-steps">
          {[0, 1, 2, 3, 4].map((s) => (
            <button key={s}
              type="button"
              className={`mcts-step-btn${step === s ? ' active' : ''}`}
              onClick={() => { setPlaying(false); setStep(s); }}
            >
              {['Reset', 'Select', 'Expand', 'Simulate', 'Backprop'][s]}
            </button>
          ))}
        </div>
        <div className="mcts-actions">
          <button type="button" className="mcts-btn mcts-btn-primary" onClick={togglePlay}>
            {playing ? '❚❚ Pause' : (step === 4 ? '↻ Replay' : '▶ Play all')}
          </button>
          <button type="button" className="mcts-btn" onClick={reset}>Clear</button>
        </div>
      </div>
    </div>
  );
}

/* ============================================================
   Widget 2 — UCB1 explorer
   Two sibling nodes. Slider for the exploration constant c.
   ============================================================ */
function Ucb1Explorer() {
  const [c, setC] = useState(Math.SQRT2);
  // Two siblings under a parent with N=30 total visits.
  // A has the better win-rate (90%), B has fewer visits and a slightly worse
  // rate (80%). At c=0 (pure exploit) A wins; at high c (pure explore) B wins
  // because UCB1 rewards under-visited nodes. The crossover is around c ≈ 0.9.
  const A = { wins: 18, visits: 20 };
  const B = { wins: 8,  visits: 10 };
  const N = 30;

  const compute = (node) => {
    const exploit = node.wins / node.visits;
    const explore = c * Math.sqrt(Math.log(N) / node.visits);
    return { exploit, explore, total: exploit + explore };
  };
  const ucbA = compute(A);
  const ucbB = compute(B);
  const winner = ucbA.total > ucbB.total ? 'A' : 'B';
  const fmt = (x) => x.toFixed(3);

  return (
    <div className="viz-panel mcts-ucb">
      <div className="mcts-ucb-controls">
        <label className="mcts-slider">
          <span>Exploration constant <strong>c</strong></span>
          <input type="range" min={0} max={3} step={0.1} value={c}
                 onChange={(e) => setC(+e.target.value)} />
          <span className="mcts-val">{c.toFixed(1)}</span>
        </label>
        <div className="mcts-ucb-formula">
          <Katex tex={'\\mathrm{UCB1} = \\frac{w}{n} + c \\cdot \\sqrt{\\frac{\\ln N}{n}}'} />
        </div>
      </div>
      <div className="mcts-ucb-grid">
        {[
          { key: 'A', node: A, ucb: ucbA, winner: winner === 'A' },
          { key: 'B', node: B, ucb: ucbB, winner: winner === 'B' },
        ].map(({ key, node, ucb, winner }) => (
          <div key={key} className={`mcts-ucb-card${winner ? ' is-winner' : ''}`}>
            <div className="mcts-ucb-card-head">
              <span className="mcts-ucb-card-id">Node {key}</span>
              <span className="mcts-ucb-card-stats">{node.wins}/{node.visits}</span>
            </div>
            <div className="mcts-ucb-row">
              <span className="mcts-ucb-row-label">exploit · w/n</span>
              <span className="mcts-ucb-row-val">{fmt(ucb.exploit)}</span>
            </div>
            <div className="mcts-ucb-row">
              <span className="mcts-ucb-row-label">explore · c·√(ln N / n)</span>
              <span className="mcts-ucb-row-val">{fmt(ucb.explore)}</span>
            </div>
            <div className="mcts-ucb-row mcts-ucb-total">
              <span className="mcts-ucb-row-label">UCB1</span>
              <span className="mcts-ucb-row-val">{fmt(ucb.total)}</span>
            </div>
          </div>
        ))}
      </div>
      <p className="viz-caption">
        Slide <em>c</em> to feel the trade-off. At <em>c = 0</em> only the win-rate
        matters and A wins (90 % vs 80 %). Push <em>c</em> up and B catches up — it
        has only half the visits, so it carries more uncertainty, and UCB1 rewards
        that. The crossover lands around <em>c ≈ 0.9</em>.
      </p>
    </div>
  );
}

/* ============================================================
   Code block (Python)
   ============================================================ */
const CODE = `import math
import random

MAX_DEPTH = 10  # Maximum depth for our simulation

class MCTSNode:
    def __init__(self, state, parent=None):
        # The state is represented as a tuple: (depth, dummy_value)
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.reward = 0.0
        # For simplicity, each node has 2 possible moves unless at max depth.
        self.untried_moves = [0, 1] if self.state[0] < MAX_DEPTH else []

    def is_terminal(self):
        # A node is terminal if we've reached our depth limit.
        return self.state[0] >= MAX_DEPTH

    def expand(self):
        # Remove a move from untried moves and create a new child node.
        move = self.untried_moves.pop()
        new_depth = self.state[0] + 1
        new_value = random.random()
        new_state = (new_depth, new_value)
        child = MCTSNode(new_state, parent=self)
        self.children.append(child)
        return child

    def rollout(self):
        # Run a simulation from this node until a terminal state is reached.
        current_depth = self.state[0]
        total_reward = 0.0
        while current_depth < MAX_DEPTH:
            total_reward += random.random()
            current_depth += 1
        return total_reward

def uct_select_child(node):
    # UCT = (child's average reward) + sqrt(2 * log(parent visits) / child visits)
    log_parent_visits = math.log(node.visits)
    def uct(child):
        return (child.reward / child.visits) + math.sqrt(2 * log_parent_visits / child.visits)
    return max(node.children, key=uct)

def mcts(root, iterations):
    for _ in range(iterations):
        node = root
        # SELECTION
        while node.untried_moves == [] and node.children != []:
            node = uct_select_child(node)
        # EXPANSION
        if node.untried_moves:
            node = node.expand()
        # SIMULATION
        reward = node.rollout()
        # BACKPROPAGATION
        while node is not None:
            node.visits += 1
            node.reward += reward
            node = node.parent
    # After running all iterations, choose the best child from the root.
    return max(root.children, key=lambda c: c.visits)
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
const ITER_TABLE = [0, 50, 100, 200, 400, 800];
const iterFormatter = (d) => ITER_TABLE[d]?.toLocaleString() ?? d;

export default function MonteCarloTreeSearch() {
  useEffect(() => {
    const prev = document.title;
    document.title = 'Monte Carlo Tree Search — Bernhard Walser';
    return () => { document.title = prev; };
  }, []);

  // mctsAgent wants raw iterations, but ChessAgentPlayer's slider passes
  // depth 1..5. Wrap so depth maps through ITER_TABLE.
  const agentWrapped = useCallback(
    (chess, depth) => mctsAgent(chess, ITER_TABLE[depth] || 250),
    [],
  );

  return (
    <article className="post-2026 mcts-post">
      <div className="post-wrap">
        <div className="post-hero">
          <div className="post-series-tag">
            <span className="post-live-dot" aria-hidden="true" />Algorithms · Part 3 of 3
          </div>
          <h1>Monte Carlo Tree Search</h1>
          <p className="post-lede">
            In the previous post, we explored the <a className="post-link" href="#/blog/mini-max">minimax algorithm</a> enhanced with <a className="post-link" href="#/blog/alpha-beta">alpha-beta pruning</a> and saw how we could cut off branches that would never influence the final decision. Now, we shift gears to a different paradigm for game tree search — Monte Carlo Tree Search (MCTS).
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
          Unlike minimax, which evaluates every branch (or prunes many), MCTS is a <strong>simulation-based</strong> search algorithm. It uses random sampling — rollouts — to estimate the value of moves. This approach is especially powerful in games like Go, where the sheer size of the search tree makes exhaustive evaluation impractical.
        </p>

        <h2 className="reveal">How does MCTS work?</h2>
        <p>
          MCTS builds the search tree incrementally and relies on four distinct steps: selection, expansion, simulation, and backpropagation. Before we dive in, let's first define what information each node stores.
        </p>
        <p>
          To decide which moves are promising, each node keeps track of how many times it has been visited and how many wins it has accumulated in simulation — always from the perspective of the player who is about to move. Hit <em>Play all</em> below to watch one full iteration.
        </p>
        <MctsIterationTree />

        <h2 className="reveal">Selection — UCB1</h2>
        <p>
          Starting at the root, the algorithm recursively descends through child nodes according to a policy. The standard choice is the <strong>Upper Confidence Bound (UCB1)</strong> formula, which balances exploring new moves against exploiting moves already known to be promising:
        </p>
        <Katex
          block
          tex={'\\mathrm{UCB1}(\\text{child}) \\;=\\; \\frac{w}{n} \\;+\\; c \\cdot \\sqrt{\\frac{\\ln N}{n}}'}
        />
        <p>
          Here <em>w</em> is the child's win count, <em>n</em> its visit count, <em>N</em> the parent's visit count, and <em>c</em> the exploration constant (commonly <Katex tex={'\\sqrt{2}'} />). Slide <em>c</em> below and watch the winner flip between two siblings — one with the better average reward, one less explored.
        </p>
        <Ucb1Explorer />

        <h2 className="reveal">Expansion, simulation, backpropagation</h2>
        <p>
          <strong>Expansion:</strong> when a leaf is reached and it is not terminal, one or more child nodes are added to the tree, opening up previously unexplored areas.
        </p>
        <p>
          <strong>Simulation (rollout):</strong> from the newly added node, a random playout is run until a terminal state — or a defined depth. The outcome is used as a crude estimate of the node's value. Random rollouts are surprisingly informative on average, even though any individual rollout is noise.
        </p>
        <p>
          <strong>Backpropagation:</strong> the result is then propagated back up the tree, updating visit counts and cumulative reward at every node along the path. Over many iterations, the statistics converge toward a useful estimate of move quality.
        </p>
        <p>
          By repeating these steps, MCTS gradually builds a more accurate picture of which moves lead to better outcomes — without ever evaluating the entire game tree. When it's time to choose, we pick the root's child with the <em>most visits</em>, not necessarily the highest reward: visit count is the more robust signal, since a high-reward child might just be a lucky outlier.
        </p>

        <h2 className="reveal">Code implementation</h2>
        <p>
          A compact Python implementation. Each node holds visits, reward, and a list of untried moves. UCT (UCB1 applied to trees) drives selection; rollouts are random.
        </p>
        <CodeBlock />

        <h2 className="reveal">Play against the algorithm</h2>
        <p>
          The same play-the-algorithm widget from earlier, now driven by MCTS with random rollouts. The slider sets the number of iterations per move — more iterations means more accurate visit-count estimates. <strong>Honest caveat:</strong> uniformly random rollouts make for a weak chess player, since random play almost never finds a tactic. Real MCTS engines (like AlphaGo's predecessors) replace random rollouts with a learned policy.
        </p>
        <ChessAgentPlayer
          agent={agentWrapped}
          algorithmName="MCTS"
          initialDepth={2}
          minDepth={1}
          maxDepth={5}
          depthLabel="Iterations"
          depthFormatter={iterFormatter}
          intro="You play white; the agent plays black using MCTS. Slide iterations from 50 up to 800 — node count grows roughly linearly with iterations, play quality grows much slower."
        />

        <h2 className="reveal">References</h2>
        <div className="post-refs">
          <div className="head ref-cite">Cite</div>
          <div className="head ref-title">Title</div>
          <div className="head ref-link">Link</div>

          <div className="ref-cite">Wikipedia</div>
          <div>
            <div className="ref-title">
              <a href="https://en.wikipedia.org/wiki/Monte_Carlo_tree_search" target="_blank" rel="noreferrer">Monte Carlo tree search</a>
            </div>
            <div className="ref-note">Overview, history, and connection to AlphaGo.</div>
          </div>
          <div className="ref-link"><a href="https://en.wikipedia.org/wiki/Monte_Carlo_tree_search" target="_blank" rel="noreferrer">wikipedia.org</a></div>

          <div className="ref-cite">Kocsis &amp; Szepesvári 2006</div>
          <div>
            <div className="ref-title">
              <a href="https://link.springer.com/chapter/10.1007/11871842_29" target="_blank" rel="noreferrer">Bandit based Monte-Carlo Planning</a>
            </div>
            <div className="ref-note">The original UCT paper.</div>
          </div>
          <div className="ref-link"><a href="https://link.springer.com/chapter/10.1007/11871842_29" target="_blank" rel="noreferrer">springer</a></div>

          <div className="ref-cite">Silver et al. 2016</div>
          <div>
            <div className="ref-title">
              <a href="https://www.nature.com/articles/nature16961" target="_blank" rel="noreferrer">Mastering the game of Go with deep neural networks and tree search</a>
            </div>
            <div className="ref-note">AlphaGo: MCTS + learned policy + value network.</div>
          </div>
          <div className="ref-link"><a href="https://www.nature.com/articles/nature16961" target="_blank" rel="noreferrer">nature</a></div>
        </div>

        <footer className="post-footer">
          <p>
            <strong>Part 3 of Algorithms</strong> · Previous:{' '}
            <a className="post-link" href="#/blog/alpha-beta">Alpha-Beta Pruning</a>
            {' · '}Start of the series:{' '}
            <a className="post-link" href="#/blog/mini-max">Minimax Search</a>.
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
