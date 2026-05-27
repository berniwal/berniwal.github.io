// src/posts/AlphaGoToAlphaZero.jsx
// Self-Improvement · Part 1 of 6 — AlphaGo to AlphaZero.
// Ported from VisualizingSelfImprovement.jsx (§1 + §2); prose preserved,
// design tokens migrated to the post-2026 redesign chrome.
//
// TAGS FOR REGISTRATION: ['self-improvement', 'alphazero', 'mcts']
// EXCERPT: How AlphaGo learned from human games, AlphaGo Zero learned from
// nothing but the rules, and AlphaZero generalised the recipe. A 2-axis map
// for the whole self-improvement series, plus an interactive PUCT search tree.

import React, { useEffect, useMemo, useRef, useState } from 'react';
import './PostChrome.css';
import './AlphaGoToAlphaZero.css';

/* =========================================================
   KaTeX wrapper — CDN-loaded global window.katex
   ========================================================= */
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
  return (
    <span
      ref={ref}
      className={`${block ? 'aga-math-block' : 'aga-math-inline'} ${className}`}
    />
  );
}

function Counter({ label, value, sub }) {
  return (
    <div className="aga-counter">
      <div className="aga-counter-label">{label}</div>
      <div className="aga-counter-value">{value}</div>
      {sub && <div className="aga-counter-sub">{sub}</div>}
    </div>
  );
}

/* =========================================================
   Widget 1 — The 2-axis map
   ========================================================= */

const MAP_METHODS = [
  { id: 'alphaevolve', name: 'AlphaEvolve',  x: 0.10, y: 0.80, color: 'search'    },
  { id: 'funsearch',   name: 'FunSearch',    x: 0.10, y: 0.92, color: 'search'    },
  { id: 'rest',        name: 'ReST / RLVR',  x: 0.65, y: 0.18, color: 'training'  },
  { id: 'star',        name: 'STaR',         x: 0.65, y: 0.30, color: 'training'  },
  { id: 'alphazero',   name: 'AlphaZero',    x: 0.65, y: 0.42, color: 'training'  },
  { id: 'rzero',       name: 'R-Zero',       x: 1.15, y: 0.18, color: 'training'  },
  { id: 'agent0',      name: 'Agent0',       x: 1.15, y: 0.30, color: 'training'  },
  { id: 'gzero',       name: 'G-Zero',       x: 1.15, y: 0.42, color: 'training'  },
  { id: 'thetaevolve', name: 'ThetaEvolve',  x: 1.65, y: 0.65, color: 'inference' },
  { id: 'ttt',         name: 'TTT-Discover', x: 1.85, y: 0.92, color: 'inference' },
];

function SectionMap({ highlightId = null }) {
  const W = 860;
  const H = 360;
  const padL = 110;
  const padR = 60;
  const padT = 40;
  const padB = 50;
  const innerW = W - padL - padR;
  const innerH = H - padT - padB;

  const toX = (xv) => padL + (xv / 2) * innerW;
  const toY = (yv) => padT + (1 - yv) * innerH;

  const labelOffsets = {
    alphazero:   [ 12,  4 ],
    star:        [ 12,  4 ],
    rest:        [ 12,  4 ],
    funsearch:   [ 12,  4 ],
    alphaevolve: [ 12,  4 ],
    thetaevolve: [ 12,  4 ],
    ttt:         [-12,  4 ],
  };

  const zoneColor = (c) =>
    c === 'search'    ? 'hsl(28, 65%, 50%)' :
    c === 'training'  ? 'hsl(148, 35%, 38%)' :
    c === 'inference' ? 'hsl(8, 55%, 50%)' :
                        'hsl(218, 50%, 45%)';

  const div1 = toX(0.50);
  const div2 = toX(1.50);
  const leftEdge  = padL - 24;
  const rightEdge = W - padR + 8;

  return (
    <div className="viz-panel aga-map">
      <div className="aga-map-svg-wrap">
        <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ maxWidth: W }}>
          {/* Background zones */}
          <rect x={leftEdge}  y={padT} width={div1 - leftEdge}   height={innerH} fill="hsl(28, 65%, 96%)" />
          <rect x={div1}      y={padT} width={div2 - div1}        height={innerH} fill="hsl(148, 35%, 96%)" />
          <rect x={div2}      y={padT} width={rightEdge - div2}   height={innerH} fill="hsl(8, 55%, 96%)" />

          {/* Y-axis label */}
          <text x={28} y={padT + innerH / 2} className="aga-map-axis-label"
                textAnchor="middle" transform={`rotate(-90, 28, ${padT + innerH / 2})`}>
            objective: ←generalise · peak-per-problem→
          </text>

          {/* X-axis labels */}
          <text x={leftEdge + 8}  y={H - 24} className="aga-map-axis-label" textAnchor="start">search only</text>
          <text x={(div1 + div2) / 2} y={H - 24} className="aga-map-axis-label" textAnchor="middle">training-time</text>
          <text x={rightEdge - 8} y={H - 24} className="aga-map-axis-label" textAnchor="end">inference-time</text>
          <text x={padL + innerW / 2} y={H - 8} className="aga-map-axis-label" textAnchor="middle">
            when do the model's weights change?
          </text>

          {/* Zone dividers */}
          <line x1={div1} y1={padT} x2={div1} y2={padT + innerH}
                stroke="hsl(218, 20%, 85%)" strokeDasharray="3 4" />
          <line x1={div2} y1={padT} x2={div2} y2={padT + innerH}
                stroke="hsl(218, 20%, 85%)" strokeDasharray="3 4" />

          {/* Inner divider in training-time zone */}
          <line x1={toX(1.0)} y1={padT + 28} x2={toX(1.0)} y2={padT + innerH}
                stroke="hsl(148, 35%, 55%)" strokeDasharray="2 3" strokeOpacity="0.8" />
          <text x={(div1 + toX(1.0)) / 2} y={padT + 16}
                className="aga-map-axis-label" textAnchor="middle"
                style={{ fontSize: 9, fill: 'hsl(148, 35%, 30%)', letterSpacing: '0.02em' }}>
            ext. verifier
          </text>
          <text x={(toX(1.0) + div2) / 2} y={padT + 16}
                className="aga-map-axis-label" textAnchor="middle"
                style={{ fontSize: 9, fill: 'hsl(148, 35%, 30%)', letterSpacing: '0.02em' }}>
            self verifier
          </text>

          {/* Y-axis midpoint grid line */}
          <line x1={leftEdge} y1={padT + innerH / 2} x2={rightEdge} y2={padT + innerH / 2}
                stroke="hsl(218, 20%, 88%)" strokeDasharray="2 3" />

          {/* Methods */}
          {MAP_METHODS.map((m) => {
            const cx = toX(m.x);
            const cy = toY(m.y);
            const off = labelOffsets[m.id] || [12, 4];
            const isHi = highlightId && m.id === highlightId;
            const dimmed = highlightId && m.id !== highlightId;
            return (
              <g key={m.id}
                 className={`aga-map-method ${dimmed ? 'dimmed' : ''} ${isHi ? 'highlighted' : ''}`}>
                <circle cx={cx} cy={cy} r={isHi ? 8 : 6}
                        fill={zoneColor(m.color)}
                        stroke="#fff" strokeWidth="2" />
                <text x={cx + off[0]} y={cy + off[1]}
                      className="aga-map-label"
                      textAnchor={off[0] < 0 ? 'end' : 'start'}>
                  {m.name}
                </text>
              </g>
            );
          })}
        </svg>
      </div>
      <p className="viz-caption" style={{ marginTop: 8 }}>
        Two axes. <strong>Horizontal</strong>: when (if ever) the model's
        weights change. <strong>Vertical</strong>: whether the objective is
        average performance across problems, or peak performance on one. Each
        method below sits somewhere on this plane.
      </p>
    </div>
  );
}

/* =========================================================
   Widget 2 — PUCT tree
   ========================================================= */

function buildBinaryTree(depth = 3) {
  const nodes = [];
  const edges = [];
  let nextId = 0;
  const root = { id: nextId++, parent: null, depth: 0, children: [] };
  nodes.push(root);
  let layer = [root];
  for (let d = 1; d <= depth; d++) {
    const nextLayer = [];
    for (const p of layer) {
      for (let k = 0; k < 2; k++) {
        const child = { id: nextId++, parent: p.id, depth: d, children: [] };
        p.children.push(child.id);
        nodes.push(child);
        edges.push({ from: p.id, to: child.id });
        nextLayer.push(child);
      }
    }
    layer = nextLayer;
  }
  const leafIds = nodes.filter((n) => n.depth === depth).map((n) => n.id);
  return { nodes, edges, leafIds };
}

function layoutTree(treeStruct, width, height, padding = 24) {
  const depthGroups = {};
  for (const n of treeStruct.nodes) {
    if (!depthGroups[n.depth]) depthGroups[n.depth] = [];
    depthGroups[n.depth].push(n.id);
  }
  const layout = {};
  const maxDepth = Math.max(...treeStruct.nodes.map((n) => n.depth));
  for (let d = 0; d <= maxDepth; d++) {
    const ids = depthGroups[d];
    ids.forEach((id, i) => {
      const x = padding + ((i + 0.5) / ids.length) * (width - 2 * padding);
      const y = padding + (d / maxDepth) * (height - 2 * padding);
      layout[id] = { x, y };
    });
  }
  return layout;
}

function initMCTSState(treeStruct, leafRewards, priors = {}) {
  const stats = {};
  for (const n of treeStruct.nodes) {
    stats[n.id] = { n: 0, w: 0, q: 0 };
  }
  const P = {};
  for (const n of treeStruct.nodes) {
    if (n.children.length === 0) continue;
    const provided = n.children.map((cid) => priors[cid]);
    const allGiven = provided.every((v) => typeof v === 'number');
    if (allGiven) {
      n.children.forEach((cid, k) => { P[cid] = provided[k]; });
    } else {
      n.children.forEach((cid) => { P[cid] = 1 / n.children.length; });
    }
  }
  return { stats, P, iteration: 0, lastPath: [], leafRewards };
}

function puctScore(parentId, childId, state, c) {
  const parent = state.stats[parentId];
  const child  = state.stats[childId];
  const N = parent.n;
  const p = state.P[childId];
  return child.q + c * p * Math.sqrt(1 + N) / (1 + child.n);
}

function bestChildByPUCT(parentId, treeStruct, state, c) {
  const parent = treeStruct.nodes[parentId];
  const scores = parent.children.map((cid) => ({
    id: cid,
    s: puctScore(parentId, cid, state, c),
  }));
  scores.sort((a, b) => b.s - a.s);
  return scores[0].id;
}

function mctsStep(state, treeStruct, c = 1.4) {
  const newStats = {};
  for (const k of Object.keys(state.stats)) {
    newStats[k] = { ...state.stats[k] };
  }
  const path = [];
  let cur = 0;
  path.push(cur);
  while (treeStruct.nodes[cur].children.length > 0) {
    cur = bestChildByPUCT(cur, treeStruct, { ...state, stats: newStats }, c);
    path.push(cur);
  }
  const v = state.leafRewards[cur];
  for (const id of path) {
    newStats[id].n += 1;
    newStats[id].w += v;
    newStats[id].q = newStats[id].w / newStats[id].n;
  }
  return {
    ...state,
    stats: newStats,
    iteration: state.iteration + 1,
    lastPath: path,
  };
}

function PUCTTreeSVG({
  treeStruct,
  layout,
  state,
  width,
  height,
  showPUCT = false,
  c = 1.4,
  leafLabelFn,
  leafRewardFmtFn,
}) {
  const pathSet = new Set(state.lastPath);
  let bestLeaf = 0;
  let bestVal = -Infinity;
  for (const lid of treeStruct.leafIds) {
    if (state.stats[lid].n > 0 && state.stats[lid].q > bestVal) {
      bestVal = state.stats[lid].q;
      bestLeaf = lid;
    }
  }
  return (
    <svg viewBox={`0 0 ${width} ${height}`} width="100%"
         style={{ maxWidth: width }}>
      {treeStruct.edges.map((e) => {
        const a = layout[e.from];
        const b = layout[e.to];
        const onPath = pathSet.has(e.from) && pathSet.has(e.to);
        return (
          <line key={`e-${e.from}-${e.to}`}
                x1={a.x} y1={a.y} x2={b.x} y2={b.y}
                className={`aga-tree-edge ${onPath ? 'path' : ''}`} />
        );
      })}

      {showPUCT && state.iteration > 0 && treeStruct.nodes[0].children.map((cid) => {
        const a = layout[0];
        const b = layout[cid];
        const mx = (a.x + b.x) / 2;
        const my = (a.y + b.y) / 2;
        const score = puctScore(0, cid, state, c);
        return (
          <text key={`puct-${cid}`}
                x={mx + 4} y={my}
                className="aga-tree-puct-label"
                textAnchor="start">
            PUCT={score.toFixed(2)}
          </text>
        );
      })}

      {treeStruct.nodes.map((n) => {
        const p = layout[n.id];
        const isLeaf = n.children.length === 0;
        const isOnPath = pathSet.has(n.id);
        const isBest = isLeaf && state.stats[n.id].n > 0 && n.id === bestLeaf;
        const visited = state.stats[n.id].n > 0;
        const cls =
          (isBest ? 'best ' : '') +
          (isOnPath ? 'path ' : '') +
          (visited && !isOnPath && !isBest ? 'visited' : '');
        const r = isLeaf ? 16 : 14;
        return (
          <g key={`n-${n.id}`}>
            <circle cx={p.x} cy={p.y} r={r}
                    className={`aga-tree-node-bg ${cls.trim()}`} />
            <text x={p.x} y={p.y} className="aga-tree-node-text">
              {isLeaf
                ? (leafRewardFmtFn ? leafRewardFmtFn(state.leafRewards[n.id]) : state.leafRewards[n.id])
                : `n=${state.stats[n.id].n}`}
            </text>
            {isLeaf && leafLabelFn && (
              <text x={p.x} y={p.y + r + 12}
                    className="aga-tree-leaf-reward"
                    textAnchor="middle">
                {leafLabelFn(n.id)}
              </text>
            )}
          </g>
        );
      })}
    </svg>
  );
}

const ALPHAZERO_LEAVES = {
  7:  -1.0,
  8:  +0.4,
  9:  +1.0,
  10: -0.6,
  11: +0.2,
  12: -1.0,
  13: +0.8,
  14: -0.4,
};

const ALPHAZERO_PRIORS = {
  1: 0.55, 2: 0.45,
  3: 0.55, 4: 0.45,
  5: 0.50, 6: 0.50,
  7: 0.40, 8: 0.60,
  9: 0.60, 10: 0.40,
  11: 0.55, 12: 0.45,
  13: 0.55, 14: 0.45,
};

function SectionAlphaZeroPUCT() {
  const treeStruct = useMemo(() => buildBinaryTree(3), []);
  const W = 600;
  const H = 280;
  const layout = useMemo(() => layoutTree(treeStruct, W, H, 28), [treeStruct]);

  const leafRewardArr = useMemo(() => {
    const arr = new Array(15).fill(0);
    for (const k of Object.keys(ALPHAZERO_LEAVES)) arr[+k] = ALPHAZERO_LEAVES[+k];
    return arr;
  }, []);

  const [state, setState] = useState(() =>
    initMCTSState(treeStruct, leafRewardArr, ALPHAZERO_PRIORS)
  );

  const step = () => setState((s) => mctsStep(s, treeStruct, 1.4));
  const reset = () => setState(initMCTSState(treeStruct, leafRewardArr, ALPHAZERO_PRIORS));
  const run10 = () => {
    setState((s) => {
      let cur = s;
      for (let i = 0; i < 10; i++) cur = mctsStep(cur, treeStruct, 1.4);
      return cur;
    });
  };

  const rootVisits = state.stats[0].n;
  const bestChild = bestChildByPUCT(0, treeStruct, state, 1.4);

  return (
    <div className="viz-panel aga-puct">
      <div className="aga-tree-controls">
        <button className="aga-btn aga-btn-primary" onClick={step}>Run 1 MCTS iteration</button>
        <button className="aga-btn" onClick={run10}>Run 10 more</button>
        <button className="aga-btn" onClick={reset}>Reset</button>
        <span className="aga-tree-status">
          iteration <strong>{state.iteration}</strong>{' '}
          {state.lastPath.length > 0 && (
            <>· last selection path: <Katex tex={state.lastPath.join(' \\to ')} /></>
          )}
        </span>
      </div>

      <div className="aga-tree-wrap">
        <PUCTTreeSVG
          treeStruct={treeStruct}
          layout={layout}
          state={state}
          width={W}
          height={H}
          showPUCT={state.iteration > 0}
          c={1.4}
          leafLabelFn={(id) => `v=${ALPHAZERO_LEAVES[id].toFixed(1)}`}
          leafRewardFmtFn={(_) => ''}
        />
      </div>

      <div className="aga-counters">
        <Counter label="Root visits" value={rootVisits} />
        <Counter label="Q at root" value={state.stats[0].q.toFixed(3)} />
        <Counter label="PUCT prefers child" value={bestChild === 1 ? 'left' : 'right'} />
        <Counter
          label="Best leaf so far"
          value={(() => {
            let bv = -Infinity, bid = '-';
            for (const lid of treeStruct.leafIds) {
              if (state.stats[lid].n > 0 && state.stats[lid].q > bv) {
                bv = state.stats[lid].q; bid = lid;
              }
            }
            return bid === '-' ? '—' : `node ${bid} (v=${bv.toFixed(1)})`;
          })()}
        />
      </div>

      <p className="viz-caption" style={{ marginTop: 12 }}>
        A binary tree, depth 3. Each leaf holds a fake "game outcome"{' '}
        <Katex tex="v \in [-1, +1]" />. Each internal node has a policy
        prior <Katex tex="P" /> (some children look more promising). On
        every step, MCTS descends from the root following the PUCT score{' '}
        <Katex tex="Q(s) + c \cdot P(s) \cdot \sqrt{1 + N} / (1 + n(s))" />,
        evaluates the leaf, and backs the value up the path — incrementing
        visit counts and refining <Katex tex="Q" /> estimates. The numbers
        labelled <strong>PUCT=…</strong> on the edges from the root are the
        full PUCT score for each child (exploit + explore combined); MCTS
        descends into whichever child has the higher score. Run a few
        iterations and watch the tree converge on the <Katex tex="v = +1.0" />{' '}
        branch.
      </p>
    </div>
  );
}

/* =========================================================
   Page
   ========================================================= */
export default function AlphaGoToAlphaZero() {
  useEffect(() => {
    const prev = document.title;
    document.title = 'AlphaGo to AlphaZero — Bernhard Walser';
    return () => { document.title = prev; };
  }, []);

  return (
    <article className="post-2026 aga-post">
      <div className="post-wrap">
        <div className="post-hero">
          <div className="post-series-tag"><span className="post-live-dot" aria-hidden="true" />Self-Improvement · Part 1 of 6</div>
          <h1>AlphaGo to AlphaZero</h1>
          <p className="post-lede">
            How does an AI <em>teach itself</em>? AlphaGo beat Lee Sedol with help
            from 30 million human moves; a year later, AlphaGo Zero learned the
            same game from scratch and surpassed it. This first post lays out the
            map for the whole series — and revisits the algorithm that started it.
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

        <h2 className="reveal">What "self-improvement" actually means</h2>
        <p>
          A model self-improves when it gets better by learning from its{' '}
          <em>own</em> outputs — proposing answers, judging them somehow, and
          training on what survived. No human labels in the loop, or at least
          not at every step. AlphaGo is the canonical example: a network that
          played itself millions of times, kept the games it won, and used
          them as training data.
        </p>
        <p>
          Before any specific method, two axes. The first asks{' '}
          <strong>when the model's weights change</strong>: never
          (search-only methods that propose programs against a verifier
          while leaving the model frozen), at training time (the
          AlphaZero / STaR / RLVR lineage), or at inference time, per
          problem (TTT-Discover).
        </p>
        <p>
          The second asks what the training objective is optimising for —{' '}
          <strong>average performance across many future problem
          instances</strong> (a reusable model is produced, then deployed
          broadly), or <strong>peak performance on one specific problem
          instance</strong> (all compute is committed to that one instance,
          no reusable model afterward). Three examples make the distinction
          concrete:
        </p>
        <ul>
          <li>
            <strong>AlphaZero</strong>: trains a network from self-play and
            then plays millions of future positions with it. The training
            loss averages across positions — the model is "good across the
            board on average," not perfect at any one opening.{' '}
            <em>Generalise.</em>
          </li>
          <li>
            <strong>STaR / ReST / RLVR</strong>: trains once on a dataset
            of math (or code) problems, then deploys the resulting model
            across many future problems it will see at inference time.{' '}
            <em>Generalise.</em>
          </li>
          <li>
            <strong>TTT-Discover</strong>: starts fresh from{' '}
            <Katex tex="\theta_0" /> for each new kernel, spends compute on{' '}
            <em>that one kernel</em>, produces "the fastest version of that
            one kernel." Next kernel resets everything. The artefact is one
            optimised kernel, not a trained model for kernel-optimisation
            in general. <em>Peak.</em>
          </li>
        </ul>
        <p>
          Most of the field is on the generalise side; TTT-Discover and the
          FunSearch / AlphaEvolve lineage are the deliberate exceptions.
          Hold this map in your head as the rest of the series fills it in.
        </p>
        <p>
          One small subtlety to flag in the chart below: the{' '}
          <em>training-time</em> band is split by a faint inner divider
          into two halves — methods whose reward comes from an{' '}
          <strong>external verifier</strong> (game rules, an
          answer-checker, a unit-test suite) on the left, and methods
          whose reward is <strong>self-generated</strong> by the model
          itself (majority-vote pseudo-labels, intrinsic signals) on the
          right. Same colour because they're all training-time and
          generalise-leaning, but they make very different trust
          assumptions about where the signal comes from.
        </p>
        <SectionMap />

        <h2 className="reveal">Where it started — AlphaGo, AlphaGo Zero, AlphaZero</h2>
        <p>
          The single canonical reference for "an AI that taught itself" is{' '}
          <a className="post-link" href="https://www.nature.com/articles/nature16961" target="_blank" rel="noreferrer">AlphaGo</a>{' '}
          (Silver et al., Nature 2016) beating Lee Sedol 4–1 in March
          2016. But AlphaGo was bootstrapped on a corpus of about{' '}
          <strong>30 million human expert moves</strong> from the KGS Go
          server. The real conceptual leap landed a year later:{' '}
          <a className="post-link" href="https://www.nature.com/articles/nature24270" target="_blank" rel="noreferrer">AlphaGo Zero</a>{' '}
          (Silver et al., Nature 2017, <em>"Mastering the game of Go
          without human knowledge"</em>) threw the human dataset away,
          started from random play, and learned purely from <strong>self-play
          against the game rules</strong> — and surpassed the original
          AlphaGo. <a className="post-link" href="https://arxiv.org/abs/1712.01815" target="_blank" rel="noreferrer">AlphaZero</a>{' '}
          (Silver et al., 2017) generalised the same algorithm to chess
          and shogi, proving the recipe wasn't Go-specific. (
          <a className="post-link" href="https://arxiv.org/abs/1911.08265" target="_blank" rel="noreferrer">MuZero</a>{' '}
          extended it to environments where even the rules are learned, but
          that's a story for another post.)
        </p>
        <p>
          Two ingredients recur in every method this series covers: a{' '}
          <strong>verifier</strong> (the game rules — a perfect, free
          reward signal) and <strong>MCTS guided by PUCT</strong>. The
          policy / value net biases search toward promising lines; search
          produces better training targets than the net could have on its
          own; the better net guides the next round of search. That loop{' '}
          <em>is</em> self-improvement.
        </p>
        <div className="aga-math-wrap">
          <Katex
            block
            tex={String.raw`
              \mathrm{PUCT}(s, a) \;=\; Q(s, a) \;+\; c \cdot P(s, a) \cdot \tfrac{\sqrt{1 + N(s)}}{1 + n(s, a)}
            `}
          />
        </div>
        <p>
          Reading the symbols, with <Katex tex="s" /> = a parent node and{' '}
          <Katex tex="a" /> = one of its child actions:
        </p>
        <ul>
          <li>
            <Katex tex="Q(s, a)" /> — the <strong>empirical value</strong>{' '}
            of taking action <Katex tex="a" /> from <Katex tex="s" />:
            average game outcome we've observed when search took this
            branch. Starts at 0 and improves with experience.
          </li>
          <li>
            <Katex tex="P(s, a)" /> — the <strong>policy prior</strong>:
            the policy net's guess at how good action <Katex tex="a" /> is{' '}
            <em>before</em> any search. Pushes search toward the moves the
            current net believes in.
          </li>
          <li>
            <Katex tex="N(s)" /> — total visit count of the{' '}
            <strong>parent</strong>: how many MCTS iterations have passed
            through <Katex tex="s" /> in total.
          </li>
          <li>
            <Katex tex="n(s, a)" /> — visit count of <em>this specific
            child</em>: how many times search has chosen action{' '}
            <Katex tex="a" /> from <Katex tex="s" />.
          </li>
          <li>
            <Katex tex="c" /> — exploration constant (≈ 1–2), trading off
            the exploit term <Katex tex="Q" /> against the explore term.
          </li>
        </ul>
        <p>
          So PUCT favours actions with high observed value (the{' '}
          <Katex tex="Q" /> term, exploit) <em>plus</em> a bonus for actions
          the prior likes that haven't been tried much yet (the{' '}
          <Katex tex="c \cdot P \cdot \sqrt{N}/(1+n)" /> term, explore). The
          square root in the numerator means the exploration bonus grows
          slowly with parent visits, so widely-explored subtrees aren't
          revisited just because they're old.
        </p>
        <p>
          Step the toy tree below to see PUCT in motion. Each leaf holds a
          fake "game outcome" <Katex tex="v \in [-1, +1]" />. Each internal
          node has a policy prior (some branches look more promising than
          others). Iteration <Katex tex="N" /> runs one full
          select-evaluate-backup cycle; the path it took is highlighted.
          Run a dozen rounds and watch the tree converge on the leaves
          with <Katex tex="v = +1" />.
        </p>
        <SectionAlphaZeroPUCT />

        <h2 className="reveal">PUCT as a refinement of vanilla MCTS</h2>
        <p>
          If the search step above feels familiar, that's because it{' '}
          <em>is</em> Monte Carlo Tree Search — with one swap. Vanilla MCTS
          estimates the value of a leaf with a <strong>random rollout</strong>:
          play random moves until the game ends, return the win/loss. AlphaZero
          replaces that rollout with a <strong>neural network</strong> that
          predicts both the value of the position and a policy prior over
          the next moves. The prior shows up in PUCT as <Katex tex="P(s, a)" />;
          the value replaces the rollout return.
        </p>
        <p>
          The structural recipe is the same — selection by upper-confidence
          score, expansion, evaluation, backup — and you may already have
          read{' '}
          <a className="post-link" href="#/blog/monte-carlo">my older post on
          Monte Carlo Tree Search</a> for the basics. PUCT is the version
          that learns its priors instead of hard-coding them.
        </p>

        <h2 className="reveal">The next question</h2>
        <p>
          AlphaZero's universe is small and self-contained: a game with a
          fixed rulebook and a free, perfect reward signal at the end.
          Self-improvement works there because the verifier is given. The
          rest of this series asks the same question in harder worlds —{' '}
          <em>how do you do AlphaGo Zero when the "game" is writing a
          correct proof, or a fast GPU kernel, instead of capturing stones?</em>
        </p>

        <h2 className="reveal">References</h2>
        <div className="post-refs">
          <div className="head ref-cite">Cite</div>
          <div className="head ref-title">Title</div>
          <div className="head ref-link">Link</div>

          <div className="ref-cite">Silver et al., 2016</div>
          <div>
            <div className="ref-title">
              <a href="https://www.nature.com/articles/nature16961" target="_blank" rel="noreferrer">Mastering the game of Go with deep neural networks and tree search</a>
            </div>
            <div className="ref-note">AlphaGo (Nature). Supervised pre-training on human play + RL self-play + MCTS.</div>
          </div>
          <div className="ref-link"><a href="https://www.nature.com/articles/nature16961" target="_blank" rel="noreferrer">nature.com</a></div>

          <div className="ref-cite">Silver et al., 2017</div>
          <div>
            <div className="ref-title">
              <a href="https://www.nature.com/articles/nature24270" target="_blank" rel="noreferrer">Mastering the game of Go without human knowledge</a>
            </div>
            <div className="ref-note">AlphaGo Zero (Nature). Pure self-play; no human games; surpassed AlphaGo.</div>
          </div>
          <div className="ref-link"><a href="https://www.nature.com/articles/nature24270" target="_blank" rel="noreferrer">nature.com</a></div>

          <div className="ref-cite">Silver et al., 2018</div>
          <div>
            <div className="ref-title">
              <a href="https://arxiv.org/abs/1712.01815" target="_blank" rel="noreferrer">A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play</a>
            </div>
            <div className="ref-note">AlphaZero (Science). One algorithm, three games. arXiv pre-print linked.</div>
          </div>
          <div className="ref-link"><a href="https://arxiv.org/abs/1712.01815" target="_blank" rel="noreferrer">arxiv.org</a></div>

          <div className="ref-cite">Earlier posts</div>
          <div>
            <div className="ref-title">
              <a href="#/blog/monte-carlo">Monte Carlo Tree Search</a>
            </div>
            <div className="ref-note">Vanilla MCTS — the algorithm AlphaZero's PUCT refines.</div>
          </div>
          <div className="ref-link"><a href="#/blog/monte-carlo">/monte-carlo</a></div>
        </div>

        <footer className="post-footer">
          <p>
            <strong>Part 1 of Self-Improvement</strong> · Next:{' '}
            <a className="post-link" href="#/blog/external-verifiers">External Verifiers (STaR, ReST, RLVR)</a>.
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
