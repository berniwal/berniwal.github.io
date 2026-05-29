// src/posts/SymbolicRegressionLlmTransfer.jsx
// Symbolic Regression · Part 2 of 2 — does the ranking transfer to an LLM?
// Ported from VisualizingSymbolicRegression.jsx (LLM-transfer half) into the
// post-2026 chrome. Prose preserved where the source had it; only chrome
// migrated.
// TAGS FOR REGISTRATION: ['symbolic-regression', 'llm', 'self-improvement']
// EXCERPT: Swap the numpy RNN proposer for an LLM behind the same ask/tell seam. Same task, same verifier, same budget — does the Part-1 ranking survive?
import React, { useEffect, useRef, useState } from 'react';
import usePageMeta from '../usePageMeta';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneLight } from 'react-syntax-highlighter/dist/esm/styles/prism';
import './PostChrome.css';
import './SymbolicRegressionLlmTransfer.css';

/* ============================ KaTeX (CDN global) ============================ */
function Katex({ tex, block = false }) {
  const ref = useRef(null);
  useEffect(() => {
    let cancelled = false;
    const render = () => {
      if (cancelled || !ref.current) return;
      if (window.katex) {
        try {
          window.katex.render(tex, ref.current, { displayMode: block, throwOnError: false });
        } catch { if (ref.current) ref.current.textContent = tex; }
      } else { setTimeout(render, 60); }
    };
    render();
    return () => { cancelled = true; };
  }, [tex, block]);
  return <span ref={ref} className={block ? 'srl-math-block' : 'srl-math-inline'} />;
}

/* ============================ data hook ============================ */
const DATA_BASE = `${process.env.PUBLIC_URL || ''}/data/symbolic-regression`;

function useJson(url) {
  const [data, setData] = useState(null);
  useEffect(() => {
    let alive = true;
    fetch(url)
      .then((r) => { if (!r.ok) throw new Error(`${r.status}`); return r.json(); })
      .then((d) => { if (alive) setData(d); })
      .catch(() => { if (alive) setData(false); });
    return () => { alive = false; };
  }, [url]);
  return data;
}

// Same shape as useJson but for JSONL — splits on newlines, parses each line.
function useJsonLines(url) {
  const [data, setData] = useState(null);
  useEffect(() => {
    let alive = true;
    fetch(url)
      .then((r) => { if (!r.ok) throw new Error(`${r.status}`); return r.text(); })
      .then((t) => {
        if (!alive) return;
        try {
          // Custom parser: -Infinity is not valid JSON; replace before parsing.
          const events = t.trim().split('\n').filter(Boolean).map((line) => (
            JSON.parse(line.replace(/-Infinity/g, '-1e308').replace(/\bInfinity\b/g, '1e308'))
          ));
          setData(events);
        } catch { setData(false); }
      })
      .catch(() => { if (alive) setData(false); });
    return () => { alive = false; };
  }, [url]);
  return data;
}

/* ============================ Layer-1 transfer widget ============================ */
const L1_ARM_ORDER = ['risk', 'best_of_n', 'entropic', 'evolution', 'greedy'];
const TARGET_FORMULA = {
  easy:   'f(x) = x^2 + 1',
  medium: 'f(x) = x^2 + \\sin(x)',
  harder: 'f(x) = x^3 - x + \\cos(2x)',
};

function Layer1Transfer() {
  const data = useJson(`${DATA_BASE}/layer1.json`);
  const [t, setT] = useState('harder');
  if (data === false) {
    return <div className="viz-panel"><p className="srl-note">Could not load Layer-1 data.</p></div>;
  }
  if (!data) {
    return <div className="viz-panel"><p className="srl-note">Loading…</p></div>;
  }
  const targets = data.targets || {};
  const tgt = targets[t];
  const arms = (tgt && tgt.arms) || {};
  const order = L1_ARM_ORDER.filter((a) => arms[a]);
  const avail = ['easy', 'medium', 'harder'].filter((k) => targets[k]);

  return (
    <div className="viz-panel">
      <div className="srl-controls">
        <label className="srl-control-label">benchmark
          <div className="srl-tabs">
            {avail.map((k) => (
              <button
                key={k}
                type="button"
                className={`srl-tab${t === k ? ' active' : ''}`}
                onClick={() => setT(k)}
              >
                {k}
              </button>
            ))}
          </div>
        </label>
        <span className="srl-note srl-tag-right">Qwen2.5-0.5B · GRPO + LoRA</span>
      </div>
      <div className="srl-l1-target">
        <span className="srl-l1-target-label">target</span>
        <Katex tex={TARGET_FORMULA[t]} />
      </div>
      <table className="srl-results">
        <thead>
          <tr><th>arm</th><th>numeric recovery</th><th>symbolic recovery</th><th>mean best</th></tr>
        </thead>
        <tbody>
          {order.map((a) => {
            const arm = arms[a];
            return (
              <tr key={a}>
                <td><span className="srl-swatch" style={{ background: arm.color }} /> {arm.label}</td>
                <td><b>{arm.numeric_solved}/{arm.seeds}</b></td>
                <td>{arm.symbolic_solved}/{arm.seeds}</td>
                <td>{arm.mean_best != null ? arm.mean_best.toFixed(3) : '—'}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="srl-note srl-note-block">
        {t === 'easy'
          ? 'On easy every arm recovers the exact closed form on every seed — x²+1 is in the model\'s prior, no objective needed.'
          : t === 'medium'
          ? 'On medium every arm recovers numerically; symbolic recovery is also near-perfect (the entropic arm misses one seed). The methods only separate on harder.'
          : 'On harder every arm gets the curve roughly right, but exact symbolic recovery nearly vanishes — best-of-N lands it once in five, the rest never.'}
      </p>
    </div>
  );
}

/* ============================ PUCT tree widget ============================ */
// Builds a tidy-tree layout from the proposer's tree_log.jsonl.
//
// What we visualise: every entry in the buffer over the run, drawn as a node,
// with edges parent -> child. A slider scrubs through rounds; nodes are only
// shown when their .timestep <= the selected round. The lineage from the
// best-found-so-far back to its initial-seed root is highlighted in colour;
// the rest of the tree fades to grey so the eye lands on the search path.
//
// Layout: classic "tidy tree" (subtree-width pass, then x-place pass).
// Pure SVG — no D3.

function buildTree(events, maxRound) {
  // Walk events in order, materialising all states ever added; filter by round.
  const states = {};   // id -> { id, parent_id, expr, value, timestep, kind: 'seed'|'child' }
  for (const ev of events) {
    if (ev.kind === 'sample') {
      for (const e of ev.entries) {
        if (!states[e.id]) {
          states[e.id] = {
            id: e.id, parent_id: e.parent_id, expr: e.expr || '',
            value: e.value, timestep: e.timestep,
            kind: e.parent_id == null ? 'seed' : 'child',
          };
        }
      }
    } else if (ev.kind === 'update') {
      for (const e of ev.entries) {
        states[e.id] = {
          id: e.id, parent_id: e.parent_id, expr: e.expr || '',
          value: e.value, timestep: e.timestep,
          kind: e.parent_id == null ? 'seed' : 'child',
        };
      }
    }
  }
  // Filter by round + build parent->children index
  const visible = Object.values(states).filter((s) => s.timestep <= maxRound);
  const byId = new Map(visible.map((s) => [s.id, s]));
  const children = new Map();
  for (const s of visible) {
    if (s.parent_id && byId.has(s.parent_id)) {
      const arr = children.get(s.parent_id) || [];
      arr.push(s);
      children.set(s.parent_id, arr);
    }
  }
  // Sort children of each parent by timestep, then value desc (so high-reward
  // siblings end up on the left -- the eye reads them first).
  for (const arr of children.values()) {
    arr.sort((a, b) => (a.timestep - b.timestep) || ((b.value || 0) - (a.value || 0)));
  }
  // Find roots (visible states with no parent or parent not visible -- the
  // initial seeds)
  const roots = visible.filter((s) => !s.parent_id || !byId.has(s.parent_id));
  return { roots, children, byId };
}

function layoutTree({ roots, children }) {
  // Standard tidy-tree recursion: compute subtreeWidth, then place.
  const NODE_W = 18;    // horizontal slot per leaf
  const LEVEL_H = 56;   // vertical gap between depths
  const widths = new Map();
  function widthOf(node) {
    if (widths.has(node.id)) return widths.get(node.id);
    const kids = children.get(node.id) || [];
    const w = kids.length === 0 ? NODE_W : kids.reduce((s, k) => s + widthOf(k), 0);
    widths.set(node.id, w);
    return w;
  }
  for (const r of roots) widthOf(r);
  const positions = new Map();
  let cursor = 0;
  function place(node, depth) {
    const kids = children.get(node.id) || [];
    if (kids.length === 0) {
      const x = cursor + NODE_W / 2;
      cursor += NODE_W;
      positions.set(node.id, { x, y: depth * LEVEL_H });
      return x;
    }
    const xs = kids.map((k) => place(k, depth + 1));
    const x = (xs[0] + xs[xs.length - 1]) / 2;
    positions.set(node.id, { x, y: depth * LEVEL_H });
    return x;
  }
  // Place each root in turn, padding between roots
  const ROOT_PAD = NODE_W * 1.5;
  for (let i = 0; i < roots.length; i++) {
    if (i > 0) cursor += ROOT_PAD;
    place(roots[i], 0);
  }
  // total width / height for the SVG viewBox
  let maxX = 0, maxY = 0;
  for (const { x, y } of positions.values()) {
    if (x > maxX) maxX = x;
    if (y > maxY) maxY = y;
  }
  return { positions, w: maxX + NODE_W, h: maxY + LEVEL_H };
}

// Round long-tail floats so the pattern jumps out instead of being buried under
// 14-digit BFGS-fit precision. Snaps to π, π/2, π/3, π/4 within a small
// tolerance; to integers and half-integers within a small tolerance; otherwise
// renders to 2 significant decimals.
const PI_SNAPS = [
  { v: Math.PI,     label: 'π'    },
  { v: -Math.PI,    label: '-π'   },
  { v: Math.PI / 2, label: 'π/2'  },
  { v: -Math.PI/2,  label: '-π/2' },
  { v: Math.PI / 3, label: 'π/3'  },
  { v: Math.PI / 4, label: 'π/4'  },
  { v: 2 * Math.PI, label: '2π'   },
];
function prettyExpr(s, tol = 0.005) {
  if (!s) return s;
  // Match floats (including in scientific notation). Negative sign is preserved
  // as a separate character in the source so we don't accidentally eat operators.
  return s.replace(/-?\d+(?:\.\d+)?(?:e[+-]?\d+)?/gi, (match) => {
    const v = parseFloat(match);
    if (!isFinite(v)) return match;
    for (const p of PI_SNAPS) {
      if (Math.abs(v - p.v) < tol) return p.label;
    }
    if (Math.abs(v - Math.round(v)) < tol) return String(Math.round(v));
    const half = Math.round(v * 2) / 2;
    if (Math.abs(v - half) < tol) return half.toString();
    // Two decimals, but trim trailing zeros so "1.50" -> "1.5".
    return parseFloat(v.toFixed(2)).toString();
  });
}

// Reward -> colour. Red at 0, amber at 0.5, green at 1.
function rewardColor(v) {
  if (v == null || !isFinite(v)) return '#cbd5e1';
  const t = Math.max(0, Math.min(1, v));
  if (t < 0.5) {
    const k = t / 0.5;
    return `rgb(${239 - Math.round(43 * k)},${Math.round(180 * k) + 68},${Math.round(60 * k) + 68})`;
  }
  const k = (t - 0.5) / 0.5;
  return `rgb(${Math.round(60 + 60 * (1 - k))},${180 + Math.round(20 * k)},${Math.round(110 + 60 * k)})`;
}

// Walk back from a leaf to its root via parent_id; returns the set of ids.
function lineageIds(leafId, byId) {
  const set = new Set();
  let cur = byId.get(leafId);
  while (cur) {
    set.add(cur.id);
    if (!cur.parent_id) break;
    cur = byId.get(cur.parent_id);
  }
  return set;
}

// Compute the visible-bounds rectangle of an entire subtree rooted at `node`,
// given the layout positions and the children adjacency. Used for "click a node
// to zoom to its subtree".
function subtreeBounds(rootId, children, positions) {
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  const stack = [rootId];
  while (stack.length) {
    const id = stack.pop();
    const p = positions.get(id);
    if (p) {
      if (p.x < minX) minX = p.x;
      if (p.x > maxX) maxX = p.x;
      if (p.y < minY) minY = p.y;
      if (p.y > maxY) maxY = p.y;
    }
    const kids = children.get(id) || [];
    for (const k of kids) stack.push(k.id);
  }
  if (!isFinite(minX)) return null;
  return { minX, maxX, minY, maxY };
}

function PuctTree() {
  const events = useJsonLines(`${DATA_BASE}/puct/tree_log.jsonl`);
  // maxRound from the data; default slider to the final round.
  const maxRoundInData = events && events.length
    ? events[events.length - 1].round : 1;
  const [round, setRound] = useState(maxRoundInData);
  // Keep the slider value in sync when data finishes loading.
  useEffect(() => { setRound(maxRoundInData); }, [maxRoundInData]);
  const [hover, setHover] = useState(null);
  // View transform (SVG viewBox). null = use default = fit-all.
  const [view, setView] = useState(null);
  const svgRef = useRef(null);
  const dragRef = useRef(null);  // { startX, startY, startView }

  if (events === false) {
    return <div className="viz-panel"><p className="srl-note">Could not load PUCT tree log.</p></div>;
  }
  if (!events) {
    return <div className="viz-panel"><p className="srl-note">Loading PUCT tree…</p></div>;
  }
  const { roots, children, byId } = buildTree(events, round);
  const { positions, w, h } = layoutTree({ roots, children });
  // Best-of-round and lineage to highlight.
  let bestNode = null;
  for (const s of byId.values()) {
    if (s.value == null || !isFinite(s.value)) continue;
    if (!bestNode || s.value > bestNode.value) bestNode = s;
  }
  const lineage = bestNode ? lineageIds(bestNode.id, byId) : new Set();
  // Hovered (or best) node info for the side panel.
  const focusNode = hover ? byId.get(hover) : bestNode;

  // Edge list for rendering: parent -> each child (only those visible).
  const edges = [];
  for (const [pid, kids] of children.entries()) {
    const p = positions.get(pid);
    if (!p) continue;
    for (const k of kids) {
      const c = positions.get(k.id);
      if (!c) continue;
      edges.push({ pid, cid: k.id, x1: p.x, y1: p.y, x2: c.x, y2: c.y });
    }
  }
  // Quick stats
  const total = byId.size;
  const nVisited = Array.from(byId.values()).filter((s) => s.value != null && isFinite(s.value)).length;

  return (
    <div className="viz-panel">
      <div className="srl-puct-targetbar">
        <span className="srl-puct-targetbar-label">target</span>
        <code className="srl-puct-targetbar-expr">
          f(x) = x³ − x + cos(2·x)
        </code>
        <span className="srl-puct-targetbar-meta">
          benchmark <code>harder</code> · Qwen3-1.7B + reasoning · seed 0
        </span>
      </div>
      <div className="srl-controls">
        <label className="srl-control-label" style={{ flex: 1 }}>
          round&nbsp;<b>{round}</b>&nbsp;/&nbsp;{maxRoundInData}
          <input
            type="range"
            min={1}
            max={maxRoundInData}
            value={round}
            onChange={(e) => setRound(Number(e.target.value))}
            style={{ width: '100%', marginTop: 6 }}
          />
        </label>
        <span className="srl-note srl-tag-right">
          buffer: {total} states · valued: {nVisited}
          {bestNode && bestNode.value != null
            ? ` · best so far ${bestNode.value.toFixed(4)}`
            : ''}
        </span>
      </div>
      <div className="srl-puct-tree">
        <div className="srl-puct-svg-wrap">
          {(() => {
            // Default viewBox: fit the whole tree with 10px margin.
            const defaultVB = { x: -10, y: -10, w: w + 20, h: h + 20 };
            const vb = view || defaultVB;

            // Convert client (screen) coords to SVG user-space coords. Used by
            // wheel-zoom-around-cursor so the point under the mouse stays fixed.
            const clientToSvg = (cx, cy) => {
              const svg = svgRef.current;
              if (!svg) return { x: 0, y: 0 };
              const rect = svg.getBoundingClientRect();
              const px = (cx - rect.left) / rect.width;
              const py = (cy - rect.top) / rect.height;
              return { x: vb.x + px * vb.w, y: vb.y + py * vb.h };
            };

            // Wheel handler: zoom in/out by factor 1.18 around cursor.
            const handleWheel = (e) => {
              e.preventDefault();
              const factor = e.deltaY > 0 ? 1.18 : (1 / 1.18);
              const c = clientToSvg(e.clientX, e.clientY);
              const newW = Math.min(Math.max(vb.w * factor, 40), defaultVB.w * 4);
              const newH = Math.min(Math.max(vb.h * factor, 40), defaultVB.h * 4);
              // Keep the cursor's world-coords stable: solve for new x,y so
              //   c.x = newX + (cx - left)/width * newW  =>  newX = c.x - px * newW
              const rect = svgRef.current.getBoundingClientRect();
              const px = (e.clientX - rect.left) / rect.width;
              const py = (e.clientY - rect.top) / rect.height;
              setView({
                x: c.x - px * newW,
                y: c.y - py * newH,
                w: newW, h: newH,
              });
            };

            // Drag-to-pan
            const handleMouseDown = (e) => {
              // Only start panning on the SVG background, not on a circle.
              if (e.target.tagName === 'circle') return;
              dragRef.current = { startX: e.clientX, startY: e.clientY, startView: vb };
              e.preventDefault();
            };
            const handleMouseMove = (e) => {
              const d = dragRef.current;
              if (!d) return;
              const rect = svgRef.current.getBoundingClientRect();
              const dxScreen = e.clientX - d.startX;
              const dyScreen = e.clientY - d.startY;
              const dxWorld = dxScreen / rect.width * d.startView.w;
              const dyWorld = dyScreen / rect.height * d.startView.h;
              setView({
                x: d.startView.x - dxWorld,
                y: d.startView.y - dyWorld,
                w: d.startView.w,
                h: d.startView.h,
              });
            };
            const handleMouseUp = () => { dragRef.current = null; };

            // Click a node to focus on its subtree. Padding 40 in world units.
            const focusNodeSubtree = (id) => {
              const bnds = subtreeBounds(id, children, positions);
              if (!bnds) return;
              const pad = 30;
              const fw = Math.max(bnds.maxX - bnds.minX + 2 * pad, 80);
              const fh = Math.max(bnds.maxY - bnds.minY + 2 * pad, 80);
              setView({ x: bnds.minX - pad, y: bnds.minY - pad, w: fw, h: fh });
            };

            return (
              <svg
                ref={svgRef}
                viewBox={`${vb.x} ${vb.y} ${vb.w} ${vb.h}`}
                preserveAspectRatio="xMidYMid meet"
                style={{
                  width: '100%', height: 460,
                  background: '#fafbfc', borderRadius: 6,
                  cursor: dragRef.current ? 'grabbing' : 'grab',
                  touchAction: 'none',
                }}
                onWheel={handleWheel}
                onMouseDown={handleMouseDown}
                onMouseMove={handleMouseMove}
                onMouseUp={handleMouseUp}
                onMouseLeave={handleMouseUp}
              >
                {/* edges first so they sit under nodes */}
                {edges.map((e) => {
                  const inLineage = lineage.has(e.pid) && lineage.has(e.cid);
                  return (
                    <line
                      key={`${e.pid}-${e.cid}`}
                      x1={e.x1} y1={e.y1} x2={e.x2} y2={e.y2}
                      stroke={inLineage ? '#0ea5e9' : '#d6dde6'}
                      strokeWidth={inLineage ? 2.2 : 0.9}
                      opacity={inLineage ? 0.95 : 0.7}
                    />
                  );
                })}
                {/* nodes */}
                {Array.from(positions.entries()).map(([id, p]) => {
                  const s = byId.get(id);
                  const isLineage = lineage.has(id);
                  const isBest = bestNode && id === bestNode.id;
                  return (
                    <circle
                      key={id}
                      cx={p.x} cy={p.y}
                      r={isBest ? 6.5 : (isLineage ? 5 : 4)}
                      fill={rewardColor(s.value)}
                      stroke={isBest ? '#0c4a6e' : (isLineage ? '#0ea5e9' : '#fff')}
                      strokeWidth={isBest ? 2 : (isLineage ? 1.5 : 0.8)}
                      onMouseEnter={() => setHover(id)}
                      onMouseLeave={() => setHover(null)}
                      onClick={(e) => { e.stopPropagation(); focusNodeSubtree(id); }}
                      style={{ cursor: 'zoom-in' }}
                    />
                  );
                })}
              </svg>
            );
          })()}
          <div className="srl-puct-svg-controls">
            <button
              type="button"
              className="srl-puct-svg-btn"
              onClick={() => setView(null)}
              aria-label="Reset view"
              title="Reset view"
            >reset view</button>
            <span className="srl-puct-svg-hint">drag · scroll to zoom · click a node to focus its subtree</span>
          </div>
        </div>
        <div className="srl-puct-info">
          <div className="srl-puct-info-h">
            {focusNode === bestNode ? 'Best so far (root → leaf path is highlighted)' : 'Hovered node'}
          </div>
          {focusNode ? (
            <>
              <div className="srl-puct-info-row">
                <span>reward</span>
                <b>{focusNode.value != null && isFinite(focusNode.value)
                    ? focusNode.value.toFixed(4) : '—'}</b>
              </div>
              <div className="srl-puct-info-row">
                <span>added at round</span>
                <b>{focusNode.timestep}</b>
              </div>
              <div className="srl-puct-info-row">
                <span>kind</span>
                <b>{focusNode.kind === 'seed' ? 'initial seed' : 'child'}</b>
              </div>
              <div className="srl-puct-expr">
                <code>{focusNode.expr ? prettyExpr(focusNode.expr) : '(initial seed — empty prompt)'}</code>
              </div>
              {focusNode.expr ? (
                <div className="srl-puct-info-raw">
                  <details>
                    <summary>raw constants</summary>
                    <code>{focusNode.expr}</code>
                  </details>
                </div>
              ) : null}
            </>
          ) : <p className="srl-note">Hover any node to inspect.</p>}
        </div>
      </div>
      <p className="srl-note srl-note-block">
        Each circle is one candidate expression in the buffer; edges run from
        parent (a previous attempt) to child (a refinement). Colour encodes the
        reward — <span style={{ color: rewardColor(0.1) }}>red</span> low,{' '}
        <span style={{ color: rewardColor(1.0) }}>green</span> high. The lineage
        from the current best back to its initial seed is traced in blue.
        Drag the slider to watch how the search expands: in the first few rounds
        all four roots get explored uniformly, then PUCT starts concentrating
        rollouts on the high-reward branches.
      </p>
    </div>
  );
}

/* ============================ Run-it-yourself code blocks ============================ */
const INSTALL_CMD = `git clone https://github.com/berniwal/berniwal.github.io
cd berniwal.github.io/experiments/self-improvement-arena
pip install -e ".[app]"           # core + Streamlit visualiser (numpy only)
streamlit run streamlit_app.py
# (add ",layer1" to the extras on Apple Silicon for the MLX LoRA arms)`;

function CodeDetails({ label, code, lang = 'bash' }) {
  return (
    <details className="post-code-details">
      <summary>
        <span className="post-code-summary-icon" aria-hidden="true">›</span>
        <span>{label}</span>
        <span className="post-code-summary-hint">click to expand</span>
      </summary>
      <SyntaxHighlighter
        style={oneLight}
        language={lang}
        PreTag="div"
        customStyle={{
          borderRadius: 10, overflow: 'auto', padding: '16px 18px',
          background: '#f1f5f9', border: '1px solid #e2e8f0',
          fontSize: 13.5, lineHeight: 1.55, margin: '12px 0 0',
        }}
        codeTagProps={{ style: { background: 'transparent', textShadow: 'none' } }}
      >
        {code}
      </SyntaxHighlighter>
    </details>
  );
}

/* ============================ the post ============================ */
export default function SymbolicRegressionLlmTransfer() {
  usePageMeta({
    title: 'Does the Ranking Transfer to an LLM?',
    description: 'Swap the proposer for an LLM and ask whether the same Layer-0 ranking survives — what the model sees, how the search budget is shared.',
    slug: 'symbolic-regression-llm-transfer',
    publishedDate: '2026-05-17',
    keywords: ['symbolic regression', 'LLM', 'transfer', 'self-improvement'],
  });

  return (
    <article className="post-2026 srl-post">
      <div className="post-wrap">
        <div className="post-hero">
          <div className="post-series-tag">
            <span className="post-live-dot" aria-hidden="true" />
            Symbolic Regression · Part 2 of 2
          </div>
          <h1>Does the Ranking Transfer to an LLM?</h1>
          <p className="post-lede">
            In <a className="post-link" href="/blog/symbolic-regression-arena">Part 1</a>{' '}
            we raced four proposers on Nguyen — evolution, greedy RL, risk-seeking RL, an
            entropic variant — under one shared verifier and one shared budget. A stable
            ranking emerged: risk-seeking on top, evolution close behind, greedy collapsing
            onto a single attractor. Now we swap the small numpy RNN for a real language
            model behind the same <code>ask()</code>/<code>tell()</code> seam, and ask
            whether the same ranking survives.
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

        <h2 className="reveal">The four proposers, side by side</h2>
        <p>
          The seam is small on purpose. The verifier scores any expression tree the same way;
          only what generates the tree differs. We line up four cards: two Layer-0 baselines
          (numpy GP, numpy RNN) and their Layer-1 LLM counterparts. The gradient-based row
          shares the exact <Katex tex="\sum_i w_i \log \pi(\tau_i)" /> objective — only the
          policy <Katex tex="\pi" /> and the way we tokenise <Katex tex="\tau_i" /> change.
          The evolutionary row shares the high-level idea (best-so-far drives the next batch)
          but the LLM version has no gradient step at all — exactly the contrast AlphaEvolve
          and FunSearch are built on.
        </p>

        <div className="viz-panel srl-fourcol">
          <div className="srl-col srl-col-l0">
            <div className="srl-col-h"><span className="srl-col-tag">Layer 0</span><strong>Evolution (GP)</strong></div>
            <p><strong>Proposer.</strong> Population of 200 expression trees, evolved with tournament selection + subtree crossover + subtree mutation.</p>
            <p><strong>Search space.</strong> Trees built from <Katex tex="\{x, +, -, \times, \div, \sin, \cos, \mathrm{const}\}" />.</p>
            <p><strong>Optimisation.</strong> None — selection pressure on the population, no gradients.</p>
          </div>
          <div className="srl-col srl-col-l1">
            <div className="srl-col-h"><span className="srl-col-tag">Layer 1</span><strong>Evolution (LLM)</strong></div>
            <p><strong>Proposer.</strong> Frozen LLM prompted each round; the K best formulas found so far are injected into the prompt as "do better than these" (AlphaEvolve / FunSearch).</p>
            <p><strong>Search space.</strong> The LLM's full vocabulary (≈150 k tokens) → free-form formula text → parsed back into the same Node tree the verifier accepts.</p>
            <p><strong>Optimisation.</strong> None — the LLM weights are frozen; selection happens implicitly through the archive in the prompt.</p>
          </div>
          <div className="srl-col srl-col-l0">
            <div className="srl-col-h"><span className="srl-col-tag">Layer 0</span><strong>RNN (policy gradient)</strong></div>
            <p><strong>Proposer.</strong> 32-hidden numpy RNN, samples expression tokens autoregressively under the four constraints; ~5 k parameters.</p>
            <p><strong>Search space.</strong> Same fixed grammar as GP, 10 tokens, structured prefix decoding.</p>
            <p><strong>Optimisation.</strong> Policy gradient — for each batch compute the advantage <Katex tex="w_i" /> (greedy / risk-seeking / entropic) and Adam-step on <Katex tex="\mathcal{J} = \sum_i w_i \log \pi(\tau_i)" />.</p>
          </div>
          <div className="srl-col srl-col-l1">
            <div className="srl-col-h"><span className="srl-col-tag">Layer 1</span><strong>LLM (LoRA + GRPO)</strong></div>
            <p><strong>Proposer.</strong> Qwen2.5-0.5B-Instruct + LoRA adapters on the attention projections. Samples free-form formula strings; we parse them back into a Node tree for the verifier.</p>
            <p><strong>Search space.</strong> Full LLM vocabulary (≈150 k tokens) instead of a 10-token grammar.</p>
            <p><strong>Optimisation.</strong> <em>The same advantage formula</em>: <code>greedy_weights</code>, <code>quantile_weights</code>, <code>entropic_weights</code> from <code>sia.objectives</code> are imported by both proposers. Adam steps on the LoRA params via PyTorch GRPO.</p>
          </div>
        </div>
        <p className="srl-caption">
          Budget stays counted in <em>verifier calls</em>, not in tokens or wall-clock. A
          large LLM proposer cannot win by simply evaluating more candidates — it gets the
          same number of seats at the verifier as the RNN.
        </p>

        <h2 className="reveal">What the LLM sees</h2>
        <p>
          For the gradient-based arms (greedy / risk / entropic / best-of-N) we hand the LLM
          a plain prompt and parse its single-line reply. For the medium target{' '}
          <Katex tex="x^2 + \sin(x)" /> with the const-placeholder mode on, the prompt is
          literally:
        </p>
        <div className="srl-prompt-wrap">
          <div className="srl-prompt-label">
            <span>prompt sent to the LLM</span>
            <span>medium · const-placeholder mode</span>
          </div>
          <pre className="srl-prompt">{`You are doing symbolic regression. Find a formula y = f(x) that fits the data.
Allowed: the variable x, operators + - * /, the functions sin and cos, and the
constant placeholder C. Write C in place of every numeric constant or coefficient
-- for example write C*x + C instead of 2.5*x + 1.3. Each C's value is chosen
automatically.
The data may be nonlinear or periodic -- consider terms like x*x, x*x*x, or
sin/cos, not just straight lines.
Reply with ONLY the formula for f(x) on a single line -- no words, no 'y =', no
code fences.

Data points:
  x = -0.900   y = -0.027
  x = -0.700   y = -0.155
  x = -0.500   y = -0.229
  x = -0.300   y = -0.205
  x = +0.000   y = +0.000
  x = +0.300   y = +0.385
  x = +0.500   y = +0.729
  x = +0.700   y = +1.134
  x = +0.900   y = +1.594

New formula for f(x):`}</pre>
        </div>
        <p>
          A correct reply is a single line like <code>x*x + sin(x)</code>, which the verifier
          parses into a Node tree and scores. With const-placeholder mode the LLM is
          encouraged to write <code>C*x*x + C*sin(x)</code> instead; BFGS then fits each{' '}
          <code>C</code> to minimise the error on the data, and the verifier scores the
          constant-substituted tree. The <strong>evolutionary arm</strong> appends one extra
          block before "New formula":
        </p>
        <div className="srl-prompt-wrap">
          <div className="srl-prompt-label">
            <span>extra block — evolution arm only</span>
            <span>archive injected each round</span>
          </div>
          <pre className="srl-prompt">{`Best formulas found so far -- propose a DIFFERENT formula that fits better than these:
  f(x) = sin(x)             (score 0.412)
  f(x) = x*x + C            (score 0.821)
  f(x) = C*x + sin(x)       (score 0.793)`}</pre>
        </div>
        <p>
          ... and that's the whole "AlphaEvolve-style" arm — frozen weights, archive in the
          prompt, do better. The gradient arms instead drop the archive block and train the
          LoRA adapters via GRPO on the same per-sample advantage <Katex tex="w_i" /> formulas
          the Layer-0 RNN uses. Every arm gets the same number of verifier calls per seed;
          the LLM's higher per-sample cost shows up in wall-clock, not in budget.
        </p>

        <h2 className="reveal">The result</h2>
        <p>
          Five seeds per arm, Qwen2.5-0.5B with LoRA on the attention projections, the same
          numeric budget as the Layer-0 sweep. Switch benchmarks below:
        </p>
        <Layer1Transfer />
        <p>
          The ranking <em>survives but only where it matters</em>. On <code>easy</code> every
          arm recovers the exact closed form on every seed — the LLM has <Katex tex="x^2 + 1" />{' '}
          in its prior and no objective is needed. On <code>medium</code> every arm still
          recovers numerically and nearly always symbolically. The methods only{' '}
          <em>separate</em> on <code>harder</code>, and there the Layer-0 order returns:
          risk-seeking is the most reliable (5/5 numeric), best-of-N and entropic are in the
          middle, greedy and evolution trail.
        </p>
        <p>
          But notice the ceiling — exact symbolic recovery on <code>harder</code> nearly
          disappears even though the curves fit. The RNN, whose vocabulary <em>is</em> the
          task grammar, hits the exact expression; the LLM samples from a 150 k-token
          vocabulary, so its search space is vastly wider and it settles for{' '}
          numerically-close. The objective dynamic transfers; the broad prior makes{' '}
          <em>exact</em> recovery harder, not easier. Five seeds is a small N — read this as
          a <em>direction</em>, not a verdict.
        </p>

        <h2 className="reveal">Will reasoning close the gap?</h2>
        <p>
          The previous section ended at a ceiling: numerical recovery is fine, but{' '}
          <em>exact</em> symbolic recovery on <code>harder</code> nearly disappears for the
          LLM proposer. The natural next question — does giving the model a thinking budget
          let it crack that ceiling?
        </p>
        <p>
          We swap Qwen2.5-0.5B for{' '}
          <a className="post-link" href="https://huggingface.co/Qwen/Qwen3-1.7B" target="_blank" rel="noreferrer">
            Qwen3-1.7B
          </a>{' '}
          (dual-mode: the same model can be prompted with or without an internal{' '}
          <code>&lt;think&gt;…&lt;/think&gt;</code> block) and run the harder target at{' '}
          <em>matched batch</em> = 8, five seeds per cell. The thinking budget is 2048 tokens,
          with a TTT-Discover-style "okay, I am out of thinking tokens" sentence spliced in if
          the model hasn't closed <code>&lt;/think&gt;</code> on its own — a forced wrap-up
          that keeps the answer in-distribution.
        </p>
        <table className="srl-results">
          <caption><code>harder</code> · Qwen3-1.7B · 5 seeds · matched batch</caption>
          <thead>
            <tr>
              <th></th>
              <th>no reasoning</th>
              <th>+ reasoning (budget 2048)</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><b>best-of-N</b> (no GRPO update)</td>
              <td>0/5 num · 0/5 DSR-sym</td>
              <td>1/5 num · 0/5 DSR-sym</td>
            </tr>
            <tr>
              <td><b>risk</b> (GRPO + quantile adv.)</td>
              <td>0/5 num · 0/5 DSR-sym</td>
              <td><b>4/5 num</b> · 1/5 DSR-sym*</td>
            </tr>
          </tbody>
        </table>
        <p>
          Two things land. <strong>First</strong>, on the diagonal you see the experiment
          that didn't have to work: <code>risk + reasoning</code> jumps from 0/5 to 4/5
          numeric recovery while the no-reasoning cells stay at zero. Neither GRPO without
          reasoning, nor reasoning without GRPO, gets across the line — both ingredients are
          required. Reasoning rollouts produce useful candidates; GRPO sharpens which ones
          the policy commits to.
        </p>
        <p>
          <strong>Second</strong>, and more honestly, the asterisk on that 1/5 DSR-symbolic
          win is doing real work. The recovered seed-2 expression was{' '}
          <code>x³ − x + sin(2x + π/2)</code>, which is mathematically{' '}
          <Katex tex="\equiv x^3 - x + \cos(2x)" /> — except the LLM also wrote a{' '}
          <code>+ ε·x²</code> padding term that BFGS happened to zero out. The DSR snap
          credits the recovery; a strict 3-term match would not. Across all four cells,{' '}
          <strong>0/20 candidates clear the strict bar</strong>. The ceiling moved, but it
          didn't break.
        </p>
        <p className="srl-caption">
          <em>Five seeds, single target, single model. Read this as a direction-of-effect,
          not a verdict. The reasoning-mode runs use ~30× more tokens per sample than the
          no-reasoning runs, so the per-call comparison above flatters reasoning on a
          per-flop basis — both framings are honest, both belong in the writeup.</em>
        </p>

        <h2 className="reveal">The missing piece: tree search</h2>
        <p>
          The reasoning experiment hints at what's missing. Watch the seed-2 trace: the
          model proposes a near-target shape (right frequency, right cubic), but every round
          it starts from the same fixed prompt and re-derives everything from scratch.
          There's no mechanism to take a 0.97-reward partial-shape candidate and{' '}
          <em>refine</em> it — the rollouts are independent draws, even when one of them
          got close.
        </p>
        <p>
          That's exactly the gap{' '}
          <a className="post-link" href="https://arxiv.org/abs/2511.23473" target="_blank" rel="noreferrer">
            TTT-Discover
          </a>{' '}
          fills, by importing an MCTS-style state buffer on top of GRPO. The mechanism is a
          slimmed{' '}
          <a className="post-link" href="https://arxiv.org/abs/1712.01815" target="_blank" rel="noreferrer">
            AlphaZero PUCT
          </a>: every candidate solution becomes a <em>state</em> in a buffer; each round,
          four states are selected and four rollouts are drawn from each, conditioned on the
          state as a starting point. Selection uses
        </p>
        <p>
          <Katex
            block
            tex={`\\text{score}(s) = Q(s) + c \\cdot P(s) \\cdot \\frac{\\sqrt{1+T}}{1+n(s)}`}
          />
        </p>
        <p>
          where <Katex tex="Q(s)" /> is the best reward any descendant of <Katex tex="s" />{' '}
          has reached (optimistic, not the mean — encourages expansion of promising
          branches), <Katex tex="P(s)" /> is a rank-based prior over the buffer's rewards,{' '}
          <Katex tex="n(s)" /> propagates visit counts up the lineage, and{' '}
          <Katex tex="T" /> is total expansions. Within a batch the four picks are{' '}
          <em>lineage-blocked</em> — once a state is chosen, its ancestors and descendants
          drop out of contention — so each round's rollouts span four genuinely different
          exploration trajectories. AlphaZero handles this via a virtual loss; TTT-Discover
          flips it to hard blocking.
        </p>
        <p>
          We port their setup directly — same PUCT formula, same lineage-blocking, same
          rank-based prior — and replace the four-arm proposer with a single{' '}
          <code>puct</code> arm. The infrastructure for the rest (Qwen3-1.7B, GRPO,
          budgeted reasoning, the forced wrap-up sentence) stays unchanged.
        </p>
        <PuctTree />
        <p>
          The widget above is one full PUCT run on seed 0 — drag the slider to watch
          the buffer grow. By round 6 the policy already proposes a candidate that
          numerically solves the target; by round 22 it has the correct three-term
          structure; by round 30 it has refined the coefficients to within{' '}
          <Katex tex="\sim 10^{-8}" /> of their integer or <Katex tex="\pi/2" /> values
          (mean-squared error <Katex tex="\sim 10^{-20}" /> on the training data).
          After SymPy simplification this is <em>exactly</em>{' '}
          <Katex tex="x^3 - x + \cos(2x)" /> — three raw terms, no padding term BFGS
          has to zero out. The blue lineage in the widget traces back from this best
          leaf to its initial-seed root: that's the actual path the search took.
        </p>

        <h3 className="reveal">How much of this is the tree search, how much is the training?</h3>
        <p>
          With one knob — <code>--lr 0</code> instead of <code>1e-5</code> — we run
          the exact same PUCT setup with a <em>frozen</em> Qwen3-1.7B (no GRPO
          updates). The buffer still grows, PUCT still selects, but the policy that
          generates the rollouts never adapts. Five seeds per cell:
        </p>
        <table className="srl-results">
          <caption><code>harder</code> · Qwen3-1.7B + reasoning · PUCT 4×4 · 5 seeds</caption>
          <thead>
            <tr>
              <th></th>
              <th>numeric</th>
              <th>DSR-sym</th>
              <th>STRICT-sym</th>
              <th>mean best</th>
              <th>mean solve calls</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><b>PUCT + lr=0</b> (no training)</td>
              <td>4/5</td>
              <td>2/5</td>
              <td><b>2/5</b></td>
              <td>0.984</td>
              <td>616</td>
            </tr>
            <tr>
              <td><b>PUCT + GRPO</b></td>
              <td><b>5/5</b></td>
              <td>4/5</td>
              <td><b>4/5</b></td>
              <td>0.984</td>
              <td><b>211</b></td>
            </tr>
          </tbody>
        </table>
        <p>
          Two findings stack. <strong>First, tree search alone — without any
          training — already breaks the ceiling.</strong> The state-buffer plus
          lineage-blocking machinery is enough to land strict-symbolic recovery on
          2 of 5 seeds with a frozen policy. Every prior arm in the arena (across
          Layer-0 RNN risk, Layer-1 LLM risk, evolution, best-of-N, reasoning +
          GRPO without tree search) finished at 0/5 on this metric. That alone is
          the missing piece.
        </p>
        <p>
          <strong>Second, GRPO sharpens the search instead of replacing it.</strong>{' '}
          Training pushes strict-symbolic to 4/5 and finds the answer{' '}
          <em>much earlier</em> — mean solve at 211 verifier calls vs 616 untrained,
          roughly 3× sooner. Seed by seed:
        </p>
        <table className="srl-results">
          <caption>per-seed: when did the policy first numerically solve?</caption>
          <thead>
            <tr>
              <th>seed</th>
              <th>trained calls</th>
              <th>untrained calls</th>
              <th>GRPO speed-up</th>
            </tr>
          </thead>
          <tbody>
            <tr><td>0</td><td><b>96</b></td><td>1152</td><td>+1056 calls (12×)</td></tr>
            <tr><td>1</td><td><b>192</b></td><td>never solved</td><td>uniquely solved by GRPO</td></tr>
            <tr><td>2</td><td><b>160</b></td><td>336</td><td>+176 calls (2.1×)</td></tr>
            <tr><td>3</td><td><b>368</b></td><td>800</td><td>+432 calls (2.2×)</td></tr>
            <tr><td>4</td><td>240</td><td><b>176</b></td><td>−64 calls (one outlier)</td></tr>
          </tbody>
        </table>
        <p>
          On 4 of 5 matched seeds, GRPO finds the solution substantially earlier; on
          seed 1, only the trained policy finds it at all. Seed 4 is the one
          exception — untrained wins by a thin 64-call margin and both finish at
          identical <Katex tex="\text{best} = 0.988" />. The headline is the
          combination: tree search supplies the structural exploration that nothing
          else in the arena had, and GRPO converts the buffer's signal into a
          faster, more reliable search.
        </p>
        <p className="srl-caption">
          <em>Five seeds is still small N — the speed-ups are large enough that the
          ranking would survive a few seed-permutations of the table, but read the
          exact multipliers as rough magnitudes. The full per-round tree of every
          run, plus the analysis script that produced this table, is in the repo at
          <code>experiments/self-improvement-arena/analyze_puct_ablation.py</code>.</em>
        </p>

        <h2 className="reveal">Run it yourself</h2>
        <p>
          Everything in both posts — the four Layer-0 proposers, the LoRA + GRPO Layer-1 LLM
          arm, the verifier, the configs, and the RunPod harness for the cloud sweeps — is in
          the same repo as this blog, under{' '}
          <a className="post-link" href="https://github.com/berniwal/berniwal.github.io/tree/main/experiments/self-improvement-arena" target="_blank" rel="noreferrer">
            experiments/self-improvement-arena
          </a>. The Layer-0 stack is pure numpy (runs anywhere); the LLM stack is PyTorch +
          transformers + peft (GPU); an on-device MLX version of the LLM arm sits alongside
          for Apple Silicon.
        </p>
        <p>
          A <strong>Streamlit visualiser</strong> ships in the same repo — same engine as the
          widgets, but with per-tab controls and the raw LLM prompt / responses on the side.
          Useful for poking at a specific seed, changing a hyperparameter live, or watching
          the policy entropy crash in real time:
        </p>
        <CodeDetails label="Install and launch the visualiser" code={INSTALL_CMD} lang="bash" />
        <p>
          Everything is seeded; reproducing any number across the two posts is one config and
          one command. The full result tables live under <code>results/</code>; the canonical{' '}
          <code>replay.json</code> bakes that power these widgets are committed.
        </p>

        <h2 className="reveal">Wrap-up</h2>
        <p>
          The shape of the post tells the story.{' '}
          <strong>The Layer-0 ranking transferred to the LLM</strong> — risk-seeking
          on top, the rest in order — and so did its ceiling: numerical recovery on{' '}
          <code>harder</code> was easy, exact symbolic recovery essentially impossible.
        </p>
        <p>
          <strong>Reasoning helped, but didn't break the ceiling.</strong>{' '}
          Qwen3-1.7B with a thinking budget got 4/5 numeric and 1/5 DSR-symbolic
          (snap-rescued); strict 3-term recovery stayed at 0/5. The model was clearly
          producing near-target shapes, just never <em>committing</em> to one and
          refining it.
        </p>
        <p>
          <strong>Tree search was the missing piece.</strong> Porting{' '}
          TTT-Discover's PUCT-over-a-state-buffer — even with the policy frozen —
          recovered the exact target on 2 of 5 seeds. Layer it with GRPO and the
          rate climbs to 4 of 5, with the solution found roughly 3× sooner. The
          two mechanisms are complementary: search supplies the structural
          exploration, training accelerates the convergence.
        </p>
        <p>
          The story arc of this two-post series wasn't planned — the Nguyen result
          ended at one ceiling and reasoning bumped against another. PUCT was the
          first thing that broke both.
        </p>

        <h2 className="reveal">References</h2>
        <div className="post-refs">
          <div className="head ref-cite">Cite</div>
          <div className="head ref-title">Title</div>
          <div className="head ref-link">Link</div>

          <div className="ref-cite">Shao et al. 2024</div>
          <div>
            <div className="ref-title">
              <a href="https://arxiv.org/abs/2402.03300" target="_blank" rel="noreferrer">DeepSeekMath / GRPO</a>
            </div>
            <div className="ref-note">Group-relative policy optimisation — the GRPO update used for the LoRA arm.</div>
          </div>
          <div className="ref-link"><a href="https://arxiv.org/abs/2402.03300" target="_blank" rel="noreferrer">arxiv 2402.03300</a></div>

          <div className="ref-cite">Petersen et al. 2021</div>
          <div>
            <div className="ref-title">
              <a href="https://arxiv.org/abs/1912.04871" target="_blank" rel="noreferrer">Deep Symbolic Regression</a>
            </div>
            <div className="ref-note">Risk-seeking policy gradient: the Layer-0 reference and the source of the const-placeholder trick.</div>
          </div>
          <div className="ref-link"><a href="https://arxiv.org/abs/1912.04871" target="_blank" rel="noreferrer">arxiv 1912.04871</a></div>

          <div className="ref-cite">Jiang et al. 2025</div>
          <div>
            <div className="ref-title">
              <a href="https://arxiv.org/abs/2511.23473" target="_blank" rel="noreferrer">TTT-Discover</a>
            </div>
            <div className="ref-note">Test-time-training discovery: budgeted reasoning + GRPO + PUCT-style state buffer. The recipe we port for the reasoning and tree-search sections.</div>
          </div>
          <div className="ref-link"><a href="https://arxiv.org/abs/2511.23473" target="_blank" rel="noreferrer">arxiv 2511.23473</a></div>

          <div className="ref-cite">Silver et al. 2017</div>
          <div>
            <div className="ref-title">
              <a href="https://arxiv.org/abs/1712.01815" target="_blank" rel="noreferrer">AlphaZero — mastering chess and shogi by self-play</a>
            </div>
            <div className="ref-note">The PUCT formula and its lineage-tracked tree expansion — what TTT-Discover (and we) adapt at the state-of-candidate-solutions level.</div>
          </div>
          <div className="ref-link"><a href="https://arxiv.org/abs/1712.01815" target="_blank" rel="noreferrer">arxiv 1712.01815</a></div>

          <div className="ref-cite">Novikov et al. 2025</div>
          <div>
            <div className="ref-title">
              <a href="https://arxiv.org/abs/2506.13131" target="_blank" rel="noreferrer">AlphaEvolve — evolutionary coding with LLMs</a>
            </div>
            <div className="ref-note">In-context program-database evolution: the template for our LLM evolution arm.</div>
          </div>
          <div className="ref-link"><a href="https://arxiv.org/abs/2506.13131" target="_blank" rel="noreferrer">arxiv 2506.13131</a></div>

          <div className="ref-cite">Romera-Paredes et al. 2023</div>
          <div>
            <div className="ref-title">
              <a href="https://www.nature.com/articles/s41586-023-06924-6" target="_blank" rel="noreferrer">FunSearch — mathematical discoveries with LLMs</a>
            </div>
            <div className="ref-note">Frozen-LLM search with an archive of best programs in the prompt.</div>
          </div>
          <div className="ref-link"><a href="https://www.nature.com/articles/s41586-023-06924-6" target="_blank" rel="noreferrer">nature</a></div>

          <div className="ref-cite">Hu et al. 2021</div>
          <div>
            <div className="ref-title">
              <a href="https://arxiv.org/abs/2106.09685" target="_blank" rel="noreferrer">LoRA — low-rank adaptation of large language models</a>
            </div>
            <div className="ref-note">The cheap-fine-tune trick that makes per-arm Qwen training fit on a single GPU.</div>
          </div>
          <div className="ref-link"><a href="https://arxiv.org/abs/2106.09685" target="_blank" rel="noreferrer">arxiv 2106.09685</a></div>

          <div className="ref-cite">Brown et al. 2020</div>
          <div>
            <div className="ref-title">
              <a href="https://arxiv.org/abs/2005.14165" target="_blank" rel="noreferrer">Language Models are Few-Shot Learners</a>
            </div>
            <div className="ref-note">The original few-shot prompting work — relevant to how the data points are placed inline in the prompt.</div>
          </div>
          <div className="ref-link"><a href="https://arxiv.org/abs/2005.14165" target="_blank" rel="noreferrer">arxiv 2005.14165</a></div>

          <div className="ref-cite">this post</div>
          <div>
            <div className="ref-title">
              <a href="https://github.com/berniwal/berniwal.github.io/tree/main/experiments/self-improvement-arena" target="_blank" rel="noreferrer">berniwal.github.io / experiments / self-improvement-arena</a>
            </div>
            <div className="ref-note">All proposers, configs, RunPod harness, Streamlit visualiser.</div>
          </div>
          <div className="ref-link"><a href="https://github.com/berniwal/berniwal.github.io/tree/main/experiments/self-improvement-arena" target="_blank" rel="noreferrer">github</a></div>

          <div className="ref-cite">Earlier posts</div>
          <div>
            <div className="ref-title">
              <a href="/blog/symbolic-regression-arena">Racing Search Algorithms on Symbolic Regression</a>
            </div>
            <div className="ref-note">Part 1 of this series — the Layer-0 arena and the Nguyen result.</div>
          </div>
          <div className="ref-link"><a href="/blog/symbolic-regression-arena">post</a></div>
        </div>

        <footer className="post-footer">
          <p>
            <strong>Part 2 of Symbolic Regression</strong> · Previous:{' '}
            <a className="post-link" href="/blog/symbolic-regression-arena">Racing Search Algorithms on Symbolic Regression</a>
            {' · '}Start of the series:{' '}
            <a className="post-link" href="/blog/symbolic-regression-arena">Racing Search Algorithms on Symbolic Regression</a>.
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
