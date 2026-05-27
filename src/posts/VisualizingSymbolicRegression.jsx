// src/posts/VisualizingSymbolicRegression.jsx
// "Visualizing ML" applied note — Symbolic Regression as a self-improvement testbed.
// Three proposers (evolution / greedy RL / risk-seeking RL) compete on the SAME task,
// SAME verifier, SAME budget; only the proposer changes. Then the numpy RNN is swapped
// for an LLM (Qwen-0.5B) behind the SAME ask/tell seam to test whether the ranking
// transfers. Data is pre-computed (experiments/self-improvement-arena) and read here
// as static JSON; this component renders, it does not train.
//
// Reuses the shared `.viz-post` design system (tokens + viz-panel / viz-tabs / viz-refs /
// viz-byline / viz-footer) defined by the other posts' CSS (globally bundled by CRA), so
// only widget-specific styles live in VisualizingSymbolicRegression.css.
import React, { useEffect, useRef, useState } from 'react';
import './VisualizingSymbolicRegression.css';

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
  return <span ref={ref} className={block ? 'viz-math-block' : 'viz-math-inline'} />;
}

/* ============================ data hooks ============================ */
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

/* ============================ tiny SVG charts ============================ */
// Generic best-so-far metric vs. verifier calls, one line per method, cursor at current step.
function MetricChart({ methods, order, field, cursorIdx, selected, onSelect, fixedMax, fmtX }) {
  const W = 380, H = 210, pad = { l: 40, r: 10, t: 12, b: 30 };
  const iw = W - pad.l - pad.r, ih = H - pad.t - pad.b;
  const refCalls = (methods[order[0]]?.checkpoints || []).map((c) => c.calls);
  const maxCalls = Math.max(1, ...refCalls);
  const allV = order.flatMap((m) => (methods[m]?.checkpoints || []).map((c) => c[field] ?? 0));
  const maxV = fixedMax ?? Math.max(0.1, ...allV);
  const X = (c) => pad.l + (c / maxCalls) * iw;
  const Y = (v) => pad.t + (1 - (v ?? 0) / maxV) * ih;
  const path = (cps) => cps.map((c, i) => `${i ? 'L' : 'M'}${X(c.calls).toFixed(1)},${Y(c[field]).toFixed(1)}`).join(' ');
  const cursorCalls = refCalls[Math.min(cursorIdx, refCalls.length - 1)] ?? 0;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img">
      <line x1={pad.l} y1={pad.t} x2={pad.l} y2={pad.t + ih} stroke="#d8d2c4" />
      <line x1={pad.l} y1={pad.t + ih} x2={pad.l + iw} y2={pad.t + ih} stroke="#d8d2c4" />
      {[0, 0.5, 1].map((f) => (
        <text key={f} x={pad.l - 6} y={Y(maxV * f) + 3} fontSize="9" fill="#9b948a" textAnchor="end">{(maxV * f).toFixed(1)}</text>
      ))}
      <text x={pad.l + iw} y={pad.t + ih + 22} fontSize="9" fill="#9b948a" textAnchor="end">{fmtX(maxCalls)}</text>
      <line x1={X(cursorCalls)} y1={pad.t} x2={X(cursorCalls)} y2={pad.t + ih} stroke="#b9b2a6" strokeDasharray="3 3" />
      {order.map((m) => {
        const cps = methods[m]?.checkpoints || [];
        if (!cps.length) return null;
        const dim = selected && selected !== m;
        return (
          <path key={m} d={path(cps)} fill="none" stroke={methods[m].color}
            strokeWidth={selected === m ? 2.4 : 1.4} opacity={dim ? 0.22 : 1}
            onClick={() => onSelect && onSelect(m)} style={{ cursor: onSelect ? 'pointer' : 'default' }} />
        );
      })}
    </svg>
  );
}

// Target curve vs the selected method's current best-fit overlay.
// When the fit closely matches the target (high reward) the fit line turns green and
// thickens, so an overlapping fit reads as "matched" rather than "missing".
const MATCH_GREEN = '#1b9e77';
function FitChart({ xGrid, yTarget, yPred, color, matched }) {
  const W = 380, H = 210, pad = { l: 36, r: 10, t: 12, b: 24 };
  const iw = W - pad.l - pad.r, ih = H - pad.t - pad.b;
  if (!xGrid || !xGrid.length) return null;
  const finite = (arr) => (arr || []).filter((v) => v != null && isFinite(v));
  const ys = [...finite(yTarget), ...finite(yPred)];
  let ymin = Math.min(...ys), ymax = Math.max(...ys);
  if (!isFinite(ymin) || !isFinite(ymax) || ymin === ymax) { ymin -= 1; ymax += 1; }
  const xmin = xGrid[0], xmax = xGrid[xGrid.length - 1];
  const X = (x) => pad.l + ((x - xmin) / (xmax - xmin)) * iw;
  const Y = (y) => pad.t + (1 - (y - ymin) / (ymax - ymin)) * ih;
  const line = (arr) => {
    let d = '', pen = false;
    arr.forEach((y, i) => {
      if (y == null || !isFinite(y)) { pen = false; return; }
      d += `${pen ? 'L' : 'M'}${X(xGrid[i]).toFixed(1)},${Y(y).toFixed(1)}`; pen = true;
    });
    return d;
  };
  const fitColor = matched ? MATCH_GREEN : color;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img">
      <line x1={pad.l} y1={pad.t} x2={pad.l} y2={pad.t + ih} stroke="#d8d2c4" />
      <line x1={pad.l} y1={pad.t + ih} x2={pad.l + iw} y2={pad.t + ih} stroke="#d8d2c4" />
      {/* target: a faint wide band so an overlapping fit sits visibly inside it, plus the dashed line */}
      <path d={line(yTarget)} fill="none" stroke="#c9c2b4" strokeWidth="6" strokeLinecap="round" opacity="0.6" />
      <path d={line(yTarget)} fill="none" stroke="#1A1A1A" strokeWidth="1.6" strokeDasharray="4 3" />
      {yPred && <path d={line(yPred)} fill="none" stroke={fitColor} strokeWidth={matched ? 2.6 : 2} />}
      <text x={pad.l + 4} y={pad.t + 10} fontSize="9" fill="#9b948a">
        target band&nbsp;&nbsp;<tspan fill={fitColor}>{matched ? '✓ fit matches' : '—— fit'}</tspan>
      </text>
    </svg>
  );
}

const fmtCalls = (n) => (n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : `${Math.round(n / 1000)}k`) + ' calls';

/* ============================ Layer-0 player ============================ */
const L0_ORDER = ['random', 'gp', 'greedy', 'cvar', 'risk', 'risk_entropic'];

function ArenaPlayer() {
  const [reward, setReward] = useState('nrmse');
  const [target, setTarget] = useState('medium');
  const [selected, setSelected] = useState('greedy');
  const [idx, setIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const data = useJson(`${DATA_BASE}/layer0_${reward}.json`);

  const tgt = data && data.targets && data.targets[target];
  const methods = (tgt && tgt.methods) || {};
  const order = L0_ORDER.filter((m) => methods[m]);
  const nFrames = (methods[order[0]] && methods[order[0]].checkpoints.length) || 0;

  useEffect(() => { setIdx(0); setPlaying(false); }, [reward, target]);
  useEffect(() => {
    if (!playing || !nFrames) return;
    const t = setInterval(() => setIdx((i) => { if (i >= nFrames - 1) { setPlaying(false); return i; } return i + 1; }), 130);
    return () => clearInterval(t);
  }, [playing, nFrames]);

  if (data === false) return <div className="viz-panel"><p className="sr-note">Could not load Layer-0 data.</p></div>;
  if (!data) return <div className="viz-panel"><p className="sr-note">Loading…</p></div>;
  if (!tgt) return <div className="viz-panel"><p className="sr-note">No data for {target}/{reward}.</p></div>;

  const cur = (methods[selected] && methods[selected].checkpoints[Math.min(idx, nFrames - 1)]) || {};
  const rows = cur.rows || [];
  const best = cur.best ?? 0;
  const matched = best >= 0.98;  // fit visually coincides with the target

  return (
    <div className="viz-panel">
      <div className="viz-controls">
        <label>target
          <div className="viz-tabs">
            {['easy', 'medium', 'harder'].map((t) => (
              <button key={t} className={`viz-tab ${target === t ? 'active' : ''}`} onClick={() => setTarget(t)}>{t}</button>
            ))}
          </div>
        </label>
        <label>reward
          <div className="viz-tabs">
            {['nrmse', 'mse'].map((r) => (
              <button key={r} className={`viz-tab ${reward === r ? 'active' : ''}`} onClick={() => setReward(r)}>{r}</button>
            ))}
          </div>
        </label>
        <button className={playing ? 'active' : ''} onClick={() => { if (idx >= nFrames - 1) setIdx(0); setPlaying((p) => !p); }}>
          {playing ? '❚❚ pause' : '▶ play'}
        </button>
        <input type="range" min={0} max={Math.max(0, nFrames - 1)} value={idx} style={{ flex: '1 1 180px' }}
          onChange={(e) => { setPlaying(false); setIdx(+e.target.value); }} />
      </div>

      <div className="sr-target">
        <span className="sr-be-label">target f(x)</span>
        <code>{tgt.target_infix}</code>
        <span className="sr-be-reward">recover this</span>
      </div>

      <div className="sr-grid3">
        <div className="sr-cell">
          <p className="sr-cap">best-so-far reward</p>
          <MetricChart methods={methods} order={order} field="best" cursorIdx={idx} selected={selected} onSelect={setSelected} fmtX={fmtCalls} />
        </div>
        <div className="sr-cell">
          <p className="sr-cap">policy entropy (lower → committed)</p>
          <MetricChart methods={methods} order={order} field="entropy" cursorIdx={idx} selected={selected} onSelect={setSelected} fixedMax={2.0} fmtX={fmtCalls} />
        </div>
        <div className="sr-cell">
          <p className="sr-cap">{methods[selected] && methods[selected].label} — fit @ {Math.round((cur.calls || 0) / 1000)}k{matched && <span className="sr-ok"> ✓ on target</span>}</p>
          <FitChart xGrid={tgt.x_grid} yTarget={tgt.y_target} yPred={cur.y_pred} color={methods[selected] && methods[selected].color} matched={matched} />
        </div>
      </div>

      <div className="sr-bestexpr">
        <span className="sr-be-label">current best</span>
        <code className={matched ? 'sr-ok-code' : ''}>{cur.best_infix || '—'}</code>
        <span className="sr-be-reward">reward {best.toFixed(3)}</span>
      </div>

      <div className="sr-legend">
        {order.map((m) => (
          <span key={m} className={`sr-chip ${selected === m ? 'active' : ''}`} onClick={() => setSelected(m)}>
            <span className="sr-swatch" style={{ background: methods[m].color }} />{methods[m].label}
          </span>
        ))}
      </div>

      <div className="sr-stats">
        <span>best <b>{(cur.best ?? 0).toFixed(3)}</b></span>
        <span>policy H <b>{(cur.entropy ?? 0).toFixed(2)}</b></span>
        <span>unique-fraction <b>{(cur.diversity ?? 0).toFixed(2)}</b></span>
        <span>solved-in-batch <b>{((cur.success_frac ?? 0) * 100).toFixed(0)}%</b></span>
      </div>

      <div className="sr-batchwrap">
        <p className="sr-cap">{methods[selected] && methods[selected].label} — top batch proposals (what the policy is sampling now)</p>
        <table className="sr-batch">
          <thead><tr><th>expression</th><th className="num">count</th><th className="num">reward</th></tr></thead>
          <tbody>
            {rows.length === 0 && <tr><td colSpan={3} className="sr-note">no proposals at this frame</td></tr>}
            {rows.map((r, i) => (
              <tr key={i}><td className="sr-expr">{r[0]}</td><td className="num">{r[1]}</td><td className="num">{r[2] == null ? '—' : r[2].toFixed(3)}</td></tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/* ============================ Layer-0 results table ============================ */
// Both reward modes, success@2M + mean best reward, 20 seeds (results/layer0_*/results.md, post-fix).
const C = { random: '#9e9e9e', gp: '#1b9e77', cvar: '#e6ab02', greedy: '#d95f02', risk: '#7570b3', entropic: '#e7298a' };
const L0 = {
  nrmse: {
    easy: [
      ['Risk-seeking RL (DSR)', C.risk, 1.00, 0.995], ['Evolution (GP)', C.gp, 1.00, 0.995],
      ['Random search', C.random, 1.00, 0.995], ['Risk-averse CVaR', C.cvar, 1.00, 0.995],
      ['Entropic RL (Jβ)', C.entropic, 0.75, 0.961], ['Greedy RL', C.greedy, 0.15, 0.841],
    ],
    medium: [
      ['Risk-seeking RL (DSR)', C.risk, 1.00, 0.994], ['Evolution (GP)', C.gp, 1.00, 0.994],
      ['Random search', C.random, 1.00, 0.994], ['Risk-averse CVaR', C.cvar, 0.90, 0.986],
      ['Entropic RL (Jβ)', C.entropic, 0.30, 0.875], ['Greedy RL', C.greedy, 0.05, 0.829],
    ],
    harder: [
      ['Risk-seeking RL (DSR)', C.risk, 0.75, 0.980], ['Evolution (GP)', C.gp, 0.15, 0.945],
      ['Random search', C.random, 0.00, 0.922], ['Entropic RL (Jβ)', C.entropic, 0.00, 0.918],
      ['Greedy RL', C.greedy, 0.00, 0.848], ['Risk-averse CVaR', C.cvar, 0.00, 0.784],
    ],
  },
  mse: {
    easy: [
      ['Risk-seeking RL (DSR)', C.risk, 1.00, 0.995], ['Evolution (GP)', C.gp, 1.00, 0.995],
      ['Random search', C.random, 1.00, 0.995], ['Risk-averse CVaR', C.cvar, 1.00, 0.995],
      ['Greedy RL', C.greedy, 0.90, 0.967], ['Entropic RL (Jβ)', C.entropic, 0.85, 0.967],
    ],
    medium: [
      ['Risk-seeking RL (DSR)', C.risk, 1.00, 0.994], ['Evolution (GP)', C.gp, 1.00, 0.993],
      ['Random search', C.random, 1.00, 0.994], ['Risk-averse CVaR', C.cvar, 0.95, 0.994],
      ['Entropic RL (Jβ)', C.entropic, 0.45, 0.843], ['Greedy RL', C.greedy, 0.25, 0.831],
    ],
    harder: [
      ['Risk-seeking RL (DSR)', C.risk, 0.10, 0.723], ['Evolution (GP)', C.gp, 0.05, 0.784],
      ['Risk-averse CVaR', C.cvar, 0.00, 0.689], ['Random search', C.random, 0.00, 0.678],
      ['Entropic RL (Jβ)', C.entropic, 0.00, 0.674], ['Greedy RL', C.greedy, 0.00, 0.655],
    ],
  },
};

function Layer0Results() {
  const [t, setT] = useState('harder');
  const [reward, setReward] = useState('nrmse');
  const rows = L0[reward][t];
  return (
    <div className="viz-panel">
      <div className="viz-controls">
        <label>benchmark
          <div className="viz-tabs">
            {['easy', 'medium', 'harder'].map((k) => (
              <button key={k} className={`viz-tab ${t === k ? 'active' : ''}`} onClick={() => setT(k)}>{k}</button>
            ))}
          </div>
        </label>
        <label>reward
          <div className="viz-tabs">
            {['nrmse', 'mse'].map((k) => (
              <button key={k} className={`viz-tab ${reward === k ? 'active' : ''}`} onClick={() => setReward(k)}>{k}</button>
            ))}
          </div>
        </label>
        <span className="sr-note" style={{ marginLeft: 'auto' }}>20 seeds · 2M calls</span>
      </div>
      <table className="sr-results">
        <thead><tr><th>method</th><th>success @2M</th><th>mean best reward</th></tr></thead>
        <tbody>
          {rows.map(([label, color, s, r]) => (
            <tr key={label}>
              <td><span className="sr-swatch" style={{ background: color }} /> {label}</td>
              <td><b>{s.toFixed(2)}</b></td>
              <td>{r.toFixed(3)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {reward === 'mse' && t === 'harder' && (
        <p className="sr-note" style={{ marginTop: 8 }}>
          Under MSE on <code>harder</code>, even risk-seeking gets only 10% — the reward saturates
          once the curve is roughly right, so the gradient signal toward exact recovery is largely
          gone. Switch back to nrmse to see the same arm at 75%.
        </p>
      )}
    </div>
  );
}

/* ============================ Nguyen 1-8 table ============================ */
// Symbolic recovery / 25 seeds, recomputed locally with SymPy from the
// nguyen-nrmse-2026-05-25 logs (L0 = {+,-,*,/,sin,cos,exp,log,x}, nrmse reward, 2M calls).
const NGUYEN = [
  ['nguyen-1', 'x³ + x² + x',           [25, 22, 24, 25]],
  ['nguyen-2', 'x⁴ + x³ + x² + x',      [22, 10, 24, 25]],
  ['nguyen-3', 'x⁵ + … + x',            [ 0,  4, 22, 23]],
  ['nguyen-4', 'x⁶ + … + x',            [ 0,  1, 19, 22]],
  ['nguyen-5', 'sin(x²)·cos(x) − 1',    [ 0,  3,  0, 11]],
  ['nguyen-6', 'sin(x) + sin(x + x²)',  [ 0, 15, 25, 25]],
  ['nguyen-7', 'log(x+1) + log(x²+1)',  [ 0,  0,  0,  0]],
  ['nguyen-8', 'sqrt(x) = exp(x/(x+x)·log x)', [ 0,  1,  0, 16]],
];
const NG_AVG = [47, 56, 114, 147];          // /200
const NG_COLORS = [C.random, C.gp, C.greedy, C.risk];
const NG_LABELS = ['Random', 'Evolution (GP)', 'Greedy (= VPG)', 'Risk-seeking'];

function NguyenTable() {
  return (
    <div className="viz-panel">
      <table className="sr-results sr-nguyen">
        <thead>
          <tr>
            <th>target</th>
            <th style={{ textAlign: 'left' }}>formula</th>
            {NG_LABELS.map((l, i) => (
              <th key={l}><span className="sr-swatch" style={{ background: NG_COLORS[i] }} /> {l}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {NGUYEN.map(([t, f, row]) => {
            const max = Math.max(...row);
            return (
              <tr key={t}>
                <td><code>{t}</code></td>
                <td style={{ textAlign: 'left' }} className="sr-expr">{f}</td>
                {row.map((v, i) => {
                  const cls = v === max && v > 0 ? 'sr-cell-best' : '';
                  const hl = t === 'nguyen-5' && i === 3 ? 'sr-cell-uniq' : '';
                  return <td key={i} className={`${cls} ${hl}`}><b>{v}</b>/25</td>;
                })}
              </tr>
            );
          })}
          <tr className="sr-row-avg">
            <td colSpan={2}><b>average (symbolic recovery)</b></td>
            {NG_AVG.map((v, i) => (
              <td key={i}><b>{v}</b>/200 <span className="sr-pct">({(v / 200 * 100).toFixed(0)}%)</span></td>
            ))}
          </tr>
        </tbody>
      </table>
      <p className="sr-note" style={{ marginTop: 8 }}>
        Symbolic recovery (exact SymPy equivalence), L₀ grammar, nrmse reward, 25 seeds, 2M calls.
        Nguyen-5 row: risk-seeking is the only method to recover it (11/25). Bold = best in row.
      </p>
    </div>
  );
}

/* ============================ Layer-1 transfer ============================ */
const L1_ARM_ORDER = ['risk', 'best_of_n', 'entropic', 'evolution', 'greedy'];

function Layer1Transfer() {
  const data = useJson(`${DATA_BASE}/layer1.json`);
  const [t, setT] = useState('harder');
  if (data === false) return <div className="viz-panel"><p className="sr-note">Could not load Layer-1 data.</p></div>;
  if (!data) return <div className="viz-panel"><p className="sr-note">Loading…</p></div>;
  const targets = data.targets || {};
  const tgt = targets[t];
  const arms = (tgt && tgt.arms) || {};
  const order = L1_ARM_ORDER.filter((a) => arms[a]);
  const avail = ['easy', 'medium', 'harder'].filter((k) => targets[k]);

  return (
    <div className="viz-panel">
      <div className="viz-controls">
        <label>benchmark
          <div className="viz-tabs">
            {avail.map((k) => (
              <button key={k} className={`viz-tab ${t === k ? 'active' : ''}`} onClick={() => setT(k)}>{k}</button>
            ))}
          </div>
        </label>
        <span className="sr-note" style={{ marginLeft: 'auto' }}>Qwen2.5-0.5B · GRPO + LoRA</span>
      </div>
      <table className="sr-results">
        <thead><tr><th>arm</th><th>numeric recovery</th><th>symbolic recovery</th><th>mean best</th></tr></thead>
        <tbody>
          {order.map((a) => {
            const arm = arms[a];
            return (
              <tr key={a}>
                <td><span className="sr-swatch" style={{ background: arm.color }} /> {arm.label}</td>
                <td><b>{arm.numeric_solved}/{arm.seeds}</b></td>
                <td>{arm.symbolic_solved}/{arm.seeds}</td>
                <td>{arm.mean_best != null ? arm.mean_best.toFixed(3) : '—'}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="sr-note" style={{ marginTop: 10 }}>
        {t === 'easy'
          ? 'On easy every arm recovers the exact closed form on every seed — x²+1 is in the model\'s prior, no objective needed.'
          : t === 'medium'
          ? 'On medium every arm recovers numerically; symbolic recovery is also near-perfect (the entropic arm misses one seed). The methods only separate on harder.'
          : 'On harder every arm gets the curve roughly right, but exact symbolic recovery nearly vanishes — best-of-N lands it once in five, the rest never.'}
      </p>
    </div>
  );
}

/* ============================ the post ============================ */
export default function VisualizingSymbolicRegression() {
  return (
    <article className="viz-post">
      <div className="viz-wide">
        <div className="viz-series-tag">Visualizing ML · Applied</div>
        <h1>Watching Search Algorithms Discover a Formula</h1>
        <p className="viz-lede">
          Symbolic regression makes a famous self-improvement dynamic visible in seconds on a
          laptop: optimizing the <em>average</em> sample collapses onto a simple-but-wrong answer,
          while optimizing the <em>best</em> sample keeps exploring and finds the peak. We race
          evolution, greedy RL, and risk-seeking RL under one shared budget — then swap the search
          policy for an LLM and ask whether the ranking survives.
        </p>
        <div className="viz-byline">
          By <strong>Bernhard Walser</strong> &amp;{' '}
          <a className="viz-link" href="https://www.anthropic.com/claude" target="_blank" rel="noreferrer"><strong>Claude</strong></a>
          {' '}(Anthropic) · co-written and co-designed ·{' '}
          <a className="viz-link" href="https://www.linkedin.com/in/bernhardwalser/" target="_blank" rel="noreferrer">LinkedIn</a>
          {' · '}
          <a className="viz-link" href="https://github.com/berniwal" target="_blank" rel="noreferrer">GitHub</a>
        </div>

        <p>
          A companion to <a className="viz-link" href="#/blog/visualizing-self-improvement">Visualizing Self-Improving AI</a>,
          which mapped the family of methods conceptually. This post is the lab bench: the same
          dynamic, but small enough to run, watch, and poke. Methods like AlphaEvolve, FunSearch
          and TTT-Discover are expensive and LLM-bound, which makes their underlying search hard
          to <em>see</em>. So we reproduce it on <strong>symbolic regression</strong> with the
          fast RNN-based setup from{' '}
          <a className="viz-link" href="https://arxiv.org/abs/1912.04871" target="_blank" rel="noreferrer">Deep Symbolic Regression</a>{' '}
          (Petersen et al. 2021) — given a handful of <Katex tex="(x, y)" /> points, recover the
          function that generated them.
        </p>

        <h2>What is symbolic regression?</h2>
        <p>
          Ordinary regression fixes a <em>form</em> in advance — say a line{' '}
          <Katex tex="y = ax + b" /> — and fits its numbers to the data. <strong>Symbolic
          regression</strong> doesn't get the form for free: it searches over the space of
          mathematical <em>expressions themselves</em> for one that both fits the points and is a
          plausible closed form — <Katex tex="x^2 + \sin x" /> rather than a 12-parameter
          polynomial that happens to pass through them. The output is an equation you can read,
          not a black box.
        </p>
        <p>
          That makes it a discrete search over a combinatorial space of expression trees: every
          internal node is an operator (<Katex tex="+,\,\times,\,\sin,\dots" />) and every leaf is a
          variable or constant. The space is enormous and non-smooth — swap one operator and the
          output changes wildly — so you cannot just follow a gradient to the right <em>structure</em>.
          That is exactly why it is a clean microcosm for the broader discovery problem: a verifier
          that scores any candidate cheaply, and a proposer that has to <em>explore</em> a huge
          combinatorial space to find the rare expression that nails it. The only thing we vary is
          how that proposer decides what to try next.
        </p>

        <h2>The setup</h2>
        <p>
          The experiment is deliberately controlled. Four search methods compete on the{' '}
          <strong>same task</strong>, scored by the <strong>same verifier</strong>, under the{' '}
          <strong>same budget</strong> — counted in <em>verifier calls</em>, so no method can win
          by simply evaluating more candidates. A candidate is an expression tree over a small
          grammar (<Katex tex="x" />, a few constants, <Katex tex="+\,-\,\times\,\div" />,{' '}
          <Katex tex="\sin,\cos" />). The verifier scores each candidate by how well it fits the
          data:
        </p>
        <p style={{ textAlign: 'center', margin: '0.5rem 0 0.8rem' }}>
          <Katex tex="\mathrm{reward} \;=\; \dfrac{1}{1 + \mathrm{NRMSE}(\hat y, y)}, \qquad \mathrm{NRMSE} = \dfrac{\sqrt{\mathrm{MSE}}}{\sigma_y}" block />
        </p>
        <p>
          where <Katex tex="\mathrm{MSE}" /> is the mean squared error between the candidate's
          predictions <Katex tex="\hat y" /> and the data <Katex tex="y" />, and{' '}
          <Katex tex="\sigma_y" /> is the standard deviation of the data — the normalisation that
          puts the error on a unit-free scale. Reward <Katex tex="= 1.0" /> is a perfect fit; the{' '}
          <Katex tex="\sqrt{\,}" /> stretches the small-error regime so the gradient signal
          stays alive even when the curve is roughly right (compare to the plain MSE reward{' '}
          <Katex tex="1/(1+\mathrm{MSE})" />, where the same fine improvement barely moves the
          number — visible when you toggle the reward in the widget below). Only the{' '}
          <strong>proposer</strong> changes between methods.
        </p>

        <h2>The four proposers</h2>
        <p>
          Three of the four arms are the <em>same</em> policy-gradient learner — an autoregressive
          policy that emits an expression token by token — trained with a group-relative update
          (the GRPO machinery from the{' '}
          <a className="viz-link" href="#/blog/visualizing-rlhf">RLHF post</a>: sample a batch,
          score each sample, push up the log-probability of the good ones). The objective each
          arm <em>maximizes</em> is always
        </p>
        <p style={{ textAlign: 'center', margin: '0.6rem 0 1rem' }}>
          <Katex tex="\mathcal{J} \;=\; \sum_i w_i \,\log \pi(\tau_i)" block />
        </p>
        <p>
          where <Katex tex="\pi(\tau_i)" /> is the policy's probability of generating the{' '}
          <Katex tex="i" />-th sampled expression <Katex tex="\tau_i" /> (the token-by-token product
          along its sequence), and <Katex tex="w_i" /> is the per-sample <strong>advantage</strong>{' '}
          — the same role it plays in GRPO, a scalar derived from that batch's rewards. Gradient
          ascent on <Katex tex="\mathcal{J}" /> pushes up <Katex tex="\log \pi(\tau_i)" /> in
          proportion to <Katex tex="w_i" />, so samples with large advantage get reinforced and
          samples with small (or zero) advantage are ignored. <strong>The reward enters only
          through <Katex tex="w_i" /></strong>; the rest of the formula — the log-probability term
          and the backprop — is identical across arms. The bullets below are just{' '}
          <em>four different rules for turning a batch of rewards into{' '}
          <Katex tex="w_i" /></em>:
        </p>
        <ul>
          <li>
            <strong>Greedy RL</strong> — maximize the <em>average</em> reward
            <Katex tex="\;\mathbb{E}[R]" />. The weight is the reward minus the batch mean,
            <Katex tex="\;w_i = R_i - \bar{R}" /> (a group-relative advantage). Every sample above
            average is reinforced, every sample below is suppressed. The problem: the fastest way to
            raise a <em>batch average</em> is to pile probability onto one simple, decent expression
            — so the policy collapses onto a single attractor and stops exploring.
          </li>
          <li>
            <strong>Risk-seeking PG (DSR)</strong> — maximize the <Katex tex="(1-\varepsilon)" />{' '}
            reward <em>quantile</em> instead of the mean. Compute the top-<Katex tex="\varepsilon" />{' '}
            threshold <Katex tex="\tilde R_\varepsilon" /> of the batch and weight only the elite:
            <Katex tex="\;w_i = (R_i - \tilde R_\varepsilon)\,\mathbb{1}[R_i \ge \tilde R_\varepsilon]" />,
            zero for everyone else. The bottom 95% contribute <em>no</em> gradient, so the policy is
            never pulled toward "good on average" — it keeps chasing the rare, excellent outliers
            where the real solution hides. This is the core idea of{' '}
            <a className="viz-link" href="https://arxiv.org/abs/1912.04871" target="_blank" rel="noreferrer">Deep Symbolic Regression</a> (Petersen et al. 2021),
            and it recently reached LLM GRPO as{' '}
            <a className="viz-link" href="https://arxiv.org/abs/2510.00911" target="_blank" rel="noreferrer">RiskPO</a> (Ren et al. 2025), which trains on a Mixed Value-at-Risk objective.
          </li>
          <li>
            <strong>Entropic <Katex tex="J_\beta" /></strong> — a soft version of the same idea:
            weight every sample by an exponential tilt, <Katex tex="\;w_i \propto e^{\beta R_i}(R_i - b)" />.
            The temperature <Katex tex="\beta" /> interpolates between greedy
            (<Katex tex="\beta\to 0" />) and "only the single best" (<Katex tex="\beta\to\infty" />).
            This exponential utility is the classical risk-sensitive objective (Howard &amp; Matheson 1972);
            it reappears as the smooth generalization in{' '}
            <a className="viz-link" href="https://arxiv.org/abs/2601.16175" target="_blank" rel="noreferrer">TTT-Discover</a>{' '}
            and, for LLM RLVR, as{' '}
            <a className="viz-link" href="https://arxiv.org/abs/2509.24261" target="_blank" rel="noreferrer">RS-GRPO</a> (Jiang et al. 2025), with the hard quantile as a limiting case.
          </li>
          <li>
            <strong>Evolution (GP)</strong> — no gradients at all. Keep a population, select parents
            by tournament, and breed via subtree crossover and mutation. The classic baseline the RL
            methods must beat; recombination in <em>solution space</em> is a strong competitor when
            the grammar is small and compositional.
          </li>
        </ul>
        <aside className="viz-subsection viz-aside">
          <h4 className="viz-subhead"><span className="viz-sub-tag viz-sub-tag-info">i</span>The lineage</h4>
          <div className="viz-aside-body">
            <p>
              The "optimize the tail, not the mean" knob has a long lineage, reached from two directions
              that rarely cite each other. The <strong>soft</strong> exponential-utility form goes back to
              risk-sensitive control (Howard &amp; Matheson 1972) and resurfaces in LLM RL as TTT-Discover
              and RS-GRPO (Jiang et al. 2025). The <strong>hard</strong> quantile form runs from CVaR
              optimization through DSR's symbolic-regression gradient (2021) to RiskPO (Ren et al. 2025),
              which swaps GRPO's mean advantage for a Mixed Value-at-Risk objective. Soft tilt and hard
              quantile are the same idea — the quantile is just a limiting case of the tilt.
            </p>
          </div>
        </aside>

        <h2>Two things that make the search well-posed</h2>
        <p>
          Before any objective can matter, two pieces of machinery have to be in place — remove
          either and the comparison stops working, regardless of which proposer you use:
        </p>
        <ul>
          <li>
            <strong>Constraints on what the policy may emit.</strong> Left unconstrained, an
            autoregressive policy happily produces non-terminating or absurd token streams. We track
            arity so every sample decodes to a finite tree, cap expression length, and forbid nested
            trig (<Katex tex="\sin(\sin(\dots))" />). A separate ablation that toggles these
            guardrails shows greedy RL collapsing <em>completely</em> without them, while
            risk-seeking only slows down — so the constraints are not cosmetic, they are part of what
            makes policy gradient learnable here at all. (This mirrors the DSR <em>in-situ</em>
            priors.)
          </li>
          <li>
            <strong>Constants handled as structure, then fit numerically.</strong> If a proposer can
            emit arbitrary floating-point numbers, it stops doing symbolic regression and starts
            doing <em>numeric</em> regression — drawing a generic curve shape and tuning its
            constants to fit, which scores well but never recovers the true form. The numpy policy
            sidesteps this with a few fixed constant tokens; the LLM (later) places a single{' '}
            <Katex tex="\texttt{const}" /> placeholder whose values are fit afterward by{' '}
            <strong>BFGS</strong>. Either way the <em>shape</em> is what's searched, and only then are
            the numbers optimized — otherwise the symbolic question becomes meaningless.
          </li>
        </ul>

        <h2>Watch them evolve</h2>
        <p>
          <strong>What you're looking at.</strong> Every RL arm in the widget is the same numpy
          RNN policy (hidden size 32) that emits an expression tree token by token from the
          grammar above, under the four constraints. One training step is: sample a batch of 200
          trees, score each with the verifier, compute each advantage <Katex tex="w_i" /> with
          the arm's rule, take one Adam step. With max length 24 and a 200-tree batch, the full
          2M-call replay you see here is 10 000 gradient steps (log-spaced frames so both the
          early collapse and the late recovery are visible in one scrub).
          Evolution (GP) has no policy — it maintains a population of 200 trees, breeds them with
          tournament + crossover + mutation, and uses the same verifier-call budget. Random search
          just draws fresh trees.
        </p>
        <p>
          A pre-computed replay (one representative seed, 2M verifier calls, log-spaced). Pick a
          target, pick a method, press play. Three views update together: <strong>best-so-far reward</strong>,{' '}
          <strong>policy entropy</strong> (per-step token-distribution entropy averaged across the
          batch — lower means the policy has committed to a sharp distribution), and the{' '}
          <strong>best-fit overlay</strong> against the target. The batch table underneath shows
          what the policy is proposing <em>right now</em> — that's the place to see structural
          lock-in directly.
        </p>
        <ArenaPlayer />
        <p className="viz-caption">
          One representative seed, full 2M-call budget, log-spaced frames so the early collapse
          and the late recovery are both visible in the same scrub. The statistical claim (20 seeds)
          is in the table below; the widget shows <em>how</em> each method moves.
        </p>
        <p style={{ marginTop: '1.4rem' }}><strong>Things to notice once you've scrubbed around:</strong></p>
        <ul>
          <li>
            <strong>Recovery on <code>harder</code> is the headline.</strong> Select risk-seeking
            on <code>harder</code> / <code>nrmse</code>, scrub to the end: the fit snaps green —
            risk recovers <Katex tex="x^3 - x + \cos(2x)" /> exactly (best reward → 0.99). Switch
            to greedy with the same target / reward — it gets to ≈0.94, finds the cubic, but never
            adds the cosine term. Same target, same budget, same gradient machinery; the only
            thing that differs is the advantage formula.
          </li>
          <li>
            <strong>The entropy curve tells the <em>why</em>.</strong> The middle chart is per-step
            policy entropy (sharpness of the next-token distribution; lower = committed). On{' '}
            <code>harder</code> / <code>nrmse</code> watch what happens around the 300k mark:
            <ul style={{ marginTop: 6, marginBottom: 0 }}>
              <li>
                <strong>Greedy</strong> drops to <Katex tex="H \approx 0.5" /> and <em>stays
                there</em> for the rest of the run. It has committed to a wrong attractor and
                cannot leave.
              </li>
              <li>
                <strong>Risk-seeking dips and then climbs back up</strong> (<Katex tex="H" />:
                1.83 → 1.45 → 1.76). It locks in on a candidate structure around 300k, decides
                that structure isn't quite right, and <em>re-explores</em>. That re-exploration is
                what carries it from the 0.94 plateau (a cubic without the cosine) to 0.99 (with
                the cosine).
              </li>
              <li>
                <strong>Entropic <Katex tex="J_\beta" /></strong> goes the other way —
                <Katex tex="H \approx 0.14" /> by 2M. It commits even harder than greedy, also
                gets stuck at 0.94.
              </li>
            </ul>
            The hard top-<Katex tex="\varepsilon" /> quantile is the only arm whose entropy
            <em> rebounds</em>; that ability to back out of a local optimum is, on this target,
            the difference between recovery and a permanent ceiling.
          </li>
          <li>
            <strong>Per-token entropy is still imperfect — read the batch table for structure.</strong>
            Even policy entropy can hide a structural collapse. On the same{' '}
            <code>harder</code> / <code>nrmse</code> frame, the greedy batch is all{' '}
            <code>sin(2.0 − x)</code>, <code>cos(0.5·x)</code>, <code>sin(1.0/x)</code>… every
            row is trig — total skeleton collapse on a target that's mostly polynomial. Risk's
            batch is <code>(x·x)·x</code>, <code>((x·x)·x) − x</code>, <code>((x·x)·x) + x</code>…
            locked on the polynomial family. The unique-string fraction is in the stats line
            (0.92 vs 0.45) but doesn't capture that. <em>The batch table is the only
            structural-collapse view in this widget.</em> A proper structural-skeleton diversity
            metric would be a cleaner next step.
          </li>
          <li>
            <strong>MSE saturates; NRMSE doesn't.</strong> Flip the reward toggle. The MSE reward{' '}
            <Katex tex="1/(1+\mathrm{MSE})" /> has a vanishing-gradient region once the curve is
            roughly right — the gradient signal toward exact recovery dies off. This shows up as a
            <em> rate gap</em> across seeds, not an absolute barrier: across 20 seeds at 2M,
            risk-seeking cracks <code>harder</code> <strong>75% of the time under NRMSE versus
            only 10% under MSE</strong>. The widget seed (seed 0) happens to recover under both,
            so you can see the same closed form in either panel — but most MSE seeds get stuck on
            wrong attractors and never reach the cosine correction. Greedy stays at 0 % recovery
            under either reward: the reward shape changes how good its "almost-right" answer
            <em>looks numerically</em>, but not its inability to escape the wrong attractor.
          </li>
          <li>
            <strong>CVaR is risk-<em>averse</em></strong> — the opposite of risk-seeking. It puts
            its gradient on the <em>bottom</em> of the batch (the worst samples) instead of the
            top. It keeps the broadest diversity (≈1.0) and learns the slowest, often by far. Good
            sanity-check arm: if even being explicitly worst-case-averse worked, the risk-seeking
            story would be much less interesting.
          </li>
          <li>
            The <strong>current-best expression</strong> below the chart is the policy's best find
            so far — watch it lock into the target shape often well before the policy stops
            sampling alternatives.
          </li>
        </ul>

        <h2>Layer 0 results</h2>
        <p>
          Over 20 seeds at the full 2M-call budget. On the easy and medium targets almost everything
          reaches the ceiling, and evolution is the most sample-efficient. The methods only separate
          on <code>harder</code> — the collapse zone — where <strong>risk-seeking recovers the
          formula three times as often as evolution, and greedy never does</strong>.
        </p>
        <Layer0Results />

        <h2>An official benchmark: Nguyen 1–8</h2>
        <p>
          Our three synthetic targets are hand-picked. The <strong>Nguyen suite</strong> is the
          standard symbolic-regression benchmark DSR reports on — a fixed set of eight ground-truth
          expressions, evaluated with the standard library{' '}
          <Katex tex="\mathcal{L}_0 = \{+, -, \times, \div, \sin, \cos, \exp, \log, x\}" /> and
          DSR's recovery metric (fraction of independent seeds whose best expression is{' '}
          <em>exactly</em> symbolically equivalent to the target, checked with SymPy). All eight
          targets are expressible in <Katex tex="\mathcal{L}_0" />, including{' '}
          <code>nguyen-8 = sqrt(x) = exp(log(x)/2)</code>. We ran all four methods, 25 seeds, 2M
          calls, nrmse reward:
        </p>
        <NguyenTable />
        <p>
          The story is clean: <strong>risk-seeking wins on average by ~17 percentage points</strong>{' '}
          (74% vs greedy 57%); evolution and random trail at 28% / 24%. The methods <em>tie</em> on
          the easy polynomials (Nguyen-1, 2, 6) and then separate sharply on the hard rows. Two are
          especially diagnostic: <strong>Nguyen-5</strong> (<Katex tex="\sin(x^2)\cos(x) - 1" />)
          and <strong>Nguyen-8</strong> (<Katex tex="\sqrt{x}" />) — both targets where greedy
          recovers <em>0 of 25</em>, and risk-seeking recovers <em>11 of 25</em> and{' '}
          <em>16 of 25</em> respectively. That is exactly the "rare excellent outlier" regime the
          risk-seeking objective is designed for: solutions reachable only from narrow neighbourhoods
          of expression-space that the mean baseline never pushes the policy toward.
        </p>
        <aside className="viz-subsection viz-aside">
          <h4 className="viz-subhead"><span className="viz-sub-tag viz-sub-tag-info">i</span>A note on the Nguyen-3/4 "VPG collapse"</h4>
          <div className="viz-aside-body">
            <p>
              The DSR paper reports vanilla policy gradient (VPG) recovering Nguyen-3 / Nguyen-4 only
              4% / 1% of the time. Our greedy arm (which is VPG) gets 88% / 76% — looks like a
              contradiction. We cross-checked against DSR's <em>own</em> currently released code (
              <a className="viz-link" href="https://github.com/dso-org/deep-symbolic-optimization" target="_blank" rel="noreferrer">dso-org/deep-symbolic-optimization</a>) by configuring it as VPG (no risk
              filter, EWMA baseline) and running it on Nguyen-3:
            </p>
            <pre className="viz-code">{`# their VPG config: training.epsilon=null, training.baseline=ewma_R
python -m dso.run config_vpg.json --b Nguyen-3 --runs 8 --seed 0
# -> success = 8 / 8     (default LSTM cell)
# -> success = 8 / 8     (BasicRNN cell ablation)`}</pre>
            <p>
              Their VPG also recovers Nguyen-3 ≈100%. We did not rigorously trace which specific
              change between the 2021 paper and their current code closes the reported gap — likely
              some combination of post-2021 drift (max length raised to 64, addition of soft-length
              and uniform-arity priors in the code, etc.). The qualitative point holds though:
              vanilla policy gradient doesn't <em>intrinsically</em> collapse on these polynomial
              Nguyens; the paper's 4%/1% is specific to the 2021 configuration, not a property of
              the algorithm. (And the headline is unchanged: risk-seeking still beats VPG on
              average and uniquely cracks Nguyen-5.) The full cross-check pipeline lives in{' '}
              <code>experiments/self-improvement-arena/runpod/run_dsr.sh</code>.
            </p>
          </div>
        </aside>
        <p>
          The Nguyen-8 row (<Katex tex="\sqrt{x}" />) is worth a closer look. The DSR paper notes
          its construction as <Katex tex="\exp\!\big(\tfrac{x}{x+x}\cdot\log x\big)" /> — a
          composition through both unary operators. Risk-seeking finds it on{' '}
          <strong>16 / 25 seeds (64%)</strong>; greedy never does. DSR reports ~96% with the same
          grammar; the remaining 64 → 96 gap points to implementation details we have not
          rigorously matched — DSR's exact training-loop schedule (entropy / learning-rate decay
          over the 2M steps), per-step sampling logic, and other code-level specifics. We share the obvious machinery — same
          grammar, same hard length cap, the same handling of invalid expressions (reward 0 for
          any NaN / overflow, e.g. <Katex tex="\log" /> of a non-positive subexpression, exactly as
          the paper specifies) — but we have not isolated which of the remaining differences
          closes the gap.
        </p>
        <p>
          Nguyen-7 (<Katex tex="\log(x+1) + \log(x^2+1)" />) is the one row none of our methods
          touch. It's expressible in <Katex tex="\mathcal{L}_0" /> but needs the additive constant
          <Katex tex="\,+1\," /> constructed (as <Katex tex="x/x" />) <em>inside</em> two separate
          <Katex tex="\log" /> arguments — a discovery sequence none of the four policies stumble
          onto in 2M calls. DSR reports ~35% on this benchmark (which is also where they tuned
          their hyperparameters), so the gap is real and roughly the same shape as Nguyen-8 — most
          likely some combination of code / training-loop details we have not replicated.
          Investigating where it closes is a next step, not a finding.
        </p>


        <h2>Does the ranking transfer to an LLM?</h2>
        <p>
          The point of the shared <code>ask()</code> / <code>tell()</code> seam is that the
          proposer is swappable. So we keep the task, the verifier, the budget — and swap the
          small numpy RNN for a real language model (Qwen2.5-0.5B), fine-tuned with{' '}
          <strong>LoRA adapters</strong> under the <em>same</em> advantage formulas. For the
          evolutionary baseline we run the same swap on the other axis: a population of 200 trees
          on Layer 0 becomes a <em>frozen LLM prompted with the best formulas found so far</em>,
          AlphaEvolve / FunSearch style.
        </p>

        <h3>The four proposers, side by side</h3>
        <div className="viz-panel sr-fourcol">
          <div className="sr-col sr-col-l0">
            <div className="sr-col-h"><span className="sr-col-tag">Layer 0</span><strong>Evolution (GP)</strong></div>
            <p><strong>Proposer.</strong> Population of 200 expression trees, evolved with tournament selection + subtree crossover + subtree mutation.</p>
            <p><strong>Search space.</strong> Trees built from <Katex tex="\{x, +, -, \times, \div, \sin, \cos, \mathrm{const}\}" />.</p>
            <p><strong>Optimisation.</strong> None — selection pressure on the population, no gradients.</p>
          </div>
          <div className="sr-col sr-col-l1">
            <div className="sr-col-h"><span className="sr-col-tag">Layer 1</span><strong>Evolution (LLM)</strong></div>
            <p><strong>Proposer.</strong> Frozen LLM prompted each round; the K best formulas found so far are injected into the prompt as "do better than these" (AlphaEvolve / FunSearch).</p>
            <p><strong>Search space.</strong> The LLM's full vocabulary (≈150 k tokens) → free-form formula text → parsed back into the same Node tree the verifier accepts.</p>
            <p><strong>Optimisation.</strong> None — the LLM weights are frozen; selection happens implicitly through the archive in the prompt.</p>
          </div>
          <div className="sr-col sr-col-l0">
            <div className="sr-col-h"><span className="sr-col-tag">Layer 0</span><strong>RNN (policy gradient)</strong></div>
            <p><strong>Proposer.</strong> 32-hidden numpy RNN, samples expression tokens autoregressively under the four constraints; ~5 k parameters.</p>
            <p><strong>Search space.</strong> Same fixed grammar as GP, 10 tokens, structured prefix decoding.</p>
            <p><strong>Optimisation.</strong> Policy gradient — for each batch compute the advantage <Katex tex="w_i" /> (greedy / risk-seeking / entropic) and Adam-step on <Katex tex="\mathcal{J} = \sum_i w_i \log \pi(\tau_i)" />.</p>
          </div>
          <div className="sr-col sr-col-l1">
            <div className="sr-col-h"><span className="sr-col-tag">Layer 1</span><strong>LLM (LoRA + GRPO)</strong></div>
            <p><strong>Proposer.</strong> Qwen2.5-0.5B-Instruct + LoRA adapters on the attention projections. Samples free-form formula strings; we parse them back into a Node tree for the verifier.</p>
            <p><strong>Search space.</strong> Full LLM vocabulary (≈150 k tokens) instead of a 10-token grammar.</p>
            <p><strong>Optimisation.</strong> <em>The same advantage formula</em>: <code>greedy_weights</code>, <code>quantile_weights</code>, <code>entropic_weights</code> from <code>sia.objectives</code> are imported by both proposers. Adam steps on the LoRA params via PyTorch GRPO.</p>
          </div>
        </div>
        <p className="viz-caption">
          The gradient-based row (RNN ↔ LLM + LoRA) shares the exact{' '}
          <Katex tex="\sum_i w_i \log \pi(\tau_i)" /> objective; only the policy <Katex tex="\pi" />{' '}
          and the way we tokenise <Katex tex="\tau_i" /> change. The evolutionary row shares the
          high-level idea (best-so-far drives the next batch) but the LLM version has no gradient
          step at all — exactly the contrast AlphaEvolve / FunSearch are built on.
        </p>

        <h3>What the LLM sees</h3>
        <p>
          For the gradient-based arms (greedy / risk / entropic / best-of-N) we hand the LLM a
          plain prompt and parse its single-line reply. For the medium target{' '}
          <Katex tex="x^2 + \sin(x)" /> with the const-placeholder mode on, the prompt is literally:
        </p>
        <div className="viz-prompt-wrap">
          <div className="viz-prompt-label">
            <span>prompt sent to the LLM</span>
            <span>medium · const-placeholder mode</span>
          </div>
          <pre className="viz-prompt">{`You are doing symbolic regression. Find a formula y = f(x) that fits the data.
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
          parses into a Node tree and scores. With const-placeholder mode the LLM is encouraged to
          write <code>C*x*x + C*sin(x)</code> instead; BFGS then fits each <code>C</code>{' '}
          to minimise the error on the data, and the verifier scores the constant-substituted
          tree. The <strong>evolutionary arm</strong> appends one extra block before "New formula":
        </p>
        <div className="viz-prompt-wrap">
          <div className="viz-prompt-label">
            <span>extra block — evolution arm only</span>
            <span>archive injected each round</span>
          </div>
          <pre className="viz-prompt">{`Best formulas found so far -- propose a DIFFERENT formula that fits better than these:
  f(x) = sin(x)             (score 0.412)
  f(x) = x*x + C            (score 0.821)
  f(x) = C*x + sin(x)       (score 0.793)`}</pre>
        </div>
        <p>
          ... and that's the whole "AlphaEvolve-style" arm — frozen weights, archive in the
          prompt, do better. The gradient arms instead drop the archive block and train the LoRA
          adapters via GRPO on the same per-sample advantage <Katex tex="w_i" /> formulas the
          Layer-0 RNN uses.
        </p>

        <Layer1Transfer />
        <p>
          The ranking <em>survives but only where it matters</em>. On <code>easy</code> every arm
          recovers the exact closed form on every seed — the LLM has <Katex tex="x^2 + 1" /> in its
          prior and no objective is needed. On <code>medium</code> every arm still recovers
          numerically and nearly always symbolically. The methods only <em>separate</em> on{' '}
          <code>harder</code>, and there the Layer-0 order returns: risk-seeking is the most
          reliable (5/5 numeric), best-of-N and entropic are in the middle, greedy and evolution
          trail. But notice the ceiling — exact symbolic recovery on <code>harder</code> nearly
          disappears even though the curves fit. The RNN, whose vocabulary <em>is</em> the task
          grammar, hits the exact expression; the LLM samples from a 150k-token vocabulary, so its
          search space is vastly wider and it settles for numerically-close. The objective dynamic
          transfers; the broad prior makes <em>exact</em> recovery harder, not easier.
        </p>

        {/*
          ============================================================
          TODO -- REASONING ADDENDUM (in flight, finishes 2026-05-27 ~PM):
          ============================================================
          New section to slot in here (between "Does the ranking transfer?"
          and "Run it yourself"). Working title: "What if the LLM also THINKS?"

          Story arc:
            1. Recap the ceiling on `harder` -- symbolic recovery ~zero.
            2. Toggle reasoning ON for Qwen3-1.7B (enable_thinking=True).
            3. Show the resulting 5-seed 2x2 (in-flight data):
                                no-reasoning b=8   |   reasoning b=8 (budgeted)
                  risk            ? / 5 num            ? / 5 num,  ? / 5 sym
                  best_of_n       ? / 5 num            ? / 5 num,  ? / 5 sym
              Headline (preliminary, from 3-seed overnight): reasoning + risk
              hit 2/3 numeric and 1/3 DSR-symbolic on harder, breaking the
              0/5 floor every numerical-only arm sits at.
            4. Caveat #1: the symbolic recovery used const-snap; under the
               strict "no junk terms" test it's 0/3. Be honest about this.
            5. Caveat #2: comparing per-call vs per-dollar of compute -- the
               reasoning arm spends ~30x more tokens per sample. Both framings
               in the writeup.

          ============================================================
          TODO -- INFO BOX: "Fitting RL training on a single GPU"
          ============================================================
          User-suggested aside. Best dropped at the moment we mention "the
          experiment OOM'd until we fixed two things" -- natural pivot from
          narrative ("we hit a wall") to mechanism ("here is what activations
          actually cost").

          Visual concept: a horizontal GPU memory bar (say A40 = 44 GB) sliced
          into colored segments, and a 3-row table showing how the activation
          slice shrinks as we add each technique.

          Row 1 -- "Naive batch=16, maxnew=2048":
            [weights 3.4GB][optim 0.2GB][KV cache 0.3GB][ACTIVATIONS ~40 GB ⚠ OOM]

          Row 2 -- "+ Micro-batch / gradient accumulation (micro=2)":
            same total batch effect, but only 2 rollouts' activations resident
            at a time during backward:
            [weights][optim][KV][activations ~5 GB][   free ~35 GB   ]
            Mechanic: 8 chunks of backward() + .grad accumulation, one step.
            Trade: wall-time (~unchanged because chunks are sequential) for memory.

          Row 3 -- "+ Gradient checkpointing":
            activations are not even fully retained per chunk; layer outputs
            are recomputed during backward (~30% extra compute, ~28x less
            activation memory for Qwen3-1.7B's 28 layers):
            [weights][optim][KV][acts 0.2GB][      free 40 GB      ]
            Trade: compute for memory.

          Optional interactive: drag a "max_new_tokens" slider and a batch
          slider; toggle (a) micro-batch, (b) checkpointing. Bar fills the
          GPU and a "fits? / OOMs" indicator flips. The point is to make the
          two knobs concretely visible as memory shifts.

          Reference snippets we already have running:
            - layer1/torch_lora_proposer.py __init__ (the three lines that
              enabled checkpointing)
            - run_layer1_torch.py --micro-batch flag exposure

          Citations for the info box:
            - Gradient checkpointing (deep-learning version):
                Chen, Xu, Zhang, Guestrin 2016 -- "Training Deep Nets with
                Sublinear Memory Cost", arXiv:1604.06174. The seminal paper
                for the O(N) -> O(sqrt(N)) memory trick by recomputing
                activations in backward.
            - Underlying classical theory (automatic differentiation):
                Griewank & Walther 2000 -- "Algorithm 799: Revolve" (TOMS).
                The optimal recompute schedule for reverse-mode AD; predates
                deep learning by decades.
            - Modern transformer refinement:
                Korthikanti et al. 2022 -- "Reducing Activation Recomputation
                in Large Transformer Models", arXiv:2205.05198 (Megatron-LM).
                Selective recomputation -- only recompute the T^2 attention
                matrix; keep the cheaper bits. ~5% overhead instead of ~30%.
            - Different attack on the same memory wall (composes with
              checkpointing):
                Dao, Fu, Ermon, Rudra, Re 2022 -- "FlashAttention",
                arXiv:2205.14135. Never materialises the T^2 attention matrix
                at all (block-wise tiling + softmax accumulator) -- O(T)
                memory for attention forward AND backward. Backward is built
                in via block-wise recomputation, basically gradient
                checkpointing baked into the attention kernel.
            - Gradient accumulation: folklore, no single seminal paper.
              Mentioned in passing by Smith et al. 2017 (arXiv:1711.00489),
              Goyal et al. 2017 (arXiv:1706.02677), and GPT-3 (Brown 2020) as
              standard tooling for large effective batch sizes.

          Composition note: FlashAttention kills the per-layer T^2 attention
          cost; checkpointing kills the per-layer-times-num-layers
          multiplier on residual / MLP / layernorm activations. Both can be
          on at the same time and they target different terms.
          ============================================================
        */}

        <h2>Run it yourself</h2>
        <p>
          Everything in this post — the four Layer-0 proposers, the LoRA + GRPO Layer-1 LLM arm,
          the verifier, the configs, and the RunPod harness for the cloud sweeps — is in the same
          repo as this blog, under{' '}
          <a className="viz-link" href="https://github.com/berniwal/berniwal.github.io/tree/main/experiments/self-improvement-arena" target="_blank" rel="noreferrer">
            experiments/self-improvement-arena
          </a>. The Layer-0 stack is pure numpy (runs anywhere); the LLM stack is PyTorch +
          transformers + peft (GPU); an on-device MLX version of the LLM arm sits alongside for
          Apple Silicon. <code>pyproject.toml</code> has the optional extras (<code>[app]</code>{' '}
          for Streamlit, <code>[layer1]</code> for the MLX LoRA arms).
        </p>
        <p>
          There's also a <strong>Streamlit visualiser</strong> for stepping the search batch by
          batch on a single task — same engine as this widget, but with per-tab controls and the
          raw LLM prompt / responses on the side. Useful if you want to poke at a specific seed,
          change a hyperparameter live, or watch the policy entropy crash in real time:
        </p>
        <pre className="viz-code">{`git clone https://github.com/berniwal/berniwal.github.io
cd berniwal.github.io/experiments/self-improvement-arena
pip install -e ".[app]"           # core + Streamlit visualiser (numpy only)
streamlit run streamlit_app.py
# (add ",layer1" to the extras on Apple Silicon for the MLX LoRA arms)`}</pre>
        <p>
          Everything is seeded; reproducing any number in the post is one config + one command.
          The full result tables live under <code>results/</code>{' '}
          (<code>layer0_mse/results.md</code>, <code>layer0_nrmse/results.md</code>,{' '}
          <code>nguyen_nrmse/results.md</code>); the per-run JSON logs are gitignored to keep the
          repo light, but the canonical <code>replay.json</code> bakes (which power this widget)
          are committed.
        </p>

        <h2>References</h2>
        <div className="viz-refs">
          <div className="head">cite</div>
          <div className="head">title</div>
          <div className="head">link</div>

          <div className="section-row">Methods</div>

          <div className="ref-cite">Howard &amp; Matheson 1972</div>
          <div>
            <div className="ref-title"><a href="https://doi.org/10.1287/mnsc.18.7.356" target="_blank" rel="noreferrer">Risk-Sensitive Markov Decision Processes</a></div>
            <div className="ref-note">The classical exponential-utility objective the entropic / risk-seeking arms descend from.</div>
          </div>
          <div className="ref-link"><a href="https://doi.org/10.1287/mnsc.18.7.356" target="_blank" rel="noreferrer">doi 10.1287/mnsc.18.7.356</a></div>

          <div className="ref-cite">Petersen et al. 2021</div>
          <div>
            <div className="ref-title"><a href="https://arxiv.org/abs/1912.04871" target="_blank" rel="noreferrer">Deep Symbolic Regression: recovering expressions via risk-seeking policy gradients</a></div>
            <div className="ref-note">The hard top-ε quantile objective and the Nguyen recovery benchmark.</div>
          </div>
          <div className="ref-link"><a href="https://arxiv.org/abs/1912.04871" target="_blank" rel="noreferrer">arxiv 1912.04871</a></div>

          <div className="ref-cite">Jiang et al. 2025</div>
          <div>
            <div className="ref-title"><a href="https://arxiv.org/abs/2601.16175" target="_blank" rel="noreferrer">TTT-Discover — entropic J_β test-time training</a></div>
            <div className="ref-note">Soft exponential-tilt generalization of risk-seeking.</div>
          </div>
          <div className="ref-link"><a href="https://arxiv.org/abs/2601.16175" target="_blank" rel="noreferrer">arxiv 2601.16175</a></div>

          <div className="ref-cite">Jiang et al. 2025</div>
          <div>
            <div className="ref-title"><a href="https://arxiv.org/abs/2509.24261" target="_blank" rel="noreferrer">Risk-Sensitive RL for Alleviating Exploration Dilemmas in Large Language Models</a></div>
            <div className="ref-note">RS-GRPO: exponential-utility risk-seeking advantage as a drop-in for LLM RLVR — the same knob, on LLMs.</div>
          </div>
          <div className="ref-link"><a href="https://arxiv.org/abs/2509.24261" target="_blank" rel="noreferrer">arxiv 2509.24261</a></div>

          <div className="ref-cite">Ren et al. 2025</div>
          <div>
            <div className="ref-title"><a href="https://arxiv.org/abs/2510.00911" target="_blank" rel="noreferrer">RiskPO: Risk-based Policy Optimization via Verifiable Reward for LLM Post-Training</a></div>
            <div className="ref-note">The quantile/VaR mirror of RS-GRPO: a Mixed Value-at-Risk advantage for GRPO — the DSR thread reaching LLMs.</div>
          </div>
          <div className="ref-link"><a href="https://arxiv.org/abs/2510.00911" target="_blank" rel="noreferrer">arxiv 2510.00911</a></div>

          <div className="ref-cite">Novikov et al. 2025</div>
          <div>
            <div className="ref-title"><a href="https://arxiv.org/abs/2506.13131" target="_blank" rel="noreferrer">AlphaEvolve — evolutionary coding with LLMs</a></div>
            <div className="ref-note">In-context program-database evolution (our LLM evolution arm).</div>
          </div>
          <div className="ref-link"><a href="https://arxiv.org/abs/2506.13131" target="_blank" rel="noreferrer">arxiv 2506.13131</a></div>

          <div className="ref-cite">Real et al. 2019</div>
          <div>
            <div className="ref-title"><a href="https://arxiv.org/abs/1802.01548" target="_blank" rel="noreferrer">Regularized Evolution for Image Classifier Architecture Search</a></div>
            <div className="ref-note">Tournament + aging evolution, the GP arm's lineage.</div>
          </div>
          <div className="ref-link"><a href="https://arxiv.org/abs/1802.01548" target="_blank" rel="noreferrer">arxiv 1802.01548</a></div>

          <div className="section-row">Code &amp; data</div>

          <div className="ref-cite">this post</div>
          <div>
            <div className="ref-title"><a href="https://github.com/berniwal/berniwal.github.io/tree/main/experiments/self-improvement-arena" target="_blank" rel="noreferrer">berniwal.github.io / experiments / self-improvement-arena</a></div>
            <div className="ref-note">All proposers, configs, RunPod harness, Streamlit visualiser. <code>pip install -e ".[app]"</code> + <code>streamlit run streamlit_app.py</code>.</div>
          </div>
          <div className="ref-link"><a href="https://github.com/berniwal/berniwal.github.io/tree/main/experiments/self-improvement-arena" target="_blank" rel="noreferrer">github</a></div>

          <div className="ref-cite">DSO</div>
          <div>
            <div className="ref-title"><a href="https://github.com/dso-org/deep-symbolic-optimization" target="_blank" rel="noreferrer">deep-symbolic-optimization</a></div>
            <div className="ref-note">DSR's reference implementation + the Nguyen suite we cross-checked against.</div>
          </div>
          <div className="ref-link"><a href="https://github.com/dso-org/deep-symbolic-optimization" target="_blank" rel="noreferrer">github</a></div>

          <div className="section-row">Earlier posts</div>

          <div className="ref-cite">Visualizing ML</div>
          <div><div className="ref-title"><a href="#/blog/visualizing-self-improvement">Visualizing Self-Improving AI: From AlphaZero to TTT-Discover</a></div></div>
          <div className="ref-link"><a href="#/blog/visualizing-self-improvement">post</a></div>

          <div className="ref-cite">Visualizing ML</div>
          <div><div className="ref-title"><a href="#/blog/visualizing-rlhf">Visualizing RLHF: PPO, DPO, and GRPO</a></div></div>
          <div className="ref-link"><a href="#/blog/visualizing-rlhf">post</a></div>
        </div>

        <footer className="viz-footer">
          <p>
            <strong>A companion to Visualizing ML</strong> · Related:{' '}
            <a className="viz-link" href="#/blog/visualizing-self-improvement">Visualizing Self-Improving AI</a>
            {' · '}Start of the series:{' '}
            <a className="viz-link" href="#/blog/visualizing-attention">Visualizing Attention</a>.
          </p>
          <p style={{ marginBottom: 0 }}>
            Bernhard Walser · ML Engineer, Digitec Galaxus · ETH Computer Science ·{' '}
            <a className="viz-link" href="https://www.linkedin.com/in/bernhardwalser/" target="_blank" rel="noreferrer">LinkedIn</a>
            {' · '}
            <a className="viz-link" href="https://github.com/berniwal" target="_blank" rel="noreferrer">GitHub</a>
          </p>
          <p style={{ marginTop: 6, marginBottom: 0, fontSize: '0.88rem', color: 'var(--ink-faint)', fontStyle: 'italic' }}>
            Co-authored with{' '}
            <a className="viz-link" href="https://www.anthropic.com/claude" target="_blank" rel="noreferrer">Claude</a>
            {' '}(Anthropic) — code, experiments, and interactive scaffolding by Claude, refined and co-designed by Bernhard.
          </p>
        </footer>
      </div>
    </article>
  );
}
