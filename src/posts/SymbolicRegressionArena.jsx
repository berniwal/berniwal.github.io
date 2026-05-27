// src/posts/SymbolicRegressionArena.jsx
// Symbolic Regression · Part 1 of 2 — Racing classical search algorithms
// (random, evolution, greedy / VPG, risk-seeking) on the same task, same
// verifier, same budget. Ported to the post-2026 chrome from the older
// VisualizingSymbolicRegression post.
//
// TAGS FOR REGISTRATION: ['symbolic-regression', 'rl', 'self-improvement']
// EXCERPT: Four classical search algorithms race to recover a hidden equation under one shared budget. Only the proposer changes — and the ranking is surprisingly sharp.
import React, { useEffect, useRef, useState } from 'react';
import usePageMeta from '../usePageMeta';
import './PostChrome.css';
import './SymbolicRegressionArena.css';

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

/* ============================ data hook ============================ */
// Keep the original data path — JSON files are not being renamed.
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
      <line x1={pad.l} y1={pad.t} x2={pad.l} y2={pad.t + ih} stroke="#e2e8f0" />
      <line x1={pad.l} y1={pad.t + ih} x2={pad.l + iw} y2={pad.t + ih} stroke="#e2e8f0" />
      {[0, 0.5, 1].map((f) => (
        <text key={f} x={pad.l - 6} y={Y(maxV * f) + 3} fontSize="9" fill="#94a3b8" textAnchor="end">{(maxV * f).toFixed(1)}</text>
      ))}
      <text x={pad.l + iw} y={pad.t + ih + 22} fontSize="9" fill="#94a3b8" textAnchor="end">{fmtX(maxCalls)}</text>
      <line x1={X(cursorCalls)} y1={pad.t} x2={X(cursorCalls)} y2={pad.t + ih} stroke="#cbd5e1" strokeDasharray="3 3" />
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
      <line x1={pad.l} y1={pad.t} x2={pad.l} y2={pad.t + ih} stroke="#e2e8f0" />
      <line x1={pad.l} y1={pad.t + ih} x2={pad.l + iw} y2={pad.t + ih} stroke="#e2e8f0" />
      <path d={line(yTarget)} fill="none" stroke="#cbd5e1" strokeWidth="6" strokeLinecap="round" opacity="0.6" />
      <path d={line(yTarget)} fill="none" stroke="#0f172a" strokeWidth="1.6" strokeDasharray="4 3" />
      {yPred && <path d={line(yPred)} fill="none" stroke={fitColor} strokeWidth={matched ? 2.6 : 2} />}
      <text x={pad.l + 4} y={pad.t + 10} fontSize="9" fill="#94a3b8">
        target band&nbsp;&nbsp;<tspan fill={fitColor}>{matched ? '✓ fit matches' : '—— fit'}</tspan>
      </text>
    </svg>
  );
}

const fmtCalls = (n) => (n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : `${Math.round(n / 1000)}k`) + ' calls';

/* ============================ Arena player ============================ */
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

  if (data === false) return <div className="viz-panel"><p className="sra-note">Could not load Layer-0 data.</p></div>;
  if (!data) return <div className="viz-panel"><p className="sra-note">Loading…</p></div>;
  if (!tgt) return <div className="viz-panel"><p className="sra-note">No data for {target}/{reward}.</p></div>;

  const cur = (methods[selected] && methods[selected].checkpoints[Math.min(idx, nFrames - 1)]) || {};
  const rows = cur.rows || [];
  const best = cur.best ?? 0;
  const matched = best >= 0.98;

  return (
    <div className="viz-panel">
      <div className="sra-controls">
        <label>target
          <div className="sra-tabs">
            {['easy', 'medium', 'harder'].map((t) => (
              <button key={t} className={`sra-tab ${target === t ? 'active' : ''}`} onClick={() => setTarget(t)}>{t}</button>
            ))}
          </div>
        </label>
        <label>reward
          <div className="sra-tabs">
            {['nrmse', 'mse'].map((r) => (
              <button key={r} className={`sra-tab ${reward === r ? 'active' : ''}`} onClick={() => setReward(r)}>{r}</button>
            ))}
          </div>
        </label>
        <button className={`sra-btn ${playing ? 'active' : ''}`}
          onClick={() => { if (idx >= nFrames - 1) setIdx(0); setPlaying((p) => !p); }}>
          {playing ? '❚❚ pause' : '▶ play'}
        </button>
        <input type="range" min={0} max={Math.max(0, nFrames - 1)} value={idx} className="sra-range"
          onChange={(e) => { setPlaying(false); setIdx(+e.target.value); }} />
      </div>

      <div className="sra-target">
        <span className="sra-be-label">target f(x)</span>
        <code>{tgt.target_infix}</code>
        <span className="sra-be-reward">recover this</span>
      </div>

      <div className="sra-grid3">
        <div className="sra-cell">
          <p className="sra-cap">best-so-far reward</p>
          <MetricChart methods={methods} order={order} field="best" cursorIdx={idx} selected={selected} onSelect={setSelected} fmtX={fmtCalls} />
        </div>
        <div className="sra-cell">
          <p className="sra-cap">policy entropy (lower → committed)</p>
          <MetricChart methods={methods} order={order} field="entropy" cursorIdx={idx} selected={selected} onSelect={setSelected} fixedMax={2.0} fmtX={fmtCalls} />
        </div>
        <div className="sra-cell">
          <p className="sra-cap">{methods[selected] && methods[selected].label} — fit @ {Math.round((cur.calls || 0) / 1000)}k{matched && <span className="sra-ok"> ✓ on target</span>}</p>
          <FitChart xGrid={tgt.x_grid} yTarget={tgt.y_target} yPred={cur.y_pred} color={methods[selected] && methods[selected].color} matched={matched} />
        </div>
      </div>

      <div className="sra-bestexpr">
        <span className="sra-be-label">current best</span>
        <code className={matched ? 'sra-ok-code' : ''}>{cur.best_infix || '—'}</code>
        <span className="sra-be-reward">reward {best.toFixed(3)}</span>
      </div>

      <div className="sra-legend">
        {order.map((m) => (
          <span key={m} className={`sra-chip ${selected === m ? 'active' : ''}`} onClick={() => setSelected(m)}>
            <span className="sra-swatch" style={{ background: methods[m].color }} />{methods[m].label}
          </span>
        ))}
      </div>

      <div className="sra-stats">
        <span>best <b>{(cur.best ?? 0).toFixed(3)}</b></span>
        <span>policy H <b>{(cur.entropy ?? 0).toFixed(2)}</b></span>
        <span>unique-fraction <b>{(cur.diversity ?? 0).toFixed(2)}</b></span>
        <span>solved-in-batch <b>{((cur.success_frac ?? 0) * 100).toFixed(0)}%</b></span>
      </div>

      <div className="sra-batchwrap">
        <p className="sra-cap">{methods[selected] && methods[selected].label} — top batch proposals (what the policy is sampling now)</p>
        <table className="sra-batch">
          <thead><tr><th>expression</th><th className="num">count</th><th className="num">reward</th></tr></thead>
          <tbody>
            {rows.length === 0 && <tr><td colSpan={3} className="sra-note">no proposals at this frame</td></tr>}
            {rows.map((r, i) => (
              <tr key={i}><td className="sra-expr">{r[0]}</td><td className="num">{r[1]}</td><td className="num">{r[2] == null ? '—' : r[2].toFixed(3)}</td></tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/* ============================ Layer-0 results table ============================ */
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
      <div className="sra-controls">
        <label>benchmark
          <div className="sra-tabs">
            {['easy', 'medium', 'harder'].map((k) => (
              <button key={k} className={`sra-tab ${t === k ? 'active' : ''}`} onClick={() => setT(k)}>{k}</button>
            ))}
          </div>
        </label>
        <label>reward
          <div className="sra-tabs">
            {['nrmse', 'mse'].map((k) => (
              <button key={k} className={`sra-tab ${reward === k ? 'active' : ''}`} onClick={() => setReward(k)}>{k}</button>
            ))}
          </div>
        </label>
        <span className="sra-note" style={{ marginLeft: 'auto' }}>20 seeds · 2M calls</span>
      </div>
      <table className="sra-results">
        <thead><tr><th>method</th><th>success @2M</th><th>mean best reward</th></tr></thead>
        <tbody>
          {rows.map(([label, color, s, r]) => (
            <tr key={label}>
              <td><span className="sra-swatch" style={{ background: color }} /> {label}</td>
              <td><b>{s.toFixed(2)}</b></td>
              <td>{r.toFixed(3)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {reward === 'mse' && t === 'harder' && (
        <p className="sra-note" style={{ marginTop: 8 }}>
          Under MSE on <code>harder</code>, even risk-seeking gets only 10% — the reward saturates
          once the curve is roughly right, so the gradient signal toward exact recovery is largely
          gone. Switch back to nrmse to see the same arm at 75%.
        </p>
      )}
    </div>
  );
}

/* ============================ Nguyen 1-8 table ============================ */
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
const NG_AVG = [47, 56, 114, 147];
const NG_COLORS = [C.random, C.gp, C.greedy, C.risk];
const NG_LABELS = ['Random', 'Evolution (GP)', 'Greedy (= VPG)', 'Risk-seeking'];

function NguyenTable() {
  return (
    <div className="viz-panel">
      <table className="sra-results sra-nguyen">
        <thead>
          <tr>
            <th>target</th>
            <th style={{ textAlign: 'left' }}>formula</th>
            {NG_LABELS.map((l, i) => (
              <th key={l}><span className="sra-swatch" style={{ background: NG_COLORS[i] }} /> {l}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {NGUYEN.map(([t, f, row]) => {
            const max = Math.max(...row);
            return (
              <tr key={t}>
                <td><code>{t}</code></td>
                <td style={{ textAlign: 'left' }} className="sra-expr">{f}</td>
                {row.map((v, i) => {
                  const cls = v === max && v > 0 ? 'sra-cell-best' : '';
                  const hl = t === 'nguyen-5' && i === 3 ? 'sra-cell-uniq' : '';
                  return <td key={i} className={`${cls} ${hl}`}><b>{v}</b>/25</td>;
                })}
              </tr>
            );
          })}
          <tr className="sra-row-avg">
            <td colSpan={2}><b>average (symbolic recovery)</b></td>
            {NG_AVG.map((v, i) => (
              <td key={i}><b>{v}</b>/200 <span className="sra-pct">({(v / 200 * 100).toFixed(0)}%)</span></td>
            ))}
          </tr>
        </tbody>
      </table>
      <p className="sra-note" style={{ marginTop: 8 }}>
        Symbolic recovery (exact SymPy equivalence), L₀ grammar, nrmse reward, 25 seeds, 2M calls.
        Nguyen-5 row: risk-seeking is the only method to recover it (11/25). Bold = best in row.
      </p>
    </div>
  );
}

/* ============================ the post ============================ */
export default function SymbolicRegressionArena() {
  usePageMeta({
    title: 'Racing Search Algorithms on Symbolic Regression',
    description: 'Symbolic regression as a self-improvement testbed: evolution, greedy RL, and risk-seeking RL race to recover a hidden equation under one shared compute budget.',
    slug: 'symbolic-regression-arena',
    publishedDate: '2026-05-10',
    keywords: ['symbolic regression', 'evolution', 'PPO', 'risk-seeking', 'self-improvement'],
  });

  return (
    <article className="post-2026 sra-post">
      <div className="post-wrap">
        <div className="post-hero">
          <div className="post-series-tag"><span className="post-live-dot" aria-hidden="true" />Symbolic Regression · Part 1 of 2</div>
          <h1>Racing Search Algorithms on Symbolic Regression</h1>
          <p className="post-lede">
            <em>Hide an equation behind a handful of <Katex tex="(x, y)" /> points and ask four
            different search algorithms to recover it — under one shared compute budget. Only
            the proposer changes. The ranking is sharper than you'd expect.</em>
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

        <h2 className="reveal">What is symbolic regression?</h2>
        <p>
          Ordinary regression fixes a <em>form</em> in advance — say a line{' '}
          <Katex tex="y = ax + b" /> — and fits its numbers to the data. <strong>Symbolic
          regression</strong> doesn't get the form for free. It searches over the space of
          mathematical <em>expressions themselves</em> for one that both fits the points and is a
          plausible closed form — <Katex tex="x^2 + \sin x" /> rather than a 12-parameter
          polynomial that happens to pass through them. The output is an equation you can read.
        </p>
        <p>
          That makes it a discrete search over a combinatorial space of expression trees: every
          internal node is an operator (<Katex tex="+,\,\times,\,\sin,\dots" />) and every leaf
          is a variable or constant. The space is enormous and non-smooth — swap one operator
          and the output changes wildly. You cannot just follow a gradient to the right{' '}
          <em>structure</em>. That's exactly why it's a clean microcosm for the broader discovery
          problem: a cheap verifier, and a proposer that has to <em>explore</em> a huge
          combinatorial space to find the rare expression that nails it.
        </p>

        <h2 className="reveal">The setup</h2>
        <p>
          Four search methods compete on the <strong>same task</strong>, scored by the{' '}
          <strong>same verifier</strong>, under the <strong>same budget</strong> — counted in{' '}
          <em>verifier calls</em>, so no method can win by simply evaluating more candidates. A
          candidate is an expression tree over a small grammar (<Katex tex="x" />, a few
          constants, <Katex tex="+\,-\,\times\,\div" />, <Katex tex="\sin,\cos" />). The verifier
          scores it by how well it fits the data:
        </p>
        <Katex tex="\mathrm{reward} \;=\; \dfrac{1}{1 + \mathrm{NRMSE}(\hat y, y)}, \qquad \mathrm{NRMSE} = \dfrac{\sqrt{\mathrm{MSE}}}{\sigma_y}" block />
        <p>
          where <Katex tex="\mathrm{MSE}" /> is the mean squared error between the candidate's
          predictions <Katex tex="\hat y" /> and the data <Katex tex="y" />, and{' '}
          <Katex tex="\sigma_y" /> is the standard deviation of <Katex tex="y" /> — the
          normalisation that puts the error on a unit-free scale. Reward <Katex tex="= 1.0" /> is
          a perfect fit; the <Katex tex="\sqrt{\,}" /> stretches the small-error regime so the
          gradient signal stays alive even when the curve is roughly right. Only the{' '}
          <strong>proposer</strong> changes between methods.
        </p>

        <h2 className="reveal">The four proposers</h2>
        <p>
          Every arm shares the same shape: a <strong>proposer</strong> emits expression trees,
          and the <strong>verifier</strong> scores them. What differs is <em>what's learned from
          the scores</em>. Random search learns nothing. Evolution (GP) keeps no model — just a
          population it breeds with tournament selection, subtree crossover and mutation. The two
          RL arms train an autoregressive policy <Katex tex="\pi" /> that samples expression
          tokens, and update it via group-relative policy gradient on the objective
        </p>
        <Katex tex="\mathcal{J} \;=\; \sum_i w_i \,\log \pi(\tau_i)" block />
        <p>
          where <Katex tex="\tau_i" /> is the <Katex tex="i" />-th sampled expression and{' '}
          <Katex tex="w_i" /> is its <strong>advantage</strong> — a scalar derived from the
          batch's rewards. Gradient ascent reinforces samples with large advantage and ignores
          the rest. The only thing that distinguishes the gradient arms is how they turn a batch
          of rewards into <Katex tex="w_i" />:
        </p>
        <div className="sra-proposers">
          <div className="sra-proposer">
            <span className="sra-prop-tag">no learning</span>
            <strong>Random search</strong>
            Sample fresh trees from the grammar. The honest baseline — if a fancier method can't
            beat random under a shared budget, it isn't doing anything useful.
          </div>
          <div className="sra-proposer">
            <span className="sra-prop-tag">population, no gradient</span>
            <strong>Evolution (GP)</strong>
            A population of 200 trees, evolved by tournament selection + subtree crossover +
            mutation. Recombination in solution space; strong when the grammar is small and
            compositional.
          </div>
          <div className="sra-proposer">
            <span className="sra-prop-tag">greedy / VPG</span>
            <strong>Greedy RL</strong>
            Maximises the <em>average</em> batch reward — set <Katex tex="w_i = R_i - \bar R" />
            {' '}(vanilla policy gradient). The fastest way to raise the mean is to pile probability
            onto one decent expression, so the policy tends to collapse onto a single attractor.
          </div>
          <div className="sra-proposer">
            <span className="sra-prop-tag">risk-seeking (DSR)</span>
            <strong>Risk-seeking RL</strong>
            Maximises the top-<Katex tex="\varepsilon" /> reward <em>quantile</em> instead of the
            mean. Only the elite samples carry gradient; the bottom 95% contribute nothing. The
            policy is never pulled toward "good on average" — it keeps chasing the rare outliers
            where the real solution hides.
          </div>
        </div>
        <p>
          The risk-seeking objective is the core idea of{' '}
          <a className="post-link" href="https://arxiv.org/abs/1912.04871" target="_blank" rel="noreferrer">Deep Symbolic Regression</a>{' '}
          (Petersen et al. 2021). An LLM proposer — Qwen2.5 fine-tuned with LoRA + GRPO under the
          same advantage formulas — is the subject of{' '}
          <a className="post-link" href="/blog/symbolic-regression-llm-transfer">Part 2</a>.
        </p>

        <h2 className="reveal">Two things that make the search well-posed</h2>
        <p>
          Before any objective can matter, two pieces of machinery have to be in place. Remove
          either and the comparison stops working, regardless of which proposer you use.
        </p>
        <p>
          <strong>Constraints on what the policy may emit.</strong> Left unconstrained, an
          autoregressive policy happily produces non-terminating or absurd token streams. We
          track arity so every sample decodes to a finite tree, cap expression length, and forbid
          nested trig (<Katex tex="\sin(\sin(\dots))" />). A separate ablation that toggles these
          guardrails shows greedy RL collapsing <em>completely</em> without them, while
          risk-seeking only slows down.
        </p>
        <p>
          <strong>Constants handled as structure, then fit numerically.</strong> If a proposer
          can emit arbitrary floating-point numbers, it stops doing symbolic regression and
          starts doing <em>numeric</em> regression — drawing a generic curve shape and tuning its
          constants. The numpy policy sidesteps this with a few fixed constant tokens; the LLM
          (in Part 2) places a single <Katex tex="\texttt{const}" /> placeholder whose values are
          fit afterward by BFGS. The <em>shape</em> is what's searched; only then are the numbers
          optimised.
        </p>

        <h2 className="reveal">Watch them evolve</h2>
        <p>
          A pre-computed replay of one representative seed, full 2M-call budget, log-spaced
          frames so the early collapse and the late recovery are both visible in the same scrub.
          Pick a target, pick a method, press play. Three views update together:{' '}
          <strong>best-so-far reward</strong>, <strong>policy entropy</strong> (lower = the
          policy has committed to a sharp distribution), and the <strong>best-fit overlay</strong>{' '}
          against the target. The batch table underneath shows what the policy is sampling{' '}
          <em>right now</em>.
        </p>
        <ArenaPlayer />
        <p style={{ marginTop: 18 }}><strong>Things to notice once you've scrubbed around:</strong></p>
        <ul>
          <li>
            <strong>Recovery on <code>harder</code> is the headline.</strong> Select risk-seeking
            on <code>harder</code> / <code>nrmse</code>, scrub to the end: the fit snaps green —
            risk recovers <Katex tex="x^3 - x + \cos(2x)" /> exactly (best reward → 0.99). Switch
            to greedy with the same target — it reaches ≈0.94, finds the cubic, but never adds
            the cosine term. Same budget, same gradient machinery; only the advantage formula
            differs.
          </li>
          <li>
            <strong>The entropy curve tells the <em>why</em>.</strong> On <code>harder</code> /{' '}
            <code>nrmse</code> around the 300k mark, greedy drops to <Katex tex="H \approx 0.5" />{' '}
            and stays there — it has committed to a wrong attractor and cannot leave.
            Risk-seeking dips and then climbs back up (1.83 → 1.45 → 1.76): it re-explores after
            locking onto a structure that isn't quite right. That rebound is what carries it from
            0.94 to 0.99.
          </li>
          <li>
            <strong>The batch table is the only structural-collapse view.</strong> On the same
            frame, greedy's batch is all trig — <code>sin(2.0 − x)</code>, <code>cos(0.5·x)</code>,{' '}
            <code>sin(1.0/x)</code> — total skeleton collapse on a target that's mostly
            polynomial. Risk's batch is <code>(x·x)·x</code>, <code>((x·x)·x) − x</code>… locked
            on the polynomial family. Per-token entropy can hide this.
          </li>
          <li>
            <strong>MSE saturates; NRMSE doesn't.</strong> Flip the reward toggle.{' '}
            <Katex tex="1/(1+\mathrm{MSE})" /> has a vanishing-gradient region once the curve is
            roughly right. Across 20 seeds at 2M, risk-seeking cracks <code>harder</code>{' '}
            <strong>75% of the time under NRMSE versus only 10% under MSE</strong>.
          </li>
        </ul>

        <h2 className="reveal">Layer 0 results</h2>
        <p>
          Over 20 seeds at the full 2M-call budget. On the easy and medium targets almost
          everything reaches the ceiling, and evolution is the most sample-efficient. The methods
          only separate on <code>harder</code> — the collapse zone — where{' '}
          <strong>risk-seeking recovers the formula three times as often as evolution, and
          greedy never does</strong>.
        </p>
        <Layer0Results />

        <h2 className="reveal">An official benchmark: Nguyen 1–8</h2>
        <p>
          Three hand-picked targets is suggestive; a standard benchmark is convincing. The{' '}
          <strong>Nguyen suite</strong> is what DSR reports on — a fixed set of eight ground-truth
          expressions, evaluated with the standard library{' '}
          <Katex tex="\mathcal{L}_0 = \{+, -, \times, \div, \sin, \cos, \exp, \log, x\}" /> and
          DSR's recovery metric (fraction of seeds whose best expression is{' '}
          <em>exactly</em> symbolically equivalent to the target, checked with SymPy). All four
          methods, 25 seeds, 2M calls, nrmse reward:
        </p>
        <NguyenTable />
        <p>
          <strong>Risk-seeking wins on average by ~17 percentage points</strong> (74% vs greedy
          57%); evolution and random trail at 28% / 24%. The methods <em>tie</em> on the easy
          polynomials (Nguyen-1, 2, 6) and separate sharply on the hard rows. Two are especially
          diagnostic: <strong>Nguyen-5</strong> (<Katex tex="\sin(x^2)\cos(x) - 1" />) and{' '}
          <strong>Nguyen-8</strong> (<Katex tex="\sqrt{x}" />) — both targets where greedy
          recovers <em>0 of 25</em>, and risk-seeking recovers <em>11 of 25</em> and{' '}
          <em>16 of 25</em> respectively. That's the "rare excellent outlier" regime the
          risk-seeking objective is designed for.
        </p>

        <h2 className="reveal">Where this goes next</h2>
        <p>
          We have a clean ranking among classical proposers on a fixed verifier-call budget:
          risk-seeking &gt; evolution ≳ greedy &gt; random, sharpening as the target gets harder.
          The obvious next question: swap the small RNN policy for a pre-trained language model.
          Does the same objective still pick the same winner, or does the LLM's prior wash the
          ranking out?{' '}
          <a className="post-link" href="/blog/symbolic-regression-llm-transfer">Part 2</a> runs
          that experiment.
        </p>

        <h2 className="reveal">References</h2>
        <div className="post-refs">
          <div className="head ref-cite">Cite</div>
          <div className="head ref-title">Title</div>
          <div className="head ref-link">Link</div>

          <div className="ref-cite">Schmidt &amp; Lipson 2009</div>
          <div>
            <div className="ref-title">
              <a href="https://www.science.org/doi/10.1126/science.1165893" target="_blank" rel="noreferrer">Distilling Free-Form Natural Laws from Experimental Data</a>
            </div>
            <div className="ref-note">The Eureqa paper — symbolic regression as scientific-law discovery via evolution.</div>
          </div>
          <div className="ref-link"><a href="https://www.science.org/doi/10.1126/science.1165893" target="_blank" rel="noreferrer">science.org</a></div>

          <div className="ref-cite">Williams 1992</div>
          <div>
            <div className="ref-title">
              <a href="https://doi.org/10.1007/BF00992696" target="_blank" rel="noreferrer">Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning</a>
            </div>
            <div className="ref-note">REINFORCE / vanilla policy gradient — the greedy arm's underlying update.</div>
          </div>
          <div className="ref-link"><a href="https://doi.org/10.1007/BF00992696" target="_blank" rel="noreferrer">doi 10.1007/BF00992696</a></div>

          <div className="ref-cite">Petersen et al. 2021</div>
          <div>
            <div className="ref-title">
              <a href="https://arxiv.org/abs/1912.04871" target="_blank" rel="noreferrer">Deep Symbolic Regression: recovering expressions via risk-seeking policy gradients</a>
            </div>
            <div className="ref-note">The hard top-ε quantile objective and the Nguyen recovery benchmark.</div>
          </div>
          <div className="ref-link"><a href="https://arxiv.org/abs/1912.04871" target="_blank" rel="noreferrer">arxiv 1912.04871</a></div>

          <div className="ref-cite">Shao et al. 2024</div>
          <div>
            <div className="ref-title">
              <a href="https://arxiv.org/abs/2402.03300" target="_blank" rel="noreferrer">DeepSeekMath — Group Relative Policy Optimization (GRPO)</a>
            </div>
            <div className="ref-note">Group-relative advantage; the gradient machinery shared by all RL arms here.</div>
          </div>
          <div className="ref-link"><a href="https://arxiv.org/abs/2402.03300" target="_blank" rel="noreferrer">arxiv 2402.03300</a></div>

          <div className="ref-cite">DSO</div>
          <div>
            <div className="ref-title">
              <a href="https://github.com/dso-org/deep-symbolic-optimization" target="_blank" rel="noreferrer">deep-symbolic-optimization</a>
            </div>
            <div className="ref-note">DSR's reference implementation and the Nguyen suite.</div>
          </div>
          <div className="ref-link"><a href="https://github.com/dso-org/deep-symbolic-optimization" target="_blank" rel="noreferrer">github</a></div>

          <div className="ref-cite">this post</div>
          <div>
            <div className="ref-title">
              <a href="https://github.com/berniwal/berniwal.github.io/tree/main/experiments/self-improvement-arena" target="_blank" rel="noreferrer">experiments/self-improvement-arena</a>
            </div>
            <div className="ref-note">All proposers, configs, and the harness that produced these results.</div>
          </div>
          <div className="ref-link"><a href="https://github.com/berniwal/berniwal.github.io/tree/main/experiments/self-improvement-arena" target="_blank" rel="noreferrer">github</a></div>
        </div>

        <footer className="post-footer">
          <p>
            <strong>Part 1 of Symbolic Regression</strong> · Next:{' '}
            <a className="post-link" href="/blog/symbolic-regression-llm-transfer">Does the Ranking Transfer to an LLM?</a>
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
