// src/posts/RotaryPositionalEmbeddings.jsx
// Positions in Transformers · Part 2 of 2 — Rotary Positional Embeddings (RoPE).
// Ported from the legacy VisualizingRoPE post (§3–§7 + closing); prose preserved,
// design tokens migrated to the post-2026 redesign chrome.
// TAGS FOR REGISTRATION: ['transformers', 'positions', 'rope']
// EXCERPT: RoPE encodes token position by rotating each query and key vector — the dot product naturally depends on relative offset, with zero new parameters and a clean path to long context.

import React, { useEffect, useMemo, useRef, useState } from 'react';
import './PostChrome.css';
import './RotaryPositionalEmbeddings.css';

/* =========================================================
   KaTeX wrapper — uses the CDN-loaded global window.katex
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
      className={`${block ? 'viz-math-block' : 'viz-math-inline'} ${className}`}
    />
  );
}

/* =========================================================
   Tooltip wrapper — hover or tap-focus reveals a breakdown
   ========================================================= */
function Tip({ children, label }) {
  return (
    <span className="viz-tip" tabIndex={0}>
      {children}
      <span className="viz-tip-content" role="tooltip">{label}</span>
    </span>
  );
}

function tipBody(formula, values, result) {
  return (
    <>
      <div className="viz-tip-row"><span className="viz-tip-key">formula</span><span>{formula}</span></div>
      <div className="viz-tip-row"><span className="viz-tip-key">values</span><span>{values}</span></div>
      <div className="viz-tip-row viz-tip-result"><span className="viz-tip-key">=</span><span>{result}</span></div>
    </>
  );
}

function dotBreakdownTip(Q, K) {
  const total = Q.reduce((s, q, i) => s + q * K[i], 0);
  const termRow = (q, k, i) => (
    <span key={i} style={{ display: 'block' }}>
      Q<sub>{i}</sub>·K<sub>{i}</sub> = {q.toFixed(2)} · {k.toFixed(2)} ={' '}
      <strong style={{ color: '#fff' }}>{(q * k).toFixed(3)}</strong>
    </span>
  );
  return (
    <>
      <div className="viz-tip-row">
        <span className="viz-tip-key">formula</span>
        <span>Q · K = Σ<sub>i</sub> Q<sub>i</sub> · K<sub>i</sub></span>
      </div>
      <div className="viz-tip-row">
        <span className="viz-tip-key">terms</span>
        <span>{Q.map((q, i) => termRow(q, K[i], i))}</span>
      </div>
      <div className="viz-tip-row viz-tip-result">
        <span className="viz-tip-key">=</span>
        <span>{Q.map((q, i) => (q * K[i]).toFixed(3)).join(' + ').replace(/\+ -/g, '− ')} = <strong>{total.toFixed(3)}</strong></span>
      </div>
    </>
  );
}

/* =========================================================
   2D rotation SVG primitive
   Math-space coords are (x, y) with x right, y UP. SVG y is flipped.
   ========================================================= */
const CANVAS_PX = 280;
const CANVAS_PAD = 24;
const AXIS_RANGE = 1.4;

function toSvgX(mx) {
  const inner = CANVAS_PX - 2 * CANVAS_PAD;
  return CANVAS_PAD + ((mx + AXIS_RANGE) / (2 * AXIS_RANGE)) * inner;
}
function toSvgY(my) {
  const inner = CANVAS_PX - 2 * CANVAS_PAD;
  return CANVAS_PAD + ((AXIS_RANGE - my) / (2 * AXIS_RANGE)) * inner;
}

function RotationAxes() {
  const xs = [-1, 0, 1];
  const ys = [-1, 0, 1];
  return (
    <g aria-hidden="true">
      <circle
        cx={toSvgX(0)}
        cy={toSvgY(0)}
        r={toSvgX(1) - toSvgX(0)}
        className="rope-svg-grid"
        fill="none"
        strokeDasharray="3 4"
      />
      <line x1={toSvgX(-AXIS_RANGE)} y1={toSvgY(0)} x2={toSvgX(AXIS_RANGE)} y2={toSvgY(0)} className="rope-svg-axis" />
      <line x1={toSvgX(0)} y1={toSvgY(-AXIS_RANGE)} x2={toSvgX(0)} y2={toSvgY(AXIS_RANGE)} className="rope-svg-axis" />
      {xs.filter((v) => v !== 0).map((v) => (
        <text key={`xt-${v}`} x={toSvgX(v)} y={toSvgY(0) + 12} textAnchor="middle" className="rope-svg-label">{v}</text>
      ))}
      {ys.filter((v) => v !== 0).map((v) => (
        <text key={`yt-${v}`} x={toSvgX(0) - 8} y={toSvgY(v) + 3} textAnchor="end" className="rope-svg-label">{v}</text>
      ))}
    </g>
  );
}

function ArrowVector({ x, y, color, label, dashed = false, opacity = 1 }) {
  const sx = toSvgX(0);
  const sy = toSvgY(0);
  const ex = toSvgX(x);
  const ey = toSvgY(y);
  const angle = Math.atan2(ey - sy, ex - sx);
  const headLen = 9;
  const headWid = 5;
  const hbx = ex - headLen * Math.cos(angle);
  const hby = ey - headLen * Math.sin(angle);
  const px = Math.cos(angle + Math.PI / 2);
  const py = Math.sin(angle + Math.PI / 2);
  const head1x = hbx + headWid * px;
  const head1y = hby + headWid * py;
  const head2x = hbx - headWid * px;
  const head2y = hby - headWid * py;
  return (
    <g style={{ opacity }}>
      <line
        x1={sx} y1={sy} x2={hbx} y2={hby}
        stroke={color}
        className="rope-svg-vector"
        strokeDasharray={dashed ? '4 4' : undefined}
      />
      <polygon
        points={`${ex},${ey} ${head1x},${head1y} ${head2x},${head2y}`}
        fill={color}
        className="rope-svg-vector-head"
      />
      {label && (
        <text
          x={ex + (Math.cos(angle) >= 0 ? 8 : -8) * (Math.abs(Math.cos(angle)) < 0.4 ? 0 : 1)}
          y={ey - 8}
          textAnchor={Math.cos(angle) >= 0 ? 'start' : 'end'}
          fill={color}
          className="rope-svg-angle-label"
        >
          {label}
        </text>
      )}
    </g>
  );
}

function AngleArc({ angle, radius = 0.32, color = 'hsl(218, 50%, 65%)' }) {
  if (Math.abs(angle) < 1e-3) return null;
  const sx = toSvgX(radius);
  const sy = toSvgY(0);
  const ex = toSvgX(radius * Math.cos(angle));
  const ey = toSvgY(radius * Math.sin(angle));
  const largeArc = Math.abs(angle) > Math.PI ? 1 : 0;
  const sweep = angle > 0 ? 0 : 1;
  const r = toSvgX(radius) - toSvgX(0);
  return (
    <g>
      <path
        d={`M ${toSvgX(0)} ${toSvgY(0)} L ${sx} ${sy} A ${r} ${r} 0 ${largeArc} ${sweep} ${ex} ${ey} Z`}
        className="rope-svg-arc-fill"
      />
      <text
        x={toSvgX((radius + 0.12) * Math.cos(angle / 2))}
        y={toSvgY((radius + 0.12) * Math.sin(angle / 2))}
        textAnchor="middle"
        className="rope-svg-angle-label"
        fill={color}
      >
        θ
      </text>
    </g>
  );
}

function RotationCanvas({ children, size = CANVAS_PX }) {
  return (
    <div className="rope-canvas" style={{ width: size, height: size }}>
      <svg viewBox={`0 0 ${CANVAS_PX} ${CANVAS_PX}`} width={size} height={size}>
        <RotationAxes />
        {children}
      </svg>
    </div>
  );
}

function rotate2D(x, y, theta) {
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  return [c * x - s * y, s * x + c * y];
}

/* =========================================================
   §1 — Rotation in a 2D plane
   ========================================================= */
function SectionRotation() {
  const [vx, setVx] = useState(0.8);
  const [vy, setVy] = useState(0.3);
  const [thetaDeg, setThetaDeg] = useState(60);
  const theta = (thetaDeg * Math.PI) / 180;
  const [rx, ry] = rotate2D(vx, vy, theta);

  const traceSteps = 18;
  const tracePoints = useMemo(() => {
    const pts = [];
    for (let i = 0; i <= traceSteps; i++) {
      const a = (theta * i) / traceSteps;
      const [tx, ty] = rotate2D(vx, vy, a);
      pts.push(`${toSvgX(tx)},${toSvgY(ty)}`);
    }
    return pts.join(' ');
  }, [vx, vy, theta]);

  const c = Math.cos(theta).toFixed(3);
  const s = Math.sin(theta).toFixed(3);

  return (
    <div className="viz-panel">
      <div className="rope-stage">
        <RotationCanvas>
          <AngleArc angle={theta} />
          <polyline points={tracePoints} className="rope-svg-trace" stroke="hsl(218, 50%, 55%)" />
          <ArrowVector x={vx} y={vy} color="hsl(218, 50%, 35%)" label="v" />
          <ArrowVector x={rx} y={ry} color="hsl(28, 65%, 45%)" label="R(θ)·v" />
        </RotationCanvas>

        <div className="rope-readout">
          <div className="rope-controls">
            <label className="rope-slider">
              <span>x</span>
              <input type="range" min={-1.2} max={1.2} step={0.05} value={vx} onChange={(e) => setVx(+e.target.value)} />
              <strong>{vx.toFixed(2)}</strong>
            </label>
            <label className="rope-slider">
              <span>y</span>
              <input type="range" min={-1.2} max={1.2} step={0.05} value={vy} onChange={(e) => setVy(+e.target.value)} />
              <strong>{vy.toFixed(2)}</strong>
            </label>
            <label className="rope-slider">
              <span>θ</span>
              <input type="range" min={-180} max={180} step={5} value={thetaDeg} onChange={(e) => setThetaDeg(+e.target.value)} />
              <strong>{thetaDeg}°</strong>
            </label>
          </div>

          <div className="rope-matrix">
            <div className="rope-matrix-head">R(θ) · v  =</div>
            <div className="rope-matrix-grid">
              <Tip label={tipBody('cos θ · x + (-sin θ) · y', `${c}·${vx.toFixed(2)} + (${(-s)})·${vy.toFixed(2)}`, `${rx.toFixed(3)}  (= x' )`)}>
                <span>{c}·x</span>
              </Tip>
              <Tip label={tipBody('+ (-sin θ) · y', `+ (${-s}) · ${vy.toFixed(2)}`, '')}>
                <span>{(-s)}·y</span>
              </Tip>
              <Tip label={tipBody('sin θ · x + cos θ · y', `${s}·${vx.toFixed(2)} + ${c}·${vy.toFixed(2)}`, `${ry.toFixed(3)}  (= y' )`)}>
                <span>{s}·x</span>
              </Tip>
              <Tip label={tipBody('+ cos θ · y', `+ ${c} · ${vy.toFixed(2)}`, '')}>
                <span>{c}·y</span>
              </Tip>
            </div>
            <div className="rope-matrix-result">
              =&nbsp;(<strong>{rx.toFixed(3)}</strong>, <strong>{ry.toFixed(3)}</strong>)
            </div>
          </div>

          <p className="viz-caption" style={{ marginTop: 12 }}>
            The rotation matrix <Katex tex="R(\theta) = \begin{bmatrix}\cos\theta & -\sin\theta\\ \sin\theta & \cos\theta\end{bmatrix}" />{' '}
            takes any 2D vector and rotates it by <Katex tex="\theta" /> radians counter-clockwise.
            Lengths are preserved; the vector just spins.
          </p>
        </div>
      </div>
    </div>
  );
}

/* =========================================================
   §2 — Pairwise dot-product invariance
   ========================================================= */
function SectionDotInvariance() {
  // Fixed demo vectors — only the angle sliders are interactive.
  const v1x = 0.9, v1y = 0.15;
  const v2x = 0.4, v2y = 0.75;
  const [t1Deg, setT1Deg] = useState(30);
  const [t2Deg, setT2Deg] = useState(30);
  const [lock, setLock] = useState(true);

  const t1 = (t1Deg * Math.PI) / 180;
  const t2 = (t2Deg * Math.PI) / 180;

  const [r1x, r1y] = rotate2D(v1x, v1y, t1);
  const [r2x, r2y] = rotate2D(v2x, v2y, t2);

  const dotOriginal = v1x * v2x + v1y * v2y;
  const dotRotated  = r1x * r2x + r1y * r2y;
  const close = Math.abs(dotOriginal - dotRotated) < 0.005;

  const onT1 = (e) => {
    const next = +e.target.value;
    if (lock) {
      const delta = next - t1Deg;
      setT2Deg((p) => p + delta);
    }
    setT1Deg(next);
  };
  const onT2 = (e) => {
    const next = +e.target.value;
    if (lock) {
      const delta = next - t2Deg;
      setT1Deg((p) => p + delta);
    }
    setT2Deg(next);
  };

  const diff = ((t1Deg - t2Deg + 540) % 360) - 180;

  return (
    <div className="viz-panel">
      <div className="rope-stage">
        <RotationCanvas>
          <ArrowVector x={v1x} y={v1y} color="hsl(218, 50%, 35%)" label="v₁" opacity={0.35} />
          <ArrowVector x={v2x} y={v2y} color="hsl(148, 35%, 28%)" label="v₂" opacity={0.35} />
          <ArrowVector x={r1x} y={r1y} color="hsl(218, 50%, 35%)" label="R(θ₁)v₁" />
          <ArrowVector x={r2x} y={r2y} color="hsl(148, 35%, 28%)" label="R(θ₂)v₂" />
        </RotationCanvas>

        <div className="rope-readout">
          <div className="rope-controls">
            <label className="rope-slider">
              <span>θ₁</span>
              <input type="range" min={-180} max={180} step={5} value={t1Deg} onChange={onT1} />
              <strong>{t1Deg}°</strong>
            </label>
            <label className="rope-slider">
              <span>θ₂</span>
              <input type="range" min={-180} max={180} step={5} value={t2Deg} onChange={onT2} />
              <strong>{t2Deg}°</strong>
            </label>
            <label className="rope-check">
              <input type="checkbox" checked={lock} onChange={(e) => setLock(e.target.checked)} />
              lock difference  θ₁ − θ₂ = <strong>{diff}°</strong>
            </label>
          </div>

          <div className="rope-dot-readout">
            <span className="label">v₁ · v₂  <em>(original)</em></span>
            <span className="value">{dotOriginal.toFixed(3)}</span>

            <span className="label">R(θ₁)v₁ · R(θ₂)v₂  <em>(rotated)</em></span>
            <span className={`value ${close ? 'match' : ''}`}>{dotRotated.toFixed(3)}</span>
          </div>

          <p className="viz-caption" style={{ marginTop: 12 }}>
            With <strong>lock difference</strong> on, both sliders move
            together: <Katex tex="\theta_1 - \theta_2" /> stays constant and so
            does the rotated dot product (it matches the original).
            Uncheck the lock and pull the two angles apart — the rotated dot
            product changes <em>only</em> when the difference does.
          </p>
        </div>
      </div>
    </div>
  );
}

/* =========================================================
   §3 — Scaling to d-dimensional Q and K (multi-pair rotation)
   ========================================================= */
const D_MODEL_DEMO = 8;
const PAIR_VECTORS = [
  [0.7,  0.35],
  [0.55, -0.45],
  [-0.5, 0.6],
  [0.6,  0.2],
];

const PAIR_PALETTE = [
  { bg: 'hsl(218, 50%, 95%)', border: 'hsl(218, 50%, 65%)', fg: 'hsl(218, 50%, 28%)', accent: 'hsl(218, 50%, 42%)' },
  { bg: 'hsl(28,  65%, 95%)', border: 'hsl(28,  65%, 65%)', fg: 'hsl(28,  65%, 28%)', accent: 'hsl(28,  65%, 45%)' },
  { bg: 'hsl(148, 38%, 95%)', border: 'hsl(148, 38%, 60%)', fg: 'hsl(148, 38%, 22%)', accent: 'hsl(148, 38%, 36%)' },
  { bg: 'hsl(280, 40%, 95%)', border: 'hsl(280, 40%, 65%)', fg: 'hsl(280, 40%, 30%)', accent: 'hsl(280, 40%, 45%)' },
];

function VectorStrip({ values, label, palette = PAIR_PALETTE, prefix = 'Q' }) {
  const d = values.length;
  return (
    <div className="rope-vstrip">
      <div className="rope-vstrip-label">{label}</div>
      <div className="rope-vstrip-grid" style={{ gridTemplateColumns: `repeat(${d}, minmax(0, 1fr))`, maxWidth: 70 * d }}>
        {values.map((v, idx) => {
          const pairIdx = Math.floor(idx / 2);
          const p = palette[pairIdx % palette.length];
          return (
            <div key={idx} className="rope-vstrip-cell" style={{
              background: p.bg,
              border: `1px solid ${p.border}`,
              color: p.fg,
            }}>
              <div className="rope-vstrip-cell-idx">{prefix}<sub>{idx}</sub></div>
              <div className="rope-vstrip-cell-val">{v.toFixed(2)}</div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function SectionMultiPair() {
  const [m, setM] = useState(16);
  const freqs = useMemo(() => {
    const out = [];
    for (let i = 0; i < D_MODEL_DEMO / 2; i++) {
      out.push(Math.pow(10000, -(2 * i) / D_MODEL_DEMO));
    }
    return out;
  }, []);

  const Q_full = PAIR_VECTORS.flatMap(([x, y]) => [x, y]);
  const Q_rotated = (() => {
    const out = new Array(D_MODEL_DEMO);
    for (let i = 0; i < D_MODEL_DEMO / 2; i++) {
      const theta = m * freqs[i];
      const [x, y] = PAIR_VECTORS[i];
      const [rx, ry] = rotate2D(x, y, theta);
      out[2 * i] = rx;
      out[2 * i + 1] = ry;
    }
    return out;
  })();

  return (
    <div className="viz-panel">
      <div className="rope-controls-row">
        <label className="rope-slider" style={{ flex: 1, minWidth: 280 }}>
          <span>position m</span>
          <input type="range" min={0} max={256} step={1} value={m} onChange={(e) => setM(+e.target.value)} />
          <strong>{m}</strong>
        </label>
      </div>

      <VectorStrip
        values={Q_full}
        label="Q  (one query vector, d = 8) — before RoPE"
      />

      <div className="rope-pipeline-arrow">
        ↓ split into <Katex tex="d/2 = 4" /> pairs · rotate pair i by <Katex tex="m \cdot \theta_i" />
      </div>

      <div className="rope-pair-grid">
        {PAIR_VECTORS.map(([vx, vy], i) => {
          const theta = m * freqs[i];
          const [rx, ry] = rotate2D(vx, vy, theta);
          const p = PAIR_PALETTE[i];
          return (
            <div key={i} className="rope-pair-card" style={{ background: p.bg, borderColor: p.border }}>
              <div className="rope-pair-title" style={{ color: p.fg }}>
                pair {i} &nbsp;·&nbsp; (Q<sub>{2 * i}</sub>, Q<sub>{2 * i + 1}</sub>)
              </div>
              <RotationCanvas size={160}>
                <ArrowVector x={vx} y={vy} color="#94a3b8" opacity={0.45} />
                <ArrowVector x={rx} y={ry} color={p.accent} />
              </RotationCanvas>
              <div className="rope-pair-freq" style={{ color: p.fg }}>
                θ<sub>{i}</sub> = {freqs[i] < 0.01 ? freqs[i].toExponential(2) : freqs[i].toFixed(3)}<br />
                m · θ<sub>{i}</sub> = {theta.toFixed(3)} rad
              </div>
            </div>
          );
        })}
      </div>

      <div className="rope-pipeline-arrow">
        ↓ concatenate the rotated pairs back into a d = 8 vector ↓
      </div>

      <VectorStrip
        values={Q_rotated}
        label={`Q'  (same vector, after RoPE at position m = ${m})`}
        prefix="Q'"
      />

      <p className="viz-caption" style={{ marginTop: 16 }}>
        The same eight numbers come out the bottom, but each pair has been
        rotated by its own angle. The leftmost pair{' '}
        <Katex tex="(i{=}0)" /> uses <Katex tex="\theta_0 = 1" /> rad/position
        and sweeps fast; the rightmost <Katex tex="(i{=}3)" /> uses{' '}
        <Katex tex="\theta_3 = 10000^{-6/8} \approx 0.001" /> rad/position and
        barely moves over hundreds of tokens. The colour-coding above ties
        each pair card to two specific cells of Q so you can see the
        disassemble → rotate → reassemble pipeline as one operation on the
        whole vector.
      </p>
    </div>
  );
}

/* =========================================================
   §4 — Attention with RoPE: only (n − m) matters
   ========================================================= */
const Q_DEMO = [0.8, 0.3, 0.4, -0.5];
const K_DEMO = [0.5, 0.6, -0.6, 0.4];
const D_RD = 4;
const RD_FREQS = (() => {
  const out = [];
  for (let i = 0; i < D_RD / 2; i++) out.push(Math.pow(10000, -(2 * i) / D_RD));
  return out;
})();

function applyRoPE(vec, pos) {
  const out = vec.slice();
  for (let i = 0; i < vec.length / 2; i++) {
    const theta = pos * RD_FREQS[i];
    const [a, b] = [vec[2 * i], vec[2 * i + 1]];
    const [ra, rb] = rotate2D(a, b, theta);
    out[2 * i] = ra;
    out[2 * i + 1] = rb;
  }
  return out;
}

function dotProduct(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

function SectionRoPEDot() {
  const [m, setM] = useState(4);
  const [n, setN] = useState(7);
  const [lock, setLock] = useState(true);

  const onM = (e) => {
    const next = +e.target.value;
    if (lock) {
      const delta = next - m;
      setN((p) => Math.max(0, Math.min(64, p + delta)));
    }
    setM(next);
  };
  const onN = (e) => {
    const next = +e.target.value;
    if (lock) {
      const delta = next - n;
      setM((p) => Math.max(0, Math.min(64, p + delta)));
    }
    setN(next);
  };

  const qRot = applyRoPE(Q_DEMO, m);
  const kRot = applyRoPE(K_DEMO, n);
  const rawDot = dotProduct(Q_DEMO, K_DEMO);
  const ropeDot = dotProduct(qRot, kRot);

  const refDiff = n - m;
  const kRotRef = applyRoPE(K_DEMO, refDiff);
  const refDot  = dotProduct(Q_DEMO, kRotRef);

  const SWEEP_RANGE = 32;
  const sweepData = useMemo(() => {
    const pts = [];
    for (let d = -SWEEP_RANGE; d <= SWEEP_RANGE; d++) {
      const kr = applyRoPE(K_DEMO, d);
      pts.push({ d, v: dotProduct(Q_DEMO, kr) });
    }
    return pts;
  }, []);

  const curveW = 460;
  const curveH = 140;
  const xPad = 28;
  const yPad = 14;
  const xMin = -SWEEP_RANGE;
  const xMax =  SWEEP_RANGE;
  const vs = sweepData.map((p) => p.v);
  const vMin = Math.min(...vs, -1);
  const vMax = Math.max(...vs,  1);
  const toCX = (d) => xPad + ((d - xMin) / (xMax - xMin)) * (curveW - 2 * xPad);
  const toCY = (v) => curveH - yPad - ((v - vMin) / (vMax - vMin)) * (curveH - 2 * yPad);
  const pathD = sweepData
    .map((p, idx) => `${idx === 0 ? 'M' : 'L'} ${toCX(p.d)} ${toCY(p.v)}`)
    .join(' ');
  const currentDiff = n - m;

  return (
    <div className="viz-panel">
      <div className="rope-controls-row">
        <label className="rope-slider" style={{ minWidth: 260 }}>
          <span>m (query)</span>
          <input type="range" min={0} max={64} step={1} value={m} onChange={onM} />
          <strong>{m}</strong>
        </label>
        <label className="rope-slider" style={{ minWidth: 260 }}>
          <span>n (key)</span>
          <input type="range" min={0} max={64} step={1} value={n} onChange={onN} />
          <strong>{n}</strong>
        </label>
        <label className="rope-check">
          <input type="checkbox" checked={lock} onChange={(e) => setLock(e.target.checked)} />
          lock (n − m) = <strong>{currentDiff}</strong>
        </label>
      </div>

      <div className="rope-dot-readout" style={{ maxWidth: 540 }}>
        <span className="label">q · k  <em>(no RoPE — constant in m, n)</em></span>
        <span className="value">{rawDot.toFixed(3)}</span>
        <span className="label">RoPE(q, m) · RoPE(k, n)  <em>(depends only on n − m)</em></span>
        <span className="value match">{ropeDot.toFixed(3)}</span>
      </div>

      <div style={{ marginTop: 18 }}>
        <div className="rope-curve-label">
          RoPE dot product as a function of (n − m), with m = {m} held fixed
        </div>
        <svg viewBox={`0 0 ${curveW} ${curveH}`} width="100%" className="rope-curve-svg" style={{ maxWidth: curveW }}>
          <line x1={xPad} y1={toCY(0)} x2={curveW - xPad} y2={toCY(0)} stroke="hsl(218, 20%, 80%)" strokeDasharray="3 3" />
          <line x1={toCX(0)} y1={yPad} x2={toCX(0)} y2={curveH - yPad} stroke="hsl(218, 20%, 80%)" />
          <path d={pathD} stroke="hsl(218, 50%, 40%)" strokeWidth="2" fill="none" />
          <circle cx={toCX(currentDiff)} cy={toCY(ropeDot)} r="4.5" fill="hsl(28, 65%, 45%)" />
          <text x={toCX(0)} y={curveH - 2} textAnchor="middle" fontSize="9" className="rope-curve-tick">0</text>
          <text x={toCX(-16)} y={curveH - 2} textAnchor="middle" fontSize="9" className="rope-curve-tick">-16</text>
          <text x={toCX(16)} y={curveH - 2} textAnchor="middle" fontSize="9" className="rope-curve-tick">+16</text>
        </svg>
      </div>

      <p className="viz-caption">
        The orange dot is the current setting. Slide <em>m</em> and <em>n</em>{' '}
        with the lock on — both rotate but their <em>difference</em>{' '}
        <Katex tex="n - m" /> doesn't change, so the dot product (and the dot
        on the curve) stays put. Uncheck the lock and slide them apart — the
        dot slides along the curve. Position has been folded into the dot
        product geometrically, without any extra learned parameters.
      </p>

      <div className="rope-aside">
        <h4 className="rope-subhead">
          <span className="rope-sub-tag">i</span>
          Invariance check: same dot product, different absolute positions
        </h4>
        <p className="rope-subhead-prose">
          Both setups have the same relative offset{' '}
          <Katex tex={`n - m = ${refDiff}`} />, so rotating both Q and K by
          their position angles leaves the dot product depending only on the
          gap. They must produce the same RoPE dot product.
        </p>

        <div className="rope-inv-row">
          <div className="rope-inv-header">
            <span className="rope-inv-tag">REF</span>
            Q at position <strong>0</strong>, K at position <strong>{refDiff}</strong>
            <span className="rope-inv-arrow">→ Q stays put, only K is rotated</span>
          </div>
          <VectorStrip values={Q_DEMO} label="Q  (position 0, no rotation)" prefix="Q" />
          <VectorStrip values={kRotRef} label={`K  (rotated by (n − m) · θ at each pair)`} prefix="K" />
          <div className="rope-inv-dot">
            Q · K&nbsp;=&nbsp;
            <Tip label={dotBreakdownTip(Q_DEMO, kRotRef)}>
              <strong>{refDot.toFixed(3)}</strong>
            </Tip>
          </div>
        </div>

        <div className="rope-inv-row" style={{ marginTop: 14 }}>
          <div className="rope-inv-header">
            <span className="rope-inv-tag rope-inv-tag-cur">CUR</span>
            Q at position <strong>{m}</strong>, K at position <strong>{n}</strong>
            <span className="rope-inv-arrow">→ both rotated, but by different amounts</span>
          </div>
          <VectorStrip values={qRot}    label={`Q  (rotated by m · θ at each pair)`} prefix="Q" />
          <VectorStrip values={kRot}    label={`K  (rotated by n · θ at each pair)`} prefix="K" />
          <div className="rope-inv-dot">
            Q · K&nbsp;=&nbsp;
            <Tip label={dotBreakdownTip(qRot, kRot)}>
              <strong className={Math.abs(refDot - ropeDot) < 0.005 ? 'match' : ''}>
                {ropeDot.toFixed(3)}
              </strong>
            </Tip>
          </div>
        </div>

        <p className="viz-caption" style={{ marginTop: 12 }}>
          Both dot products are identical even though the actual vectors in
          the strips look different — the rotated Q in the lower row has
          nothing in common with the un-rotated Q above. The model still
          sees the same score because everything except the angle{' '}
          <em>difference</em> cancels out in the dot product.
        </p>
      </div>
    </div>
  );
}

/* =========================================================
   §5 — Long-context extension: vanilla / PI / YaRN
   ========================================================= */
const LONG_CTX_FORMULAS = {
  vanilla:
    String.raw`\theta_{m,\,i} \;=\; m \cdot \theta_i, \qquad \theta_i \;=\; 10000^{-2i/d}`,
  pi:
    String.raw`\theta_{m,\,i} \;=\; \tfrac{m}{s} \cdot \theta_i, \qquad s \;=\; \tfrac{T_{\mathrm{eval}}}{T_{\mathrm{train}}} \;\;\text{(same } s \text{ for every pair } i\text{)}`,
  yarn:
    String.raw`\theta_{m,\,i} \;=\; m \cdot \theta_i \cdot \Big[\tfrac{1-\gamma_i}{s} + \gamma_i\Big], \qquad s \;=\; \tfrac{T_{\mathrm{eval}}}{T_{\mathrm{train}}}, \;\; \gamma_i \;=\; \mathrm{ramp}\!\big(\tfrac{\lambda_i}{T_{\mathrm{train}}};\,\alpha,\beta\big), \;\; \lambda_i \;=\; \tfrac{2\pi}{\theta_i}`,
};

const LONG_CTX_CAPTIONS = {
  vanilla: (
    <>
      The original RoPE schedule: each pair <Katex tex="i" /> rotates at its own
      fixed frequency <Katex tex="\theta_i" />, with the angle growing linearly
      in the absolute position <Katex tex="m" />. At <Katex tex="m = T_{\mathrm{eval}}" />{' '}
      a slow pair (small <Katex tex="\theta_i" />) has accumulated an angle the
      model never saw during training — that's the OOD failure mode.
    </>
  ),
  pi: (
    <>
      <a className="post-link" href="https://arxiv.org/abs/2306.15595" target="_blank" rel="noreferrer">Position Interpolation</a>{' '}
      (Chen et al., 2023) divides <em>every</em> position by{' '}
      <Katex tex="s = T_{\mathrm{eval}}/T_{\mathrm{train}}" /> before feeding it
      to RoPE. Every pair's effective frequency is shrunk by{' '}
      <Katex tex="1/s" />, so at <Katex tex="m = T_{\mathrm{eval}}" /> the
      slowest pair has only swept the same angle range it saw at{' '}
      <Katex tex="T_{\mathrm{train}}" />. Cheap (one constant, ~1k fine-tuning
      steps) but blunt — the fast pairs get squashed even though they didn't
      need fixing.
    </>
  ),
  yarn: (
    <>
      <a className="post-link" href="https://arxiv.org/abs/2309.00071" target="_blank" rel="noreferrer">YaRN</a>{' '}
      (Peng et al., 2023) makes the scale <em>per-pair</em>. Each pair's
      wavelength <Katex tex="\lambda_i = 2\pi/\theta_i" /> (the number of
      tokens it takes to complete one full rotation) is compared to{' '}
      <Katex tex="T_{\mathrm{train}}" />. The ramp <Katex tex="\gamma_i" /> is
      <strong> 1</strong> for fast pairs (<Katex tex="\lambda_i \ll T_{\mathrm{train}}" />,
      already cycled through every angle during training — leave them alone)
      and <strong>0</strong> for slow pairs (<Katex tex="\lambda_i \gg T_{\mathrm{train}}" />,
      genuinely OOD — apply full PI compression <Katex tex="1/s" />), with a
      linear ramp between thresholds <Katex tex="\alpha" /> and{' '}
      <Katex tex="\beta" /> (typically 1 and 32). On the slowest pair shown
      here, <Katex tex="\gamma_i = 0" /> and YaRN coincides with PI; on a fast
      pair, <Katex tex="\gamma_i = 1" /> and YaRN coincides with vanilla.
    </>
  ),
};

function SectionLongContext() {
  const [mode, setMode] = useState('vanilla');
  const tTrain = 2048;
  const tEval = 16384;
  const pairI = D_MODEL_DEMO / 2 - 1;
  const baseFreq = Math.pow(10000, -(2 * pairI) / D_MODEL_DEMO);

  const scaleFactor = tEval / tTrain;
  const angleAt = (pos) => {
    switch (mode) {
      case 'vanilla': return pos * baseFreq;
      case 'pi':      return (pos / scaleFactor) * baseFreq;
      case 'yarn': {
        return (pos / scaleFactor) * baseFreq;
      }
      default: return pos * baseFreq;
    }
  };

  const curveW = 560;
  const curveH = 200;
  const xPad = 36;
  const yPad = 20;
  const numPts = 80;
  const ptsScreen = useMemo(() => {
    const out = [];
    for (let k = 0; k <= numPts; k++) {
      const pos = (k / numPts) * tEval;
      out.push({ pos, ang: angleAt(pos) });
    }
    return out;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode]);

  const vanillaMax = tEval * baseFreq;
  const trainedMax = tTrain * baseFreq;
  const yMax = vanillaMax * 1.05;

  const toCX = (pos) => xPad + (pos / tEval) * (curveW - 2 * xPad);
  const toCY = (ang) => curveH - yPad - (ang / yMax) * (curveH - 2 * yPad);
  const pathD = ptsScreen
    .map((p, idx) => `${idx === 0 ? 'M' : 'L'} ${toCX(p.pos)} ${toCY(p.ang)}`)
    .join(' ');

  const trainX = toCX(tTrain);
  const trainAngleY = toCY(trainedMax);

  return (
    <div className="viz-panel">
      <div className="rope-controls-row">
        <div className="rope-tabs" role="tablist">
          {[
            { id: 'vanilla', label: 'Vanilla RoPE' },
            { id: 'pi',      label: 'Position Interpolation' },
            { id: 'yarn',    label: 'YaRN (frequency-aware)' },
          ].map((opt) => (
            <button
              key={opt.id}
              role="tab"
              className={`rope-tab ${mode === opt.id ? 'active' : ''}`}
              onClick={() => setMode(opt.id)}
            >
              {opt.label}
            </button>
          ))}
        </div>
      </div>

      <div className="viz-math-block" style={{ marginTop: 10 }}>
        <Katex block tex={LONG_CTX_FORMULAS[mode]} />
      </div>

      <p className="rope-mode-caption">
        {LONG_CTX_CAPTIONS[mode]}
      </p>

      <div style={{ marginTop: 12 }}>
        <div className="rope-curve-label">
          rotation angle at one mid-frequency pair (rad) vs position
        </div>
        <svg viewBox={`0 0 ${curveW} ${curveH}`} width="100%" className="rope-curve-svg" style={{ maxWidth: curveW }}>
          <rect x={xPad} y={yPad} width={curveW - 2 * xPad} height={trainAngleY - yPad} fill="hsl(8, 60%, 92%)" />
          <rect x={xPad} y={trainAngleY} width={curveW - 2 * xPad} height={(curveH - yPad) - trainAngleY} fill="hsl(148, 45%, 88%)" />
          <line x1={xPad} y1={trainAngleY} x2={curveW - xPad} y2={trainAngleY} stroke="hsl(8, 55%, 50%)" strokeWidth="1.5" strokeDasharray="4 4" />
          <text x={curveW - xPad - 4} y={trainAngleY - 4} textAnchor="end" fontSize="10" fill="hsl(8, 55%, 35%)" className="rope-curve-tick">
            trained-angle max = {trainedMax.toFixed(2)} rad
          </text>
          <line x1={trainX} y1={yPad} x2={trainX} y2={curveH - yPad} stroke="hsl(218, 30%, 70%)" strokeWidth="1" strokeDasharray="2 3" />
          <text x={trainX + 4} y={curveH - yPad - 4} fontSize="10" fill="hsl(218, 30%, 45%)" className="rope-curve-tick">T_train = {tTrain.toLocaleString()}</text>
          <line x1={xPad} y1={toCY(0)} x2={curveW - xPad} y2={toCY(0)} stroke="hsl(218, 20%, 80%)" />
          <path d={pathD} stroke="hsl(218, 50%, 40%)" strokeWidth="2" fill="none" />
          <text x={xPad} y={curveH - 4} fontSize="10" className="rope-curve-tick">0</text>
          <text x={curveW - xPad} y={curveH - 4} textAnchor="end" fontSize="10" className="rope-curve-tick">{tEval.toLocaleString()}</text>
          <text x={xPad - 4} y={toCY(0) + 3} textAnchor="end" fontSize="10" className="rope-curve-tick">0</text>
          <text x={xPad - 4} y={toCY(yMax) + 9} textAnchor="end" fontSize="10" className="rope-curve-tick">
            {yMax.toFixed(1)}
          </text>
        </svg>
      </div>

      <p className="viz-caption">
        Chart shows the slowest pair (<Katex tex={`i = ${pairI}`} />) — the
        one where the OOD problem actually bites. Green band = rotation-angle
        range the model saw during training; red band above the dashed line
        = angles the model has never been asked to interpret. The vertical
        dotted line marks <Katex tex="T_\mathrm{train}" /> as a position
        reference.
      </p>
    </div>
  );
}

/* =========================================================
   Code expander — compact per-pair rotation snippet
   ========================================================= */
const CODE = `# RoPE — rotate each (Q[2i], Q[2i+1]) pair by m * theta_i
import math, torch

def rotate2d(x, y, theta):
    c, s = math.cos(theta), math.sin(theta)
    return c * x - s * y,  s * x + c * y

def apply_rope(vec, pos, d):
    out = vec.clone()
    for i in range(d // 2):
        theta = pos * (10000.0 ** (-2 * i / d))   # pair i's frequency
        a, b  = vec[2*i], vec[2*i + 1]
        out[2*i], out[2*i + 1] = rotate2d(a, b, theta)
    return out

# Attention with RoPE: depends only on (n - m)
score = apply_rope(q, m, d) @ apply_rope(k, n, d)
`;

function CodeBlock() {
  return (
    <details className="post-code-details">
      <summary>
        <span className="post-code-summary-icon" aria-hidden="true">›</span>
        <span>Show the rotation in Python</span>
        <span className="post-code-summary-hint">click to expand</span>
      </summary>
      <pre className="rope-code-pre">{CODE}</pre>
    </details>
  );
}

/* =========================================================
   Page
   ========================================================= */
export default function RotaryPositionalEmbeddings() {
  useEffect(() => {
    const prev = document.title;
    document.title = 'Rotary Positional Embeddings (RoPE) — Bernhard Walser';
    return () => { document.title = prev; };
  }, []);

  return (
    <article className="post-2026 rope-post">
      <div className="post-wrap">
        <div className="post-hero">
          <div className="post-series-tag"><span className="post-live-dot" aria-hidden="true" />Positions in Transformers · Part 2 of 2</div>
          <h1>Rotary Positional Embeddings (RoPE)</h1>
          <p className="post-lede">
            Self-attention doesn't know which token came first. Earlier schemes
            patched this by <em>adding</em> a position vector or a position bias.
            RoPE does something different: it <em>rotates</em> each query and
            key by an angle proportional to its position. The dot product then
            depends only on the gap between two tokens — and the whole trick
            costs zero new parameters.
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

        <h2 className="reveal">The geometric core: rotation in a 2D plane</h2>
        <p>
          The next interactives all build on one primitive: rotating a 2D
          vector. Pick any vector <Katex tex="v = (x, y)" /> and any angle{' '}
          <Katex tex="\theta" />, and watch{' '}
          <Katex tex="R(\theta)" /> send it to its rotated copy{' '}
          <Katex tex="R(\theta) \cdot v" />. Hover any entry of the matrix
          to see the dot product that produced it.
        </p>
        <SectionRotation />

        <h2 className="reveal">Pairwise dot-product invariance</h2>
        <p>
          Here's the key fact RoPE leans on. If you have two 2D vectors and
          rotate each by its own angle, the dot product depends{' '}
          <em>only on the difference</em> of the two angles, not on the
          individual angles themselves:
        </p>
        <div className="viz-math-block">
          <Katex block tex="(R(\theta_1)\,v_1) \cdot (R(\theta_2)\,v_2) \;=\; v_1 \cdot R(\theta_2 - \theta_1)\,v_2" />
        </div>
        <p>
          Below: two vectors and two angle sliders. Tick{' '}
          <strong>lock difference</strong> and drag — the rotated dot product
          stays constant, because <Katex tex="\theta_1 - \theta_2" /> stays
          constant. Untick and pull them apart — the dot product moves only
          when the difference moves.
        </p>
        <SectionDotInvariance />

        <h2 className="reveal">Scaling to d-dimensional Q and K</h2>
        <p>
          Two-dimensional rotations are a building block. To handle real Q
          and K vectors (which live in <Katex tex="d_{head}" />-dim, typically
          64 or 128), RoPE <strong>pairs up dimensions</strong> and applies
          a different rotation to each pair. Dim 0 and dim 1 are pair 0; dim
          2 and dim 3 are pair 1; and so on. Each pair gets its own angular
          frequency:
        </p>
        <div className="viz-math-block">
          <Katex block tex="\theta_i \;=\; 10000^{-2i/d}, \quad i = 0, 1, \ldots, d/2 - 1" />
        </div>
        <p>
          At token position <Katex tex="m" />, pair <Katex tex="i" /> is
          rotated by <Katex tex="m \cdot \theta_i" />. Pair 0 uses{' '}
          <Katex tex="\theta_0 = 1" /> and sweeps fast across positions; the
          last pair uses something near <Katex tex="10^{-4}" /> and crawls.
          Slide <em>m</em> below — every pair rotates, but at completely
          different speeds. This is exactly the Fourier multi-resolution
          intuition: at any position, the model has both fine-grained
          (high-frequency) and coarse-grained (low-frequency) views of where
          it is.
        </p>
        <SectionMultiPair />
        <p>
          A helpful intuition: think of the <Katex tex="d/2" /> pairs as the
          hands on a clock. The fastest pair (<Katex tex="i=0" />) is the
          second hand — one full revolution every six tokens or so, perfect
          for distinguishing immediate neighbours but useless for long
          ranges (it cycles through the same angle every six positions, so{' '}
          <Katex tex="m=6" /> and <Katex tex="m=12" /> look identical to it
          modulo <Katex tex="2\pi" />). The slowest pair is the hour hand —
          it barely moves between adjacent tokens, but over hundreds of
          positions it traces out clearly distinct angles, so it
          disambiguates long offsets the fast pair confuses. The model can
          read whichever scale of relative distance matters for the head's
          job: short-range syntactic dependencies use the fast pairs;
          long-range semantic dependencies use the slow ones.
        </p>

        <h2 className="reveal">Attention with RoPE — what the model sees</h2>
        <p>
          Now we plug the rotations into attention. For a query at position{' '}
          <Katex tex="m" /> and a key at position <Katex tex="n" />, RoPE
          rotates them and computes the dot product:
        </p>
        <div className="viz-math-block">
          <Katex block tex="\mathrm{score}(q_m, k_n) \;=\; (R_m\, q) \cdot (R_n\, k)" />
        </div>
        <p>
          Because the rotation matrices are orthogonal and each pair of
          dimensions rotates by an angle proportional to position, the score
          depends only on the <em>difference</em> <Katex tex="n - m" /> — by
          the same argument as the 2D case, scaled up to many pairs. Slide{' '}
          <em>m</em> and <em>n</em> below with the lock on: both positions
          slide but the dot product stays put. Untick the lock and change the
          gap: the dot product slides along the curve.
        </p>
        <SectionRoPEDot />

        <CodeBlock />

        <h2 className="reveal">Long-context extension: PI and YaRN</h2>
        <p>
          One subtlety from the clock-face intuition: the model is trained
          at some maximum context length <Katex tex="T_\mathrm{train}" />{' '}
          (say 2048), and the OOD problem at longer eval contexts isn't
          uniform across pairs. The <em>fast</em> pairs cycled through every
          possible <Katex tex="(\cos, \sin)" /> value many times during
          training — going further in position just produces more of the
          same, modulo <Katex tex="2\pi" />. The <em>slow</em> pairs are the
          ones in trouble: they only swept a thin arc during training, and
          at long contexts they reach angles the model has genuinely never
          seen.
        </p>
        <p>
          Two techniques fix this without retraining.{' '}
          <a className="post-link" href="https://arxiv.org/abs/2306.15595" target="_blank" rel="noreferrer">
            Position Interpolation
          </a>{' '}
          (Chen et al. 2023) was first: rescale <em>all</em> position
          indices uniformly by <Katex tex="T_\mathrm{train}/T_\mathrm{eval}" />,
          so every pair's angles stay inside the trained range — but it
          also compresses the fast pairs that didn't need it, costing
          short-range resolution.{' '}
          <a className="post-link" href="https://arxiv.org/abs/2309.00071" target="_blank" rel="noreferrer">
            YaRN
          </a>{' '}
          (Peng et al. 2023) is the frequency-aware version: leave the fast
          pairs alone, apply PI-style compression to the slow pairs
          (where the real OOD is), ramp smoothly in between. (The bridge
          between the two — scaling RoPE's base constant instead of
          positions to get frequency-awareness from a single number — is
          known as <em>NTK-aware scaling</em>, and YaRN's piecewise
          refinement is sometimes called <em>NTK-by-parts</em>.)
        </p>
        <SectionLongContext />
        <p>
          A fair question: doesn't rescaling positions <em>change</em> what
          relative offsets mean to the model? How can this work without
          retraining? The answer is that the model never learned "offset 5
          means X" — it learned to react to rotation-angle <em>signatures</em>.
          With PI at <Katex tex="S = 8" />, an eval offset of 8 now produces
          the geometric signal that an offset of 1 used to. The model's heads
          still see distance-monotone signals, the dot product still encodes{' '}
          <em>some</em> relative-offset structure, and the original training
          distribution is approximately preserved.
        </p>
        <p>
          What you give up is <strong>resolution</strong>: with positions
          compressed 8×, the model can't tell "1 token apart" from "8 tokens
          apart" any more — both look like the same angle to it. Fine-grained
          syntactic behaviour suffers. YaRN's frequency-aware design is the
          response to this: leave the fast pairs (where short-range
          resolution lives) untouched, and only compress the slow pairs
          (where the OOD problem actually was). You still lose some quality
          at extreme extensions, which is why production deployments
          typically pair PI/YaRN with a few thousand steps of fine-tuning on
          long-context data — the heads adapt to the new effective offsets
          and recover most of the degradation. Post-hoc context extension
          works, but it isn't free, and the "free" part has a ceiling.
        </p>

        <h2 className="reveal">Why everyone uses it</h2>
        <p>
          Position turns out to be a geometric quantity, not a learned one.
          The earlier schemes (absolute embeddings, Shaw, T5, Swin) bolted it
          on as a separate vector or a separate bias table — extra parameters,
          bounded by training length, awkward to extrapolate. RoPE replaces
          all of that with a single rotation per dimension pair: position{' '}
          <Katex tex="m" /> becomes an angle, and the dot product between{' '}
          <Katex tex="R_m q" /> and <Katex tex="R_n k" /> naturally encodes
          the relative offset <Katex tex="n - m" />. Zero new parameters, no
          table to look up, and (with PI / YaRN) it stretches well past the
          training horizon.
        </p>
        <p>
          That's why <a className="post-link" href="https://arxiv.org/abs/2302.13971" target="_blank" rel="noreferrer">Llama</a>,{' '}
          <a className="post-link" href="https://arxiv.org/abs/2310.06825" target="_blank" rel="noreferrer">Mistral</a>,{' '}
          <a className="post-link" href="https://arxiv.org/abs/2309.16609" target="_blank" rel="noreferrer">Qwen</a>, and{' '}
          <a className="post-link" href="https://arxiv.org/abs/2412.19437" target="_blank" rel="noreferrer">DeepSeek</a>{' '}
          all use RoPE. The 2D rotation from the first section is the right
          unit for talking about position; everything else — multi-frequency
          pairs, scaling tricks for long context — is bookkeeping on top of
          that one idea.
        </p>

        <h2 className="reveal">References</h2>
        <div className="post-refs">
          <div className="head ref-cite">Cite</div>
          <div className="head ref-title">Title</div>
          <div className="head ref-link">Link</div>

          <div className="ref-cite">Su et al. 2021</div>
          <div>
            <div className="ref-title">
              <a href="https://arxiv.org/abs/2104.09864" target="_blank" rel="noreferrer">RoFormer: Enhanced Transformer with Rotary Position Embedding</a>
            </div>
            <div className="ref-note">The original RoPE paper. Used by Llama, Mistral, Qwen, DeepSeek.</div>
          </div>
          <div className="ref-link"><a href="https://arxiv.org/abs/2104.09864" target="_blank" rel="noreferrer">arxiv 2104.09864</a></div>

          <div className="ref-cite">Vaswani et al. 2017</div>
          <div>
            <div className="ref-title">
              <a href="https://arxiv.org/abs/1706.03762" target="_blank" rel="noreferrer">Attention Is All You Need</a>
            </div>
            <div className="ref-note">The Transformer paper; sinusoidal absolute positional embeddings as a starting point.</div>
          </div>
          <div className="ref-link"><a href="https://arxiv.org/abs/1706.03762" target="_blank" rel="noreferrer">arxiv 1706.03762</a></div>

          <div className="ref-cite">Chen et al. 2023</div>
          <div>
            <div className="ref-title">
              <a href="https://arxiv.org/abs/2306.15595" target="_blank" rel="noreferrer">Extending Context Window of Large Language Models via Position Interpolation</a>
            </div>
            <div className="ref-note">Uniform position rescaling by T_train / T_eval.</div>
          </div>
          <div className="ref-link"><a href="https://arxiv.org/abs/2306.15595" target="_blank" rel="noreferrer">arxiv 2306.15595</a></div>

          <div className="ref-cite">Peng et al. 2023</div>
          <div>
            <div className="ref-title">
              <a href="https://arxiv.org/abs/2309.00071" target="_blank" rel="noreferrer">YaRN: Efficient Context Window Extension</a>
            </div>
            <div className="ref-note">Frequency-aware NTK-by-parts scaling; leaves fast pairs alone.</div>
          </div>
          <div className="ref-link"><a href="https://arxiv.org/abs/2309.00071" target="_blank" rel="noreferrer">arxiv 2309.00071</a></div>

          <div className="ref-cite">Press et al. 2021</div>
          <div>
            <div className="ref-title">
              <a href="https://arxiv.org/abs/2108.12409" target="_blank" rel="noreferrer">ALiBi: Attention with Linear Biases</a>
            </div>
            <div className="ref-note">Parameter-free linear-distance bias — the other big "rotate or bias" alternative.</div>
          </div>
          <div className="ref-link"><a href="https://arxiv.org/abs/2108.12409" target="_blank" rel="noreferrer">arxiv 2108.12409</a></div>

          <div className="ref-cite">Earlier post</div>
          <div>
            <div className="ref-title">
              <a href="#/blog/positional-encodings">Positional Encodings: A Tour</a>
            </div>
            <div className="ref-note">Part 1 of this series — absolute, Shaw, T5, Swin: the predecessors RoPE replaces.</div>
          </div>
          <div className="ref-link"><a href="#/blog/positional-encodings">post</a></div>
        </div>

        <footer className="post-footer">
          <p>
            <strong>Part 2 of Positions in Transformers</strong> · Previous:{' '}
            <a className="post-link" href="#/blog/positional-encodings">Positional Encodings: A Tour</a>
            {' · '}Start of the series:{' '}
            <a className="post-link" href="#/blog/positional-encodings">Positional Encodings: A Tour</a>.
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
