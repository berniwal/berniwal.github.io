// src/posts/VisualizingRoPE.jsx
// Part 3 of "Visualizing ML" — RoPE, geometrically.
import React, { useEffect, useMemo, useRef, useState } from 'react';
import './VisualizingRoPE.css';

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

/* =========================================================
   Info box — reusable aside callout
   ========================================================= */

function InfoBox({ title, children }) {
  return (
    <aside className="viz-subsection viz-aside">
      <h4 className="viz-subhead">
        <span className="viz-sub-tag viz-sub-tag-info">i</span>
        {title}
      </h4>
      <div className="viz-aside-body">{children}</div>
    </aside>
  );
}

/* =========================================================
   2D rotation SVG primitive
   Math-space coords are (x, y) with x right, y UP. SVG y is flipped.
   ========================================================= */

const CANVAS_PX = 280;     // total SVG width/height
const CANVAS_PAD = 24;     // border padding inside the SVG
const AXIS_RANGE = 1.4;    // math-space axis range from -AXIS_RANGE to +AXIS_RANGE

function toSvgX(mx) {
  const inner = CANVAS_PX - 2 * CANVAS_PAD;
  return CANVAS_PAD + ((mx + AXIS_RANGE) / (2 * AXIS_RANGE)) * inner;
}

function toSvgY(my) {
  const inner = CANVAS_PX - 2 * CANVAS_PAD;
  return CANVAS_PAD + ((AXIS_RANGE - my) / (2 * AXIS_RANGE)) * inner;
}

function RotationAxes({ size = CANVAS_PX }) {
  // Grid lines at -1, 0, 1
  const xs = [-1, 0, 1];
  const ys = [-1, 0, 1];
  return (
    <g aria-hidden="true">
      {/* unit-circle hint */}
      <circle
        cx={toSvgX(0)}
        cy={toSvgY(0)}
        r={toSvgX(1) - toSvgX(0)}
        className="viz-rope-svg-grid"
        fill="none"
        strokeDasharray="3 4"
      />
      {/* axes */}
      <line x1={toSvgX(-AXIS_RANGE)} y1={toSvgY(0)} x2={toSvgX(AXIS_RANGE)} y2={toSvgY(0)} className="viz-rope-svg-axis" />
      <line x1={toSvgX(0)} y1={toSvgY(-AXIS_RANGE)} x2={toSvgX(0)} y2={toSvgY(AXIS_RANGE)} className="viz-rope-svg-axis" />
      {/* tick labels */}
      {xs.filter((v) => v !== 0).map((v) => (
        <text key={`xt-${v}`} x={toSvgX(v)} y={toSvgY(0) + 12} textAnchor="middle" className="viz-rope-svg-label">{v}</text>
      ))}
      {ys.filter((v) => v !== 0).map((v) => (
        <text key={`yt-${v}`} x={toSvgX(0) - 8} y={toSvgY(v) + 3} textAnchor="end" className="viz-rope-svg-label">{v}</text>
      ))}
    </g>
  );
}

function ArrowVector({ x, y, color, label, dashed = false, opacity = 1 }) {
  // Draw an arrow from origin to (x, y) in math-space.
  const sx = toSvgX(0);
  const sy = toSvgY(0);
  const ex = toSvgX(x);
  const ey = toSvgY(y);
  // Arrowhead is a small triangle perpendicular to the vector.
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
        className="viz-rope-svg-vector"
        strokeDasharray={dashed ? '4 4' : undefined}
      />
      <polygon
        points={`${ex},${ey} ${head1x},${head1y} ${head2x},${head2y}`}
        fill={color}
        className="viz-rope-svg-vector-head"
      />
      {label && (
        <text
          x={ex + (Math.cos(angle) >= 0 ? 8 : -8) * (Math.abs(Math.cos(angle)) < 0.4 ? 0 : 1)}
          y={ey - 8}
          textAnchor={Math.cos(angle) >= 0 ? 'start' : 'end'}
          fill={color}
          className="viz-rope-svg-angle-label"
        >
          {label}
        </text>
      )}
    </g>
  );
}

function AngleArc({ angle, radius = 0.32, color = 'hsl(218, 50%, 65%)' }) {
  // Arc from angle 0 (positive x-axis) to `angle` (radians).
  if (Math.abs(angle) < 1e-3) return null;
  const sx = toSvgX(radius);
  const sy = toSvgY(0);
  const ex = toSvgX(radius * Math.cos(angle));
  const ey = toSvgY(radius * Math.sin(angle));
  // SVG arc flags
  const largeArc = Math.abs(angle) > Math.PI ? 1 : 0;
  const sweep = angle > 0 ? 0 : 1; // SVG y is flipped, so swap sweep
  const r = toSvgX(radius) - toSvgX(0);
  return (
    <g>
      <path
        d={`M ${toSvgX(0)} ${toSvgY(0)} L ${sx} ${sy} A ${r} ${r} 0 ${largeArc} ${sweep} ${ex} ${ey} Z`}
        className="viz-rope-svg-arc-fill"
      />
      <text
        x={toSvgX((radius + 0.12) * Math.cos(angle / 2))}
        y={toSvgY((radius + 0.12) * Math.sin(angle / 2))}
        textAnchor="middle"
        className="viz-rope-svg-angle-label"
        fill={color}
      >
        θ
      </text>
    </g>
  );
}

function RotationCanvas({ children, size = CANVAS_PX }) {
  return (
    <div className="viz-rope-canvas" style={{ width: size, height: size }}>
      <svg viewBox={`0 0 ${size} ${size}`} width={size} height={size}>
        <RotationAxes size={size} />
        {children}
      </svg>
    </div>
  );
}

/* =========================================================
   §3 — The geometric core: rotation in a 2D plane
   ========================================================= */

function rotate2D(x, y, theta) {
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  return [c * x - s * y, s * x + c * y];
}

function SectionRotation() {
  const [vx, setVx] = useState(0.8);
  const [vy, setVy] = useState(0.3);
  const [thetaDeg, setThetaDeg] = useState(60);
  const theta = (thetaDeg * Math.PI) / 180;
  const [rx, ry] = rotate2D(vx, vy, theta);

  // Build a faint trace of intermediate rotations from 0 → theta in 20 steps.
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
      <div className="viz-rope-stage">
        <RotationCanvas>
          <AngleArc angle={theta} />
          {/* Trace of intermediate rotations */}
          <polyline points={tracePoints} className="viz-rope-svg-trace" stroke="hsl(218, 50%, 55%)" />
          {/* Original vector */}
          <ArrowVector x={vx} y={vy} color="hsl(218, 50%, 35%)" label="v" />
          {/* Rotated vector */}
          <ArrowVector x={rx} y={ry} color="hsl(28, 65%, 45%)" label="R(θ)·v" />
        </RotationCanvas>

        <div className="viz-rope-readout">
          <div className="viz-rope-controls">
            <label className="viz-rope-slider">
              <span>x</span>
              <input type="range" min={-1.2} max={1.2} step={0.05} value={vx} onChange={(e) => setVx(+e.target.value)} />
              <strong>{vx.toFixed(2)}</strong>
            </label>
            <label className="viz-rope-slider">
              <span>y</span>
              <input type="range" min={-1.2} max={1.2} step={0.05} value={vy} onChange={(e) => setVy(+e.target.value)} />
              <strong>{vy.toFixed(2)}</strong>
            </label>
            <label className="viz-rope-slider">
              <span>θ</span>
              <input type="range" min={-180} max={180} step={5} value={thetaDeg} onChange={(e) => setThetaDeg(+e.target.value)} />
              <strong>{thetaDeg}°</strong>
            </label>
          </div>

          <div className="viz-rope-matrix">
            <div style={{ fontFamily: 'var(--mono)', fontSize: '0.78rem', color: 'var(--ink-faint)', marginBottom: 4 }}>
              R(θ) · v  =
            </div>
            <div className="viz-rope-matrix-grid">
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
            <div style={{ marginTop: 6, fontFamily: 'var(--mono)', fontSize: '0.78rem', color: 'var(--ink-soft)' }}>
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

      <pre className="viz-code">
{`# 2D rotation — the building block of RoPE
import torch
def rotate2d(x, y, theta):
    c, s = torch.cos(theta), torch.sin(theta)
    return c * x - s * y,  s * x + c * y`}
      </pre>
    </div>
  );
}

/* =========================================================
   §4 — Pairwise dot-product invariance
   ========================================================= */

function SectionDotInvariance() {
  // Two fixed-ish vectors; the user adjusts angles θ₁, θ₂.
  const [v1x, setV1x] = useState(0.9);
  const [v1y, setV1y] = useState(0.15);
  const [v2x, setV2x] = useState(0.4);
  const [v2y, setV2y] = useState(0.75);
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

  // Lock-difference handlers — slide both together.
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
      <div className="viz-rope-stage">
        <RotationCanvas>
          {/* Original vectors (low opacity) */}
          <ArrowVector x={v1x} y={v1y} color="hsl(218, 50%, 35%)" label="v₁" opacity={0.35} />
          <ArrowVector x={v2x} y={v2y} color="hsl(148, 35%, 28%)" label="v₂" opacity={0.35} />
          {/* Rotated vectors */}
          <ArrowVector x={r1x} y={r1y} color="hsl(218, 50%, 35%)" label="R(θ₁)v₁" />
          <ArrowVector x={r2x} y={r2y} color="hsl(148, 35%, 28%)" label="R(θ₂)v₂" />
        </RotationCanvas>

        <div className="viz-rope-readout">
          <div className="viz-rope-controls">
            <label className="viz-rope-slider">
              <span>θ₁</span>
              <input type="range" min={-180} max={180} step={5} value={t1Deg} onChange={onT1} />
              <strong>{t1Deg}°</strong>
            </label>
            <label className="viz-rope-slider">
              <span>θ₂</span>
              <input type="range" min={-180} max={180} step={5} value={t2Deg} onChange={onT2} />
              <strong>{t2Deg}°</strong>
            </label>
            <label style={{ display: 'inline-flex', alignItems: 'center', gap: 8, fontSize: '0.88rem', color: 'var(--ink-soft)' }}>
              <input type="checkbox" checked={lock} onChange={(e) => setLock(e.target.checked)} />
              lock difference  θ₁ − θ₂ = <strong style={{ fontFamily: 'var(--mono)', color: 'var(--accent)', marginLeft: 4 }}>{diff}°</strong>
            </label>
          </div>

          <div className="viz-rope-dot-readout">
            <span className="label">v₁ · v₂  <em style={{ fontSize: '0.8rem' }}>(original)</em></span>
            <span className="value">{dotOriginal.toFixed(3)}</span>

            <span className="label">R(θ₁)v₁ · R(θ₂)v₂  <em style={{ fontSize: '0.8rem' }}>(rotated)</em></span>
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

      <pre className="viz-code">
{`# Pairwise dot product after rotating each vector by its own angle
q_rot = rotate2d(q, theta1)
k_rot = rotate2d(k, theta2)
score = (q_rot * k_rot).sum()    # depends only on (theta1 - theta2)`}
      </pre>
    </div>
  );
}

/* =========================================================
   §5 — Scaling to d-dimensional Q and K (multi-pair rotation)
   ========================================================= */

const D_MODEL_DEMO = 8;
const PAIR_VECTORS = [
  [0.7,  0.35],
  [0.55, -0.45],
  [-0.5, 0.6],
  [0.6,  0.2],
];

function SectionMultiPair() {
  const [m, setM] = useState(16);
  // Standard RoPE base = 10000; θ_i = 10000^(-2i/d) for i = 0 .. d/2-1
  const freqs = useMemo(() => {
    const out = [];
    for (let i = 0; i < D_MODEL_DEMO / 2; i++) {
      out.push(Math.pow(10000, -(2 * i) / D_MODEL_DEMO));
    }
    return out;
  }, []);

  return (
    <div className="viz-panel">
      <div className="viz-controls">
        <label className="viz-rope-slider" style={{ flex: 1, minWidth: 280 }}>
          <span>position m</span>
          <input type="range" min={0} max={256} step={1} value={m} onChange={(e) => setM(+e.target.value)} />
          <strong>{m}</strong>
        </label>
      </div>

      <div className="viz-rope-pair-grid">
        {PAIR_VECTORS.map(([vx, vy], i) => {
          const theta = m * freqs[i];
          const [rx, ry] = rotate2D(vx, vy, theta);
          return (
            <div key={i} className="viz-rope-pair-card">
              <div className="viz-rope-pair-title">
                pair (dim {2 * i}, dim {2 * i + 1})
              </div>
              <RotationCanvas size={160}>
                <ArrowVector x={vx} y={vy} color="hsl(218, 50%, 35%)" opacity={0.3} />
                <ArrowVector x={rx} y={ry} color="hsl(28, 65%, 45%)" />
              </RotationCanvas>
              <div className="viz-rope-pair-freq">
                θᵢ = {freqs[i] < 0.01 ? freqs[i].toExponential(2) : freqs[i].toFixed(3)}<br />
                m·θᵢ = {theta.toFixed(3)} rad
              </div>
            </div>
          );
        })}
      </div>

      <p className="viz-caption" style={{ marginTop: 12 }}>
        Each pair rotates at a different speed. The leftmost pair{' '}
        <Katex tex="(i{=}0)" /> uses <Katex tex="\theta_0 = 1" /> rad/position
        and sweeps fast; the rightmost <Katex tex="(i{=}3)" /> uses{' '}
        <Katex tex="\theta_3 = 10000^{-6/8} \approx 0.001" /> rad/position and
        barely moves over hundreds of tokens. The choice is exactly Fourier
        multi-resolution: different positions produce different rotation
        patterns across all frequencies, so the model can read off both nearby
        word order (high-freq pairs change) and long-range structure
        (low-freq pairs change).
      </p>

      <pre className="viz-code">
{`# RoPE frequencies for d_model = d
import torch
i = torch.arange(0, d, 2)              # 0, 2, 4, ..., d-2
freqs = 10000.0 ** (-i / d)            # θ_0 = 1, θ_1 = 1/10, ..., θ_(d/2-1) ≈ 1/10000`}
      </pre>
    </div>
  );
}

/* =========================================================
   §6 — Attention with RoPE: only (m − n) matters
   ========================================================= */

// Small pre-baked Q, K vectors, 4-dim = 2 pairs, so the math fits the page.
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

  // Sweep n − m from -32 to +32 with m fixed at the current value to
  // produce the dot-product-as-function-of-relative-position curve.
  const sweepData = useMemo(() => {
    const pts = [];
    const range = 32;
    for (let d = -range; d <= range; d++) {
      const nLocal = m + d;
      if (nLocal < 0 || nLocal > 256) { pts.push(null); continue; }
      const qr = applyRoPE(Q_DEMO, m);
      const kr = applyRoPE(K_DEMO, nLocal);
      pts.push({ d, v: dotProduct(qr, kr) });
    }
    return pts;
  }, [m]);

  // Curve SVG
  const curveW = 460;
  const curveH = 140;
  const xPad = 28;
  const yPad = 14;
  const xs = sweepData.filter(Boolean).map((p) => p.d);
  const vs = sweepData.filter(Boolean).map((p) => p.v);
  const vMin = Math.min(...vs, -1);
  const vMax = Math.max(...vs, 1);
  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);
  const toCX = (d) => xPad + ((d - xMin) / (xMax - xMin)) * (curveW - 2 * xPad);
  const toCY = (v) => curveH - yPad - ((v - vMin) / (vMax - vMin)) * (curveH - 2 * yPad);
  const pathD = sweepData
    .map((p, idx) => p ? `${idx === 0 ? 'M' : 'L'} ${toCX(p.d)} ${toCY(p.v)}` : '')
    .filter(Boolean)
    .join(' ');
  const currentDiff = n - m;

  return (
    <div className="viz-panel">
      <div className="viz-controls">
        <label className="viz-rope-slider" style={{ minWidth: 260 }}>
          <span>m (query)</span>
          <input type="range" min={0} max={64} step={1} value={m} onChange={onM} />
          <strong>{m}</strong>
        </label>
        <label className="viz-rope-slider" style={{ minWidth: 260 }}>
          <span>n (key)</span>
          <input type="range" min={0} max={64} step={1} value={n} onChange={onN} />
          <strong>{n}</strong>
        </label>
        <label style={{ display: 'inline-flex', alignItems: 'center', gap: 8, fontSize: '0.88rem', color: 'var(--ink-soft)' }}>
          <input type="checkbox" checked={lock} onChange={(e) => setLock(e.target.checked)} />
          lock (n − m) = <strong style={{ fontFamily: 'var(--mono)', color: 'var(--accent)', marginLeft: 4 }}>{currentDiff}</strong>
        </label>
      </div>

      <div className="viz-rope-dot-readout" style={{ maxWidth: 540 }}>
        <span className="label">q · k  <em style={{ fontSize: '0.8rem' }}>(no RoPE — constant in m, n)</em></span>
        <span className="value">{rawDot.toFixed(3)}</span>
        <span className="label">RoPE(q, m) · RoPE(k, n)  <em style={{ fontSize: '0.8rem' }}>(depends only on n − m)</em></span>
        <span className="value match">{ropeDot.toFixed(3)}</span>
      </div>

      <div style={{ marginTop: 18 }}>
        <div style={{ fontFamily: 'var(--mono)', fontSize: '0.78rem', color: 'var(--ink-faint)', marginBottom: 4 }}>
          RoPE dot product as a function of (n − m), with m = {m} held fixed
        </div>
        <svg viewBox={`0 0 ${curveW} ${curveH}`} width="100%" style={{ maxWidth: curveW, background: 'hsl(218, 30%, 98%)', border: '1px solid hsl(218, 30%, 90%)', borderRadius: 8 }}>
          {/* y = 0 line */}
          <line x1={xPad} y1={toCY(0)} x2={curveW - xPad} y2={toCY(0)} stroke="hsl(218, 20%, 80%)" strokeDasharray="3 3" />
          {/* x = 0 vertical */}
          <line x1={toCX(0)} y1={yPad} x2={toCX(0)} y2={curveH - yPad} stroke="hsl(218, 20%, 80%)" />
          {/* curve */}
          <path d={pathD} stroke="hsl(218, 50%, 40%)" strokeWidth="2" fill="none" />
          {/* current marker */}
          <circle cx={toCX(currentDiff)} cy={toCY(ropeDot)} r="4.5" fill="hsl(28, 65%, 45%)" />
          {/* x-axis labels */}
          <text x={toCX(0)} y={curveH - 2} textAnchor="middle" fontSize="9" fill="var(--ink-faint)" fontFamily="var(--mono)">0</text>
          <text x={toCX(-16)} y={curveH - 2} textAnchor="middle" fontSize="9" fill="var(--ink-faint)" fontFamily="var(--mono)">-16</text>
          <text x={toCX(16)} y={curveH - 2} textAnchor="middle" fontSize="9" fill="var(--ink-faint)" fontFamily="var(--mono)">+16</text>
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

      <pre className="viz-code">
{`# Apply RoPE to one head's q and k, then score
def apply_rope(x, pos, freqs):           # x: (d,)
    out = x.clone()
    for i, theta in enumerate(freqs):
        a, b = x[2*i], x[2*i + 1]
        out[2*i]     = math.cos(pos*theta)*a - math.sin(pos*theta)*b
        out[2*i + 1] = math.sin(pos*theta)*a + math.cos(pos*theta)*b
    return out

score = apply_rope(q, m, freqs) @ apply_rope(k, n, freqs)   # depends only on (n − m)`}
      </pre>
    </div>
  );
}

/* =========================================================
   §7 — Long-context extension: vanilla / PI / YaRN
   ========================================================= */

function SectionLongContext() {
  const [mode, setMode] = useState('vanilla');
  const tTrain = 2048;
  const tEval = 16384;
  const pairI = 1;  // mid-frequency pair so the wrap-around is visible
  const baseFreq = Math.pow(10000, -(2 * pairI) / D_MODEL_DEMO);

  // Compute angle vs position for the chosen mode.
  const scaleFactor = tEval / tTrain;
  const angleAt = (pos) => {
    switch (mode) {
      case 'vanilla': return pos * baseFreq;
      case 'pi':      return (pos / scaleFactor) * baseFreq;
      case 'yarn': {
        // Toy YaRN: high-freq dimensions scale more aggressively than low-freq.
        // Real YaRN uses NTK-aware scaling; we sketch the qualitative shape.
        const blend = 1 / Math.pow(scaleFactor, 0.7);
        return pos * baseFreq * blend;
      }
      default: return pos * baseFreq;
    }
  };

  // Build the curve SVG
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

  const maxAng = Math.max(...ptsScreen.map((p) => p.ang), 1);
  const minAng = 0;
  const toCX = (pos) => xPad + (pos / tEval) * (curveW - 2 * xPad);
  const toCY = (ang) => curveH - yPad - ((ang - minAng) / (maxAng - minAng)) * (curveH - 2 * yPad);
  const pathD = ptsScreen
    .map((p, idx) => `${idx === 0 ? 'M' : 'L'} ${toCX(p.pos)} ${toCY(p.ang)}`)
    .join(' ');

  // Trained-range box
  const trainX = toCX(tTrain);

  return (
    <div className="viz-panel">
      <div className="viz-controls">
        <div className="viz-tabs" role="tablist">
          {[
            { id: 'vanilla', label: 'Vanilla RoPE' },
            { id: 'pi',      label: 'Position Interpolation' },
            { id: 'yarn',    label: 'YaRN (frequency-aware)' },
          ].map((opt) => (
            <button
              key={opt.id}
              role="tab"
              className={`viz-tab ${mode === opt.id ? 'active' : ''}`}
              onClick={() => setMode(opt.id)}
            >
              {opt.label}
            </button>
          ))}
        </div>
      </div>

      <div style={{ marginTop: 12 }}>
        <div style={{ fontFamily: 'var(--mono)', fontSize: '0.78rem', color: 'var(--ink-faint)', marginBottom: 4 }}>
          rotation angle at one mid-frequency pair (rad) vs position
        </div>
        <svg viewBox={`0 0 ${curveW} ${curveH}`} width="100%" style={{ maxWidth: curveW, background: 'hsl(218, 30%, 98%)', border: '1px solid hsl(218, 30%, 90%)', borderRadius: 8 }}>
          {/* trained range box */}
          <rect x={xPad} y={yPad} width={trainX - xPad} height={curveH - 2 * yPad} fill="hsl(148, 35%, 96%)" />
          {/* eval range box */}
          <rect x={trainX} y={yPad} width={curveW - xPad - trainX} height={curveH - 2 * yPad} fill="hsl(8, 55%, 98%)" />
          {/* vertical line at T_train */}
          <line x1={trainX} y1={yPad} x2={trainX} y2={curveH - yPad} stroke="hsl(8, 55%, 50%)" strokeWidth="1.5" strokeDasharray="4 4" />
          <text x={trainX + 4} y={yPad + 12} fontSize="10" fill="hsl(8, 55%, 35%)" fontFamily="var(--mono)">T_train = {tTrain.toLocaleString()}</text>
          {/* zero line */}
          <line x1={xPad} y1={toCY(0)} x2={curveW - xPad} y2={toCY(0)} stroke="hsl(218, 20%, 80%)" />
          {/* curve */}
          <path d={pathD} stroke="hsl(218, 50%, 40%)" strokeWidth="2" fill="none" />
          {/* x-axis labels */}
          <text x={xPad} y={curveH - 4} fontSize="10" fill="var(--ink-faint)" fontFamily="var(--mono)">0</text>
          <text x={curveW - xPad} y={curveH - 4} textAnchor="end" fontSize="10" fill="var(--ink-faint)" fontFamily="var(--mono)">{tEval.toLocaleString()}</text>
          {/* y-axis labels */}
          <text x={xPad - 4} y={toCY(0) + 3} textAnchor="end" fontSize="10" fill="var(--ink-faint)" fontFamily="var(--mono)">0</text>
          <text x={xPad - 4} y={toCY(maxAng) + 3} textAnchor="end" fontSize="10" fill="var(--ink-faint)" fontFamily="var(--mono)">
            {maxAng.toFixed(1)}
          </text>
        </svg>
      </div>

      <p className="viz-caption">
        The green region is the training-time position range. Past{' '}
        <Katex tex="T_\mathrm{train}" /> (red zone), the model has never seen
        these rotation angles.{' '}
        {mode === 'vanilla' && <>Vanilla RoPE keeps growing — the angles at far positions look like nothing in training, and the model breaks.</>}
        {mode === 'pi' && <><a className="viz-link" href="https://arxiv.org/abs/2306.15595" target="_blank" rel="noreferrer">Position Interpolation</a> rescales positions by <Katex tex="T_\mathrm{eval}/T_\mathrm{train}" /> so the angles stay inside the trained range — the cost is slightly squashed resolution everywhere.</>}
        {mode === 'yarn' && <><a className="viz-link" href="https://arxiv.org/abs/2309.00071" target="_blank" rel="noreferrer">YaRN</a> scales high-frequency pairs more aggressively than low-frequency ones, preserving fine-grained position resolution while still keeping out-of-distribution angles tame.</>}
      </p>

      <pre className="viz-code">
{`# Vanilla:    angle = pos * theta_i
# PI:         angle = (pos / S) * theta_i           where S = T_eval / T_train
# YaRN:       angle = pos * theta_i * f(i, S)       where f is frequency-aware`}
      </pre>
    </div>
  );
}

/* =========================================================
   §1 — Permutation invariance
   ========================================================= */

// Reusing the shared toy world from posts #1 and #2.
const EMB = {
  the:  [ 0.2, -0.1,  0.3,  0.0,  0.1, -0.2,  0.4,  0.1],
  cat:  [ 0.5,  0.3, -0.2,  0.4,  0.1,  0.0, -0.1,  0.2],
  sat:  [-0.1,  0.4,  0.5,  0.2, -0.3,  0.1,  0.2, -0.4],
  on:   [ 0.1, -0.2,  0.1, -0.3,  0.4,  0.2, -0.1,  0.0],
  mat:  [ 0.4,  0.2, -0.1,  0.3, -0.2,  0.5,  0.0,  0.1],
};

const W_Q = [
  [ 1.4,  0.0, -0.9,  0.5],
  [ 0.5,  1.8,  0.0, -1.4],
  [-0.9,  0.5,  1.4,  0.0],
  [ 0.0, -1.4,  1.8,  0.9],
  [ 1.8,  0.9, -0.5,  0.0],
  [-0.5,  0.0,  0.9,  1.4],
  [ 0.9,  1.4,  0.0, -0.5],
  [ 0.0,  0.5, -0.9,  1.8],
];
const W_K = [
  [ 0.9,  0.5,  0.0, -1.4],
  [ 0.0,  1.4,  0.9,  0.5],
  [ 1.4, -0.5,  0.0,  0.9],
  [-0.9,  0.0,  1.8,  0.5],
  [ 0.5,  1.8,  0.0, -0.9],
  [ 0.0,  0.9, -1.4,  0.0],
  [ 1.8,  0.0,  0.5,  1.4],
  [-0.5,  0.0,  0.9, -0.5],
];

const PROMPT = ['the', 'cat', 'sat', 'on', 'mat'];

function matmul2(A, B) {
  const m = A.length, k = A[0].length, n = B[0].length;
  const C = Array.from({ length: m }, () => new Array(n).fill(0));
  for (let i = 0; i < m; i++)
    for (let j = 0; j < n; j++) {
      let s = 0;
      for (let p = 0; p < k; p++) s += A[i][p] * B[p][j];
      C[i][j] = s;
    }
  return C;
}

const transpose = (A) => A[0].map((_, j) => A.map((r) => r[j]));

function softmaxRows(M) {
  return M.map((row) => {
    const m = Math.max(...row);
    const exps = row.map((v) => Math.exp(v - m));
    const s = exps.reduce((a, b) => a + b, 0);
    return exps.map((e) => e / s);
  });
}

function SectionPermutation() {
  const [order, setOrder] = useState([0, 1, 2, 3, 4]);
  const tokens = order.map((i) => PROMPT[i]);

  // Compute attention for the current order.
  const X = tokens.map((t) => EMB[t]);
  const Q = matmul2(X, W_Q);
  const K = matmul2(X, W_K);
  const scaled = matmul2(Q, transpose(K)).map((row) => row.map((v) => v / 2));
  const A = softmaxRows(scaled);

  const shuffle = () => {
    const next = [...order];
    for (let i = next.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [next[i], next[j]] = [next[j], next[i]];
    }
    setOrder(next);
  };
  const reset = () => setOrder([0, 1, 2, 3, 4]);

  const cellSize = 56;

  return (
    <div className="viz-panel">
      <div className="viz-controls">
        <button onClick={shuffle} className="active">Shuffle tokens</button>
        <button onClick={reset}>Reset</button>
      </div>

      <div className="viz-token-row" style={{ marginBottom: 14 }}>
        {tokens.map((t, i) => (
          <div className="viz-token-pill" key={`${t}-${i}`}>
            <div className="viz-token-text">{t}</div>
            <div className="viz-token-pos">pos {i}</div>
          </div>
        ))}
      </div>

      <div style={{ overflowX: 'auto' }}>
        <div style={{
          display: 'grid',
          gridTemplateColumns: `60px repeat(${tokens.length}, ${cellSize}px)`,
          gap: 2,
          background: 'var(--rule)',
          padding: 2,
          borderRadius: 4,
          width: 'fit-content',
        }}>
          <div />
          {tokens.map((t, j) => (
            <div key={`col-${j}`} style={{ background: 'transparent', color: 'var(--ink-faint)', fontSize: '0.72rem', fontStyle: 'italic', textAlign: 'center', padding: '4px 4px' }}>{t}</div>
          ))}
          {tokens.map((tRow, i) => (
            <React.Fragment key={`row-${i}`}>
              <div style={{ background: 'transparent', color: 'var(--ink-faint)', fontSize: '0.72rem', fontStyle: 'italic', textAlign: 'right', padding: '4px 6px', alignSelf: 'center' }}>{tRow}</div>
              {A[i].map((v, j) => {
                const sat = 18 + Math.round(v * 60);
                const light = 96 - Math.round(v * 50);
                return (
                  <div
                    key={`a-${i}-${j}`}
                    style={{
                      background: `hsl(218, ${sat}%, ${light}%)`,
                      color: v > 0.55 ? '#fff' : 'var(--ink)',
                      fontFamily: 'var(--mono)',
                      fontSize: '0.72rem',
                      padding: '4px 4px',
                      textAlign: 'right',
                      width: cellSize,
                    }}
                  >
                    {v.toFixed(2)}
                  </div>
                );
              })}
            </React.Fragment>
          ))}
        </div>
      </div>

      <p className="viz-caption" style={{ marginTop: 12 }}>
        Click <strong>Shuffle</strong>: the token pills above reorder, and the
        attention matrix reorders <em>with them</em> — same numbers, just at
        different row/column labels. Self-attention has no built-in notion of
        which token came first, second, or last. Position has to be injected
        from outside.
      </p>
    </div>
  );
}

/* =========================================================
   §2 — Position encoding zoo
   ========================================================= */

// Hand-picked vectors per relative offset for Shaw Key (a^K) and Shaw Value (a^V).
// d_k = d_v = 4 to match the Q/K dimension of our toy model.
const A_K = {
  '-3': [-0.20,  0.10,  0.00, -0.10],
  '-2': [-0.10, -0.05,  0.15, -0.10],
  '-1': [ 0.05, -0.10,  0.00,  0.15],
  '0':  [ 0.25,  0.00,  0.00,  0.20],
  '1':  [-0.05,  0.10, -0.10,  0.00],
  '2':  [-0.10,  0.05,  0.15, -0.10],
  '3':  [-0.20,  0.10,  0.00, -0.10],
};
const A_V = {
  '-3': [ 0.10, -0.20,  0.05,  0.00],
  '-2': [ 0.15, -0.10,  0.00,  0.05],
  '-1': [ 0.10, -0.05, -0.10,  0.00],
  '0':  [ 0.00,  0.20,  0.20,  0.00],
  '1':  [-0.05, -0.10,  0.10,  0.00],
  '2':  [-0.10,  0.00, -0.05,  0.15],
  '3':  [-0.20,  0.05,  0.10,  0.00],
};
function aK(off) { return A_K[String(off)] ?? [0, 0, 0, 0]; }
function aV(off) { return A_V[String(off)] ?? [0, 0, 0, 0]; }

// T5 scalar bias per relative offset.
const T5_BIAS = { '-3': -0.4, '-2': -0.2, '-1': 0.1, '0': 0.5, '1': 0.1, '2': -0.2, '3': -0.4 };

// Build the effective bias matrix added to scores under each mode (or null when
// the mode doesn't add a score-level bias — that's Shaw Value).
function effectiveBias(mode, Q) {
  const T = Q.length;
  const M = Array.from({ length: T }, () => new Array(T).fill(0));
  if (mode === 'shaw-key') {
    // B[i][j] = q_i · a^K_{i-j}   (the bias *is* content-aware via the dot with q)
    for (let i = 0; i < T; i++) {
      for (let j = 0; j < T; j++) {
        const a = aK(i - j);
        let s = 0;
        for (let k = 0; k < Q[i].length; k++) s += Q[i][k] * a[k];
        M[i][j] = s;
      }
    }
    return M;
  }
  if (mode === 'shaw-value') {
    // No bias on scores under Shaw Value.
    return null;
  }
  if (mode === 't5') {
    for (let i = 0; i < T; i++)
      for (let j = 0; j < T; j++) M[i][j] = T5_BIAS[String(i - j)] ?? 0;
    return M;
  }
  if (mode === 'swin') {
    const lookup = {
      '0,0': 0.6,
      '0,1': 0.0, '0,-1': 0.0,
      '1,0': -0.1, '-1,0': -0.1,
      '1,1': -0.3, '-1,-1': -0.3,
      '1,-1': -0.4, '-1,1': -0.4,
    };
    for (let i = 0; i < T; i++)
      for (let j = 0; j < T; j++) {
        const di = Math.floor(i / 2) - Math.floor(j / 2);
        const dj = (i % 2) - (j % 2);
        M[i][j] = lookup[`${di},${dj}`] ?? 0;
      }
    return M;
  }
  return M;
}

// Small heatmap of an arbitrary numeric matrix (diverging blue/red for signed, single-hue for [0,1]).
function MiniMatrix({ data, title, cellSize = 50, diverging = true, range = 0.6, vmax }) {
  const T = data.length;
  const cols = data[0].length;
  return (
    <div>
      {title && (
        <div style={{ fontFamily: 'var(--mono)', fontSize: '0.78rem', color: 'var(--ink-faint)', marginBottom: 6 }}>
          {title}
        </div>
      )}
      <div style={{
        display: 'grid',
        gridTemplateColumns: `repeat(${cols}, ${cellSize}px)`,
        gap: 2,
        background: 'var(--rule)',
        padding: 2,
        borderRadius: 4,
      }}>
        {data.flatMap((row, i) =>
          row.map((v, j) => {
            let bg;
            if (diverging) {
              const a = Math.min(1, Math.abs(v) / range);
              bg = v >= 0
                ? `hsl(218, ${20 + a * 60}%, ${96 - a * 40}%)`
                : `hsl(8, ${20 + a * 60}%, ${96 - a * 40}%)`;
            } else {
              const norm = Math.min(1, Math.max(0, v / (vmax || 1)));
              const sat = 18 + Math.round(norm * 60);
              const light = 96 - Math.round(norm * 50);
              bg = `hsl(218, ${sat}%, ${light}%)`;
            }
            const ink = !diverging && v / (vmax || 1) > 0.55 ? '#fff' : 'var(--ink)';
            return (
              <div key={`mm-${i}-${j}`} style={{
                background: bg,
                color: ink,
                fontFamily: 'var(--mono)',
                fontSize: '0.72rem',
                padding: '4px 4px',
                textAlign: 'right',
                width: cellSize,
              }}>
                {v.toFixed(2)}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}

function SectionPosEncodingZoo() {
  const [mode, setMode] = useState('shaw-key');
  const T = 4;
  const tokens = PROMPT.slice(0, T);
  const X = tokens.map((t) => EMB[t]);
  const Q = matmul2(X, W_Q);
  const K = matmul2(X, W_K);
  const raw = matmul2(Q, transpose(K)).map((row) => row.map((v) => v / 2));

  const B = effectiveBias(mode, Q);
  const scores = B
    ? raw.map((row, i) => row.map((v, j) => v + B[i][j]))
    : raw;
  const A = softmaxRows(scores);

  // For Shaw Value: build the per-row "added value contribution" matrix
  //   ΔV[i] = Σ_j α_ij · a^V_{i-j}    ∈ R^{d_v}
  // Shows what gets *added* to each output row by the value-side relative term.
  const dV = 4;
  const deltaV = mode === 'shaw-value'
    ? A.map((aRow, i) => {
        const out = new Array(dV).fill(0);
        for (let j = 0; j < T; j++) {
          const a = aV(i - j);
          for (let k = 0; k < dV; k++) out[k] += aRow[j] * a[k];
        }
        return out;
      })
    : null;

  const formulas = {
    'shaw-key':   'e_{ij} = q_i \\cdot (k_j + a^K_{i-j})^\\top / \\sqrt{d}     \\;\\;\\Longleftrightarrow\\;\\; q_i \\!\\cdot\\! k_j + q_i \\!\\cdot\\! a^K_{i-j}',
    'shaw-value': 'z_i = \\sum_{j} \\alpha_{ij}\\,(v_j + a^V_{i-j})    \\quad\\text{(scores unchanged; position added to values)}',
    't5':         'e_{ij} = q_i \\cdot k_j / \\sqrt{d} + b_{i-j} \\quad\\text{(learned scalar per offset)}',
    'swin':       'e_{ij} = q_i \\cdot k_j / \\sqrt{d} + b_{\\Delta i,\\,\\Delta j} \\quad\\text{(2D index for image patches)}',
  };

  const captions = {
    'shaw-key':
      'Shaw 2018 eq (4) — the relative-position info is a *vector* a^K_{i-j} added to k_j *before* the dot product. The contribution to the score is q_i · a^K_{i-j}, which depends on the query, so different queries weight the same relative offset differently. Notice in the bias matrix below: each *row* looks different even though the index pattern is Toeplitz — that\'s the content dependence.',
    'shaw-value':
      'Shaw 2018 eq (3) — the relative-position info is a vector a^V_{i-j} added to the *value* during the weighted sum. Attention scores (and the attention pattern) are completely unchanged: it\'s pure positional information flowing into the output. The right panel shows ΔV_i, the per-row positional vector added to each output row.',
    't5':
      'T5 2019 (and Swin in 2D) simplified Shaw\'s vector bias to a *learned scalar* per relative offset. Same Toeplitz indexing as Shaw, but the bias is content-independent — the same value gets added regardless of which q, k are involved. Cheap, effective, and the form most people think of as "relative position bias".',
    'swin':
      'Swin 2021 ports T5\'s scalar bias to 2D image patches: index by (Δi, Δj) within an M × M window. Same upside as T5 (relative, simple), same downside (bounded by the trained window — anything past the M × M window falls off the table).',
  };

  return (
    <div className="viz-panel">
      <div className="viz-controls">
        <div className="viz-tabs" role="tablist">
          {[
            { id: 'shaw-key',   label: 'Shaw Key' },
            { id: 'shaw-value', label: 'Shaw Value' },
            { id: 't5',         label: 'T5 Scalar' },
            { id: 'swin',       label: 'Swin 2D' },
          ].map((opt) => (
            <button
              key={opt.id}
              role="tab"
              className={`viz-tab ${mode === opt.id ? 'active' : ''}`}
              onClick={() => setMode(opt.id)}
            >
              {opt.label}
            </button>
          ))}
        </div>
      </div>

      <div className="viz-math-block">
        <Katex block tex={formulas[mode]} />
      </div>

      <p style={{ fontSize: '0.94rem', color: 'var(--ink-soft)', margin: '8px 0 14px' }}>
        {captions[mode]}
      </p>

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 24 }}>
        {/* Left panel: bias matrix (or value-side contribution for Shaw Value) */}
        {mode === 'shaw-value' ? (
          <MiniMatrix
            data={deltaV}
            title="ΔV_i = Σ_j α_ij · a^V_{i-j}  (added to each output row)"
            diverging
            range={0.15}
          />
        ) : B ? (
          <MiniMatrix
            data={B}
            title={
              mode === 'shaw-key'
                ? 'effective bias  q_i · a^K_{i-j}  (content-aware)'
                : 'bias matrix B  (constant per relative offset)'
            }
            diverging
            range={mode === 'shaw-key' ? 0.4 : 0.6}
          />
        ) : null}

        {/* Right panel: resulting attention pattern */}
        <MiniMatrix
          data={A}
          title={
            mode === 'shaw-value'
              ? 'attention pattern A  (unchanged — no bias on scores)'
              : 'resulting attention pattern A  (rows sum to 1)'
          }
          diverging={false}
          vmax={1}
        />
      </div>

      <p className="viz-caption" style={{ marginTop: 14 }}>
        Three of these are <strong>additive on scores</strong> (Shaw Key, T5,
        Swin) — they reshape the attention pattern itself. One is{' '}
        <strong>additive on values</strong> (Shaw Value) — the attention
        pattern is untouched; position flows into the output through a
        different door. None of them rotate Q or K. RoPE will be different on
        every axis: <em>multiplicative</em>, <em>on Q and K directly</em>,
        and <em>zero learned parameters</em>. That's §3.
      </p>
    </div>
  );
}

/* =========================================================
   The post
   ========================================================= */

export default function VisualizingRoPE() {
  useEffect(() => {
    const prevTitle = document.title;
    document.title = 'Visualizing RoPE: Rotary Positional Embeddings, Geometrically — Bernhard Walser';

    const setMeta = (name, content, attr = 'name') => {
      let el = document.querySelector(`meta[${attr}="${name}"]`);
      if (!el) {
        el = document.createElement('meta');
        el.setAttribute(attr, name);
        document.head.appendChild(el);
      }
      el.setAttribute('content', content);
    };

    const description =
      'A geometric walk through Rotary Positional Embeddings (RoPE): 2D rotation, pairwise dot-product invariance, scaling to d-dimensional Q and K, attention with RoPE, and the long-context extensions (Position Interpolation, YaRN).';

    setMeta('description', description);
    setMeta('og:title', 'Visualizing RoPE', 'property');
    setMeta('og:description', description, 'property');
    setMeta('og:type', 'article', 'property');
    setMeta('twitter:card', 'summary_large_image');
    setMeta('twitter:title', 'Visualizing RoPE');
    setMeta('twitter:description', description);

    return () => { document.title = prevTitle; };
  }, []);

  return (
    <article className="viz-post">
      <div className="viz-wide">
        <div className="viz-series-tag">Visualizing ML · Part 3</div>
        <h1>Visualizing RoPE: Rotary Positional Embeddings, Geometrically</h1>
        <p className="viz-lede">
          <em>"The cat ate the fish"</em> and <em>"the fish ate the cat"</em>{' '}
          mean different things, so the model has to know which word came when.
          Every modern open LLM — Llama, Mistral, Qwen, DeepSeek — solves this
          with Rotary Positional Embeddings (RoPE), a geometric trick: rotate
          each query and key in 2D pairs by an angle proportional to position.
          The relative-position information then falls out of the dot product
          for free.
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
          Post #3 of <em>Visualizing ML</em>. If you haven't read{' '}
          <a className="viz-link" href="#/blog/visualizing-attention">post #1</a>{' '}
          (attention) or{' '}
          <a className="viz-link" href="#/blog/visualizing-kv-cache">post #2</a>{' '}
          (the KV cache), the prose stands alone — but the toy world (small
          dimensions, hand-rounded numbers) is the same one we've been using.
        </p>

        <h2>1. The positional encoding problem</h2>
        <p>
          Self-attention is permutation-invariant. If you shuffle the tokens
          of a sequence, the model produces the <em>same</em> outputs in
          shuffled order — it has no built-in notion of which token came
          first or last. <em>"The cat sat on the mat"</em> and{' '}
          <em>"on the mat the cat sat"</em> would, without intervention,
          score identical to a vanilla attention block.
        </p>
        <p>
          Click <strong>Shuffle</strong> below and watch: the attention
          matrix keeps the same numbers, only the row and column labels
          move. Position information has to come from somewhere outside the
          attention math itself.
        </p>
        <SectionPermutation />
        <p>
          The early transformer fixed this by <em>adding</em> a positional
          vector to each token's embedding — either learned absolute
          positions or fixed sinusoidal ones. Later work (Shaw 2018, Swin
          2021) moved to relative position biases <em>added</em> to the
          attention score itself. All of them shared one property: the
          position information was bolted onto the existing structure as an
          extra term. RoPE is a different idea — bake position into Q and K
          geometrically, so it falls out of the dot product for free. The
          next section makes the contrast concrete.
        </p>

        <h2>2. The predecessor family: relative position bias</h2>
        <p>
          Between absolute embeddings and RoPE there's a small family of
          schemes that encode position as an <strong>additive term</strong>{' '}
          somewhere inside attention. Four variants are worth seeing
          side-by-side because each makes a slightly different design
          choice:{' '}
          <a className="viz-link" href="https://arxiv.org/abs/1803.02155" target="_blank" rel="noreferrer">
            Shaw et al. 2018
          </a>{' '}
          actually proposed two forms (one adds a vector to keys, one adds
          a vector to values);{' '}
          <a className="viz-link" href="https://arxiv.org/abs/1910.10683" target="_blank" rel="noreferrer">
            T5
          </a>{' '}
          (Raffel et al. 2019) simplified Shaw's key-vector to a learned
          scalar per offset; and{' '}
          <a className="viz-link" href="https://arxiv.org/abs/2103.14030" target="_blank" rel="noreferrer">
            Swin
          </a>{' '}
          (Liu et al. 2021) lifted the T5 scalar to 2D image patches.
          Toggle between them below — note the structural differences in the
          bias panel as much as the result.
        </p>
        <SectionPosEncodingZoo />
        <p>
          Three of these add a bias to the attention <em>scores</em>; one
          adds it to the <em>values</em>. None of them touch Q or K directly.
          And all of them depend on a table indexed by relative offset that
          is bounded by the trained range. RoPE breaks both habits at once:
          it modifies Q and K themselves (<em>multiplicatively</em>, via a
          rotation), and the relative-position dependence falls out of the
          dot product without any extra learned parameters. The next four
          sections build that idea up from a single 2D rotation.
        </p>

        <h2>3. The geometric core: rotation in a 2D plane</h2>
        <p>
          The next interactives all build on one primitive: rotating a 2D
          vector. Pick any vector <Katex tex="v = (x, y)" /> and any angle{' '}
          <Katex tex="\theta" />, and watch{' '}
          <Katex tex="R(\theta)" /> send it to its rotated copy{' '}
          <Katex tex="R(\theta) \cdot v" />. Hover any entry of the matrix
          to see the dot product that produced it.
        </p>
        <SectionRotation />

        <h2>4. Pairwise dot-product invariance</h2>
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

        <h2>5. Scaling to d-dimensional Q and K</h2>
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

        <h2>6. Attention with RoPE — what the model sees</h2>
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
            the same argument as §4, scaled up to many pairs. Slide{' '}
          <em>m</em> and <em>n</em> below with the lock on: both positions
          slide but the dot product stays put. Untick the lock and change the
          gap: the dot product slides along the curve.
        </p>
        <SectionRoPEDot />

        <h2>7. Long-context extension: PI and YaRN</h2>
        <p>
          One subtlety: the model is trained at some maximum context length{' '}
          <Katex tex="T_\mathrm{train}" /> (say 2048). Past that, RoPE's
          rotation angles enter unprecedented territory — the high-frequency
          pairs have wound around the unit circle thousands of times, but at
          angles the model has never been asked to interpret. So when you try
          to evaluate at <Katex tex="T_\mathrm{eval} = 16k" />, vanilla RoPE
          breaks.
        </p>
        <p>
          Two techniques fix this without retraining.{' '}
          <a className="viz-link" href="https://arxiv.org/abs/2306.15595" target="_blank" rel="noreferrer">
            Position Interpolation
          </a>{' '}
          (Chen et al. 2023) simply rescales position indices so the angles
          stay inside the trained range.{' '}
          <a className="viz-link" href="https://arxiv.org/abs/2309.00071" target="_blank" rel="noreferrer">
            YaRN
          </a>{' '}
          (Peng et al. 2023) does the same idea but{' '}
          <em>frequency-aware</em>: it scales high-frequency pairs more
          aggressively than low-frequency ones, preserving fine resolution
          where it matters. Toggle between the three modes below.
        </p>
        <SectionLongContext />

        <h2>8. What this isn't (yet)</h2>
        <p>
          <strong>ALiBi (Attention with Linear Biases).</strong>{' '}
          <a className="viz-link" href="https://arxiv.org/abs/2108.12409" target="_blank" rel="noreferrer">
            Press et al. 2021
          </a>{' '}
          uses an even simpler additive bias: subtract a linear function of
          the distance <Katex tex="|i - j|" /> from the score, with no
          learned parameters at all. Trades some short-context quality for
          parameter-free extrapolation. BLOOM and MPT used it.
        </p>
        <p>
          <strong>NoPE (no positional encoding).</strong>{' '}
          <a className="viz-link" href="https://arxiv.org/abs/2305.19466" target="_blank" rel="noreferrer">
            Kazemnejad et al. 2023
          </a>{' '}
          showed that decoder-only transformers with causal masking can{' '}
          <em>learn</em> position information from the mask alone, sometimes
          generalising better than RoPE on synthetic length tasks. Not
          mainstream in production yet, but a clean "what if you didn't
          encode position at all" datapoint.
        </p>
        <p>
          <strong>Long context, more broadly.</strong> PI and YaRN are the
          first wave of post-training context extension. The frontier (NTK
          scaling, dynamic NTK, LongRoPE, RingAttention, sparse-attention
          variants) is its own ecosystem.{' '}
          <em>Future post in this series.</em>
        </p>

        <h2>9. Back to the question</h2>
        <p>
          Why does Llama, Mistral, Qwen, DeepSeek all use RoPE?
        </p>
        <p>
          Because position information turns out to be a geometric quantity,
          not a learned one. The earliest schemes (absolute embeddings,
          Shaw, Swin) bolted it on as a separate vector or a separate bias
          table — extra parameters, bounded by training length, awkward to
          extrapolate. RoPE replaces all of that with a single rotation per
          dimension pair: position <Katex tex="m" /> becomes an angle, and
          the dot product between <Katex tex="R_m q" /> and{' '}
          <Katex tex="R_n k" /> naturally encodes the relative offset{' '}
          <Katex tex="n - m" />. Zero new parameters, no table to look up,
          and (with PI / YaRN) it stretches well past the training horizon.
        </p>
        <p>
          That's the entire trick. Everything else — multi-frequency pairs
          to give the model multi-resolution position information, scaling
          tricks for long context — is bookkeeping on top of one idea: the
          2D rotation in §3 is the right unit for talking about position.
        </p>

        <h2>What comes next</h2>
        <p>
          <strong>The RLHF stack.</strong> Post #4 is about how a base LM
          becomes a chat model — supervised fine-tuning, reward models, PPO
          and DPO, the whole loop. Mostly geometric in its own way: each
          stage moves the model's distribution around in policy space.{' '}
          <em>Coming next.</em>
        </p>

        <footer className="viz-footer">
          <p>
            <strong>Part 3 of Visualizing ML</strong> · Previous:{' '}
            <a className="viz-link" href="#/blog/visualizing-kv-cache">Visualizing the KV Cache</a>
            . Next: <em>The RLHF Stack</em>.
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
            {' '}(Anthropic) — initial drafts and interactive scaffolding by Claude,
            refined and co-designed by Bernhard.
          </p>
        </footer>
      </div>
    </article>
  );
}
