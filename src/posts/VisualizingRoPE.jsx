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

// Tooltip body for a dot product, showing each per-component term then the sum.
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
  // The viewBox is always in math-space-pixel coords (CANVAS_PX); the SVG is
  // rendered at `size` px and the browser scales the content to fit. This lets
  // §5's smaller cards reuse the exact same toSvgX/toSvgY helpers as §3 / §4.
  return (
    <div className="viz-rope-canvas" style={{ width: size, height: size }}>
      <svg viewBox={`0 0 ${CANVAS_PX} ${CANVAS_PX}`} width={size} height={size}>
        <RotationAxes />
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

// One colour per pair, used both in the vector strips and the pair cards so
// the connection between "two cells of Q" and "one 2D rotation card" is visual.
const PAIR_PALETTE = [
  { bg: 'hsl(218, 50%, 95%)', border: 'hsl(218, 50%, 65%)', fg: 'hsl(218, 50%, 28%)', accent: 'hsl(218, 50%, 42%)' },
  { bg: 'hsl(28,  65%, 95%)', border: 'hsl(28,  65%, 65%)', fg: 'hsl(28,  65%, 28%)', accent: 'hsl(28,  65%, 45%)' },
  { bg: 'hsl(148, 38%, 95%)', border: 'hsl(148, 38%, 60%)', fg: 'hsl(148, 38%, 22%)', accent: 'hsl(148, 38%, 36%)' },
  { bg: 'hsl(280, 40%, 95%)', border: 'hsl(280, 40%, 65%)', fg: 'hsl(280, 40%, 30%)', accent: 'hsl(280, 40%, 45%)' },
];

function VectorStrip({ values, label, palette = PAIR_PALETTE, prefix = 'Q' }) {
  const d = values.length;
  // Pair-coloring assumes consecutive pairs of dims share the same colour.
  return (
    <div style={{ margin: '4px 0 8px' }}>
      <div style={{ fontFamily: 'var(--mono)', fontSize: '0.78rem', color: 'var(--ink-faint)', marginBottom: 6 }}>
        {label}
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: `repeat(${d}, minmax(0, 1fr))`, gap: 4, maxWidth: 70 * d }}>
        {values.map((v, idx) => {
          const pairIdx = Math.floor(idx / 2);
          const p = palette[pairIdx % palette.length];
          return (
            <div key={idx} style={{
              background: p.bg,
              border: `1px solid ${p.border}`,
              padding: '6px 4px',
              borderRadius: 4,
              textAlign: 'center',
              fontFamily: 'var(--mono)',
              color: p.fg,
            }}>
              <div style={{ fontSize: '0.6rem', opacity: 0.75 }}>{prefix}<sub>{idx}</sub></div>
              <div style={{ fontWeight: 600, fontVariantNumeric: 'tabular-nums', fontSize: '0.82rem' }}>
                {v.toFixed(2)}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

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

  // Flatten PAIR_VECTORS into one d=8 vector — this is the "real" Q we'll
  // disassemble, rotate, and reassemble.
  const Q_full     = PAIR_VECTORS.flatMap(([x, y]) => [x, y]);
  const Q_rotated  = (() => {
    const out = new Array(D_MODEL_DEMO);
    for (let i = 0; i < D_MODEL_DEMO / 2; i++) {
      const theta = m * freqs[i];
      const [x, y] = PAIR_VECTORS[i];
      const [rx, ry] = rotate2D(x, y, theta);
      out[2 * i]     = rx;
      out[2 * i + 1] = ry;
    }
    return out;
  })();

  return (
    <div className="viz-panel">
      <div className="viz-controls">
        <label className="viz-rope-slider" style={{ flex: 1, minWidth: 280 }}>
          <span>position m</span>
          <input type="range" min={0} max={256} step={1} value={m} onChange={(e) => setM(+e.target.value)} />
          <strong>{m}</strong>
        </label>
      </div>

      <VectorStrip
        values={Q_full}
        label="Q  (one query vector, d = 8) — before RoPE"
      />

      <div style={{
        textAlign: 'center',
        fontFamily: 'var(--mono)',
        fontSize: '0.78rem',
        color: 'var(--ink-faint)',
        margin: '6px 0',
      }}>
        ↓ split into <Katex tex="d/2 = 4" /> pairs · rotate pair i by <Katex tex="m \cdot \theta_i" />
      </div>

      <div className="viz-rope-pair-grid">
        {PAIR_VECTORS.map(([vx, vy], i) => {
          const theta = m * freqs[i];
          const [rx, ry] = rotate2D(vx, vy, theta);
          const p = PAIR_PALETTE[i];
          return (
            <div key={i} className="viz-rope-pair-card" style={{ background: p.bg, borderColor: p.border }}>
              <div className="viz-rope-pair-title" style={{ color: p.fg }}>
                pair {i} &nbsp;·&nbsp; (Q<sub>{2 * i}</sub>, Q<sub>{2 * i + 1}</sub>)
              </div>
              <RotationCanvas size={160}>
                <ArrowVector x={vx} y={vy} color="var(--ink-faint)" opacity={0.45} />
                <ArrowVector x={rx} y={ry} color={p.accent} />
              </RotationCanvas>
              <div className="viz-rope-pair-freq" style={{ color: p.fg }}>
                θ<sub>{i}</sub> = {freqs[i] < 0.01 ? freqs[i].toExponential(2) : freqs[i].toFixed(3)}<br />
                m · θ<sub>{i}</sub> = {theta.toFixed(3)} rad
              </div>
            </div>
          );
        })}
      </div>

      <div style={{
        textAlign: 'center',
        fontFamily: 'var(--mono)',
        fontSize: '0.78rem',
        color: 'var(--ink-faint)',
        margin: '8px 0 6px',
      }}>
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

      <pre className="viz-code">
{`# RoPE frequencies for d_model = d, then applied to a single token's Q
import torch
i = torch.arange(0, d, 2)              # 0, 2, 4, ..., d-2
freqs = 10000.0 ** (-i / d)            # θ_0 = 1, θ_1 = 1/10, ..., θ_(d/2-1) ≈ 1/10000

def apply_rope_to_vec(x, pos):         # x: shape (d,)
    out = x.clone()
    for i, t in enumerate(freqs):
        c, s = math.cos(pos * t), math.sin(pos * t)
        a, b = x[2*i], x[2*i + 1]
        out[2*i]     = c * a - s * b
        out[2*i + 1] = s * a + c * b
    return out                          # same shape, just rotated in pairs`}
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

  // Reference scenario: Q at position 0 (no rotation), K at position (n − m).
  // By the §4 invariance, this should give the same dot product as Q at m, K at n.
  const refDiff = n - m;
  const kRotRef = applyRoPE(K_DEMO, refDiff);
  const refDot  = dotProduct(Q_DEMO, kRotRef);

  // The §4 / §6 invariance says the RoPE dot product depends only on the
  // relative offset d = n − m. So we can compute the whole sweep with Q at
  // position 0 and K at position d — independent of m. The curve is static
  // (same shape no matter what m is); only the marker dot below moves.
  const SWEEP_RANGE = 32;
  const sweepData = useMemo(() => {
    const pts = [];
    for (let d = -SWEEP_RANGE; d <= SWEEP_RANGE; d++) {
      const kr = applyRoPE(K_DEMO, d);
      pts.push({ d, v: dotProduct(Q_DEMO, kr) });
    }
    return pts;
  }, []);

  // Curve SVG — fixed axes, never re-scale with m.
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

      <div className="viz-subsection" style={{ marginTop: 16 }}>
        <h4 className="viz-subhead">
          <span className="viz-sub-tag viz-sub-tag-info">i</span>
          Invariance check: same dot product, different absolute positions
        </h4>
        <p className="viz-subhead-prose">
          Both setups have the same relative offset{' '}
          <Katex tex={`n - m = ${refDiff}`} />, so the §4 identity says
          they must produce the same RoPE dot product.
        </p>

        <div className="viz-rope-inv-row">
          <div className="viz-rope-inv-header">
            <span className="viz-rope-inv-tag">REF</span>
            Q at position <strong>0</strong>, K at position <strong>{refDiff}</strong>
            <span className="viz-rope-inv-arrow">→ Q stays put, only K is rotated</span>
          </div>
          <VectorStrip values={Q_DEMO} label="Q  (position 0, no rotation)" prefix="Q" />
          <VectorStrip values={kRotRef} label={`K  (rotated by (n − m) · θ at each pair)`} prefix="K" />
          <div className="viz-rope-inv-dot">
            Q · K&nbsp;=&nbsp;
            <Tip label={dotBreakdownTip(Q_DEMO, kRotRef)}>
              <strong>{refDot.toFixed(3)}</strong>
            </Tip>
          </div>
        </div>

        <div className="viz-rope-inv-row" style={{ marginTop: 14 }}>
          <div className="viz-rope-inv-header">
            <span className="viz-rope-inv-tag viz-rope-inv-tag-cur">CUR</span>
            Q at position <strong>{m}</strong>, K at position <strong>{n}</strong>
            <span className="viz-rope-inv-arrow">→ both rotated, but by different amounts</span>
          </div>
          <VectorStrip values={qRot}    label={`Q  (rotated by m · θ at each pair)`} prefix="Q" />
          <VectorStrip values={kRot}    label={`K  (rotated by n · θ at each pair)`} prefix="K" />
          <div className="viz-rope-inv-dot">
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
      <a className="viz-link" href="https://arxiv.org/abs/2306.15595" target="_blank" rel="noreferrer">Position Interpolation</a>{' '}
      (Chen et al., 2023) divides <em>every</em> position by{' '}
      <Katex tex="s = T_{\mathrm{eval}}/T_{\mathrm{train}}" /> before feeding it
      to RoPE. Equivalently, every pair's effective frequency is shrunk by{' '}
      <Katex tex="1/s" />, so at <Katex tex="m = T_{\mathrm{eval}}" /> the
      slowest pair has only swept the same angle range it saw at{' '}
      <Katex tex="T_{\mathrm{train}}" />. Cheap (one constant, ~1k fine-tuning
      steps) but blunt — the fast pairs get squashed even though they didn't
      need fixing.
    </>
  ),
  yarn: (
    <>
      <a className="viz-link" href="https://arxiv.org/abs/2309.00071" target="_blank" rel="noreferrer">YaRN</a>{' '}
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
  // Lowest-frequency pair (i = d/2 − 1) — this is the one whose angle
  // genuinely drifts out of the trained range at long contexts. High-freq
  // pairs wrap around so many times they've already seen every angle during
  // training; there's no OOD to fix there.
  const pairI = D_MODEL_DEMO / 2 - 1;
  const baseFreq = Math.pow(10000, -(2 * pairI) / D_MODEL_DEMO);

  // Compute angle vs position for the chosen mode.
  const scaleFactor = tEval / tTrain;
  const angleAt = (pos) => {
    switch (mode) {
      case 'vanilla': return pos * baseFreq;
      case 'pi':      return (pos / scaleFactor) * baseFreq;
      case 'yarn': {
        // For a low-frequency pair (what we visualise) YaRN's NTK-aware
        // schedule reduces to PI — slow dimensions get the full compression.
        // High-freq dimensions (not shown) would be left untouched.
        return (pos / scaleFactor) * baseFreq;
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

  // Static y-axis: always span the vanilla curve's full range, so PI and
  // YaRN visibly *flatten* relative to the green band. The auto-scaling
  // version made all three modes look identical.
  const vanillaMax = tEval * baseFreq;
  const trainedMax = tTrain * baseFreq;
  const yMax = vanillaMax * 1.05;

  const toCX = (pos) => xPad + (pos / tEval) * (curveW - 2 * xPad);
  const toCY = (ang) => curveH - yPad - (ang / yMax) * (curveH - 2 * yPad);
  const pathD = ptsScreen
    .map((p, idx) => `${idx === 0 ? 'M' : 'L'} ${toCX(p.pos)} ${toCY(p.ang)}`)
    .join(' ');

  // Trained position threshold (vertical guide line)
  const trainX = toCX(tTrain);
  // Trained angle threshold (horizontal boundary between safe / OOD)
  const trainAngleY = toCY(trainedMax);

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

      <div className="viz-math-block" style={{ marginTop: 10 }}>
        <Katex block tex={LONG_CTX_FORMULAS[mode]} />
      </div>

      <p style={{ fontSize: '0.94rem', color: 'var(--ink-soft)', margin: '8px 0 14px' }}>
        {LONG_CTX_CAPTIONS[mode]}
      </p>

      <div style={{ marginTop: 12 }}>
        <div style={{ fontFamily: 'var(--mono)', fontSize: '0.78rem', color: 'var(--ink-faint)', marginBottom: 4 }}>
          rotation angle at one mid-frequency pair (rad) vs position
        </div>
        <svg viewBox={`0 0 ${curveW} ${curveH}`} width="100%" style={{ maxWidth: curveW, background: 'hsl(218, 30%, 98%)', border: '1px solid hsl(218, 30%, 90%)', borderRadius: 8 }}>
          {/* OOD angle band — top */}
          <rect x={xPad} y={yPad} width={curveW - 2 * xPad} height={trainAngleY - yPad} fill="hsl(8, 60%, 92%)" />
          {/* Trained angle band — bottom */}
          <rect x={xPad} y={trainAngleY} width={curveW - 2 * xPad} height={(curveH - yPad) - trainAngleY} fill="hsl(148, 45%, 88%)" />
          {/* horizontal line at trained-angle threshold */}
          <line x1={xPad} y1={trainAngleY} x2={curveW - xPad} y2={trainAngleY} stroke="hsl(8, 55%, 50%)" strokeWidth="1.5" strokeDasharray="4 4" />
          <text x={curveW - xPad - 4} y={trainAngleY - 4} textAnchor="end" fontSize="10" fill="hsl(8, 55%, 35%)" fontFamily="var(--mono)">
            trained-angle max = {trainedMax.toFixed(2)} rad
          </text>
          {/* vertical guide at T_train (position annotation) */}
          <line x1={trainX} y1={yPad} x2={trainX} y2={curveH - yPad} stroke="hsl(218, 30%, 70%)" strokeWidth="1" strokeDasharray="2 3" />
          <text x={trainX + 4} y={curveH - yPad - 4} fontSize="10" fill="hsl(218, 30%, 45%)" fontFamily="var(--mono)">T_train = {tTrain.toLocaleString()}</text>
          {/* zero line */}
          <line x1={xPad} y1={toCY(0)} x2={curveW - xPad} y2={toCY(0)} stroke="hsl(218, 20%, 80%)" />
          {/* curve */}
          <path d={pathD} stroke="hsl(218, 50%, 40%)" strokeWidth="2" fill="none" />
          {/* x-axis labels */}
          <text x={xPad} y={curveH - 4} fontSize="10" fill="var(--ink-faint)" fontFamily="var(--mono)">0</text>
          <text x={curveW - xPad} y={curveH - 4} textAnchor="end" fontSize="10" fill="var(--ink-faint)" fontFamily="var(--mono)">{tEval.toLocaleString()}</text>
          {/* y-axis labels */}
          <text x={xPad - 4} y={toCY(0) + 3} textAnchor="end" fontSize="10" fill="var(--ink-faint)" fontFamily="var(--mono)">0</text>
          <text x={xPad - 4} y={toCY(yMax) + 9} textAnchor="end" fontSize="10" fill="var(--ink-faint)" fontFamily="var(--mono)">
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

// Sinusoidal positional embeddings for absolute mode: P_i[2k] = sin(i/10000^(2k/d)),
// P_i[2k+1] = cos(i/10000^(2k/d)). Same d as our token embeddings (8).
function sinusoidalPE(pos, d) {
  const out = new Array(d).fill(0);
  for (let k = 0; k < d / 2; k++) {
    const freq = 1 / Math.pow(10000, (2 * k) / d);
    out[2 * k]     = Math.sin(pos * freq);
    out[2 * k + 1] = Math.cos(pos * freq);
  }
  return out;
}

function SectionPosEncodingZoo() {
  const [mode, setMode] = useState('absolute');
  const T = 4;
  const tokens = PROMPT.slice(0, T);
  const X_base = tokens.map((t) => EMB[t]);

  // For Absolute mode, position vectors are added to the embedding *before* Q/K projections.
  // For the other modes, position info appears later (as score bias / value bias / nothing).
  const dModel = X_base[0].length;
  const P = Array.from({ length: T }, (_, i) => sinusoidalPE(i, dModel));
  const X_abs = X_base.map((x, i) => x.map((v, k) => v + P[i][k]));
  const X_for_q = mode === 'absolute' ? X_abs : X_base;

  const Q = matmul2(X_for_q, W_Q);
  const K = matmul2(X_for_q, W_K);
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
    'absolute':   'A_{ij} = \\mathrm{softmax}\\!\\left( (q_i + p_i) \\cdot (k_j + p_j) / \\sqrt{d_k} \\right)',
    'shaw-key':   'A_{ij} = \\mathrm{softmax}\\!\\left( q_i \\cdot (k_j + a^K_{i-j}) / \\sqrt{d_k} \\right)',
    'shaw-value': 'A_{ij} = \\mathrm{softmax}\\!\\left( q_i \\cdot k_j / \\sqrt{d_k} \\right), \\quad z_i = \\sum_{j} A_{ij}\\,(v_j + a^V_{i-j})',
    't5':         'A_{ij} = \\mathrm{softmax}\\!\\left( q_i \\cdot k_j / \\sqrt{d_k} + b_{i-j} \\right)',
    'swin':       'A_{ij} = \\mathrm{softmax}\\!\\left( q_i \\cdot k_j / \\sqrt{d_k} + b_{\\Delta i,\\,\\Delta j} \\right)',
  };

  const captions = {
    'absolute': (
      <>
        The simplest scheme — and the one the original Transformer paper
        (Vaswani et al. 2017) used. A position vector <Katex tex="p_i" /> is
        added to each token embedding <em>before</em> the Q/K projections.
        The attention math itself is unchanged; positional information
        arrives baked into Q and K. Sinusoidal or learned, both bounded by
        the trained sequence length — past it, the <Katex tex="p_i" /> are
        undefined.
      </>
    ),
    'shaw-key': (
      <>
        Shaw 2018 eq (4) — the relative-position info is a <em>vector</em>{' '}
        <Katex tex="a^K_{i-j}" /> added to <Katex tex="k_j" />{' '}
        <em>before</em> the dot product. The{' '}
        <Katex tex="a^K" /> values are <strong>learnable parameters</strong>:
        a small table of <Katex tex="d_k" />-dimensional vectors, one per
        relative offset, trained jointly with everything else. They're
        initialized like any other embedding table — a small random Gaussian
        (Xavier / Glorot-style) — and gradient descent shapes them from
        there. Shaw clipped the offset range (typically{' '}
        <Katex tex="\pm 16" />) so the table stays a fixed size — offsets
        beyond the clip share the boundary entry.
        {' '}The contribution to the score is{' '}
        <Katex tex="q_i \cdot a^K_{i-j}" />, which depends on the query — so
        different queries weight the same relative offset differently.
        Notice in the bias matrix below: each <em>row</em> looks different
        even though the index pattern is Toeplitz. That's the content
        dependence.
      </>
    ),
    'shaw-value': (
      <>
        Shaw 2018 eq (3) — the relative-position info is a vector{' '}
        <Katex tex="a^V_{i-j}" /> added to the <em>value</em> during the
        weighted sum.{' '}
        <Katex tex="a^V" /> is the same shape as <Katex tex="a^K" />: a
        learnable lookup table of <Katex tex="d_v" />-dimensional vectors
        indexed by clipped relative offset, learned alongside the model
        weights. Shaw used both <Katex tex="a^K" /> and <Katex tex="a^V" />{' '}
        but found the value-side gains were modest, and most later work
        (T5, Swin, …) dropped <Katex tex="a^V" /> entirely.
        {' '}Attention scores (and the attention pattern) are completely
        unchanged: it's pure positional information flowing into the output.
        The right panel shows <Katex tex="\Delta V_i" />, the per-row
        positional vector added to each output row.
      </>
    ),
    't5': (
      <>
        T5 2019 (and Swin in 2D) simplified Shaw's vector bias to a{' '}
        <em>learned scalar</em> per relative offset. Same Toeplitz indexing
        as Shaw, but the bias is content-independent — the same value gets
        added regardless of which <Katex tex="q, k" /> are involved. Cheap,
        effective, and the form most people think of as "relative position
        bias".
      </>
    ),
    'swin': (
      <>
        Swin 2021 ports T5's scalar bias to 2D image patches: index by{' '}
        <Katex tex="(\Delta i, \Delta j)" /> within an{' '}
        <Katex tex="M \times M" /> window. Same upside as T5 (relative,
        simple), same downside (bounded by the trained window — anything
        past the <Katex tex="M \times M" /> window falls off the table).
      </>
    ),
  };

  return (
    <div className="viz-panel">
      <div className="viz-controls">
        <div className="viz-tabs" role="tablist">
          {[
            { id: 'absolute',   label: 'Absolute' },
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
        {/* Left panel: bias matrix, position embeddings, or value-side contribution */}
        {mode === 'absolute' ? (
          <MiniMatrix
            data={P}
            title={`positional embeddings p_i  (sinusoidal, d = ${dModel})`}
            cellSize={36}
            diverging
            range={1}
          />
        ) : mode === 'shaw-value' ? (
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
              : mode === 'absolute'
                ? 'attention pattern A  (with position baked into Q, K)'
                : 'resulting attention pattern A  (rows sum to 1)'
          }
          diverging={false}
          vmax={1}
        />
      </div>

      <p className="viz-caption" style={{ marginTop: 14 }}>
        Five different places to inject position: into the input
        embedding (<strong>Absolute</strong>), into the key vector before
        the dot product (<strong>Shaw Key</strong>), into the value vector
        in the weighted sum (<strong>Shaw Value</strong>), as a learned
        scalar bias on the score (<strong>T5 Scalar</strong>), as the same
        scalar but indexed by a 2D offset (<strong>Swin 2D</strong>). None
        of them <em>rotate</em> Q or K. RoPE will be different on every
        axis: <em>multiplicative</em>, <em>on Q and K directly</em>, and
        <em> zero learned parameters</em>. That's §3.
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

        <h2>2. The predecessors: absolute and relative position bias</h2>
        <p>
          Before RoPE, every approach injected position as an{' '}
          <strong>additive term</strong> somewhere inside attention. Five
          variants are worth seeing side-by-side because each chooses a
          different place to inject it.{' '}
          <strong>Absolute</strong> (Vaswani et al. 2017, the original
          Transformer) adds a position vector to the input embedding before
          Q/K projections.{' '}
          <a className="viz-link" href="https://arxiv.org/abs/1803.02155" target="_blank" rel="noreferrer">
            Shaw et al. 2018
          </a>{' '}
          proposed two relative forms: one adds a vector to keys before the
          dot product (Shaw Key), one adds a vector to values inside the
          weighted sum (Shaw Value).{' '}
          <a className="viz-link" href="https://arxiv.org/abs/1910.10683" target="_blank" rel="noreferrer">
            T5
          </a>{' '}
          (Raffel et al. 2019) simplified Shaw Key's vector to a learned
          scalar per relative offset, and{' '}
          <a className="viz-link" href="https://arxiv.org/abs/2103.14030" target="_blank" rel="noreferrer">
            Swin
          </a>{' '}
          (Liu et al. 2021) ported that scalar to 2D image patches. Toggle
          between them below — note the structural differences in the left
          panel as much as the right.
        </p>
        <SectionPosEncodingZoo />
        <p>
          Three of these add a bias to the attention <em>scores</em>
          {' '}(Shaw Key, T5, Swin); one adds it to the <em>values</em>
          {' '}(Shaw Value); one adds it to the <em>inputs</em> (Absolute).
          None of them touch Q or K via a rotation, and all of them rely on
          a table indexed by absolute position or relative offset that's
          bounded by the trained range. RoPE breaks both habits at once: it
          modifies Q and K themselves <em>multiplicatively</em>, via a
          rotation, and the relative-position dependence falls out of the
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
        <p>
          This is also exactly why §7's long-context tricks exist — but
          counterintuitively, it's the <em>slow</em> pairs that drift out of
          distribution first. The fast pairs wind around the unit circle so
          many times even within the trained range that the model has
          already seen every possible <Katex tex="(\cos, \sin)" /> value;
          extending position just produces more of the same. The slow pairs,
          on the other hand, only ever traced a thin arc during training —
          at long evaluation contexts they reach angles the model has
          genuinely never been asked to interpret.
        </p>

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
          One subtlety from §5's clock-face intuition: the model is trained
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
          <a className="viz-link" href="https://arxiv.org/abs/2306.15595" target="_blank" rel="noreferrer">
            Position Interpolation
          </a>{' '}
          (Chen et al. 2023) was first: rescale <em>all</em> position
          indices uniformly by <Katex tex="T_\mathrm{train}/T_\mathrm{eval}" />,
          so every pair's angles stay inside the trained range — but it
          also compresses the fast pairs that didn't need it, costing
          short-range resolution.{' '}
          <a className="viz-link" href="https://arxiv.org/abs/2309.00071" target="_blank" rel="noreferrer">
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
          A fair question at this point: doesn't rescaling positions{' '}
          <em>change</em> what relative offsets mean to the model? How can
          this possibly work without retraining? The answer is that the
          model never learned "offset 5 means X" — it learned to react to
          rotation-angle <em>signatures</em>. With PI at{' '}
          <Katex tex="S = 8" />, an eval offset of 8 now produces the
          geometric signal that an offset of 1 used to. The model's heads
          still see distance-monotone signals, the dot product still
          encodes <em>some</em> relative-offset structure, and the model's
          original training distribution is approximately preserved.
        </p>
        <p>
          What you give up is <strong>resolution</strong>: with positions
          compressed 8×, the model can't tell "1 token apart" from "8
          tokens apart" any more — both look like the same angle to it.
          Fine-grained syntactic behaviour suffers. YaRN's frequency-aware
          design is exactly the response to this: it leaves the fast pairs
          (where short-range resolution lives) untouched, and only
          compresses the slow pairs (where the OOD problem actually was).
          You still lose some quality at extreme extensions, which is why
          production deployments typically pair PI/YaRN with a few
          thousand steps of fine-tuning on long-context data — the heads
          adapt to the new effective offsets and recover most of the
          degradation. So: post-hoc context extension works, but it isn't
          free, and the "free" part has a ceiling.
        </p>

        <h2>8. Related work</h2>
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
          first wave of post-training context extension. The frontier — NTK
          scaling, dynamic NTK, LongRoPE, RingAttention, sparse-attention
          variants — is its own ecosystem, all built on the same rotation
          idea from §3.
        </p>
        <p>
          <strong>Earlier in this series.</strong>{' '}
          <a className="viz-link" href="#/blog/visualizing-attention">Post #1 — Visualizing Attention</a>{' '}
          covers Q/K/V, multi-head, and causal masking — the substrate RoPE
          modifies. <a className="viz-link" href="#/blog/visualizing-kv-cache">Post #2 — Visualizing the KV Cache</a>{' '}
          shows why inference is bandwidth-bound and how the cache shapes
          every long-context engineering decision (PI/YaRN included — they
          stretch the rotation schedule, not the cache).
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

        <footer className="viz-footer">
          <p>
            <strong>Part 3 of Visualizing ML</strong> · Previous:{' '}
            <a className="viz-link" href="#/blog/visualizing-kv-cache">Visualizing the KV Cache</a>
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
            {' '}(Anthropic) — initial drafts and interactive scaffolding by Claude,
            refined and co-designed by Bernhard.
          </p>
        </footer>
      </div>
    </article>
  );
}
