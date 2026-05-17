// src/posts/VisualizingAttention.jsx
// Part 1 of "Visualizing ML" — Q/K/V, Multi-Head, Causal Masking
import React, { useEffect, useMemo, useRef, useState } from 'react';
import './VisualizingAttention.css';

/* =========================================================
   Math helpers — kept naive (T, d are tiny by design)
   ========================================================= */

const round = (x, n = 2) => {
  const k = 10 ** n;
  return Math.round(x * k) / k;
};

const matmul = (A, B) => {
  const m = A.length, k = A[0].length, n = B[0].length;
  const C = Array.from({ length: m }, () => new Array(n).fill(0));
  for (let i = 0; i < m; i++)
    for (let j = 0; j < n; j++) {
      let s = 0;
      for (let p = 0; p < k; p++) s += A[i][p] * B[p][j];
      C[i][j] = s;
    }
  return C;
};

const transpose = (A) => A[0].map((_, j) => A.map((r) => r[j]));

const scaleMat = (A, s) => A.map((r) => r.map((v) => v * s));

const softmaxRow = (row) => {
  const m = Math.max(...row);
  const exps = row.map((v) => Math.exp(v - m));
  const s = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / s);
};

const softmaxRows = (A) => A.map(softmaxRow);

const applyCausalMask = (A) =>
  A.map((row, i) => row.map((v, j) => (j > i ? -Infinity : v)));

const rowEntropy = (p) => {
  let h = 0;
  for (const x of p) if (x > 0) h -= x * Math.log2(x);
  return h;
};

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
        // KaTeX script not yet loaded — retry shortly
        setTimeout(render, 60);
      }
    };
    render();
    return () => {
      cancelled = true;
    };
  }, [tex, block]);
  return (
    <span
      ref={ref}
      className={`${block ? 'viz-math-block' : 'viz-math-inline'} ${className}`}
    />
  );
}

/* =========================================================
   Precomputed data — sentences, embeddings, projections
   ========================================================= */

// 8-dim embedding per token. Deterministic and hand-rounded so
// every number on the page is one a reader can verify by eye.
const EMB = {
  the:    [ 0.2, -0.1,  0.3,  0.0,  0.1, -0.2,  0.4,  0.1],
  cat:    [ 0.5,  0.3, -0.2,  0.4,  0.1,  0.0, -0.1,  0.2],
  sat:    [-0.1,  0.4,  0.5,  0.2, -0.3,  0.1,  0.2, -0.4],
  on:     [ 0.1, -0.2,  0.1, -0.3,  0.4,  0.2, -0.1,  0.0],
  mat:    [ 0.4,  0.2, -0.1,  0.3, -0.2,  0.5,  0.0,  0.1],
  she:    [ 0.6,  0.1,  0.0,  0.2,  0.3, -0.1,  0.4, -0.2],
  gave:   [-0.2,  0.5,  0.3, -0.1,  0.1,  0.4, -0.3,  0.2],
  him:    [ 0.4, -0.1,  0.2,  0.5, -0.2,  0.0,  0.3,  0.1],
  a:      [ 0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1],
  book:   [-0.3,  0.2,  0.5, -0.1,  0.4,  0.2, -0.2,  0.3],
  quick:  [ 0.3,  0.4, -0.2,  0.1,  0.5, -0.1,  0.2,  0.0],
  brown:  [ 0.2, -0.3,  0.1,  0.4, -0.1,  0.3,  0.0,  0.2],
  fox:    [-0.1,  0.5,  0.2, -0.2,  0.3,  0.1,  0.4, -0.3],
  jumps:  [ 0.0,  0.2, -0.3,  0.5,  0.1, -0.2,  0.3,  0.4],
  code:   [ 0.5, -0.2,  0.4,  0.1, -0.3,  0.2,  0.1,  0.0],
  that:   [ 0.1,  0.3, -0.1,  0.0,  0.2, -0.2,  0.4,  0.1],
  writes: [-0.2,  0.4,  0.3,  0.5,  0.0,  0.1, -0.3,  0.2],
  itself: [ 0.3, -0.1,  0.5,  0.2, -0.2,  0.4,  0.0, -0.1],
};

const SENTENCES = [
  { id: 'cat',   tokens: ['the', 'cat', 'sat', 'on', 'the', 'mat'] },
  { id: 'she',   tokens: ['she', 'gave', 'him', 'a', 'book'] },
  { id: 'fox',   tokens: ['a', 'quick', 'brown', 'fox', 'jumps'] },
  { id: 'code',  tokens: ['code', 'that', 'writes', 'itself'] },
];

// W_Q, W_K, W_V — 8x4, hand-picked so projections have visible structure.
// W_Q and W_K use a larger range than W_V so the resulting QKᵀ dot products
// are big enough to give visibly non-uniform softmax. With d_k=4 this needs
// a bit of help — a real model achieves it through training, not by hand.
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

const W_V = [
  [ 0.1,  0.3,  0.0,  0.2],
  [ 0.4,  0.0, -0.1,  0.0],
  [ 0.0,  0.2,  0.3, -0.1],
  [ 0.2, -0.1,  0.0,  0.4],
  [-0.2,  0.0,  0.3,  0.1],
  [ 0.0,  0.4,  0.1, -0.2],
  [ 0.3,  0.1,  0.0,  0.0],
  [ 0.1, -0.2,  0.4,  0.2],
];

const buildXForSentence = (sent) => sent.tokens.map((t) => EMB[t]);

/* =========================================================
   Numeric matrix grid — every cell tappable for explanation
   ========================================================= */

function MatrixGrid({
  data,
  label,
  rowLabels,
  colLabels,
  decimals = 2,
  cellWidth = 46,
  onCellEnter,
  onCellLeave,
  onCellTap,
  highlight, // { row, col } either may be null
  trace,      // { rowOfFirst, colOfSecond } -> draws X row and W col highlights
  whichTrace, // 'x' | 'w' | 'both'
}) {
  const cols = data[0].length;
  const hasRowLabels = !!rowLabels;
  const hasColLabels = !!colLabels;
  const gridCols = `${hasRowLabels ? '38px ' : ''}repeat(${cols}, ${cellWidth}px)`;

  return (
    <div className="viz-mat">
      {label && <div className="viz-mat-label">{label}</div>}
      <div
        className="viz-mat-grid"
        style={{ gridTemplateColumns: gridCols }}
        role="grid"
      >
        {hasColLabels && (
          <>
            {hasRowLabels && <div className="viz-cell viz-col-label" />}
            {colLabels.map((c, j) => (
              <div key={`c-${j}`} className="viz-cell viz-col-label">{c}</div>
            ))}
          </>
        )}
        {data.map((row, i) => (
          <React.Fragment key={`r-${i}`}>
            {hasRowLabels && (
              <div className="viz-cell viz-row-label">{rowLabels[i]}</div>
            )}
            {row.map((v, j) => {
              const isFocus = highlight && highlight.row === i && highlight.col === j;
              const isRowHi = highlight && highlight.row === i && highlight.col == null;
              const isColHi = highlight && highlight.col === j && highlight.row == null;
              let traceClass = '';
              if (trace) {
                if (whichTrace === 'x' && i === trace.row) traceClass = 'viz-trace-x';
                if (whichTrace === 'w' && j === trace.col) traceClass = 'viz-trace-w';
              }
              const display = Number.isFinite(v)
                ? v.toFixed(decimals)
                : (v === -Infinity ? '−∞' : '·');
              const interactive = !!(onCellEnter || onCellTap);
              return (
                <div
                  key={`c-${i}-${j}`}
                  className={[
                    'viz-cell',
                    interactive ? 'viz-cell-tappable' : '',
                    isFocus ? 'viz-focus' : '',
                    isRowHi ? 'viz-row-hi' : '',
                    isColHi ? 'viz-col-hi' : '',
                    traceClass,
                  ].filter(Boolean).join(' ')}
                  style={{ width: cellWidth }}
                  onMouseEnter={onCellEnter ? () => onCellEnter(i, j) : undefined}
                  onMouseLeave={onCellLeave ? () => onCellLeave(i, j) : undefined}
                  onClick={onCellTap ? () => onCellTap(i, j) : undefined}
                  tabIndex={interactive ? 0 : -1}
                  onKeyDown={
                    interactive && onCellTap
                      ? (e) => {
                          if (e.key === 'Enter' || e.key === ' ') {
                            e.preventDefault();
                            onCellTap(i, j);
                          }
                        }
                      : undefined
                  }
                >
                  {display}
                </div>
              );
            })}
          </React.Fragment>
        ))}
      </div>
    </div>
  );
}

/* =========================================================
   Heatmap grid — saturation encodes magnitude
   ========================================================= */

function HeatmapGrid({
  data,
  label,
  rowLabels,
  colLabels,
  decimals = 2,
  cellWidth = 46,
  vmin,
  vmax,
  onRowEnter,
  onRowLeave,
  onRowTap,
  activeRow,
  showValues = true,
  diverging = false,
}) {
  const cols = data[0].length;
  const flat = data.flat().filter(Number.isFinite);
  const lo = vmin != null ? vmin : Math.min(...flat);
  const hi = vmax != null ? vmax : Math.max(...flat);
  const span = hi - lo || 1;
  const gridCols = `38px repeat(${cols}, ${cellWidth}px)`;

  const cellColor = (v) => {
    if (!Number.isFinite(v)) return '#2A2A2A';
    if (diverging) {
      const norm = (v - lo) / span; // 0..1
      const t = (norm - 0.5) * 2; // -1..1
      if (t >= 0) {
        const sat = 18 + Math.round(t * 60);
        const light = 96 - Math.round(t * 46);
        return `hsl(var(--accent-h), ${sat}%, ${light}%)`;
      } else {
        const sat = 18 + Math.round(-t * 50);
        const light = 96 - Math.round(-t * 40);
        return `hsl(8, ${sat}%, ${light}%)`;
      }
    }
    const norm = (v - lo) / span;
    const sat = 18 + Math.round(norm * 60);
    const light = 96 - Math.round(norm * 50);
    return `hsl(var(--accent-h), ${sat}%, ${light}%)`;
  };

  const textColor = (v) => {
    if (!Number.isFinite(v)) return '#FFF';
    const norm = (v - lo) / span;
    return norm > 0.55 ? '#FFF' : 'var(--ink)';
  };

  return (
    <div className="viz-mat">
      {label && <div className="viz-mat-label">{label}</div>}
      <div
        className="viz-mat-grid"
        style={{ gridTemplateColumns: gridCols }}
      >
        <div className="viz-cell viz-col-label" />
        {colLabels.map((c, j) => (
          <div key={`hc-${j}`} className="viz-cell viz-col-label">{c}</div>
        ))}
        {data.map((row, i) => (
          <React.Fragment key={`hr-${i}`}>
            <div className="viz-cell viz-row-label">{rowLabels[i]}</div>
            {row.map((v, j) => {
              const isActive = activeRow === i;
              const display = Number.isFinite(v)
                ? v.toFixed(decimals)
                : '−∞';
              return (
                <div
                  key={`hc-${i}-${j}`}
                  className={[
                    'viz-cell',
                    'viz-heat',
                    'viz-cell-tappable',
                    isActive ? 'viz-row-hi' : '',
                  ].filter(Boolean).join(' ')}
                  style={{
                    width: cellWidth,
                    background: cellColor(v),
                    color: textColor(v),
                  }}
                  onMouseEnter={onRowEnter ? () => onRowEnter(i) : undefined}
                  onMouseLeave={onRowLeave ? () => onRowLeave(i) : undefined}
                  onClick={onRowTap ? () => onRowTap(i) : undefined}
                  tabIndex={0}
                  onKeyDown={
                    onRowTap
                      ? (e) => {
                          if (e.key === 'Enter' || e.key === ' ') {
                            e.preventDefault();
                            onRowTap(i);
                          }
                        }
                      : undefined
                  }
                >
                  {showValues ? display : ''}
                </div>
              );
            })}
          </React.Fragment>
        ))}
      </div>
    </div>
  );
}

/* =========================================================
   Interactive #1 — Tokens -> Embeddings -> Q, K, V
   ========================================================= */

function Section1QKV() {
  const [sentenceId, setSentenceId] = useState('she');
  const [hover, setHover] = useState(null); // { mat: 'Q'|'K'|'V', i, j }
  const sentence = SENTENCES.find((s) => s.id === sentenceId);

  const { X, Q, K, V } = useMemo(() => {
    const X = buildXForSentence(sentence);
    return {
      X,
      Q: matmul(X, W_Q),
      K: matmul(X, W_K),
      V: matmul(X, W_V),
    };
  }, [sentence]);

  const Wmap = { Q: W_Q, K: W_K, V: W_V };

  const explanation = (() => {
    if (!hover) return null;
    const W = Wmap[hover.mat];
    const xRow = X[hover.i];
    const wCol = W.map((r) => r[hover.j]);
    const terms = xRow.map((x, p) => `${round(x).toFixed(2)}·${round(wCol[p]).toFixed(2)}`);
    const total = xRow.reduce((s, x, p) => s + x * wCol[p], 0);
    return (
      <>
        <strong>{hover.mat}[{hover.i},{hover.j}] = X[row {hover.i}] · {hover.mat === 'Q' ? 'W_Q' : hover.mat === 'K' ? 'W_K' : 'W_V'}[col {hover.j}]</strong>
        {' = '}
        {terms.join(' + ')} = <strong>{round(total).toFixed(3)}</strong>
      </>
    );
  })();

  const buildHi = (mat) => (hover && hover.mat === mat ? { row: hover.i, col: hover.j } : null);

  const colLabelsX = ['e0','e1','e2','e3','e4','e5','e6','e7'];
  const colLabelsW = ['c0','c1','c2','c3'];

  return (
    <div className="viz-panel">
      <div className="viz-controls">
        <label>
          Sentence
          <select
            value={sentenceId}
            onChange={(e) => { setSentenceId(e.target.value); setHover(null); }}
          >
            {SENTENCES.map((s) => (
              <option key={s.id} value={s.id}>{s.tokens.join(' ')}</option>
            ))}
          </select>
        </label>
        <span style={{ color: 'var(--ink-faint)', fontSize: '0.85rem' }}>
          d<sub>model</sub> = 8 &nbsp;·&nbsp; d<sub>k</sub> = 4
        </span>
      </div>

      <div className="viz-subsection">
        <h4 className="viz-subhead">
          <span className="viz-sub-tag viz-sub-tag-q">Q</span>
          Queries — <em>"what am I looking for?"</em>
        </h4>
        <div className="viz-matrix-row">
          <MatrixGrid
            data={X}
            label="X  (T × d_model)"
            rowLabels={sentence.tokens}
            colLabels={colLabelsX}
            cellWidth={42}
            highlight={hover && hover.mat === 'Q' ? { row: hover.i, col: null } : null}
          />
          <div className="viz-matrix-op">·</div>
          <MatrixGrid
            data={W_Q}
            label="W_Q  (d_model × d_k)"
            colLabels={colLabelsW}
            cellWidth={48}
            highlight={hover && hover.mat === 'Q' ? { row: null, col: hover.j } : null}
          />
          <div className="viz-matrix-op">=</div>
          <MatrixGrid
            data={Q}
            label="Q  (T × d_k)"
            rowLabels={sentence.tokens}
            colLabels={colLabelsW}
            cellWidth={50}
            highlight={buildHi('Q')}
            onCellEnter={(i, j) => setHover({ mat: 'Q', i, j })}
            onCellLeave={() => setHover(null)}
            onCellTap={(i, j) => setHover({ mat: 'Q', i, j })}
          />
        </div>
      </div>

      <div className="viz-subsection">
        <h4 className="viz-subhead">
          <span className="viz-sub-tag viz-sub-tag-k">K</span>
          Keys — <em>"what do I offer?"</em>
        </h4>
        <div className="viz-matrix-row">
          <MatrixGrid
            data={X}
            label="X  (T × d_model)"
            rowLabels={sentence.tokens}
            colLabels={colLabelsX}
            cellWidth={42}
            highlight={hover && hover.mat === 'K' ? { row: hover.i, col: null } : null}
          />
          <div className="viz-matrix-op">·</div>
          <MatrixGrid
            data={W_K}
            label="W_K  (d_model × d_k)"
            colLabels={colLabelsW}
            cellWidth={48}
            highlight={hover && hover.mat === 'K' ? { row: null, col: hover.j } : null}
          />
          <div className="viz-matrix-op">=</div>
          <MatrixGrid
            data={K}
            label="K  (T × d_k)"
            rowLabels={sentence.tokens}
            colLabels={colLabelsW}
            cellWidth={50}
            highlight={buildHi('K')}
            onCellEnter={(i, j) => setHover({ mat: 'K', i, j })}
            onCellLeave={() => setHover(null)}
            onCellTap={(i, j) => setHover({ mat: 'K', i, j })}
          />
        </div>
      </div>

      <div className="viz-subsection">
        <h4 className="viz-subhead">
          <span className="viz-sub-tag viz-sub-tag-v">V</span>
          Values — <em>"what I'll pass along if matched"</em>
        </h4>
        <div className="viz-matrix-row">
          <MatrixGrid
            data={X}
            label="X  (T × d_model)"
            rowLabels={sentence.tokens}
            colLabels={colLabelsX}
            cellWidth={42}
            highlight={hover && hover.mat === 'V' ? { row: hover.i, col: null } : null}
          />
          <div className="viz-matrix-op">·</div>
          <MatrixGrid
            data={W_V}
            label="W_V  (d_model × d_k)"
            colLabels={colLabelsW}
            cellWidth={48}
            highlight={hover && hover.mat === 'V' ? { row: null, col: hover.j } : null}
          />
          <div className="viz-matrix-op">=</div>
          <MatrixGrid
            data={V}
            label="V  (T × d_k)"
            rowLabels={sentence.tokens}
            colLabels={colLabelsW}
            cellWidth={50}
            highlight={buildHi('V')}
            onCellEnter={(i, j) => setHover({ mat: 'V', i, j })}
            onCellLeave={() => setHover(null)}
            onCellTap={(i, j) => setHover({ mat: 'V', i, j })}
          />
        </div>
      </div>

      <div className="viz-explain">
        {explanation || (
          <span className="viz-explain-empty">
            Hover or tap any cell in Q, K, or V to see the dot product that produced it.
          </span>
        )}
      </div>

      <pre className="viz-code">
{`# Three linear projections, all from the same X
Q = X @ W_Q     # (T, d_k)
K = X @ W_K     # (T, d_k)
V = X @ W_V     # (T, d_k)`}
      </pre>
    </div>
  );
}

/* =========================================================
   Softmax mini-demo (used inline in §5)
   ========================================================= */

const SOFTMAX_PRESETS = [
  { label: 'all equal',     vals: [1.0, 1.0, 1.0, 1.0] },
  { label: 'mild winner',   vals: [3.0, 1.0, 1.0, 1.0] },
  { label: 'sharp winner',  vals: [5.0, 1.0, 0.5, 0.0] },
  { label: 'two winners',   vals: [3.0, 3.0, 0.0, 0.0] },
];

function SoftmaxDemo() {
  const [presetIdx, setPresetIdx] = useState(1);
  const vals = SOFTMAX_PRESETS[presetIdx].vals;
  const probs = softmaxRow(vals);
  const inMin = Math.min(0, ...vals);
  const inMax = Math.max(...vals);
  const inSpan = (inMax - inMin) || 1;
  const maxBar = 180;

  return (
    <div className="viz-softmax-demo">
      <div className="viz-softmax-tabs">
        {SOFTMAX_PRESETS.map((p, i) => (
          <button
            key={i}
            className={`viz-tab ${i === presetIdx ? 'active' : ''}`}
            onClick={() => setPresetIdx(i)}
          >
            {p.label}
          </button>
        ))}
      </div>
      <div className="viz-softmax-row">
        <div className="viz-softmax-side">
          <div className="viz-softmax-side-label">input numbers</div>
          {vals.map((v, i) => (
            <div className="viz-softmax-bar-row" key={i}>
              <span className="viz-softmax-num">{v.toFixed(1)}</span>
              <span
                className="viz-softmax-bar viz-softmax-bar-in"
                style={{ width: Math.max(2, ((v - inMin) / inSpan) * maxBar) }}
              />
            </div>
          ))}
        </div>
        <div className="viz-softmax-arrow">softmax →</div>
        <div className="viz-softmax-side">
          <div className="viz-softmax-side-label">probabilities (sum = 1.0)</div>
          {probs.map((p, i) => (
            <div className="viz-softmax-bar-row" key={i}>
              <span className="viz-softmax-num">{(p * 100).toFixed(1)}%</span>
              <span
                className="viz-softmax-bar viz-softmax-bar-out"
                style={{ width: Math.max(2, p * maxBar) }}
              />
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

/* =========================================================
   Interactive #5 — Scores QK^T / sqrt(d_k) and softmax
   ========================================================= */

const TEMPS = [
  { label: 'no scaling (÷1)',     div: 1 },
  { label: '÷ √d_k/2 ≈ 1.41',     div: Math.sqrt(2) },
  { label: '÷ √d_k = 2  (paper)', div: 2 },
  { label: '÷ 2·√d_k = 4',        div: 4 },
];

function Section2Scores() {
  const [sentenceId, setSentenceId] = useState('she');
  const [tempIdx, setTempIdx] = useState(2);
  const [hoverRow, setHoverRow] = useState(null);
  const sentence = SENTENCES.find((s) => s.id === sentenceId);

  const colLabelsW = ['c0','c1','c2','c3'];

  const { Q, K, Kt, raw, scaled, A } = useMemo(() => {
    const X = buildXForSentence(sentence);
    const Q = matmul(X, W_Q);
    const K = matmul(X, W_K);
    const Kt = transpose(K);
    const raw = matmul(Q, Kt);
    const scaled = scaleMat(raw, 1 / TEMPS[tempIdx].div);
    const A = softmaxRows(scaled);
    return { Q, K, Kt, raw, scaled, A };
  }, [sentence, tempIdx]);

  const meta = (() => {
    if (hoverRow == null) return null;
    const probs = A[hoverRow];
    const top = probs
      .map((p, j) => ({ p, j }))
      .sort((a, b) => b.p - a.p)[0];
    return (
      <>
        Row <strong>{sentence.tokens[hoverRow]}</strong> attends most to{' '}
        <strong>{sentence.tokens[top.j]}</strong> ({(top.p * 100).toFixed(1)}%).
        Entropy = {rowEntropy(probs).toFixed(2)} bits.
      </>
    );
  })();

  return (
    <div className="viz-panel">
      <div className="viz-controls">
        <label>
          Sentence
          <select
            value={sentenceId}
            onChange={(e) => { setSentenceId(e.target.value); setHoverRow(null); }}
          >
            {SENTENCES.map((s) => (
              <option key={s.id} value={s.id}>{s.tokens.join(' ')}</option>
            ))}
          </select>
        </label>
      </div>

      <div className="viz-subsection">
        <h4 className="viz-subhead">
          <span className="viz-sub-tag viz-sub-tag-step">1</span>
          Compute raw scores: <Katex tex="Q \cdot K^{\top}" />
        </h4>
        <p className="viz-subhead-prose">
          Every query row in <Katex tex="Q" /> is dotted with every key row in{' '}
          <Katex tex="K" /> (i.e. every <em>column</em> of <Katex tex="K^{\top}" />).
          The result is a <Katex tex="T \times T" /> matrix where entry{' '}
          <Katex tex="(i, j)" /> is a single number: <em>how compatible is
          token i's question with token j's offering?</em>
        </p>
        <div className="viz-matrix-row">
          <MatrixGrid
            data={Q}
            label="Q  (T × d_k)"
            rowLabels={sentence.tokens}
            colLabels={colLabelsW}
            cellWidth={50}
          />
          <div className="viz-matrix-op">·</div>
          <MatrixGrid
            data={Kt}
            label="Kᵀ  (d_k × T)"
            rowLabels={colLabelsW}
            colLabels={sentence.tokens}
            cellWidth={50}
          />
          <div className="viz-matrix-op">=</div>
          <HeatmapGrid
            data={raw}
            label="scores = Q·Kᵀ"
            rowLabels={sentence.tokens}
            colLabels={sentence.tokens}
            cellWidth={56}
            diverging
          />
        </div>
      </div>

      <div className="viz-subsection">
        <h4 className="viz-subhead">
          <span className="viz-sub-tag viz-sub-tag-step">2</span>
          Scale, then softmax → attention matrix <Katex tex="A" />
        </h4>
        <p className="viz-subhead-prose">
          Divide the raw scores by <Katex tex="\sqrt{d_k}" />, then apply{' '}
          <strong>softmax</strong> row-by-row. Softmax turns each row of
          arbitrary real numbers into a probability distribution (every entry
          positive, every row summing to 1). The output <Katex tex="A" /> is
          the <strong>attention matrix</strong>: row <Katex tex="i" /> is
          literally "how much should token <Katex tex="i" /> pay attention to
          each other token?" — those weights are what we'll use in the next
          step to mix the value vectors.
        </p>
        <div className="viz-controls" style={{ marginTop: 4 }}>
          <label style={{ flex: 1, minWidth: 260 }}>
            Scaling: <strong style={{ marginLeft: 4 }}>{TEMPS[tempIdx].label}</strong>
            <input
              type="range"
              min={0}
              max={TEMPS.length - 1}
              step={1}
              value={tempIdx}
              onChange={(e) => setTempIdx(parseInt(e.target.value, 10))}
              style={{ marginLeft: 8, width: 200 }}
            />
          </label>
        </div>
        <div className="viz-matrix-row">
          <HeatmapGrid
            data={scaled}
            label={`scaled = scores / ${TEMPS[tempIdx].div.toFixed(2)}`}
            rowLabels={sentence.tokens}
            colLabels={sentence.tokens}
            cellWidth={56}
            activeRow={hoverRow}
            onRowEnter={(i) => setHoverRow(i)}
            onRowLeave={() => setHoverRow(null)}
            onRowTap={(i) => setHoverRow(i)}
            diverging
          />
          <div className="viz-matrix-op">softmax →</div>
          <HeatmapGrid
            data={A}
            label="A  (rows sum to 1)"
            rowLabels={sentence.tokens}
            colLabels={sentence.tokens}
            cellWidth={56}
            vmin={0}
            vmax={1}
            activeRow={hoverRow}
            onRowEnter={(i) => setHoverRow(i)}
            onRowLeave={() => setHoverRow(null)}
            onRowTap={(i) => setHoverRow(i)}
          />
        </div>
        <div className="viz-row-meta">
          {meta || (
            <span style={{ fontStyle: 'italic' }}>
              Hover or tap any row of <strong>A</strong> to see what that token attends to.
            </span>
          )}
        </div>
      </div>

      <div className="viz-subsection">
        <h4 className="viz-subhead">
          <span className="viz-sub-tag viz-sub-tag-star">★</span>
          Step — what is softmax, exactly?
        </h4>
        <p className="viz-subhead-prose">
          You just used softmax — let's unpack it. Softmax turns a list of
          arbitrary real numbers into a probability distribution. Why that
          function, specifically? Two reasons. First, attention weights must
          be non-negative — you can't attend by a negative amount. Second,
          they need to sum to 1, so the output in the next section is a clean
          weighted average instead of arbitrarily scaled. Softmax delivers
          both via:
        </p>
        <div className="viz-math-block">
          <Katex block tex="\mathrm{softmax}(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}}" />
        </div>
        <p className="viz-subhead-prose">
          Exponentiating makes everything positive; dividing by the sum makes
          the result sum to 1. Larger inputs dominate exponentially — a
          winner-take-most behaviour, controlled by the spread of the inputs.
          Try the presets:
        </p>
        <SoftmaxDemo />
        <p className="viz-caption" style={{ marginTop: 12 }}>
          That winner-take-most behaviour is also why the{' '}
          <Katex tex="\sqrt{d_k}" /> divisor in Step 2 is not cosmetic: without
          it, dot products grow with <Katex tex="d_k" /> and softmax saturates
          into one-hot, killing the gradient. Drag the slider in Step 2 to
          watch the attention sharpen or flatten as the scaling changes.
        </p>
      </div>

      <pre className="viz-code">
{`import math
scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)   # (T, T)
A = scores.softmax(dim=-1)                          # rows sum to 1`}
      </pre>
    </div>
  );
}

/* =========================================================
   Interactive #3 — Output = A · V with animated mixing
   ========================================================= */

function Section3Output() {
  const [sentenceId, setSentenceId] = useState('she');
  const [pickedRow, setPickedRow] = useState(0);
  const [animKey, setAnimKey] = useState(0);
  const sentence = SENTENCES.find((s) => s.id === sentenceId);

  const { V, A, Y } = useMemo(() => {
    const X = buildXForSentence(sentence);
    const Q = matmul(X, W_Q);
    const K = matmul(X, W_K);
    const V = matmul(X, W_V);
    const scores = scaleMat(matmul(Q, transpose(K)), 1 / 2);
    const A = softmaxRows(scores);
    const Y = matmul(A, V);
    return { V, A, Y };
  }, [sentence]);

  const replay = () => setAnimKey((k) => k + 1);

  const T = sentence.tokens.length;
  const weights = A[pickedRow];

  return (
    <div className="viz-panel">
      <div className="viz-controls">
        <label>
          Sentence
          <select
            value={sentenceId}
            onChange={(e) => { setSentenceId(e.target.value); setPickedRow(0); replay(); }}
          >
            {SENTENCES.map((s) => (
              <option key={s.id} value={s.id}>{s.tokens.join(' ')}</option>
            ))}
          </select>
        </label>
        <label>
          Output row
          <select
            value={pickedRow}
            onChange={(e) => { setPickedRow(parseInt(e.target.value, 10)); replay(); }}
          >
            {sentence.tokens.map((t, i) => (
              <option key={i} value={i}>{t}</option>
            ))}
          </select>
        </label>
        <button onClick={replay}>Replay animation</button>
      </div>

      <div className="viz-matrix-row" style={{ overflowX: 'auto' }}>
        <HeatmapGrid
          data={A}
          label="A"
          rowLabels={sentence.tokens}
          colLabels={sentence.tokens}
          cellWidth={50}
          vmin={0}
          vmax={1}
          activeRow={pickedRow}
          onRowTap={(i) => { setPickedRow(i); replay(); }}
          onRowEnter={(i) => setPickedRow(i)}
        />
        <div className="viz-matrix-op">·</div>
        <MatrixGrid
          data={V}
          label="V"
          rowLabels={sentence.tokens}
          colLabels={['v0','v1','v2','v3']}
          cellWidth={50}
        />
        <div className="viz-matrix-op">=</div>
        <MatrixGrid
          data={Y}
          label="Y = A·V"
          rowLabels={sentence.tokens}
          colLabels={['y0','y1','y2','y3']}
          cellWidth={50}
          highlight={{ row: pickedRow, col: null }}
        />
      </div>

      <div className="viz-mix-stage" key={animKey}>
        <div style={{ fontFamily: 'var(--mono)', fontSize: '0.82rem', color: 'var(--ink-faint)', marginBottom: 8 }}>
          Y[<strong style={{ color: 'var(--ink)' }}>{sentence.tokens[pickedRow]}</strong>] = Σⱼ A[{pickedRow},j] · V[j]
        </div>
        {weights.map((w, j) => (
          <MixRow
            key={`${animKey}-${j}`}
            delay={120 + j * 220}
            token={sentence.tokens[j]}
            weight={w}
            vRow={V[j]}
          />
        ))}
        <OutputMixRow
          delay={120 + T * 220 + 200}
          token={sentence.tokens[pickedRow]}
          yRow={Y[pickedRow]}
        />
      </div>

      <pre className="viz-code">
{`Y = A @ V                  # (T, d_k)
# Each output row is a convex combination of V's rows,
# weighted by that row of A.`}
      </pre>
    </div>
  );
}

function MixRow({ delay, token, weight, vRow }) {
  const [on, setOn] = useState(false);
  useEffect(() => {
    const t = setTimeout(() => setOn(true), delay);
    return () => clearTimeout(t);
  }, [delay]);
  const barWidth = Math.max(2, Math.round(weight * 220));
  return (
    <div className={`viz-mix-row${on ? ' viz-mix-active' : ''}`}>
      <span className="viz-mix-weight">{(weight * 100).toFixed(1)}%</span>
      <span style={{ display: 'inline-block', width: 240 }}>
        <span className="viz-mix-bar" style={{ display: 'block', width: barWidth }} />
      </span>
      <span style={{ minWidth: 60, fontStyle: 'italic' }}>{token}</span>
      <span>
        [{vRow.map((v) => v.toFixed(2)).join(', ')}]
      </span>
    </div>
  );
}

function OutputMixRow({ delay, token, yRow }) {
  const [on, setOn] = useState(false);
  useEffect(() => {
    const t = setTimeout(() => setOn(true), delay);
    return () => clearTimeout(t);
  }, [delay]);
  return (
    <div className={`viz-mix-row viz-mix-output${on ? ' viz-mix-active' : ''}`}>
      <span className="viz-mix-weight">=</span>
      <span style={{ display: 'inline-block', width: 240 }} />
      <span style={{ minWidth: 60 }}>{token}</span>
      <span>[{yRow.map((v) => v.toFixed(2)).join(', ')}]</span>
    </div>
  );
}

/* =========================================================
   Interactive #4 — Multi-head (hand-crafted patterns)
   ========================================================= */

// All patterns are pre-baked for "she gave him a book" (T=5) and
// then mapped onto other sentences by tiling. They are illustrative:
// they show *what kinds* of patterns heads tend to learn, not an
// actual trained model.
const HEAD_PATTERNS = [
  {
    name: 'head 0 — diagonal / identity',
    desc: 'A weak baseline: each token mostly looks at itself. Often the first thing heads learn at init.',
    build: (T) => {
      const A = Array.from({ length: T }, () => new Array(T).fill(0));
      for (let i = 0; i < T; i++) {
        for (let j = 0; j < T; j++) {
          A[i][j] = i === j ? 0.7 : 0.3 / (T - 1);
        }
      }
      return A;
    },
  },
  {
    name: 'head 1 — previous-token',
    desc: 'A classic head: each token attends to the one immediately before it. Useful for bigram-style features.',
    build: (T) => {
      const A = Array.from({ length: T }, () => new Array(T).fill(0));
      for (let i = 0; i < T; i++) {
        for (let j = 0; j < T; j++) {
          if (j === i - 1) A[i][j] = 0.75;
          else if (j === i) A[i][j] = 0.15;
          else A[i][j] = 0.1 / Math.max(1, T - 2);
        }
      }
      return A;
    },
  },
  {
    name: 'head 2 — subject ↔ verb',
    desc: 'Picks out a long-range syntactic dependency: verbs look back at their subject. Real heads often specialize like this.',
    build: (T) => {
      // hand-tuned: each row mostly attends to position 1 (the verb)
      const A = Array.from({ length: T }, () => new Array(T).fill(0));
      const target = Math.min(1, T - 1);
      for (let i = 0; i < T; i++) {
        for (let j = 0; j < T; j++) {
          if (j === target) A[i][j] = 0.6;
          else if (j === i) A[i][j] = 0.2;
          else A[i][j] = 0.2 / Math.max(1, T - 2);
        }
      }
      return A;
    },
  },
  {
    name: 'head 3 — broadcast',
    desc: 'Roughly uniform across the sequence: a "context-vector" head that mixes everything. Useful for global signals.',
    build: (T) => Array.from({ length: T }, () => new Array(T).fill(1 / T)),
  },
  {
    name: 'head 4 — next-token (acausal)',
    desc: 'Looks forward — only possible without a causal mask. Encoders use these; decoders cannot.',
    build: (T) => {
      const A = Array.from({ length: T }, () => new Array(T).fill(0));
      for (let i = 0; i < T; i++) {
        for (let j = 0; j < T; j++) {
          if (j === i + 1) A[i][j] = 0.75;
          else if (j === i) A[i][j] = 0.15;
          else A[i][j] = 0.1 / Math.max(1, T - 2);
        }
      }
      return A;
    },
  },
  {
    name: 'head 5 — pronoun → antecedent',
    desc: 'Bidirectional coreference: late tokens look back to noun phrases. Famous from BERT analyses.',
    build: (T) => {
      const A = Array.from({ length: T }, () => new Array(T).fill(0));
      for (let i = 0; i < T; i++) {
        for (let j = 0; j < T; j++) {
          if (j === 0) A[i][j] = 0.55;
          else if (j === i) A[i][j] = 0.25;
          else A[i][j] = 0.2 / Math.max(1, T - 2);
        }
      }
      return A;
    },
  },
  {
    name: 'head 6 — sink to first token',
    desc: 'Many trained transformers learn an "attention sink": a head that dumps unused mass on position 0.',
    build: (T) => {
      const A = Array.from({ length: T }, () => new Array(T).fill(0));
      for (let i = 0; i < T; i++) {
        for (let j = 0; j < T; j++) {
          if (j === 0) A[i][j] = 0.8;
          else A[i][j] = 0.2 / (T - 1);
        }
      }
      return A;
    },
  },
  {
    name: 'head 7 — local window (±1)',
    desc: 'Tight local mixing — close to a convolution. Common in low layers.',
    build: (T) => {
      const A = Array.from({ length: T }, () => new Array(T).fill(0));
      for (let i = 0; i < T; i++) {
        let s = 0;
        for (let j = 0; j < T; j++) {
          const d = Math.abs(i - j);
          A[i][j] = d === 0 ? 0.5 : d === 1 ? 0.22 : 0.02;
          s += A[i][j];
        }
        for (let j = 0; j < T; j++) A[i][j] /= s;
      }
      return A;
    },
  },
];

function Section4MultiHead() {
  const [numHeads, setNumHeads] = useState(4);
  const [sentenceId, setSentenceId] = useState('she');
  const sentence = SENTENCES.find((s) => s.id === sentenceId);
  const T = sentence.tokens.length;
  const heads = HEAD_PATTERNS.slice(0, numHeads).map((h) => ({
    ...h,
    A: h.build(T),
  }));

  return (
    <div className="viz-panel">
      <div className="viz-controls">
        <label>
          Sentence
          <select
            value={sentenceId}
            onChange={(e) => setSentenceId(e.target.value)}
          >
            {SENTENCES.map((s) => (
              <option key={s.id} value={s.id}>{s.tokens.join(' ')}</option>
            ))}
          </select>
        </label>
        <label>
          Heads: <strong style={{ marginLeft: 4 }}>{numHeads}</strong>
          <input
            type="range"
            min={1}
            max={8}
            step={1}
            value={numHeads}
            onChange={(e) => setNumHeads(parseInt(e.target.value, 10))}
            style={{ marginLeft: 8, width: 180 }}
          />
        </label>
      </div>

      <div className="viz-head-grid">
        {heads.map((h, idx) => {
          // Card inner width ≈ 213px (240 minmax minus 12px×2 padding).
          // Grid overhead: 38px row label + 4px grid padding + 2px×T gaps.
          // Solve for cell size: floor((213 − 42 − 2T) / T), clamped to [20, 36].
          const headCell = Math.max(20, Math.min(36, Math.floor((171 - 2 * T) / T)));
          return (
            <div className="viz-head-card" key={idx}>
              <div className="viz-head-title">{h.name}</div>
              <HeatmapGrid
                data={h.A}
                rowLabels={sentence.tokens}
                colLabels={sentence.tokens}
                cellWidth={headCell}
                vmin={0}
                vmax={1}
                showValues={false}
              />
              <div className="viz-head-desc">{h.desc}</div>
            </div>
          );
        })}
      </div>

      <p className="viz-caption">
        These patterns are hand-crafted to be readable — in a real model they
        emerge from gradient descent on next-token loss. The point is the{' '}
        <em>diversity</em>: heads decompose attention into specialised circuits.
      </p>

      <pre className="viz-code">
{`# Multi-head attention: split d_model into h heads of size d_k = d_model // h
Q = (X @ W_Q).view(T, h, d_k).transpose(0, 1)   # (h, T, d_k)
K = (X @ W_K).view(T, h, d_k).transpose(0, 1)
V = (X @ W_V).view(T, h, d_k).transpose(0, 1)
A = (Q @ K.transpose(-2, -1) / d_k**0.5).softmax(-1)
Y = (A @ V).transpose(0, 1).reshape(T, h * d_k)`}
      </pre>
    </div>
  );
}

/* =========================================================
   Interactive #5 — Causal masking
   ========================================================= */

function Section5Causal() {
  const [sentenceId, setSentenceId] = useState('she');
  const [masked, setMasked] = useState(true);
  const sentence = SENTENCES.find((s) => s.id === sentenceId);

  const { scores, A, maskMat } = useMemo(() => {
    const X = buildXForSentence(sentence);
    const Q = matmul(X, W_Q);
    const K = matmul(X, W_K);
    const raw = scaleMat(matmul(Q, transpose(K)), 1 / 2);
    const T = sentence.tokens.length;
    const maskMat = Array.from({ length: T }, (_, i) =>
      Array.from({ length: T }, (_, j) => (j > i ? -Infinity : 0))
    );
    const scoresM = masked ? applyCausalMask(raw) : raw;
    const A = softmaxRows(scoresM);
    return { scores: scoresM, A, maskMat };
  }, [sentence, masked]);

  return (
    <div className="viz-panel">
      <div className="viz-controls">
        <label>
          Sentence
          <select
            value={sentenceId}
            onChange={(e) => setSentenceId(e.target.value)}
          >
            {SENTENCES.map((s) => (
              <option key={s.id} value={s.id}>{s.tokens.join(' ')}</option>
            ))}
          </select>
        </label>
        <label className="viz-toggle">
          <input
            type="checkbox"
            checked={masked}
            onChange={(e) => setMasked(e.target.checked)}
          />
          Apply causal mask
        </label>
      </div>

      <div className="viz-matrix-row" style={{ overflowX: 'auto' }}>
        <HeatmapGrid
          data={maskMat}
          label="mask  (upper triangle = −∞)"
          rowLabels={sentence.tokens}
          colLabels={sentence.tokens}
          cellWidth={50}
          showValues={false}
        />
        <div className="viz-matrix-op">+</div>
        <HeatmapGrid
          data={scores}
          label="masked scores"
          rowLabels={sentence.tokens}
          colLabels={sentence.tokens}
          cellWidth={50}
          diverging
        />
        <div className="viz-matrix-op">softmax →</div>
        <HeatmapGrid
          data={A}
          label="A  (rows still sum to 1)"
          rowLabels={sentence.tokens}
          colLabels={sentence.tokens}
          cellWidth={50}
          vmin={0}
          vmax={1}
        />
      </div>

      <p className="viz-caption">
        After masking, position <em>i</em> can only attend to positions ≤ <em>i</em>.
        That is exactly what lets the decoder predict <strong>every</strong> next
        token in a single forward pass during training: each position's loss already
        depends only on prior context, so gradients don't leak future information.
      </p>

      <pre className="viz-code">
{`mask = torch.triu(torch.full((T, T), float("-inf")), diagonal=1)
scores = (Q @ K.transpose(-2, -1) / d_k**0.5) + mask
A = scores.softmax(dim=-1)
Y = A @ V`}
      </pre>
    </div>
  );
}

/* =========================================================
   §1 — Tokenization (precomputed splits, illustrative)
   ========================================================= */

const TOKENIZER_PRESETS = [
  {
    id: 'aiayn',
    text: 'Attention is all you need',
    splits: {
      char: 'Attention is all you need'.split(''),
      word: ['Attention', 'is', 'all', 'you', 'need'],
      subword: ['Att', 'ention', ' is', ' all', ' you', ' need'],
    },
  },
  {
    id: 'unhappy',
    text: 'unhappiness',
    splits: {
      char: 'unhappiness'.split(''),
      word: ['unhappiness'],
      subword: ['un', 'happi', 'ness'],
    },
  },
  {
    id: 'trans',
    text: 'transformer-based',
    splits: {
      char: 'transformer-based'.split(''),
      word: ['transformer-based'],
      subword: ['transform', 'er', '-', 'based'],
    },
  },
  {
    id: 'gpt',
    text: 'GPT-4 costs $0.03',
    splits: {
      char: 'GPT-4 costs $0.03'.split(''),
      word: ['GPT-4', 'costs', '$0.03'],
      subword: ['GPT', '-', '4', ' costs', ' $', '0', '.', '03'],
    },
  },
];

// Stable fake token IDs — same string always gets the same number.
const fakeTokenId = (s) => {
  let h = 0;
  for (let i = 0; i < s.length; i++) {
    h = (h * 31 + s.charCodeAt(i)) >>> 0;
  }
  return h % 49997;
};

function SectionTokens() {
  const [presetId, setPresetId] = useState('aiayn');
  const [mode, setMode] = useState('subword');
  const preset = TOKENIZER_PRESETS.find((p) => p.id === presetId);
  const tokens = preset.splits[mode];

  return (
    <div className="viz-panel">
      <div className="viz-controls">
        <label>
          Input
          <select
            value={presetId}
            onChange={(e) => setPresetId(e.target.value)}
          >
            {TOKENIZER_PRESETS.map((p) => (
              <option key={p.id} value={p.id}>{p.text}</option>
            ))}
          </select>
        </label>
        <div className="viz-tabs" role="tablist">
          {['char', 'word', 'subword'].map((m) => (
            <button
              key={m}
              role="tab"
              className={`viz-tab ${mode === m ? 'active' : ''}`}
              onClick={() => setMode(m)}
            >
              {m}
            </button>
          ))}
        </div>
      </div>

      <div className="viz-token-source">"{preset.text}"</div>

      <div className="viz-token-row">
        {tokens.map((t, i) => (
          <div className="viz-token-pill" key={i}>
            <div className="viz-token-text">{t === ' ' ? '␣' : t.replace(/^ /, '·')}</div>
            <div className="viz-token-id">{fakeTokenId(t)}</div>
          </div>
        ))}
      </div>

      <p className="viz-caption">
        <strong>{tokens.length} token{tokens.length === 1 ? '' : 's'}.</strong>{' '}
        Each pill is one entry in the model's vocabulary. The integer below it
        is the token ID — that's what actually flows into the network.
        Leading-space markers (<code>·</code>) are how real BPE tokenizers
        preserve word boundaries.
      </p>
    </div>
  );
}

/* =========================================================
   §2 — Architecture comparison: RNN / CNN / Attention
   ========================================================= */

const ARCHES = [
  {
    id: 'rnn',
    label: 'RNN',
    receptive: (pos, T) => {
      // RNN at position t sees all previous tokens, but with decaying strength.
      const arr = new Array(T).fill(0);
      for (let j = 0; j <= pos; j++) {
        arr[j] = Math.pow(0.7, pos - j); // decay backwards
      }
      return arr;
    },
    parallelSteps: (T) => T,
    description:
      'Reads left-to-right, carrying a hidden state. Token i can technically reach all earlier tokens, but information about token 1 is squashed through (i-1) updates by the time it arrives — that\'s the long-range decay (fading colour below).',
    flaw: 'Sequential: step i waits for step i−1. T tokens → T wall-clock steps.',
  },
  {
    id: 'cnn',
    label: 'CNN (k=3)',
    receptive: (pos, T) => {
      const arr = new Array(T).fill(0);
      for (let j = 0; j < T; j++) {
        if (Math.abs(j - pos) <= 1) arr[j] = 1;
      }
      return arr;
    },
    parallelSteps: () => 1,
    description:
      'Each layer sees only a small fixed window (kernel size 3 → ±1 neighbour). Reaching position 1 from position 100 requires stacking many layers — the receptive field grows linearly with depth.',
    flaw: 'Parallel within a layer, but global context demands depth.',
  },
  {
    id: 'attn',
    label: 'Attention',
    receptive: (pos, T) => new Array(T).fill(1),
    parallelSteps: () => 1,
    description:
      'Every token can attend to every other token in a single layer. No decay, no kernel limit — the receptive field is the whole sequence from layer 1.',
    flaw: 'Costs O(T²) in compute and memory. Manageable up to ~thousands of tokens.',
  },
];

function SectionHistory() {
  const [archId, setArchId] = useState('attn');
  const T = 7;
  const [focus, setFocus] = useState(3);
  const arch = ARCHES.find((a) => a.id === archId);
  const intensities = arch.receptive(focus, T);

  return (
    <div className="viz-panel">
      <div className="viz-controls">
        <div className="viz-tabs" role="tablist">
          {ARCHES.map((a) => (
            <button
              key={a.id}
              role="tab"
              className={`viz-tab ${archId === a.id ? 'active' : ''}`}
              onClick={() => setArchId(a.id)}
            >
              {a.label}
            </button>
          ))}
        </div>
        <span style={{ color: 'var(--ink-faint)', fontSize: '0.85rem' }}>
          Tap a token to see what <strong style={{ color: 'var(--ink)' }}>it</strong> can reach in <em>one</em> layer.
        </span>
      </div>

      <div className="viz-arch-stage">
        <div className="viz-arch-label">Receptive field (1 layer)</div>
        <div className="viz-arch-row">
          {Array.from({ length: T }, (_, i) => {
            const intensity = intensities[i];
            const isFocus = i === focus;
            return (
              <button
                key={i}
                className={`viz-arch-node ${isFocus ? 'viz-arch-focus' : ''}`}
                style={{
                  background: isFocus
                    ? 'var(--accent)'
                    : intensity > 0
                    ? `hsl(var(--accent-h), ${20 + intensity * 60}%, ${94 - intensity * 50}%)`
                    : '#fff',
                  color: isFocus ? '#fff' : intensity > 0.55 ? '#fff' : 'var(--ink)',
                }}
                onClick={() => setFocus(i)}
              >
                t{i + 1}
              </button>
            );
          })}
        </div>

        <div className="viz-arch-label" style={{ marginTop: 22 }}>Sequential dependency chain (T = {T})</div>
        <div className="viz-arch-row">
          {Array.from({ length: T }, (_, i) => (
            <React.Fragment key={i}>
              <div className="viz-arch-step">{i + 1}</div>
              {i < T - 1 && (
                <div className={`viz-arch-arrow ${archId === 'rnn' ? 'viz-arch-arrow-on' : 'viz-arch-arrow-off'}`}>→</div>
              )}
            </React.Fragment>
          ))}
        </div>
        <div className="viz-arch-clock">
          Wall-clock steps to compute one layer:{' '}
          <strong>{arch.parallelSteps(T)}</strong>
          {arch.parallelSteps(T) === 1 ? ' (all positions in parallel)' : ' (one at a time)'}
        </div>

        <p className="viz-caption" style={{ marginTop: 14 }}>
          <strong>{arch.label}.</strong> {arch.description} <em>{arch.flaw}</em>
        </p>
      </div>
    </div>
  );
}

/* =========================================================
   §3 — Embeddings: vocabulary lookup table
   ========================================================= */

function SectionEmbeddings() {
  const [sentenceId, setSentenceId] = useState('she');
  const [picked, setPicked] = useState(0);
  const sentence = SENTENCES.find((s) => s.id === sentenceId);
  const pickedTok = sentence.tokens[picked];

  // Show the unique vocabulary of the chosen sentence as the lookup table.
  const vocab = Array.from(new Set(sentence.tokens));

  return (
    <div className="viz-panel">
      <div className="viz-controls">
        <label>
          Sentence
          <select
            value={sentenceId}
            onChange={(e) => { setSentenceId(e.target.value); setPicked(0); }}
          >
            {SENTENCES.map((s) => (
              <option key={s.id} value={s.id}>{s.tokens.join(' ')}</option>
            ))}
          </select>
        </label>
        <span style={{ color: 'var(--ink-faint)', fontSize: '0.85rem' }}>
          Tap a token to look up its embedding.
        </span>
      </div>

      <div className="viz-token-row">
        {sentence.tokens.map((t, i) => (
          <button
            key={i}
            className={`viz-token-pill viz-token-pill-button ${i === picked ? 'viz-token-pill-active' : ''}`}
            onClick={() => setPicked(i)}
          >
            <div className="viz-token-text">{t}</div>
            <div className="viz-token-id">{fakeTokenId(t)}</div>
          </button>
        ))}
      </div>

      <div className="viz-matrix-row" style={{ marginTop: 18 }}>
        <div className="viz-mat">
          <div className="viz-mat-label">embedding table  (|V| × d_model)</div>
          <div
            className="viz-mat-grid"
            style={{ gridTemplateColumns: `64px repeat(8, 44px)` }}
          >
            <div className="viz-cell viz-col-label" />
            {['e0','e1','e2','e3','e4','e5','e6','e7'].map((c, j) => (
              <div key={j} className="viz-cell viz-col-label">{c}</div>
            ))}
            {vocab.map((tok) => {
              const isHit = tok === pickedTok;
              return (
                <React.Fragment key={tok}>
                  <div className={`viz-cell viz-row-label ${isHit ? 'viz-row-hi' : ''}`}>{tok}</div>
                  {EMB[tok].map((v, j) => (
                    <div
                      key={j}
                      className={`viz-cell ${isHit ? 'viz-trace-x' : ''}`}
                    >
                      {v.toFixed(2)}
                    </div>
                  ))}
                </React.Fragment>
              );
            })}
          </div>
        </div>
        <div className="viz-matrix-op">→</div>
        <MatrixGrid
          data={sentence.tokens.map((t) => EMB[t])}
          label="X  (one row per token)"
          rowLabels={sentence.tokens}
          colLabels={['e0','e1','e2','e3','e4','e5','e6','e7']}
          cellWidth={42}
          highlight={{ row: picked, col: null }}
        />
      </div>

      <p className="viz-caption">
        Each token ID indexes one row of the embedding table. Stack those rows
        in sentence order and you get the matrix <Katex tex="X" />. From here on,
        every operation is linear algebra — no strings, no integers.
      </p>
    </div>
  );
}

/* =========================================================
   §9 — Positional embeddings (sinusoidal, illustrative)
   ========================================================= */

// Original Transformer-paper sinusoidal positional encoding.
// PE[p, 2i]   = sin(p / 10000^(2i / d_model))
// PE[p, 2i+1] = cos(p / 10000^(2i / d_model))
function sinusoidalPE(pos, d) {
  const v = new Array(d);
  for (let i = 0; i < d; i++) {
    const pair = Math.floor(i / 2);
    const freq = Math.pow(10000, -(2 * pair) / d);
    v[i] = i % 2 === 0 ? Math.sin(pos * freq) : Math.cos(pos * freq);
  }
  return v;
}

// Precompute the first 16 positions for d_model = 8.
const POS_EMB = Array.from({ length: 16 }, (_, p) => sinusoidalPE(p, 8));

function SectionPositional() {
  const [sentenceId, setSentenceId] = useState('she');
  const [hoverRow, setHoverRow] = useState(null);
  const sentence = SENTENCES.find((s) => s.id === sentenceId);
  const T = sentence.tokens.length;
  const tokenEmb = sentence.tokens.map((t) => EMB[t]);
  const posEmb = Array.from({ length: T }, (_, p) => POS_EMB[p]);
  const xFinal = tokenEmb.map((row, i) =>
    row.map((v, j) => v + posEmb[i][j])
  );

  const colsE = ['e0','e1','e2','e3','e4','e5','e6','e7'];
  const posLabels = Array.from({ length: T }, (_, i) => `pos ${i}`);

  return (
    <div className="viz-panel">
      <div className="viz-controls">
        <label>
          Sentence
          <select
            value={sentenceId}
            onChange={(e) => { setSentenceId(e.target.value); setHoverRow(null); }}
          >
            {SENTENCES.map((s) => (
              <option key={s.id} value={s.id}>{s.tokens.join(' ')}</option>
            ))}
          </select>
        </label>
        <span style={{ color: 'var(--ink-faint)', fontSize: '0.85rem' }}>
          Hover or tap any row.
        </span>
      </div>

      <div className="viz-matrix-row">
        <MatrixGrid
          data={tokenEmb}
          label="token embedding"
          rowLabels={sentence.tokens}
          colLabels={colsE}
          cellWidth={42}
          highlight={hoverRow != null ? { row: hoverRow, col: null } : null}
          onCellEnter={(i) => setHoverRow(i)}
          onCellLeave={() => setHoverRow(null)}
          onCellTap={(i) => setHoverRow(i)}
        />
        <div className="viz-matrix-op">+</div>
        <MatrixGrid
          data={posEmb}
          label="positional embedding"
          rowLabels={posLabels}
          colLabels={colsE}
          cellWidth={42}
          highlight={hoverRow != null ? { row: hoverRow, col: null } : null}
          onCellEnter={(i) => setHoverRow(i)}
          onCellLeave={() => setHoverRow(null)}
          onCellTap={(i) => setHoverRow(i)}
        />
        <div className="viz-matrix-op">=</div>
        <MatrixGrid
          data={xFinal}
          label="X  (input to attention)"
          rowLabels={sentence.tokens}
          colLabels={colsE}
          cellWidth={46}
          highlight={hoverRow != null ? { row: hoverRow, col: null } : null}
          onCellEnter={(i) => setHoverRow(i)}
          onCellLeave={() => setHoverRow(null)}
          onCellTap={(i) => setHoverRow(i)}
        />
      </div>

      <p className="viz-caption">
        Each position has its own deterministic vector. Position <em>p</em>
        uses sines and cosines at different frequencies — the leftmost
        columns oscillate fast, the rightmost slowly, so any two positions
        produce a distinct fingerprint. The sum is what actually flows into
        the Q / K / V projections. (Yes, this means we glossed over a step
        back in §3 — apologies.)
      </p>

      <pre className="viz-code">
{`# Sinusoidal positional encoding (Vaswani et al. 2017)
pos = torch.arange(T).unsqueeze(1)          # (T, 1)
i   = torch.arange(0, d_model, 2)           # (d_model/2,)
freq = 10000 ** (-i / d_model)              # (d_model/2,)
PE = torch.zeros(T, d_model)
PE[:, 0::2] = torch.sin(pos * freq)
PE[:, 1::2] = torch.cos(pos * freq)
X = token_emb + PE                          # input to attention`}
      </pre>
    </div>
  );
}

/* =========================================================
   The post
   ========================================================= */

export default function VisualizingAttention() {
  // SEO + sharing
  useEffect(() => {
    const prevTitle = document.title;
    document.title = 'Visualizing Attention: Q/K/V, Multi-Head, and Causal Masking — Bernhard Walser';

    const setMeta = (name, content, attr = 'name') => {
      let el = document.querySelector(`meta[${attr}="${name}"]`);
      if (!el) {
        el = document.createElement('meta');
        el.setAttribute(attr, name);
        document.head.appendChild(el);
      }
      el.setAttribute('content', content);
      return el;
    };

    const description =
      'An interactive walk through self-attention: tokens, Q/K/V projections, scaled dot-product, multi-head, and causal masking — every matrix small enough to read.';

    setMeta('description', description);
    setMeta('og:title', 'Visualizing Attention: Q/K/V, Multi-Head, and Causal Masking', 'property');
    setMeta('og:description', description, 'property');
    setMeta('og:type', 'article', 'property');
    setMeta('twitter:card', 'summary_large_image');
    setMeta('twitter:title', 'Visualizing Attention');
    setMeta('twitter:description', description);

    return () => {
      document.title = prevTitle;
      // Leave meta tags in place — overwritten on next route change.
    };
  }, []);

  return (
    <article className="viz-post">
      <div className="viz-wide">
        <div className="viz-series-tag">Visualizing ML · Part 1</div>
        <h1>Visualizing Attention: Q/K/V, Multi-Head, and Causal Masking</h1>
        <p className="viz-lede">
          Consider the sentence <em>"The trophy didn't fit in the suitcase
          because it was too small."</em> What does <em>"it"</em> refer to?
          Swap <em>"small"</em> for <em>"big"</em> and the answer flips. To
          get this right, a model has to look across the whole sentence and
          decide, for every word, <strong>which other words matter</strong>.
          That mechanism is called <strong>attention</strong> — and it is the
          single most important idea in modern language models.
        </p>
        <div className="viz-byline">
          By <strong>Bernhard Walser</strong> · Senior ML Engineer ·{' '}
          <a className="viz-link" href="https://www.linkedin.com/in/bernhardwalser/" target="_blank" rel="noreferrer">LinkedIn</a>
          {' · '}
          <a className="viz-link" href="https://github.com/berniwal" target="_blank" rel="noreferrer">GitHub</a>
        </div>

        <p>
          Attention won out over the alternatives for two reasons. First, it is{' '}
          <strong>direct</strong>: any token can look at any other token in a
          single step, no matter how far apart they are in the sequence. Second,
          it is <strong>parallel</strong>: every token's lookup is computed
          independently and can be run at the same time on a GPU, instead of
          one-after-the-other like the recurrent networks that came before.
          Direct + parallel is what made transformers scale from millions of
          parameters to trillions.
        </p>
        <p>
          The post is built around interactive widgets. If you already know
          transformers, skip the prose — every cell on the page is one you can
          verify by eye, and every projection, softmax and mask is computed
          live in your browser. If you don't, the prose between widgets is
          enough to follow end-to-end without external reading.
        </p>

        <h2>1. What is a token?</h2>
        <p>
          A neural network does not see text. It sees integers. Before any
          computation, an input string is chopped into pieces called{' '}
          <strong>tokens</strong>, and each piece is mapped to an ID drawn
          from a fixed vocabulary (typically 30k–200k entries). The whole
          sequence becomes a list of IDs.
        </p>
        <p>
          How the string gets chopped is its own design choice. Whole-word
          tokenizers blow up on unseen words; character tokenizers make
          sequences long and slow; modern models use a learned middle ground
          called subword tokenization. Toggle the modes below to compare. The
          subword splits here are precomputed for illustration — real
          tokenizers learn the merges from data using{' '}
          <a className="viz-link" href="https://en.wikipedia.org/wiki/Byte_pair_encoding" target="_blank" rel="noreferrer">Byte-Pair Encoding</a>
          {' '}or similar. (A future post in this series will visualise that.)
        </p>
        <SectionTokens />

        <h2>2. Where attention came from</h2>
        <p>
          Attention did not appear out of nowhere. Before transformers, two
          families dominated sequence modelling, and each had a flaw that
          attention fixes directly.
        </p>
        <p>
          <strong>Recurrent networks (RNNs, LSTMs).</strong> Read the sequence
          one step at a time, carrying a hidden state forward. Powerful, but
          two problems: information about token 1 has to survive being
          repeatedly squashed through a tiny hidden state by the time it
          reaches token 100 — long-range dependencies decay. And training is
          sequential by construction: you cannot start step{' '}
          <Katex tex="t" /> until you finish step <Katex tex="t-1" />,
          which leaves modern parallel hardware idle.
        </p>
        <p>
          <strong>Convolutions.</strong>{' '}
          <a className="viz-link" href="https://arxiv.org/abs/1705.03122" target="_blank" rel="noreferrer">Convolutional sequence models</a>
          {' '}(Gehring et al., 2017) replace recurrence with stacked 1D
          convolutions. Each layer sees only a small fixed window (the kernel),
          so reaching across a long sentence requires stacking many layers.
          Parallel within a layer, but the receptive field is still local
          unless you go deep.
        </p>
        <p>
          <strong>Bahdanau attention.</strong>{' '}
          <a className="viz-link" href="https://arxiv.org/abs/1409.0473" target="_blank" rel="noreferrer">Bahdanau, Cho and Bengio (2014)</a>
          {' '}introduced attention as a side-mechanism: an RNN decoder could{' '}
          <em>look back</em> at any encoder position when producing each output
          word. That solved the long-range problem, but attention still rode
          on top of an RNN — so training stayed sequential.
        </p>
        <p>
          <strong>"Attention Is All You Need".</strong>{' '}
          <a className="viz-link" href="https://arxiv.org/abs/1706.03762" target="_blank" rel="noreferrer">Vaswani et al. (2017)</a>
          {' '}made the leap: keep attention, drop the recurrence and the
          convolutions entirely. The transformer is built almost only of
          attention layers and feed-forward layers. Compare the three below —
          tap any position to see what it can "reach" in a single layer, and
          which steps must happen sequentially.
        </p>
        <SectionHistory />

        <h2>3. From tokens to vectors: embeddings</h2>
        <p>
          A token ID by itself is just a number — and integers don't capture
          meaning (token <em>42</em> is not more "similar" to token <em>43</em>{' '}
          than to <em>9000</em>). So before attention can do anything, each
          token ID is converted into a vector of real numbers by looking it up
          in a learned table. That vector is the token's <strong>embedding</strong>.
        </p>
        <p>
          The embedding table is a matrix with one row per vocabulary entry,
          width <Katex tex="d_{model}" />. During training, gradient descent
          pushes embeddings of similar-meaning tokens close together in this
          high-dimensional space. The model never sees the original string
          again — everything downstream operates on these vectors.
        </p>
        <p>
          We use <Katex tex="d_{model}=8" /> here so every number is visible
          on the page. Real models use <Katex tex="d_{model}" /> of 768–12288.
          Tap any token in the sentence to see its lookup.
        </p>
        <SectionEmbeddings />

        <h2>4. Query, Key, Value</h2>
        <p>
          Now we have a matrix <Katex tex="X \in \mathbb{R}^{T \times d_{model}}" /> —
          one embedding row per token. Attention's first move is to project
          each row into three different vectors using three learned matrices:
        </p>
        <div className="viz-math-block">
          <Katex block tex="Q = X W_Q,\quad K = X W_K,\quad V = X W_V \;\in\; \mathbb{R}^{T \times d_k}" />
        </div>
        <p>
          The names matter. Think of <strong>queries</strong> as <em>"what am
          I looking for?"</em>, <strong>keys</strong> as <em>"what do I offer?"</em>,
          and <strong>values</strong> as <em>"what I will pass along if matched."</em>{' '}
          Each token plays all three roles at once: it is asking a question,
          advertising what it knows, and holding content to share. Three
          projections so the same token can do three different jobs.
        </p>
        <p>
          Hover or tap any cell of <Katex tex="Q" />, <Katex tex="K" />, or{' '}
          <Katex tex="V" /> to see the exact dot product that produced it.
        </p>
        <Section1QKV />

        <h2>5. Attention scores and why we divide by √d_k</h2>
        <p>
          Now we have <Katex tex="Q" />, <Katex tex="K" />, <Katex tex="V" />.
          The attention machinery has two jobs left: (1) decide, for every
          token, <em>how much</em> to attend to every other token, and (2) use
          those amounts to mix the values. This section does (1). The output
          is a matrix called <Katex tex="A" />, and the whole formula is:
        </p>
        <div className="viz-math-block">
          <Katex block tex="A = \mathrm{softmax}\!\left(\frac{Q K^{\top}}{\sqrt{d_k}}\right) \in \mathbb{R}^{T \times T}" />
        </div>
        <p>
          We'll do it in three pieces: two computational steps inside the
          widget, plus a quick detour to unpack softmax itself.
        </p>
        <Section2Scores />

        <h2>6. The output is a weighted mixture of V</h2>
        <p>
          A row of <Katex tex="A" /> sums to one, so the output for token{' '}
          <Katex tex="i" /> is a convex combination of value rows:
        </p>
        <div className="viz-math-block">
          <Katex block tex="Y_i = \sum_{j=1}^{T} A_{ij}\, V_j" />
        </div>
        <p>
          That is the whole computation. Pick an output row and watch the
          mixture build up: each value row's contribution is scaled by its
          attention weight, then summed. High-attention rows are visibly heavier;
          low-attention rows barely move the output.
        </p>
        <Section3Output />

        <h2>7. Multi-head: many attentions in parallel</h2>
        <p>
          One head can only model one kind of relationship at a time. Multi-head
          attention runs <Katex tex="h" /> attentions in parallel, each with its
          own projection matrices, then concatenates the outputs. The cost is
          identical — you split <Katex tex="d_{model}" /> across heads — but the
          model gets <Katex tex="h" /> independent ways to mix tokens.
        </p>
        <p>
          The heads below are <strong>hand-crafted</strong>, not trained, but
          they reflect well-documented patterns from actual models: a previous-token
          head, a subject↔verb head, a broadcast head, an attention sink. Slide
          to add heads.
        </p>
        <Section4MultiHead />

        <h2>8. Causal masking: the trick that makes parallel training work</h2>
        <p>
          A language model predicts the next token given the previous ones. If
          attention were unrestricted, token <Katex tex="i" />'s representation
          would already see token <Katex tex="i+1" />, and the prediction would
          be trivial. The fix is a causal mask: set the upper triangle of the
          score matrix to <Katex tex="-\infty" /> before softmax, so future
          tokens contribute exactly zero weight.
        </p>
        <p>
          Crucially, this is what makes training parallel. Every position's loss
          depends only on positions <Katex tex="\leq i" />, so the whole sequence
          can be processed in a single forward pass — yet each token is supervised
          on its own next-token prediction.
        </p>
        <Section5Causal />

        <h2>9. Positional embeddings: telling the model about order</h2>
        <p>
          There's one quiet problem with everything we've built. Attention is{' '}
          <strong>permutation-equivariant</strong>: if you shuffle the rows of{' '}
          <Katex tex="X" /> (the order of the tokens), every downstream
          quantity — <Katex tex="Q" />, <Katex tex="K" />, <Katex tex="V" />,
          the scores, the output — gets the same rows shuffled the same way.
          Nothing actually depends on <em>where</em> a token sits. That means
          "the cat sat on the mat" and "mat the cat sat on the" would produce
          the same set of output vectors. For a language model, that's a
          dealbreaker.
        </p>
        <p>
          The fix is dead simple: add a <strong>position vector</strong> to
          each token's embedding before any attention happens. Same dimension{' '}
          <Katex tex="d_{model}" />, one per position. After the addition,
          token <em>"the"</em> at position 0 is a different input vector from
          token <em>"the"</em> at position 4, so attention can tell them
          apart.
        </p>
        <p>
          Where do these position vectors come from? Two common choices.
          GPT and BERT use a <em>learned</em> table — just another lookup,
          one row per position, trained jointly with everything else. The
          original Transformer paper used a fixed{' '}
          <strong>sinusoidal</strong> pattern: sines and cosines at a range
          of frequencies, so each position gets a unique "fingerprint". We
          show the sinusoidal version below — it has visible structure you
          can see in the matrix.
        </p>
        <SectionPositional />

        <h2>What comes next</h2>
        <p>
          <strong>Better positional embeddings (RoPE).</strong> Both schemes
          above encode <em>absolute</em> positions, so a token at position 50
          has no innate relationship to one at position 51. Modern models use{' '}
          <strong>Rotary Position Embeddings</strong> (RoPE), which inject
          position by rotating <Katex tex="Q" /> and <Katex tex="K" /> in 2D
          pairs — the dot product then naturally encodes <em>relative</em>{' '}
          position. <em>Subject of the next post in this series.</em>
        </p>
        <p>
          <strong>KV cache.</strong> At training time we attend over the full
          sequence in one shot. At inference we generate one token at a time —
          and most of <Katex tex="K" /> and <Katex tex="V" /> hasn't changed
          since the last step. The KV cache stores them so each new token costs{' '}
          <Katex tex="O(T)" />, not <Katex tex="O(T^2)" />. <em>Coming in part 3.</em>
        </p>
        <p>
          <strong>Attention variants.</strong> Everything above is "vanilla"
          softmax attention with <Katex tex="O(T^2)" /> compute and memory.
          FlashAttention restructures the compute to be IO-aware without
          changing the math; sparse attention drops entries from the score
          matrix; mixture-of-experts routes whole token streams to different
          sub-networks. <em>All future posts in this series.</em>
        </p>

        <footer className="viz-footer">
          <p>
            <strong>Part 1 of Visualizing ML</strong> · Next: <em>The KV Cache —
            why inference is not training</em>. RSS feed coming with post #2.
          </p>
          <p style={{ marginBottom: 0 }}>
            Bernhard Walser · ML Engineer, Digitec Galaxus · ETH Computer Science ·{' '}
            <a className="viz-link" href="https://www.linkedin.com/in/bernhardwalser/" target="_blank" rel="noreferrer">LinkedIn</a>
            {' · '}
            <a className="viz-link" href="https://github.com/berniwal" target="_blank" rel="noreferrer">GitHub</a>
          </p>
        </footer>
      </div>
    </article>
  );
}
