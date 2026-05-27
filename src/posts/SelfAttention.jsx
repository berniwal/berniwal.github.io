// TAGS FOR REGISTRATION: ['attention', 'transformers']
// EXCERPT: How a single self-attention layer works — three projections (Q, K, V), a scaled dot-product, a softmax, and a weighted sum. Every number on the page is live; hover any cell to see the math that produced it.

// src/posts/SelfAttention.jsx
// Transformers · Part 2 of 3 — Inside Self-Attention: Q, K, V, and Softmax
// Ported from the original VisualizingAttention.jsx; prose preserved,
// design tokens aligned with the post-2026 redesign.
import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneLight } from 'react-syntax-highlighter/dist/esm/styles/prism';
import './PostChrome.css';
import './SelfAttention.css';

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
        setTimeout(render, 60);
      }
    };
    render();
    return () => { cancelled = true; };
  }, [tex, block]);
  return (
    <span
      ref={ref}
      className={`${block ? 'sa-math-block' : 'sa-math-inline'} ${className}`}
    />
  );
}

/* =========================================================
   Precomputed data — sentences, embeddings, projections
   ========================================================= */

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
   MatrixGrid — every cell tappable
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
  highlight,
}) {
  const cols = data[0].length;
  const hasRowLabels = !!rowLabels;
  const hasColLabels = !!colLabels;
  const gridCols = `${hasRowLabels ? '38px ' : ''}repeat(${cols}, ${cellWidth}px)`;

  return (
    <div className="sa-mat">
      {label && <div className="sa-mat-label">{label}</div>}
      <div
        className="sa-mat-grid"
        style={{ gridTemplateColumns: gridCols }}
        role="grid"
      >
        {hasColLabels && (
          <>
            {hasRowLabels && <div className="sa-cell sa-col-label" />}
            {colLabels.map((c, j) => (
              <div key={`c-${j}`} className="sa-cell sa-col-label">{c}</div>
            ))}
          </>
        )}
        {data.map((row, i) => (
          <React.Fragment key={`r-${i}`}>
            {hasRowLabels && (
              <div className="sa-cell sa-row-label">{rowLabels[i]}</div>
            )}
            {row.map((v, j) => {
              const isFocus = highlight && highlight.row === i && highlight.col === j;
              const isRowHi = highlight && highlight.row === i && highlight.col == null;
              const isColHi = highlight && highlight.col === j && highlight.row == null;
              const display = Number.isFinite(v)
                ? v.toFixed(decimals)
                : (v === -Infinity ? '−∞' : '·');
              const interactive = !!(onCellEnter || onCellTap);
              return (
                <div
                  key={`c-${i}-${j}`}
                  className={[
                    'sa-cell',
                    interactive ? 'sa-cell-tappable' : '',
                    isFocus ? 'sa-focus' : '',
                    isRowHi ? 'sa-row-hi' : '',
                    isColHi ? 'sa-col-hi' : '',
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
   HeatmapGrid — saturation encodes magnitude
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
      const norm = (v - lo) / span;
      const t = (norm - 0.5) * 2;
      if (t >= 0) {
        const sat = 18 + Math.round(t * 60);
        const light = 96 - Math.round(t * 46);
        return `hsl(218, ${sat}%, ${light}%)`;
      } else {
        const sat = 18 + Math.round(-t * 50);
        const light = 96 - Math.round(-t * 40);
        return `hsl(8, ${sat}%, ${light}%)`;
      }
    }
    const norm = (v - lo) / span;
    const sat = 18 + Math.round(norm * 60);
    const light = 96 - Math.round(norm * 50);
    return `hsl(218, ${sat}%, ${light}%)`;
  };

  const textColor = (v) => {
    if (!Number.isFinite(v)) return '#FFF';
    const norm = (v - lo) / span;
    return norm > 0.55 ? '#FFF' : 'var(--post-text)';
  };

  return (
    <div className="sa-mat">
      {label && <div className="sa-mat-label">{label}</div>}
      <div
        className="sa-mat-grid"
        style={{ gridTemplateColumns: gridCols }}
      >
        <div className="sa-cell sa-col-label" />
        {colLabels.map((c, j) => (
          <div key={`hc-${j}`} className="sa-cell sa-col-label">{c}</div>
        ))}
        {data.map((row, i) => (
          <React.Fragment key={`hr-${i}`}>
            <div className="sa-cell sa-row-label">{rowLabels[i]}</div>
            {row.map((v, j) => {
              const isActive = activeRow === i;
              const display = Number.isFinite(v)
                ? v.toFixed(decimals)
                : '−∞';
              return (
                <div
                  key={`hc-${i}-${j}`}
                  className={[
                    'sa-cell',
                    'sa-heat',
                    'sa-cell-tappable',
                    isActive ? 'sa-row-hi' : '',
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
   Widget 1 — Tokens -> Embeddings -> Q, K, V
   ========================================================= */

function Section1QKV() {
  const [sentenceId, setSentenceId] = useState('she');
  const [hover, setHover] = useState(null);
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

  const subsections = [
    { mat: 'Q', tag: 'q', title: 'Queries', sub: '"what am I looking for?"', W: W_Q, M: Q, wLabel: 'W_Q', mLabel: 'Q' },
    { mat: 'K', tag: 'k', title: 'Keys',    sub: '"what do I offer?"',       W: W_K, M: K, wLabel: 'W_K', mLabel: 'K' },
    { mat: 'V', tag: 'v', title: 'Values',  sub: "\"what I'll pass along if matched\"", W: W_V, M: V, wLabel: 'W_V', mLabel: 'V' },
  ];

  return (
    <div className="viz-panel">
      <div className="sa-controls">
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
        <span className="sa-dim-note">
          d<sub>model</sub> = 8 &nbsp;·&nbsp; d<sub>k</sub> = 4
        </span>
      </div>

      {subsections.map((s) => (
        <div className="sa-subsection" key={s.mat}>
          <h4 className="sa-subhead">
            <span className={`sa-sub-tag sa-sub-tag-${s.tag}`}>{s.mat}</span>
            {s.title} — <em>{s.sub}</em>
          </h4>
          <div className="sa-matrix-row">
            <MatrixGrid
              data={X}
              label="X  (T × d_model)"
              rowLabels={sentence.tokens}
              colLabels={colLabelsX}
              cellWidth={42}
              highlight={hover && hover.mat === s.mat ? { row: hover.i, col: null } : null}
            />
            <div className="sa-matrix-op">·</div>
            <MatrixGrid
              data={s.W}
              label={`${s.wLabel}  (d_model × d_k)`}
              colLabels={colLabelsW}
              cellWidth={48}
              highlight={hover && hover.mat === s.mat ? { row: null, col: hover.j } : null}
            />
            <div className="sa-matrix-op">=</div>
            <MatrixGrid
              data={s.M}
              label={`${s.mLabel}  (T × d_k)`}
              rowLabels={sentence.tokens}
              colLabels={colLabelsW}
              cellWidth={50}
              highlight={buildHi(s.mat)}
              onCellEnter={(i, j) => setHover({ mat: s.mat, i, j })}
              onCellLeave={() => setHover(null)}
              onCellTap={(i, j) => setHover({ mat: s.mat, i, j })}
            />
          </div>
        </div>
      ))}

      <div className="sa-explain">
        {explanation || (
          <span className="sa-explain-empty">
            Hover or tap any cell in Q, K, or V to see the dot product that produced it.
          </span>
        )}
      </div>
    </div>
  );
}

/* =========================================================
   Softmax mini-demo
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
    <div className="sa-softmax-demo">
      <div className="sa-softmax-tabs">
        {SOFTMAX_PRESETS.map((p, i) => (
          <button
            key={i}
            className={`sa-tab ${i === presetIdx ? 'active' : ''}`}
            onClick={() => setPresetIdx(i)}
          >
            {p.label}
          </button>
        ))}
      </div>
      <div className="sa-softmax-row">
        <div className="sa-softmax-side">
          <div className="sa-softmax-side-label">input numbers</div>
          {vals.map((v, i) => (
            <div className="sa-softmax-bar-row" key={i}>
              <span className="sa-softmax-num">{v.toFixed(1)}</span>
              <span
                className="sa-softmax-bar sa-softmax-bar-in"
                style={{ width: Math.max(2, ((v - inMin) / inSpan) * maxBar) }}
              />
            </div>
          ))}
        </div>
        <div className="sa-softmax-arrow">softmax →</div>
        <div className="sa-softmax-side">
          <div className="sa-softmax-side-label">probabilities (sum = 1.0)</div>
          {probs.map((p, i) => (
            <div className="sa-softmax-bar-row" key={i}>
              <span className="sa-softmax-num">{(p * 100).toFixed(1)}%</span>
              <span
                className="sa-softmax-bar sa-softmax-bar-out"
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
   Widget 2 — Scores QK^T / sqrt(d_k) and softmax
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

  const { Q, Kt, raw, scaled, A } = useMemo(() => {
    const X = buildXForSentence(sentence);
    const Q = matmul(X, W_Q);
    const K = matmul(X, W_K);
    const Kt = transpose(K);
    const raw = matmul(Q, Kt);
    const scaled = scaleMat(raw, 1 / TEMPS[tempIdx].div);
    const A = softmaxRows(scaled);
    return { Q, Kt, raw, scaled, A };
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
      <div className="sa-controls">
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

      <div className="sa-subsection">
        <h4 className="sa-subhead">
          <span className="sa-sub-tag sa-sub-tag-step">1</span>
          Compute raw scores: <Katex tex="Q \cdot K^{\top}" />
        </h4>
        <p className="sa-subhead-prose">
          Every query row in <Katex tex="Q" /> is dotted with every key row in{' '}
          <Katex tex="K" /> (i.e. every <em>column</em> of <Katex tex="K^{\top}" />).
          The result is a <Katex tex="T \times T" /> matrix where entry{' '}
          <Katex tex="(i, j)" /> is a single number: <em>how compatible is
          token i's question with token j's offering?</em>
        </p>
        <div className="sa-matrix-row">
          <MatrixGrid
            data={Q}
            label="Q  (T × d_k)"
            rowLabels={sentence.tokens}
            colLabels={colLabelsW}
            cellWidth={50}
          />
          <div className="sa-matrix-op">·</div>
          <MatrixGrid
            data={Kt}
            label="Kᵀ  (d_k × T)"
            rowLabels={colLabelsW}
            colLabels={sentence.tokens}
            cellWidth={50}
          />
          <div className="sa-matrix-op">=</div>
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

      <div className="sa-subsection">
        <h4 className="sa-subhead">
          <span className="sa-sub-tag sa-sub-tag-step">2</span>
          Scale, then softmax → attention matrix <Katex tex="A" />
        </h4>
        <p className="sa-subhead-prose">
          Divide the raw scores by <Katex tex="\sqrt{d_k}" />, then apply{' '}
          <strong>softmax</strong> row-by-row. Softmax turns each row of
          arbitrary real numbers into a probability distribution (every entry
          positive, every row summing to 1). The output <Katex tex="A" /> is
          the <strong>attention matrix</strong>: row <Katex tex="i" /> is
          literally "how much should token <Katex tex="i" /> pay attention to
          each other token?" — those weights are what we'll use in the next
          step to mix the value vectors.
        </p>
        <div className="sa-controls" style={{ marginTop: 4 }}>
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
        <div className="sa-matrix-row">
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
          <div className="sa-matrix-op">softmax →</div>
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
        <div className="sa-row-meta">
          {meta || (
            <span style={{ fontStyle: 'italic' }}>
              Hover or tap any row of <strong>A</strong> to see what that token attends to.
            </span>
          )}
        </div>
      </div>

      <div className="sa-subsection">
        <h4 className="sa-subhead">
          <span className="sa-sub-tag sa-sub-tag-star">★</span>
          What is softmax, exactly?
        </h4>
        <p className="sa-subhead-prose">
          You just used softmax — let's unpack it. Softmax turns a list of
          arbitrary real numbers into a probability distribution. Why that
          function, specifically? Two reasons. First, attention weights must
          be non-negative — you can't attend by a negative amount. Second,
          they need to sum to 1, so the output in the next section is a clean
          weighted average instead of arbitrarily scaled. Softmax delivers
          both via:
        </p>
        <div className="sa-math-block">
          <Katex block tex="\mathrm{softmax}(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}}" />
        </div>
        <p className="sa-subhead-prose">
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
    </div>
  );
}

/* =========================================================
   Widget 3 — Output = A · V with animated mixing
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
      <div className="sa-controls">
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

      <div className="sa-matrix-row" style={{ overflowX: 'auto' }}>
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
        <div className="sa-matrix-op">·</div>
        <MatrixGrid
          data={V}
          label="V"
          rowLabels={sentence.tokens}
          colLabels={['v0','v1','v2','v3']}
          cellWidth={50}
        />
        <div className="sa-matrix-op">=</div>
        <MatrixGrid
          data={Y}
          label="Y = A·V"
          rowLabels={sentence.tokens}
          colLabels={['y0','y1','y2','y3']}
          cellWidth={50}
          highlight={{ row: pickedRow, col: null }}
        />
      </div>

      <div className="sa-mix-stage" key={animKey}>
        <div className="sa-mix-header">
          Y[<strong>{sentence.tokens[pickedRow]}</strong>] = Σⱼ A[{pickedRow},j] · V[j]
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
    <div className={`sa-mix-row${on ? ' sa-mix-active' : ''}`}>
      <span className="sa-mix-weight">{(weight * 100).toFixed(1)}%</span>
      <span style={{ display: 'inline-block', width: 240 }}>
        <span className="sa-mix-bar" style={{ display: 'block', width: barWidth }} />
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
    <div className={`sa-mix-row sa-mix-output${on ? ' sa-mix-active' : ''}`}>
      <span className="sa-mix-weight">=</span>
      <span style={{ display: 'inline-block', width: 240 }} />
      <span style={{ minWidth: 60 }}>{token}</span>
      <span>[{yRow.map((v) => v.toFixed(2)).join(', ')}]</span>
    </div>
  );
}

/* =========================================================
   Code block — canonical Python self-attention
   ========================================================= */

const CODE = `import math
import torch.nn.functional as F

# X: (T, d_model), W_Q / W_K / W_V: (d_model, d_k)
Q = X @ W_Q                                  # (T, d_k)
K = X @ W_K                                  # (T, d_k)
V = X @ W_V                                  # (T, d_k)

scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)   # (T, T)
A = F.softmax(scores, dim=-1)                       # rows sum to 1
Y = A @ V                                           # (T, d_k)
`;

function CodeBlock() {
  return (
    <details className="post-code-details">
      <summary>
        <span className="post-code-summary-icon" aria-hidden="true">›</span>
        <span>Show self-attention in PyTorch</span>
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

/* =========================================================
   Page
   ========================================================= */

export default function SelfAttention() {
  useEffect(() => {
    const prev = document.title;
    document.title = 'Inside Self-Attention — Bernhard Walser';
    return () => { document.title = prev; };
  }, []);

  return (
    <article className="post-2026 sa-post">
      <div className="post-wrap">
        <div className="post-hero">
          <div className="post-series-tag">
            <span className="post-live-dot" aria-hidden="true" />Transformers · Part 2 of 3
          </div>
          <h1>Inside Self-Attention: Q, K, V, and Softmax</h1>
          <p className="post-lede">
            <em>"The trophy didn't fit in the suitcase because it was too small."</em>{' '}
            Which word does <em>it</em> refer to? Swap "small" for "big" and the answer
            flips. To get this right, a model has to decide, for every word in the
            sentence, which other words matter — and how much. That decision is the
            job of self-attention.
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
          In <a className="post-link" href="#/blog/nlp-history">Part 1</a> we
          walked the road from one-hot vectors and TF-IDF up to the moment
          attention arrives. This post zooms into one self-attention layer —
          three matrices, a dot product, a softmax, a weighted sum — and shows
          every number along the way. Hover anything to see the math.
        </p>

        <h2 className="reveal">Query, Key, Value</h2>
        <p>
          Self-attention starts by projecting each token's embedding into{' '}
          <em>three</em> different views of itself. Three learned weight
          matrices — <Katex tex="W_Q" />, <Katex tex="W_K" />, <Katex tex="W_V" /> —
          turn the input <Katex tex="X" /> (shape <Katex tex="T \times d_{model}" />)
          into <Katex tex="Q" />, <Katex tex="K" />, and <Katex tex="V" />:
        </p>
        <div className="sa-math-block">
          <Katex block tex="Q = X W_Q, \quad K = X W_K, \quad V = X W_V" />
        </div>
        <p>
          The names come from databases, and the metaphor sticks:{' '}
          <strong>Q</strong> is what the current token is <em>looking for</em>;{' '}
          <strong>K</strong> is what each token <em>advertises</em> to others;{' '}
          <strong>V</strong> is the content it <em>contributes</em> when matched.
          Same input, three roles — and the model gets to learn what each role
          should encode.
        </p>
        <Section1QKV />
        <p>
          The three matrices are independent. Nothing forces <Katex tex="Q" /> and{' '}
          <Katex tex="K" /> to look alike — and that's the point. Training pushes
          them apart so the dot product in the next step picks up on{' '}
          <em>relationships between words</em>, not just word identity.
        </p>

        <h2 className="reveal">Scores, scaling, and the softmax</h2>
        <p>
          With <Katex tex="Q" /> and <Katex tex="K" /> in hand, the model asks a
          single question for every pair of tokens: <em>does query i match key j?</em>{' '}
          A dot product answers it. Stack all pairs into a matrix and you get{' '}
          <Katex tex="QK^{\top}" />, a <Katex tex="T \times T" /> grid of compatibility scores.
        </p>
        <p>
          Two more steps turn raw scores into usable weights. First, divide by{' '}
          <Katex tex="\sqrt{d_k}" />. As <Katex tex="d_k" /> grows, dot products
          grow with it, and softmax saturates into a near-one-hot distribution —
          all the mass on one token, gradients vanish. The <Katex tex="\sqrt{d_k}" />{' '}
          divisor keeps the variance roughly constant so training stays healthy.
          Second, apply softmax row-by-row so every row sums to 1:
        </p>
        <div className="sa-math-block">
          <Katex block tex="A = \mathrm{softmax}\!\left(\frac{Q K^{\top}}{\sqrt{d_k}}\right)" />
        </div>
        <p>
          That matrix <Katex tex="A" /> is the <strong>attention matrix</strong>.
          Row <Katex tex="i" /> tells you how much token <Katex tex="i" /> attends
          to every other token — a probability distribution over the sentence.
          The widget below lets you slide the scaling and watch the rows sharpen
          or flatten in real time.
        </p>
        <Section2Scores />
        <p>
          Drag the slider from "no scaling" to "÷ 2·√d_k". With no scaling, one
          token wins almost every row and the rest get nothing. Scale too
          aggressively and every row becomes nearly uniform — the model attends
          to everything equally, which is the same as attending to nothing.
          The Vaswani choice of <Katex tex="\sqrt{d_k}" /> sits in the middle.
        </p>

        <h2 className="reveal">Compute the output</h2>
        <p>
          The last step is the easy one. We have weights ({" "}
          <Katex tex="A" />) and content (<Katex tex="V" />). Multiply:
        </p>
        <div className="sa-math-block">
          <Katex block tex="Y = A V" />
        </div>
        <p>
          Each row of <Katex tex="Y" /> is a weighted sum of <Katex tex="V" />'s
          rows, with weights from the corresponding row of <Katex tex="A" />.
          That's it — that's self-attention. Pick an output row in the widget
          and watch it assemble itself from the value vectors of every other
          token.
        </p>
        <Section3Output />
        <p>
          Every output token is now a custom blend of the whole sentence,
          weighted by relevance. The pronoun <em>it</em> in our opening sentence
          ends up looking partly like <em>trophy</em> and partly like{' '}
          <em>suitcase</em> — whichever the attention scores favoured. That blend
          is what gets passed up to the next layer.
        </p>

        <h2 className="reveal">Code</h2>
        <p>
          The whole layer is about six lines of PyTorch. Same matrices, same
          softmax, same weighted sum.
        </p>
        <CodeBlock />

        <h2 className="reveal">What's missing</h2>
        <p>
          What you just saw is one head of attention, in one layer, with no
          positional information and no masking. A real transformer stacks{' '}
          <strong>multi-head attention</strong> (several Q/K/V projections in
          parallel, each free to learn a different pattern),{' '}
          <strong>causal masking</strong> (so a token can't peek at the future),{' '}
          <strong>positional encodings</strong> (so order survives the
          permutation-invariant dot product), and an <strong>MLP</strong> after
          each attention block. Part 3 puts the pieces together.
        </p>

        <h2 className="reveal">References</h2>
        <div className="post-refs">
          <div className="head ref-cite">Cite</div>
          <div className="head ref-title">Title</div>
          <div className="head ref-link">Link</div>

          <div className="ref-cite">Vaswani et al., 2017</div>
          <div>
            <div className="ref-title">
              <a href="https://arxiv.org/abs/1706.03762" target="_blank" rel="noreferrer">Attention Is All You Need</a>
            </div>
            <div className="ref-note">The transformer paper. Scaled dot-product attention is §3.2.1.</div>
          </div>
          <div className="ref-link"><a href="https://arxiv.org/abs/1706.03762" target="_blank" rel="noreferrer">arxiv.org</a></div>

          <div className="ref-cite">Bahdanau, Cho, Bengio, 2014</div>
          <div>
            <div className="ref-title">
              <a href="https://arxiv.org/abs/1409.0473" target="_blank" rel="noreferrer">Neural Machine Translation by Jointly Learning to Align and Translate</a>
            </div>
            <div className="ref-note">The original attention paper — pre-transformer, in an encoder–decoder RNN.</div>
          </div>
          <div className="ref-link"><a href="https://arxiv.org/abs/1409.0473" target="_blank" rel="noreferrer">arxiv.org</a></div>

          <div className="ref-cite">Alammar, 2018</div>
          <div>
            <div className="ref-title">
              <a href="https://jalammar.github.io/illustrated-transformer/" target="_blank" rel="noreferrer">The Illustrated Transformer</a>
            </div>
            <div className="ref-note">A friendly visual walkthrough that influenced this series.</div>
          </div>
          <div className="ref-link"><a href="https://jalammar.github.io/illustrated-transformer/" target="_blank" rel="noreferrer">jalammar.github.io</a></div>
        </div>

        <footer className="post-footer">
          <p>
            <strong>Part 2 of Transformers</strong> · Previous:{' '}
            <a className="post-link" href="#/blog/nlp-history">From TF-IDF to Attention</a>
            {' · '}Next:{' '}
            <a className="post-link" href="#/blog/from-attention-to-transformer">From Attention to Transformer</a>.
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
