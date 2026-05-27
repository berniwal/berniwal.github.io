// TAGS FOR REGISTRATION: ['transformers', 'architecture']
// EXCERPT: A single attention operation isn't yet a transformer. Add multiple heads, an MLP per token, a causal mask, and positional information — and you have the block that powers every modern language model.

// src/posts/FromAttentionToTransformer.jsx
// Transformers · Part 3 of 3 — From Attention to Transformer.
// Ported from the original VisualizingAttention post: multi-head, causal
// masking, positional embeddings. Adds a new MLP section.
import React, { useEffect, useMemo, useRef, useState } from 'react';
import usePageMeta from '../usePageMeta';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneLight } from 'react-syntax-highlighter/dist/esm/styles/prism';
import './PostChrome.css';
import './FromAttentionToTransformer.css';

/* =========================================================
   Math helpers — kept naive (T, d are tiny by design)
   ========================================================= */
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
            displayMode: block, throwOnError: false,
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
      className={`${block ? 'fat-math-block' : 'fat-math-inline'} ${className}`}
    />
  );
}

/* =========================================================
   Precomputed embeddings, projections (carried over so we can
   show real masked-attention numbers).
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

const buildXForSentence = (sent) => sent.tokens.map((t) => EMB[t]);

/* =========================================================
   MatrixGrid — numeric matrix (used by positional widget)
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
    <div className="fat-mat">
      {label && <div className="fat-mat-label">{label}</div>}
      <div
        className="fat-mat-grid"
        style={{ gridTemplateColumns: gridCols }}
        role="grid"
      >
        {hasColLabels && (
          <>
            {hasRowLabels && <div className="fat-cell fat-col-label" />}
            {colLabels.map((c, j) => (
              <div key={`c-${j}`} className="fat-cell fat-col-label">{c}</div>
            ))}
          </>
        )}
        {data.map((row, i) => (
          <React.Fragment key={`r-${i}`}>
            {hasRowLabels && (
              <div className="fat-cell fat-row-label">{rowLabels[i]}</div>
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
                    'fat-cell',
                    interactive ? 'fat-cell-tappable' : '',
                    isFocus ? 'fat-focus' : '',
                    isRowHi ? 'fat-row-hi' : '',
                    isColHi ? 'fat-col-hi' : '',
                  ].filter(Boolean).join(' ')}
                  style={{ width: cellWidth }}
                  onMouseEnter={onCellEnter ? () => onCellEnter(i, j) : undefined}
                  onMouseLeave={onCellLeave ? () => onCellLeave(i, j) : undefined}
                  onClick={onCellTap ? () => onCellTap(i, j) : undefined}
                  tabIndex={interactive ? 0 : -1}
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
    if (!Number.isFinite(v)) return '#1e293b';
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
    return norm > 0.55 ? '#FFF' : '#0f172a';
  };

  return (
    <div className="fat-mat">
      {label && <div className="fat-mat-label">{label}</div>}
      <div
        className="fat-mat-grid"
        style={{ gridTemplateColumns: gridCols }}
      >
        <div className="fat-cell fat-col-label" />
        {colLabels.map((c, j) => (
          <div key={`hc-${j}`} className="fat-cell fat-col-label">{c}</div>
        ))}
        {data.map((row, i) => (
          <React.Fragment key={`hr-${i}`}>
            <div className="fat-cell fat-row-label">{rowLabels[i]}</div>
            {row.map((v, j) => {
              const isActive = activeRow === i;
              const display = Number.isFinite(v) ? v.toFixed(decimals) : '−∞';
              return (
                <div
                  key={`hc-${i}-${j}`}
                  className={[
                    'fat-cell', 'fat-heat',
                    onRowTap ? 'fat-cell-tappable' : '',
                    isActive ? 'fat-row-hi' : '',
                  ].filter(Boolean).join(' ')}
                  style={{
                    width: cellWidth,
                    background: cellColor(v),
                    color: textColor(v),
                  }}
                  onMouseEnter={onRowEnter ? () => onRowEnter(i) : undefined}
                  onMouseLeave={onRowLeave ? () => onRowLeave(i) : undefined}
                  onClick={onRowTap ? () => onRowTap(i) : undefined}
                  tabIndex={onRowTap ? 0 : -1}
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
   Widget 1 — Multi-head attention
   ========================================================= */
const HEAD_PATTERNS = [
  {
    name: 'head 0 — diagonal / identity',
    desc: 'A weak baseline: each token mostly looks at itself. Often the first thing heads learn at init.',
    build: (T) => {
      const A = Array.from({ length: T }, () => new Array(T).fill(0));
      for (let i = 0; i < T; i++)
        for (let j = 0; j < T; j++) A[i][j] = i === j ? 0.7 : 0.3 / (T - 1);
      return A;
    },
  },
  {
    name: 'head 1 — previous-token',
    desc: 'A classic head: each token attends to the one immediately before it. Useful for bigram-style features.',
    build: (T) => {
      const A = Array.from({ length: T }, () => new Array(T).fill(0));
      for (let i = 0; i < T; i++)
        for (let j = 0; j < T; j++) {
          if (j === i - 1) A[i][j] = 0.75;
          else if (j === i) A[i][j] = 0.15;
          else A[i][j] = 0.1 / Math.max(1, T - 2);
        }
      return A;
    },
  },
  {
    name: 'head 2 — subject ↔ verb',
    desc: 'Picks out a long-range syntactic dependency: verbs look back at their subject. Real heads often specialize like this.',
    build: (T) => {
      const A = Array.from({ length: T }, () => new Array(T).fill(0));
      const target = Math.min(1, T - 1);
      for (let i = 0; i < T; i++)
        for (let j = 0; j < T; j++) {
          if (j === target) A[i][j] = 0.6;
          else if (j === i) A[i][j] = 0.2;
          else A[i][j] = 0.2 / Math.max(1, T - 2);
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
      for (let i = 0; i < T; i++)
        for (let j = 0; j < T; j++) {
          if (j === i + 1) A[i][j] = 0.75;
          else if (j === i) A[i][j] = 0.15;
          else A[i][j] = 0.1 / Math.max(1, T - 2);
        }
      return A;
    },
  },
  {
    name: 'head 5 — pronoun → antecedent',
    desc: 'Bidirectional coreference: late tokens look back to noun phrases. Famous from BERT analyses.',
    build: (T) => {
      const A = Array.from({ length: T }, () => new Array(T).fill(0));
      for (let i = 0; i < T; i++)
        for (let j = 0; j < T; j++) {
          if (j === 0) A[i][j] = 0.55;
          else if (j === i) A[i][j] = 0.25;
          else A[i][j] = 0.2 / Math.max(1, T - 2);
        }
      return A;
    },
  },
  {
    name: 'head 6 — sink to first token',
    desc: 'Many trained transformers learn an "attention sink": a head that dumps unused mass on position 0.',
    build: (T) => {
      const A = Array.from({ length: T }, () => new Array(T).fill(0));
      for (let i = 0; i < T; i++)
        for (let j = 0; j < T; j++)
          A[i][j] = j === 0 ? 0.8 : 0.2 / (T - 1);
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

function MultiHeadWidget() {
  const [numHeads, setNumHeads] = useState(4);
  const [sentenceId, setSentenceId] = useState('she');
  const sentence = SENTENCES.find((s) => s.id === sentenceId);
  const T = sentence.tokens.length;
  const heads = HEAD_PATTERNS.slice(0, numHeads).map((h) => ({
    ...h, A: h.build(T),
  }));

  return (
    <div className="viz-panel">
      <div className="fat-controls">
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
            type="range" min={1} max={8} step={1} value={numHeads}
            onChange={(e) => setNumHeads(parseInt(e.target.value, 10))}
            style={{ marginLeft: 8, width: 180 }}
          />
        </label>
      </div>

      <div className="fat-head-grid">
        {heads.map((h, idx) => {
          const headCell = Math.max(20, Math.min(36, Math.floor((171 - 2 * T) / T)));
          return (
            <div className="fat-head-card" key={idx}>
              <div className="fat-head-title">{h.name}</div>
              <HeatmapGrid
                data={h.A}
                rowLabels={sentence.tokens}
                colLabels={sentence.tokens}
                cellWidth={headCell}
                vmin={0}
                vmax={1}
                showValues={false}
              />
              <div className="fat-head-desc">{h.desc}</div>
            </div>
          );
        })}
      </div>

      <p className="viz-caption">
        These patterns are hand-crafted to be readable — in a real model they
        emerge from gradient descent on next-token loss. The point is the{' '}
        <em>diversity</em>: heads decompose attention into specialised circuits.
      </p>
    </div>
  );
}

/* =========================================================
   Widget 2 — MLP (new). A small SVG: Linear → GeLU → Linear.
   Slide d_model to see the parameter count of one MLP block.
   ========================================================= */
function MLPWidget() {
  const [d, setD] = useState(512);
  const dff = d * 4;
  const params = d * dff + dff * d; // two weight matrices (bias ignored)

  // sample GeLU curve
  const gelu = (x) => 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3)));
  const xs = Array.from({ length: 41 }, (_, i) => -3 + (i * 6) / 40);
  const pts = xs.map((x) => [x, gelu(x)]);
  const px = (x) => 20 + ((x + 3) / 6) * 100;
  const py = (y) => 60 - ((y + 1) / 4) * 50;
  const path = pts
    .map(([x, y], i) => `${i === 0 ? 'M' : 'L'} ${px(x).toFixed(1)} ${py(y).toFixed(1)}`)
    .join(' ');

  return (
    <div className="viz-panel fat-mlp">
      <div className="fat-mlp-diagram">
        <svg viewBox="0 0 640 160" width="100%" preserveAspectRatio="xMidYMid meet">
          {/* boxes */}
          <g className="fat-mlp-box">
            <rect x="20" y="50" width="100" height="60" rx="10" />
            <text x="70" y="78" className="fat-mlp-box-label">token</text>
            <text x="70" y="98" className="fat-mlp-box-dim">d = {d}</text>
          </g>
          <line x1="120" y1="80" x2="180" y2="80" className="fat-mlp-arrow" />

          <g className="fat-mlp-op">
            <rect x="180" y="50" width="120" height="60" rx="10" />
            <text x="240" y="78" className="fat-mlp-op-label">Linear</text>
            <text x="240" y="98" className="fat-mlp-op-dim">d → 4d</text>
          </g>
          <line x1="300" y1="80" x2="340" y2="80" className="fat-mlp-arrow" />

          {/* GeLU mini-plot */}
          <g className="fat-mlp-gelu" transform="translate(340, 30)">
            <rect x="0" y="0" width="140" height="100" rx="10" />
            <line x1="20" y1="60" x2="120" y2="60" className="fat-mlp-axis" />
            <line x1="70" y1="10" x2="70" y2="100" className="fat-mlp-axis" />
            <path d={path} className="fat-mlp-curve" />
            <text x="70" y="22" className="fat-mlp-op-label">GeLU</text>
          </g>
          <line x1="480" y1="80" x2="520" y2="80" className="fat-mlp-arrow" />

          <g className="fat-mlp-op">
            <rect x="520" y="50" width="100" height="60" rx="10" />
            <text x="570" y="78" className="fat-mlp-op-label">Linear</text>
            <text x="570" y="98" className="fat-mlp-op-dim">4d → d</text>
          </g>
        </svg>
      </div>

      <div className="fat-controls">
        <label style={{ flex: 1, minWidth: 280 }}>
          Hidden size <strong>d</strong>: <strong style={{ marginLeft: 4 }}>{d}</strong>
          <input
            type="range" min={64} max={4096} step={64} value={d}
            onChange={(e) => setD(parseInt(e.target.value, 10))}
            style={{ marginLeft: 8, width: 220 }}
          />
        </label>
        <div className="fat-mlp-stats">
          <span><strong>{dff.toLocaleString('en-US')}</strong> hidden units</span>
          <span><strong>{params.toLocaleString('en-US')}</strong> parameters per MLP block</span>
        </div>
      </div>

      <p className="viz-caption">
        The MLP is applied <em>per token, independently</em>: same two
        matrices, every position. Attention mixes information <em>across</em>{' '}
        tokens; the MLP mixes it <em>within</em> each token's representation.
        With <Katex tex="d_{ff} = 4d" /> (the convention since the original
        paper), the two linear layers account for <Katex tex="8d^2" /> weights
        per block — roughly two-thirds of a modern transformer's parameters
        live here.
      </p>
    </div>
  );
}

/* =========================================================
   Widget 3 — Causal masking (ported from Section5Causal)
   ========================================================= */
function CausalWidget() {
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
      <div className="fat-controls">
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
        <label className="fat-toggle">
          <input
            type="checkbox" checked={masked}
            onChange={(e) => setMasked(e.target.checked)}
          />
          Apply causal mask
        </label>
      </div>

      <div className="fat-matrix-row">
        <HeatmapGrid
          data={maskMat}
          label="mask  (upper triangle = −∞)"
          rowLabels={sentence.tokens}
          colLabels={sentence.tokens}
          cellWidth={50}
          showValues={false}
        />
        <div className="fat-matrix-op">+</div>
        <HeatmapGrid
          data={scores}
          label="masked scores"
          rowLabels={sentence.tokens}
          colLabels={sentence.tokens}
          cellWidth={50}
          diverging
        />
        <div className="fat-matrix-op">softmax →</div>
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
    </div>
  );
}

/* =========================================================
   Widget 4 — Positional embeddings
   ========================================================= */
function sinusoidalPE(pos, d) {
  const v = new Array(d);
  for (let i = 0; i < d; i++) {
    const pair = Math.floor(i / 2);
    const freq = Math.pow(10000, -(2 * pair) / d);
    v[i] = i % 2 === 0 ? Math.sin(pos * freq) : Math.cos(pos * freq);
  }
  return v;
}
const POS_EMB = Array.from({ length: 16 }, (_, p) => sinusoidalPE(p, 8));

function PositionalWidget() {
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
      <div className="fat-controls">
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
        <span style={{ color: 'var(--post-text-mut)', fontSize: '0.85rem' }}>
          Hover or tap any row.
        </span>
      </div>

      <div className="fat-matrix-row">
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
        <div className="fat-matrix-op">+</div>
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
        <div className="fat-matrix-op">=</div>
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
        Each position has its own deterministic vector. The leftmost columns
        oscillate fast, the rightmost slowly, so any two positions produce a
        distinct fingerprint. The sum is what flows into the Q / K / V
        projections.
      </p>
    </div>
  );
}

/* =========================================================
   Code block — one transformer block in PyTorch-ish pseudocode
   ========================================================= */
const BLOCK_CODE = `def transformer_block(x):
    # x: (T, d) — T tokens, hidden size d
    # 1. Mix information across tokens via multi-head causal attention.
    h = x + multi_head_attention(layer_norm(x), causal_mask=True)
    # 2. Mix information within each token via a per-token MLP.
    h = h + mlp(layer_norm(h))  #  mlp(z) = W2 @ gelu(W1 @ z)
    return h

# A transformer is just N of these stacked, with an input embedding
# (token + positional) at the bottom and a softmax-over-vocab head on top.
`;

function CodeBlock() {
  return (
    <details className="post-code-details">
      <summary>
        <span className="post-code-summary-icon" aria-hidden="true">›</span>
        <span>Show one transformer block in pseudocode</span>
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
        {BLOCK_CODE}
      </SyntaxHighlighter>
    </details>
  );
}

/* =========================================================
   Block diagram — small SVG summarising the post.
   ========================================================= */
function BlockDiagram() {
  // Horizontal pre-norm transformer block:
  //   x → [LN → Attn] → ⊕ → [LN → MLP] → ⊕ → h
  // The two ⊕s are residual sums; each sub-block has a curved skip arc
  // (dashed) carrying x and the post-attention stream around its sub-layer.
  const y = 140;        // main flow baseline
  const sky = 60;       // arc apex (how high the residuals rise)
  return (
    <div className="viz-panel fat-block">
      <svg viewBox="0 0 920 230" width="100%" preserveAspectRatio="xMidYMid meet">
        <defs>
          <marker id="fat-arrow" viewBox="0 0 10 10" refX="9" refY="5"
            markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" className="fat-block-arrowhead" />
          </marker>
        </defs>

        {/* Input x */}
        <text x="20" y={y + 5} className="fat-block-io">x</text>
        <line x1="34" y1={y} x2="72" y2={y} className="fat-block-edge" markerEnd="url(#fat-arrow)" />

        {/* Sub-block 1: LayerNorm + Attention + residual sum */}
        <rect x="80" y={y - 22} width="86" height="44" rx="8" className="fat-block-ln" />
        <text x="123" y={y + 5} className="fat-block-label">LayerNorm</text>
        <line x1="166" y1={y} x2="186" y2={y} className="fat-block-edge" markerEnd="url(#fat-arrow)" />

        <rect x="194" y={y - 22} width="186" height="44" rx="8" className="fat-block-attn" />
        <text x="287" y={y + 5} className="fat-block-label">Multi-head attention</text>
        <text x="287" y={y - 28} className="fat-block-sublabel">(causal during training)</text>
        <line x1="380" y1={y} x2="406" y2={y} className="fat-block-edge" markerEnd="url(#fat-arrow)" />

        <circle cx="420" cy={y} r="14" className="fat-block-add" />
        <text x="420" y={y + 5} className="fat-block-add-label">+</text>

        {/* First residual: from input wire over to the first ⊕ */}
        <path
          d={`M 50 ${y} C 50 ${sky}, 420 ${sky}, 420 ${y - 14}`}
          className="fat-block-residual"
          fill="none"
          markerEnd="url(#fat-arrow)"
        />
        <text x="235" y={sky - 6} className="fat-block-residual-label">residual</text>

        {/* Sub-block 2: LayerNorm + MLP + residual sum */}
        <line x1="434" y1={y} x2="460" y2={y} className="fat-block-edge" markerEnd="url(#fat-arrow)" />
        <rect x="468" y={y - 22} width="86" height="44" rx="8" className="fat-block-ln" />
        <text x="511" y={y + 5} className="fat-block-label">LayerNorm</text>
        <line x1="554" y1={y} x2="574" y2={y} className="fat-block-edge" markerEnd="url(#fat-arrow)" />

        <rect x="582" y={y - 22} width="186" height="44" rx="8" className="fat-block-mlp" />
        <text x="675" y={y + 5} className="fat-block-label">MLP (per token)</text>
        <line x1="768" y1={y} x2="794" y2={y} className="fat-block-edge" markerEnd="url(#fat-arrow)" />

        <circle cx="808" cy={y} r="14" className="fat-block-add" />
        <text x="808" y={y + 5} className="fat-block-add-label">+</text>

        {/* Second residual: from after first ⊕ over to the second ⊕ */}
        <path
          d={`M 438 ${y} C 438 ${sky}, 808 ${sky}, 808 ${y - 14}`}
          className="fat-block-residual"
          fill="none"
          markerEnd="url(#fat-arrow)"
        />
        <text x="623" y={sky - 6} className="fat-block-residual-label">residual</text>

        {/* Output h */}
        <line x1="822" y1={y} x2="860" y2={y} className="fat-block-edge" markerEnd="url(#fat-arrow)" />
        <text x="876" y={y + 5} className="fat-block-io">h</text>

        {/* ×N stack indicator */}
        <g transform={`translate(40, ${y + 50})`}>
          <text x="0" y="0" className="fat-block-stack-inline">repeat this whole block <tspan className="fat-block-stack-times">× N</tspan> (e.g. N = 12 for GPT-2 small, 96 for GPT-3)</text>
        </g>
      </svg>
      <p className="viz-caption">
        One transformer block, read left to right: each input vector{' '}
        <em>x</em> first goes through <strong>LayerNorm → multi-head attention</strong>{' '}
        (which mixes information <em>across tokens</em>), and the result is
        added back to <em>x</em> via the residual skip. That sum then goes
        through <strong>LayerNorm → MLP</strong> (which mixes information{' '}
        <em>within</em> each token), and is again added to its own input.
        Stack <em>N</em> of these blocks and you have a transformer.
      </p>
    </div>
  );
}

/* =========================================================
   Page
   ========================================================= */
export default function FromAttentionToTransformer() {
  usePageMeta({
    title: 'From Attention to Transformer',
    description: 'Multiple heads, an MLP per token, a causal mask, and positional information — the block that powers every modern language model.',
    slug: 'from-attention-to-transformer',
    publishedDate: '2026-02-12',
    keywords: ['transformer', 'multi-head attention', 'MLP', 'LayerNorm', 'causal mask'],
  });

  return (
    <article className="post-2026 fat-post">
      <div className="post-wrap">
        <div className="post-hero">
          <div className="post-series-tag">
            <span className="post-live-dot" aria-hidden="true" />
            Transformers · Part 3 of 3
          </div>
          <h1>From Attention to Transformer</h1>
          <p className="post-lede">
            Part 2 built a single attention operation: queries, keys, values,
            scaled dot-products, softmax. That is the core of what a transformer
            does — but a single attention operation isn't yet a transformer. To
            get a transformer block you need four more ingredients: multiple
            heads, feed-forward layers, causal masking, and positional
            information.
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

        <h2 className="reveal">Multiple heads — different relationships in parallel</h2>
        <p>
          One attention matrix can only express one kind of relationship at a
          time. Real sentences need several at once: "the cat sat on the mat"
          has syntactic structure (subject–verb), local order ("the" attaches
          to "cat"), and coreference (which "the" refers to which noun).
        </p>
        <p>
          <strong>Multi-head attention</strong> runs <em>h</em> attention
          operations in parallel, each with its own learned{' '}
          <Katex tex="W_Q, W_K, W_V" />. The model splits the residual stream
          into <em>h</em> subspaces of width{' '}
          <Katex tex="d_k = d_{model} / h" />, runs attention inside each
          subspace, then concatenates the outputs. Different heads learn
          different <em>kinds</em> of relationships — syntactic, lexical,
          positional. Hover the heads below.
        </p>
        <MultiHeadWidget />
        <p>
          Mechanistic-interpretability work has found heads that specialise on
          induction (copy a token that appeared after this one before), heads
          that broadcast a global signal, and heads that act as "attention
          sinks" — dumping unused probability mass on a fixed early position.
          The patterns above are hand-drawn caricatures of what trained heads
          look like, not the real thing.
        </p>

        <h2 className="reveal">The other half of a block — the MLP</h2>
        <p>
          Attention mixes information <em>across</em> tokens. But once each
          token has assembled a context-aware vector, the model still needs to
          do something <em>with</em> it — combine the dimensions, apply
          nonlinear features, store learned facts. That is the job of the{' '}
          <strong>feed-forward layer</strong>, also called the MLP.
        </p>
        <p>
          It is the same two-layer MLP at every position, applied{' '}
          <em>independently</em> to each token's vector. Two linear maps with a
          nonlinearity in between (GeLU, in modern models); the hidden layer is
          usually <Katex tex="4 \times" /> wider than the residual stream.
          Trivial to describe; the bulk of the parameters lives here.
        </p>
        <MLPWidget />
        <p>
          A useful intuition (Geva et al., 2020): the MLP behaves like a
          key-value memory. The first linear layer picks out which "facts" to
          fire for this token; the GeLU keeps only the strongly-firing ones;
          the second linear layer adds their stored values back into the
          residual stream. Attention <em>routes</em> information; the MLP{' '}
          <em>remembers</em> it.
        </p>

        <h2 className="reveal">Causal masking — letting the model train on every position at once</h2>
        <p>
          Training a language model means predicting the next token. The
          natural way to do that would be one position at a time: feed the
          model tokens 1…t, ask it to predict token t+1, then move on. That
          works, but it wastes the parallelism of attention — every position
          would need its own forward pass.
        </p>
        <p>
          The trick is <strong>causal masking</strong>. Feed the whole
          sequence in once, compute attention, and before the softmax, zero
          out (i.e. set to <Katex tex="-\infty" />) every entry above the
          diagonal. Now when we read out token <em>t</em>'s prediction, it
          mathematically depended only on tokens 1…<em>t</em>. We can compute
          the loss at every position in a single forward pass without
          information leaking from the future.
        </p>
        <CausalWidget />

        <h2 className="reveal">Positional information — because attention has none</h2>
        <p>
          There's a subtle problem with attention: it doesn't actually know
          the <em>order</em> of the tokens. Shuffle the input and the same
          dot products come out — the attention scores would be identical, just
          permuted. That's clearly wrong for language: "dog bites man" and
          "man bites dog" should not produce the same representations.
        </p>
        <p>
          The fix is to bake position into the input. The original transformer
          added a fixed <strong>sinusoidal positional embedding</strong> to
          each token's embedding before any attention runs — different
          frequencies in different dimensions so every position gets a unique
          fingerprint.
        </p>
        <PositionalWidget />
        <p>
          Modern models mostly use{' '}
          <a className="post-link" href="/blog/visualizing-rope">rotary positional embeddings (RoPE)</a>{' '}
          instead, which rotate the Q and K vectors inside attention rather
          than adding a separate vector at the input. Same goal, cleaner
          extrapolation to longer sequences. We covered RoPE in its own post.
        </p>

        <h2 className="reveal">Putting it together</h2>
        <p>
          A transformer block is the four ingredients above, wired together
          with two residual connections and two LayerNorms:{' '}
          <Katex tex="\mathrm{LN} \to \text{Multi-head attention} \to \text{residual} \to \mathrm{LN} \to \text{MLP} \to \text{residual}" />.
          Stack <em>N</em> of these blocks — between 6 and 100+, depending on
          the model — and that's a transformer.
        </p>
        <BlockDiagram />
        <aside className="fat-info">
          <div className="fat-info-label">What does LayerNorm do?</div>
          <p>
            For every token vector going in, <strong>LayerNorm</strong>{' '}
            re-scales it so its values have mean <Katex tex="0" /> and
            standard deviation <Katex tex="1" /> across the hidden dimension.
            Concretely, for each vector <Katex tex="x \in \mathbb{R}^{d}" />:
          </p>
          <Katex
            block
            className="fat-info-eq"
            tex={'\\mathrm{LN}(x) \\;=\\; \\underbrace{\\,\\gamma\\,}_{\\text{gain}} \\,\\odot\\, \\underbrace{\\frac{x - \\mu}{\\sqrt{\\sigma^{2} + \\varepsilon}}}_{\\text{normalise to mean 0, var 1}} \\;+\\; \\underbrace{\\,\\beta\\,}_{\\text{bias}}'}
          />
          <p>
            <Katex tex="\mu" /> and <Katex tex="\sigma^{2}" /> are the mean and
            variance of the <em>d</em> numbers inside <em>x</em> — computed
            per-token, not across the batch. The small{' '}
            <Katex tex="\varepsilon" /> avoids dividing by zero. The two
            learned vectors <Katex tex="\gamma" /> and <Katex tex="\beta" />{' '}
            let the layer re-scale and re-shift the result if it wants to;
            the network can effectively cancel the normalisation by learning{' '}
            <Katex tex="\gamma = \sigma" /> and{' '}
            <Katex tex="\beta = \mu" />, but in practice it doesn't.
          </p>
          <p>
            The reason it's there: each sub-layer (attention, MLP) trains
            much more stably when its input has predictable magnitudes.
            Without it, deep transformers were nearly impossible to train.
            Modern Llama-family models use a slightly cheaper variant called{' '}
            <strong>RMSNorm</strong> — same idea but it skips the mean-centering
            step (only divides by RMS).
          </p>
        </aside>
        <CodeBlock />
        <p>
          Every modern large language model — GPT, Llama, Claude, Gemini,
          Mistral, DeepSeek — is a stack of this block, scaled up. The
          differences are details: which normalisation (RMSNorm vs LayerNorm),
          which positional scheme (RoPE vs absolute), which activation (GeLU vs
          SwiGLU), pre-norm vs post-norm. The skeleton is the same.
        </p>

        <h2 className="reveal">References</h2>
        <div className="post-refs">
          <div className="head ref-cite">Cite</div>
          <div className="head ref-title">Title</div>
          <div className="head ref-link">Link</div>

          <div className="ref-cite">Vaswani et al. 2017</div>
          <div>
            <div className="ref-title">
              <a href="https://arxiv.org/abs/1706.03762" target="_blank" rel="noreferrer">Attention Is All You Need</a>
            </div>
            <div className="ref-note">Original transformer paper. Defines multi-head attention, sinusoidal positional encoding, and the block structure used in this post.</div>
          </div>
          <div className="ref-link"><a href="https://arxiv.org/abs/1706.03762" target="_blank" rel="noreferrer">arxiv.org</a></div>

          <div className="ref-cite">Geva et al. 2020</div>
          <div>
            <div className="ref-title">
              <a href="https://arxiv.org/abs/2012.14913" target="_blank" rel="noreferrer">Transformer Feed-Forward Layers Are Key-Value Memories</a>
            </div>
            <div className="ref-note">The "MLPs as key-value memories" framing referenced in the MLP section.</div>
          </div>
          <div className="ref-link"><a href="https://arxiv.org/abs/2012.14913" target="_blank" rel="noreferrer">arxiv.org</a></div>

          <div className="ref-cite">Elhage et al. 2021</div>
          <div>
            <div className="ref-title">
              <a href="https://transformer-circuits.pub/2021/framework/index.html" target="_blank" rel="noreferrer">A Mathematical Framework for Transformer Circuits</a>
            </div>
            <div className="ref-note">The residual-stream view of the block, plus the discovery of specialised attention heads.</div>
          </div>
          <div className="ref-link"><a href="https://transformer-circuits.pub/2021/framework/index.html" target="_blank" rel="noreferrer">transformer-circuits.pub</a></div>

          <div className="ref-cite">Earlier posts</div>
          <div>
            <div className="ref-title">
              <a href="/blog/visualizing-rope" target="_blank" rel="noreferrer">Visualizing RoPE</a>
            </div>
            <div className="ref-note">Rotary positional embeddings, in detail.</div>
          </div>
          <div className="ref-link"><a href="/blog/visualizing-rope" target="_blank" rel="noreferrer">this site</a></div>
        </div>

        <footer className="post-footer">
          <p>
            <strong>Part 3 of Transformers</strong> · Previous:{' '}
            <a className="post-link" href="/blog/self-attention">Inside Self-Attention</a>
            {' · '}Start of the series:{' '}
            <a className="post-link" href="/blog/nlp-history">From TF-IDF to Attention</a>.
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
