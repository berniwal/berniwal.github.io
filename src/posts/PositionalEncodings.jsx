// src/posts/PositionalEncodings.jsx
// Positions in Transformers · Part 1 of 2 — Positional Encodings: A Tour.
// Ported from the original VisualizingRoPE post; permutation + encoding-zoo
// widgets carried over, surrounding prose trimmed per the 2026 redesign.
// TAGS FOR REGISTRATION: ['transformers', 'positions', 'history']
// EXCERPT: Self-attention is permutation-invariant. Five additive position-encoding schemes — absolute, Shaw, T5, Swin, ALiBi — and where each one injects position.
import React, { useEffect, useRef, useState } from 'react';
import './PostChrome.css';
import './PositionalEncodings.css';

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
   Toy world — small embeddings + projections, so every
   widget can compute attention live in the browser.
   ========================================================= */
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

/* =========================================================
   Widget 1 — Permutation invariance of self-attention
   ========================================================= */
function SectionPermutation() {
  const [order, setOrder] = useState([0, 1, 2, 3, 4]);
  const tokens = order.map((i) => PROMPT[i]);

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
    <div className="viz-panel pe-perm">
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
        <div
          className="pe-grid"
          style={{
            gridTemplateColumns: `60px repeat(${tokens.length}, ${cellSize}px)`,
          }}
        >
          <div />
          {tokens.map((t, j) => (
            <div key={`col-${j}`} className="pe-grid-head">{t}</div>
          ))}
          {tokens.map((tRow, i) => (
            <React.Fragment key={`row-${i}`}>
              <div className="pe-grid-rowhead">{tRow}</div>
              {A[i].map((v, j) => {
                const sat = 18 + Math.round(v * 60);
                const light = 96 - Math.round(v * 50);
                return (
                  <div
                    key={`a-${i}-${j}`}
                    className="pe-grid-cell"
                    style={{
                      background: `hsl(218, ${sat}%, ${light}%)`,
                      color: v > 0.55 ? '#fff' : 'var(--post-text)',
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
   Widget 2 — Position-encoding zoo
   ========================================================= */

// Hand-picked vectors per relative offset for Shaw Key (a^K) and Shaw Value (a^V).
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

const T5_BIAS = { '-3': -0.4, '-2': -0.2, '-1': 0.1, '0': 0.5, '1': 0.1, '2': -0.2, '3': -0.4 };

// ALiBi slope — chosen so the bias looks visible on a 4-token toy.
const ALIBI_SLOPE = 0.25;

function effectiveBias(mode, Q) {
  const T = Q.length;
  const M = Array.from({ length: T }, () => new Array(T).fill(0));
  if (mode === 'shaw-key') {
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
  if (mode === 'alibi') {
    for (let i = 0; i < T; i++)
      for (let j = 0; j < T; j++) M[i][j] = -ALIBI_SLOPE * Math.abs(i - j);
    return M;
  }
  return M;
}

function MiniMatrix({ data, title, cellSize = 50, diverging = true, range = 0.6, vmax }) {
  const cols = data[0].length;
  return (
    <div>
      {title && (
        <div className="pe-mm-title">{title}</div>
      )}
      <div
        className="pe-grid"
        style={{ gridTemplateColumns: `repeat(${cols}, ${cellSize}px)` }}
      >
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
            const ink = !diverging && v / (vmax || 1) > 0.55 ? '#fff' : 'var(--post-text)';
            return (
              <div
                key={`mm-${i}-${j}`}
                className="pe-grid-cell"
                style={{ background: bg, color: ink, width: cellSize }}
              >
                {v.toFixed(2)}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}

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
    'alibi':      'A_{ij} = \\mathrm{softmax}\\!\\left( q_i \\cdot k_j / \\sqrt{d_k} - m \\cdot |i-j| \\right)',
  };

  const captions = {
    'absolute': (
      <>
        The original Transformer (<a className="post-link" href="https://arxiv.org/abs/1706.03762" target="_blank" rel="noreferrer">Vaswani et al. 2017</a>)
        adds a position vector <Katex tex="p_i" /> to each token embedding
        <em> before</em> the Q/K projections. The attention math itself is
        unchanged; position arrives baked into Q and K. Sinusoidal or learned,
        both are bounded by the trained sequence length.
      </>
    ),
    'shaw-key': (
      <>
        <a className="post-link" href="https://arxiv.org/abs/1803.02155" target="_blank" rel="noreferrer">Shaw et al. 2018</a> — a
        learnable vector <Katex tex="a^K_{i-j}" /> added to <Katex tex="k_j" />{' '}
        <em>before</em> the dot product, indexed by clipped relative offset.
        The score contribution <Katex tex="q_i \cdot a^K_{i-j}" /> depends on
        the query, so different queries weight the same offset differently.
      </>
    ),
    'shaw-value': (
      <>
        Shaw 2018's second form — a learnable <Katex tex="a^V_{i-j}" /> added
        to the <em>value</em> during the weighted sum. Attention scores are
        untouched. The right panel shows <Katex tex="\Delta V_i" />, the
        per-row positional vector added to each output. Most later work
        dropped this branch.
      </>
    ),
    't5': (
      <>
        <a className="post-link" href="https://arxiv.org/abs/1910.10683" target="_blank" rel="noreferrer">T5</a> (Raffel et al. 2019)
        collapses Shaw's vector to a <em>learned scalar</em> per relative
        offset, added directly to the score. Cheap, content-independent, and
        the form most people picture when they hear "relative position bias".
      </>
    ),
    'swin': (
      <>
        <a className="post-link" href="https://arxiv.org/abs/2103.14030" target="_blank" rel="noreferrer">Swin</a> (Liu et al. 2021)
        ports T5's scalar bias to 2D image patches: indexed by{' '}
        <Katex tex="(\Delta i, \Delta j)" /> within an{' '}
        <Katex tex="M \times M" /> window. Same upside (relative, simple),
        same downside (bounded by the trained window).
      </>
    ),
    'alibi': (
      <>
        <a className="post-link" href="https://arxiv.org/abs/2108.12409" target="_blank" rel="noreferrer">ALiBi</a> (Press et al. 2021)
        drops the lookup table entirely: subtract a slope <Katex tex="m" />{' '}
        times <Katex tex="|i-j|" /> from each score. A fixed linear penalty
        for distance — zero learned parameters, and the bias is defined for
        any offset, so the model extrapolates past its training length.
      </>
    ),
  };

  const tabs = [
    { id: 'absolute',   label: 'Absolute' },
    { id: 'shaw-key',   label: 'Shaw Key' },
    { id: 'shaw-value', label: 'Shaw Value' },
    { id: 't5',         label: 'T5 Scalar' },
    { id: 'swin',       label: 'Swin 2D' },
    { id: 'alibi',      label: 'ALiBi' },
  ];

  return (
    <div className="viz-panel pe-zoo">
      <div className="viz-controls">
        <div className="viz-tabs" role="tablist">
          {tabs.map((opt) => (
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

      <p className="pe-zoo-caption">{captions[mode]}</p>

      <div className="pe-zoo-panels">
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
                : mode === 'alibi'
                  ? 'bias matrix  −m · |i − j|  (fixed linear penalty)'
                  : 'bias matrix B  (constant per relative offset)'
            }
            diverging
            range={mode === 'shaw-key' ? 0.4 : mode === 'alibi' ? 0.8 : 0.6}
          />
        ) : null}

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
        Six places to inject position: into the input embedding
        (<strong>Absolute</strong>), into the key vector before the dot
        product (<strong>Shaw Key</strong>), into the value vector in the
        weighted sum (<strong>Shaw Value</strong>), as a learned scalar on
        the score (<strong>T5</strong>), as the same scalar indexed by a 2D
        offset (<strong>Swin</strong>), or as a fixed linear penalty in
        distance (<strong>ALiBi</strong>). Different surface, same shape:
        position appears as something <em>added</em> to attention.
      </p>
    </div>
  );
}

/* =========================================================
   Page
   ========================================================= */
export default function PositionalEncodings() {
  useEffect(() => {
    const prev = document.title;
    document.title = 'Positional Encodings: A Tour — Bernhard Walser';
    return () => { document.title = prev; };
  }, []);

  return (
    <article className="post-2026 pe-post">
      <div className="post-wrap">
        <div className="post-hero">
          <div className="post-series-tag">
            <span className="post-live-dot" aria-hidden="true" />
            Positions in Transformers · Part 1 of 2
          </div>
          <h1>Positional Encodings: A Tour</h1>
          <p className="post-lede">
            <em>"The cat sat on the mat"</em> and <em>"on the mat the cat
            sat"</em> would score identical to a vanilla attention block —
            same words, same numbers, just shuffled. Self-attention is
            blind to order. This post walks through the five additive
            tricks people tried before RoPE to fix that.
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

        <h2 className="reveal">The positional encoding problem</h2>
        <p>
          Self-attention is <strong>permutation-invariant</strong>. Shuffle
          the tokens of a sequence and the model produces the <em>same</em>{' '}
          outputs in shuffled order. It has no built-in notion of which
          token came first.
        </p>
        <p>
          Click <strong>Shuffle</strong> below and watch: the attention
          matrix keeps the same numbers, only the row and column labels
          move.
        </p>
        <SectionPermutation />
        <p>
          Position information has to come from somewhere outside the
          attention math itself. The question is <em>where</em> to add it.
        </p>

        <h2 className="reveal">The predecessors — five variants of additive bias</h2>
        <p>
          Before RoPE, every scheme injected position as an{' '}
          <strong>additive term</strong> somewhere inside attention. The
          left panel of the widget below shows that term; the right panel
          shows the attention pattern it produces.
        </p>
        <p>
          The variants differ only in <em>where</em> the addition happens:
          on the input embedding before Q/K projections (Absolute), on the
          key vector before the dot product (Shaw Key), on the value vector
          in the weighted sum (Shaw Value), on the attention score as a
          learned scalar (T5), the same scalar in 2D (Swin), or a
          parameter-free linear penalty on the score (ALiBi).
        </p>
        <SectionPosEncodingZoo />
        <p>
          Different surface, same shape: position is something the model{' '}
          <em>adds</em> on top of attention. <a className="post-link" href="#/blog/rope">Part 2</a> throws that
          habit out — instead of adding a bias, it{' '}
          <em>rotates</em> Q and K, so relative position falls out of the
          dot product for free.
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
            <div className="ref-note">Original Transformer; sinusoidal absolute positions added to the input embedding.</div>
          </div>
          <div className="ref-link"><a href="https://arxiv.org/abs/1706.03762" target="_blank" rel="noreferrer">arxiv.org</a></div>

          <div className="ref-cite">Shaw et al. 2018</div>
          <div>
            <div className="ref-title">
              <a href="https://arxiv.org/abs/1803.02155" target="_blank" rel="noreferrer">Self-Attention with Relative Position Representations</a>
            </div>
            <div className="ref-note">Learnable vectors added to keys (Shaw Key) and values (Shaw Value), indexed by clipped relative offset.</div>
          </div>
          <div className="ref-link"><a href="https://arxiv.org/abs/1803.02155" target="_blank" rel="noreferrer">arxiv.org</a></div>

          <div className="ref-cite">Raffel et al. 2019</div>
          <div>
            <div className="ref-title">
              <a href="https://arxiv.org/abs/1910.10683" target="_blank" rel="noreferrer">Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5)</a>
            </div>
            <div className="ref-note">Collapses Shaw's vector to a learned scalar bias per relative offset.</div>
          </div>
          <div className="ref-link"><a href="https://arxiv.org/abs/1910.10683" target="_blank" rel="noreferrer">arxiv.org</a></div>

          <div className="ref-cite">Liu et al. 2021</div>
          <div>
            <div className="ref-title">
              <a href="https://arxiv.org/abs/2103.14030" target="_blank" rel="noreferrer">Swin Transformer</a>
            </div>
            <div className="ref-note">Ports the T5 scalar bias to 2D image patches within a windowed attention block.</div>
          </div>
          <div className="ref-link"><a href="https://arxiv.org/abs/2103.14030" target="_blank" rel="noreferrer">arxiv.org</a></div>

          <div className="ref-cite">Press et al. 2021</div>
          <div>
            <div className="ref-title">
              <a href="https://arxiv.org/abs/2108.12409" target="_blank" rel="noreferrer">Train Short, Test Long: Attention with Linear Biases (ALiBi)</a>
            </div>
            <div className="ref-note">A parameter-free linear penalty on the attention score that extrapolates beyond the training length.</div>
          </div>
          <div className="ref-link"><a href="https://arxiv.org/abs/2108.12409" target="_blank" rel="noreferrer">arxiv.org</a></div>
        </div>

        <footer className="post-footer">
          <p>
            <strong>Part 1 of Positions in Transformers</strong> · Next:{' '}
            <a className="post-link" href="#/blog/rope">Rotary Positional Embeddings (RoPE)</a>.
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
