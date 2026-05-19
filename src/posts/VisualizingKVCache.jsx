// src/posts/VisualizingKVCache.jsx
// Part 2 of "Visualizing ML" — Prefill, Decode, KV Cache economics.
import React, { useEffect, useMemo, useRef, useState } from 'react';
import './VisualizingKVCache.css';

/* =========================================================
   Math helpers — copied from post #1 (T, d are tiny by design)
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
   Shared toy world — same tokens/embeddings/weights as post #1
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
  { id: 'she',  tokens: ['she', 'gave', 'him', 'a', 'book'] },
  { id: 'cat',  tokens: ['the', 'cat', 'sat', 'on', 'the', 'mat'] },
  { id: 'fox',  tokens: ['a', 'quick', 'brown', 'fox', 'jumps'] },
  { id: 'code', tokens: ['code', 'that', 'writes', 'itself'] },
];

// Same W_Q / W_K / W_V from post #1 — readers carry numbers between posts.
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

// What gets "generated" after each prompt — deterministic for the demo.
const CONTINUATIONS = {
  she:  ['the', 'cat', 'sat', 'on'],
  cat:  ['she', 'gave', 'him', 'a'],
  fox:  ['the', 'mat', 'on', 'book'],
  code: ['a', 'fox', 'jumps', 'on'],
};

const buildX = (tokens) => tokens.map((t) => EMB[t]);

/* =========================================================
   MatrixGrid — copied from post #1
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
  freshRow,
}) {
  if (!data || data.length === 0) {
    return (
      <div className="viz-mat">
        {label && <div className="viz-mat-label">{label}</div>}
        <div
          style={{
            border: '1px dashed var(--rule)',
            borderRadius: 4,
            padding: '14px 18px',
            color: 'var(--ink-faint)',
            fontFamily: 'var(--mono)',
            fontSize: '0.78rem',
            fontStyle: 'italic',
            minWidth: 120,
          }}
        >
          empty
        </div>
      </div>
    );
  }
  const cols = data[0].length;
  const hasRowLabels = !!rowLabels;
  const hasColLabels = !!colLabels;
  const gridCols = `${hasRowLabels ? '52px ' : ''}repeat(${cols}, ${cellWidth}px)`;

  return (
    <div className="viz-mat">
      {label && <div className="viz-mat-label">{label}</div>}
      <div className="viz-mat-grid" style={{ gridTemplateColumns: gridCols }}>
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
              <div className={`viz-cell viz-row-label ${i === freshRow ? 'viz-row-hi' : ''}`}>
                {rowLabels[i]}
              </div>
            )}
            {row.map((v, j) => {
              const isFocus = highlight && highlight.row === i && highlight.col === j;
              const isRowHi = highlight && highlight.row === i && highlight.col == null;
              const isColHi = highlight && highlight.col === j && highlight.row == null;
              const isFresh = i === freshRow;
              const display = Number.isFinite(v) ? v.toFixed(decimals) : '·';
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
                    isFresh ? 'viz-row-hi' : '',
                  ].filter(Boolean).join(' ')}
                  style={{ width: cellWidth }}
                  onMouseEnter={onCellEnter ? () => onCellEnter(i, j) : undefined}
                  onMouseLeave={onCellLeave ? () => onCellLeave(i, j) : undefined}
                  onClick={onCellTap ? () => onCellTap(i, j) : undefined}
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
   Counter chip — small KPI strip beneath each interactive
   ========================================================= */

function Counter({ label, value, sub }) {
  return (
    <div className="viz-counter">
      <div className="viz-counter-label">{label}</div>
      <div className="viz-counter-value">{value}</div>
      {sub && <div className="viz-counter-sub">{sub}</div>}
    </div>
  );
}

/* =========================================================
   §1 — Interactive A: Prefill → Decode step-through
   ========================================================= */

function SectionPrefillDecode() {
  const [presetId, setPresetId] = useState('she');
  // step encodes: 0 = idle, 1 = post-prefill, 2..N = decoded (step-1) tokens
  const [step, setStep] = useState(0);

  const preset = SENTENCES.find((s) => s.id === presetId);
  const promptTokens = preset.tokens;
  const T_prompt = promptTokens.length;
  const continuation = CONTINUATIONS[presetId] || [];
  const maxDecodeSteps = continuation.length;

  const decodedSoFar = Math.max(0, step - 1);
  const generatedTokens = continuation.slice(0, decodedSoFar);
  const allTokens = step === 0 ? [] : [...promptTokens, ...generatedTokens];

  const { K, V, currentQ, freshRowIdx } = useMemo(() => {
    if (allTokens.length === 0) {
      return { K: [], V: [], currentQ: [], freshRowIdx: -1 };
    }
    const X = buildX(allTokens);
    const K = matmul(X, W_K);
    const V = matmul(X, W_V);
    // currentQ: prefill shows Q for all prompt rows; decode shows just the new row
    const isDecode = step >= 2;
    if (isDecode) {
      const lastRow = X[X.length - 1];
      const Qlast = matmul([lastRow], W_Q);
      return { K, V, currentQ: Qlast, freshRowIdx: K.length - 1 };
    }
    // prefill
    const Q = matmul(X, W_Q);
    return { K, V, currentQ: Q, freshRowIdx: -1 };
  }, [allTokens, step]);

  const phase = step === 0 ? 'idle' : step === 1 ? 'prefill' : 'decode';
  const cacheBytes = K.length === 0 ? 0 : K.length * 4 * 2 * 2; // T × d_k × 2 (K & V) × 2 bytes (fp16)
  const newRowsThisStep = step === 1 ? T_prompt : step >= 2 ? 1 : 0;
  const cacheBytesReadThisStep = step >= 2 ? K.length * 4 * 2 * 2 : 0;

  const canDecode = step >= 1 && decodedSoFar < maxDecodeSteps;
  const canPrefill = step === 0;

  const colLabelsK = ['k0','k1','k2','k3'];
  const colLabelsV = ['v0','v1','v2','v3'];

  return (
    <div className="viz-panel">
      <div className="viz-controls">
        <label>
          Prompt
          <select
            value={presetId}
            onChange={(e) => { setPresetId(e.target.value); setStep(0); }}
          >
            {SENTENCES.map((s) => (
              <option key={s.id} value={s.id}>{s.tokens.join(' ')}</option>
            ))}
          </select>
        </label>
        <button
          className={canPrefill ? 'active' : ''}
          disabled={!canPrefill}
          onClick={() => setStep(1)}
        >
          Run prefill
        </button>
        <button
          className={canDecode ? 'active' : ''}
          disabled={!canDecode}
          onClick={() => setStep((s) => s + 1)}
        >
          Decode one step
        </button>
        <button onClick={() => setStep(0)}>Reset</button>
      </div>

      <div className="viz-token-row">
        {promptTokens.map((t, i) => (
          <div className="viz-token-pill viz-token-prompt" key={`p-${i}`}>
            <div className="viz-token-text">{t}</div>
            <div className="viz-token-tag">prompt</div>
          </div>
        ))}
        {generatedTokens.map((t, i) => (
          <div className="viz-token-pill viz-token-generated" key={`g-${i}`}>
            <div className="viz-token-text">{t}</div>
            <div className="viz-token-tag">gen</div>
          </div>
        ))}
        {step === 0 && (
          <span style={{ alignSelf: 'center', color: 'var(--ink-faint)', fontStyle: 'italic', fontSize: '0.88rem' }}>
            press <strong>Run prefill</strong> to begin
          </span>
        )}
      </div>

      <div className="viz-matrix-row" style={{ marginTop: 16 }}>
        <MatrixGrid
          data={currentQ}
          label={phase === 'decode'
            ? 'Q this step  (just the new token)'
            : phase === 'prefill'
              ? 'Q this step  (all prompt rows, in parallel)'
              : 'Q this step  (none yet)'}
          rowLabels={
            phase === 'decode'
              ? [generatedTokens[generatedTokens.length - 1] || '·']
              : promptTokens
          }
          colLabels={['q0','q1','q2','q3']}
          cellWidth={46}
        />
        <div className="viz-matrix-op">·</div>
        <MatrixGrid
          data={K}
          label={`K cache  (${K.length} × d_k)`}
          rowLabels={allTokens}
          colLabels={colLabelsK}
          cellWidth={46}
          freshRow={freshRowIdx}
        />
        <div className="viz-matrix-op">·</div>
        <MatrixGrid
          data={V}
          label={`V cache  (${V.length} × d_k)`}
          rowLabels={allTokens}
          colLabels={colLabelsV}
          cellWidth={46}
          freshRow={freshRowIdx}
        />
      </div>

      <div className="viz-counters">
        <Counter
          label="Phase"
          value={phase}
          sub={phase === 'idle' ? 'press prefill →' : phase === 'prefill' ? 'one big matmul' : `step ${decodedSoFar} of ${maxDecodeSteps}`}
        />
        <Counter
          label="Tokens in cache"
          value={K.length}
          sub={`${T_prompt} prompt + ${decodedSoFar} generated`}
        />
        <Counter
          label="New cache rows this step"
          value={newRowsThisStep}
          sub={phase === 'prefill' ? 'one shot, parallel' : phase === 'decode' ? 'exactly one' : '—'}
        />
        <Counter
          label="Cache read this step"
          value={`${cacheBytesReadThisStep} B`}
          sub={phase === 'prefill' ? 'cache was empty' : phase === 'decode' ? 'fp16 · the whole cache' : '—'}
        />
        <Counter
          label="Cache total size"
          value={`${cacheBytes} B`}
          sub={`${K.length} × d_k × 2 (K,V) × fp16`}
        />
      </div>

      <p className="viz-caption">
        <strong>Prefill</strong> processes every prompt token in one parallel pass —
        the cache fills in a single shot. <strong>Decode</strong> appends exactly
        one row to K and V per step, and to score the new token it re-reads
        the <em>entire</em> cache. That asymmetric read-vs-write pattern is the
        whole story of inference cost.
      </p>

      <pre className="viz-code">
{`# Prefill — single parallel matmul over the whole prompt
K_cache = X_prompt @ W_K        # (T_prompt, d_k)
V_cache = X_prompt @ W_V

# Decode — one new token per step, append to cache, re-read all of it
for _ in range(max_new_tokens):
    q  = x_new @ W_Q                              # (1, d_k)
    k  = x_new @ W_K;  v  = x_new @ W_V
    K_cache = torch.cat([K_cache, k], dim=0)      # grow by 1 row
    V_cache = torch.cat([V_cache, v], dim=0)
    A  = (q @ K_cache.T / d_k**0.5).softmax(-1)   # read ALL of K
    y  = A @ V_cache                              # read ALL of V`}
      </pre>
    </div>
  );
}

/* =========================================================
   §1 / §4 — TransformerLayerDiagram
   Static block-diagram of one transformer layer with per-op FLOPs.
   mode='forward'  — clean, per-token costs (§1)
   mode='decode'   — same diagram, K & V highlighted as "saved", plus KV cache memory block (§4)
   ========================================================= */

// Reference numbers for the diagram annotations (Llama-7B-shaped, MHA).
const ARCH_D     = 4096;
const ARCH_D_KV  = 4096;
const ARCH_T     = 2048;
const ARCH_DTYPE = 2; // fp16

function ArchBlock({
  name, shape, flops, numeric, bytes, bytesNumeric,
  kind = 'op', highlight, badge,
}) {
  return (
    <div className={`viz-arch-block viz-arch-block-${kind}${highlight ? ' viz-arch-block-saved' : ''}`}>
      {badge && <div className="viz-arch-badge">{badge}</div>}
      <div className="viz-arch-name">{name}</div>
      {shape && <div className="viz-arch-shape">{shape}</div>}
      {flops && (
        <div className="viz-arch-flops">
          <span className="viz-arch-formula">{flops}</span>
          {numeric && <span className="viz-arch-numeric">{numeric}</span>}
        </div>
      )}
      {bytes && (
        <div className="viz-arch-flops viz-arch-flops-mem">
          <span className="viz-arch-formula">{bytes}</span>
          {bytesNumeric && <span className="viz-arch-numeric">{bytesNumeric}</span>}
        </div>
      )}
    </div>
  );
}

function ArchArrow() {
  return <div className="viz-arch-arrow" aria-hidden="true">▼</div>;
}

function TransformerLayerDiagram({ mode = 'forward' }) {
  const d = ARCH_D, dkv = ARCH_D_KV, T = ARCH_T, dtype = ARCH_DTYPE;
  const num = (f) => fmtNum(f, 'FLOP');
  const numB = (b) => fmtNum(b, 'B');
  const cacheMode = mode === 'decode';

  // FLOPs per op (matmul = 2·M·N·P, applied to 1 new token).
  const flopsQ   = 2 * d * d;
  const flopsK   = 2 * d * dkv;
  const flopsV   = 2 * d * dkv;
  const flopsAtt = 4 * T * d;
  const flopsO   = 2 * d * d;
  const flopsUp  = 8 * d * d;
  const flopsDn  = 8 * d * d;
  const totalFlops = flopsQ + flopsK + flopsV + flopsAtt + flopsO + flopsUp + flopsDn;
  // Naive (no-cache) decode would re-project K and V for ALL T tokens every step,
  // turning the K, V terms into 2T·d·d_kv each (the T-factor blow-up).
  const naiveFlops = flopsQ + flopsAtt + flopsO + flopsUp + flopsDn
                   + 2 * T * d * dkv + 2 * T * d * dkv;
  const savings = naiveFlops / totalFlops;

  // Weight bytes per op (params × dtype, read once per step).
  const bytesQ   = d * d * dtype;
  const bytesK   = d * dkv * dtype;
  const bytesV   = d * dkv * dtype;
  const bytesO   = d * d * dtype;
  const bytesUp  = 4 * d * d * dtype;
  const bytesDn  = 4 * d * d * dtype;
  // Attention has no learnable weights, but in decode mode it reads the KV
  // cache from HBM — that's the cost we annotate on the attention block.
  const bytesAttn = cacheMode ? 2 * dkv * T * dtype : 0;
  const totalBytes = bytesQ + bytesK + bytesV + bytesO + bytesUp + bytesDn + bytesAttn;

  const cacheBytes = 2 * dkv * T * dtype; // KV cache storage per layer per request

  return (
    <div className="viz-arch-diagram">
      <ArchBlock name="Input X" shape="(1, d)" kind="io" />
      <ArchArrow />

      <div className="viz-arch-section">
        <div className="viz-arch-section-label">Attention block</div>
        <div className="viz-arch-parallel">
          <ArchBlock
            name="Q proj"
            shape="(1, d) · (d, d)"
            flops="2d²"          numeric={num(flopsQ)}
            bytes="d²·dtype"      bytesNumeric={numB(bytesQ)}
          />
          <ArchBlock
            name="K proj"
            shape="(1, d) · (d, d_kv)"
            flops="2·d·d_kv"          numeric={num(flopsK)}
            bytes="d·d_kv·dtype"      bytesNumeric={numB(bytesK)}
            highlight={cacheMode}
            badge={cacheMode ? '1 of T' : null}
          />
          <ArchBlock
            name="V proj"
            shape="(1, d) · (d, d_kv)"
            flops="2·d·d_kv"          numeric={num(flopsV)}
            bytes="d·d_kv·dtype"      bytesNumeric={numB(bytesV)}
            highlight={cacheMode}
            badge={cacheMode ? '1 of T' : null}
          />
          {cacheMode && (
            <div className="viz-arch-cache" role="note">
              <div className="viz-arch-badge viz-arch-badge-memory">memory</div>
              <div className="viz-arch-name">KV cache</div>
              <div className="viz-arch-shape">2 · d_kv · T · fp16</div>
              <div className="viz-arch-flops">
                <span className="viz-arch-numeric">{fmtNum(cacheBytes, 'B')} / layer</span>
              </div>
            </div>
          )}
        </div>
        <ArchArrow />
        <ArchBlock
          name="Attention"
          shape="q · Kᵀ  +  A · V"
          flops="4T·d"        numeric={num(flopsAtt)}
          bytes={cacheMode ? '2·d_kv·T·dtype (KV cache)' : null}
          bytesNumeric={cacheMode ? numB(bytesAttn) : null}
        />
        <ArchArrow />
        <ArchBlock
          name="O proj"
          shape="(1, d) · (d, d)"
          flops="2d²"       numeric={num(flopsO)}
          bytes="d²·dtype"  bytesNumeric={numB(bytesO)}
        />
      </div>

      <ArchArrow />

      <div className="viz-arch-section">
        <div className="viz-arch-section-label">FFN block</div>
        <ArchBlock
          name="FFN up"   shape="(1, d) · (d, 4d)"
          flops="8d²"        numeric={num(flopsUp)}
          bytes="4d²·dtype"  bytesNumeric={numB(bytesUp)}
        />
        <ArchArrow />
        <ArchBlock
          name="FFN down" shape="(1, 4d) · (4d, d)"
          flops="8d²"        numeric={num(flopsDn)}
          bytes="4d²·dtype"  bytesNumeric={numB(bytesDn)}
        />
      </div>

      <ArchArrow />
      <ArchBlock name="Output" shape="(1, d)" kind="io" />

      <div className="viz-arch-total">
        {cacheMode ? (
          <>
            <div className="viz-arch-total-line viz-arch-total-naive">
              <span className="viz-arch-total-key">Naive compute</span>
              <span className="viz-arch-total-formula">
                4d² +{' '}
                <span className="viz-arch-bad-term">4T·d·d_kv <em>← K,V × T tokens (redundant)</em></span>
                {' '}+ 16d² + 4T·d
              </span>
              <span className="viz-arch-total-value">≈ <strong>{num(naiveFlops)}</strong></span>
            </div>
            <div className="viz-arch-total-line viz-arch-total-cached">
              <span className="viz-arch-total-key">Cached compute</span>
              <span className="viz-arch-total-formula">
                4d² +{' '}
                <span className="viz-arch-good-term">4·d·d_kv <em>← K,V × 1 token (saved!)</em></span>
                {' '}+ 16d² + 4T·d
              </span>
              <span className="viz-arch-total-value">
                ≈ <strong>{num(totalFlops)}</strong>
                <span className="viz-arch-savings">
                  {savings >= 100 ? Math.round(savings).toLocaleString() : savings.toFixed(1)}× less
                </span>
              </span>
            </div>
          </>
        ) : (
          <div className="viz-arch-total-line">
            <span className="viz-arch-total-key">Compute</span>
            <span className="viz-arch-total-formula">4d² + 4·d·d_kv + 16d² + 4T·d</span>
            <span className="viz-arch-total-value">≈ <strong>{num(totalFlops)}</strong></span>
          </div>
        )}
        <div className="viz-arch-total-line viz-arch-total-mem">
          <span className="viz-arch-total-key">Memory{cacheMode ? ' (weights + KV read)' : ' (weights)'}</span>
          <span className="viz-arch-total-formula">
            {cacheMode ? (
              <>
                (10d² + 2·d·d_kv{' '}
                <span className="viz-arch-new-term">+ 2·d_kv·T <em>← new: KV read</em></span>
                ) · dtype
              </>
            ) : (
              '(10d² + 2·d·d_kv) · dtype'
            )}
          </span>
          <span className="viz-arch-total-value">≈ <strong>{numB(totalBytes)}</strong></span>
        </div>
        <span className="viz-arch-total-note">
          per layer per token · d = {d}, d_kv = {dkv}, T = {T} · MHA · fp16
        </span>
      </div>

      {cacheMode && (
        <p className="viz-caption" style={{ marginTop: 14 }}>
          The two compute lines above are the whole story of caching at a
          glance: every term is identical except K, V — without the cache
          you'd re-project them for all <strong>T</strong> tokens every step,
          with it you project them for the one new token. The single boxed
          term shrinks by a factor of T; the FFN, attention compute, Q and O
          projections are untouched. The cost: the orange{' '}
          <strong>KV cache</strong> block on the right is now live in HBM,
          adding {fmtNum(cacheBytes, 'B')} per layer per request to the
          memory bill — which is exactly the "+ 2·d_kv·T" term in the memory
          line below.
        </p>
      )}
    </div>
  );
}

/* =========================================================
   §2 — Cost breakdown columns (static comparison)
   ========================================================= */

const COST_D_OPTIONS = [256, 512, 1024, 2048, 4096, 8192, 12288];
const COST_T_OPTIONS = [128, 512, 1024, 2048, 4096, 8192, 16384, 32768];
const COST_N_OPTIONS = [1, 16, 64, 256, 1024, 4096, 16384];

function CostBreakdown() {
  const [dIdx, setDIdx] = useState(4); // 4096
  const [tIdx, setTIdx] = useState(3); // 2048
  const [nIdx, setNIdx] = useState(3); // 256

  const d = COST_D_OPTIONS[dIdx];
  const T = COST_T_OPTIONS[tIdx];
  const N = COST_N_OPTIONS[nIdx];

  // Per-step FLOPs (treating T as the current context length).
  // Matches the §1 diagram exactly: matmul = 2·M·N·P FLOPs, includes Q+O+K+V,
  // FFN (16d²), and attention compute (4T·d). MHA is assumed (d_kv = d).
  //
  //   per-op FLOPs (one new token):
  //     Q proj           : 2d²
  //     O proj           : 2d²
  //     K proj (cache)   : 2d²              | K proj (no cache) : 2T·d²
  //     V proj (cache)   : 2d²              | V proj (no cache) : 2T·d²
  //     FFN up + down    : 16d²
  //     attention q·Kᵀ+A·V: 4T·d
  //
  //   total cached   = 24d² + 4T·d
  //   total no-cache = 20d² + 4T·d² + 4T·d
  const perStepCache    = 24 * d * d + 4 * T * d;
  const perStepNoCache  = 20 * d * d + 4 * T * d * d + 4 * T * d;
  const cumNoCache      = N * perStepNoCache;
  const cumCache        = N * perStepCache;
  const ratio           = perStepNoCache / perStepCache;

  // Memory cost of the cache itself (per layer, fp16, MHA-shaped).
  // 2 × T × d × 2 bytes = 4·T·d bytes.
  const cacheBytesPerLayer = 4 * T * d;
  // The cache grows by one row of K and V per decode step, so after generating
  // N more tokens it holds (T + N) rows per layer.
  const cacheBytesPerLayerAfterN = 4 * (T + N) * d;
  // Common open-weights models live around 32 layers; show the all-layers
  // total too so the reader sees real-world cache magnitudes.
  const N_LAYERS_REF = 32;
  const cacheBytesAllLayers       = cacheBytesPerLayer       * N_LAYERS_REF;
  const cacheBytesAllLayersAfterN = cacheBytesPerLayerAfterN * N_LAYERS_REF;

  return (
    <div className="viz-panel">
      <div className="viz-cost-pair">
        <div className="viz-cost-col viz-cost-no-cache">
          <div className="viz-cost-title">
            <span className="viz-cost-badge viz-cost-badge-no">✗</span>
            Without cache · per decode step
          </div>
          <div className="viz-cost-row">
            <span>Recompute K for <strong>all T tokens</strong></span>
            <span className="viz-cost-mag">2T · d²</span>
          </div>
          <div className="viz-cost-row">
            <span>Recompute V for <strong>all T tokens</strong></span>
            <span className="viz-cost-mag">2T · d²</span>
          </div>
          <div className="viz-cost-row">
            <span>Project Q for the 1 new token</span>
            <span className="viz-cost-mag">2d²</span>
          </div>
          <div className="viz-cost-row">
            <span>Project O for the 1 new token</span>
            <span className="viz-cost-mag">2d²</span>
          </div>
          <div className="viz-cost-row">
            <span>Attention scores  q · Kᵀ</span>
            <span className="viz-cost-mag">2T · d</span>
          </div>
          <div className="viz-cost-row">
            <span>Weighted sum  A · V</span>
            <span className="viz-cost-mag">2T · d</span>
          </div>
          <div className="viz-cost-row">
            <span>FFN up + down  (8d² each)</span>
            <span className="viz-cost-mag">16d²</span>
          </div>
          <div className="viz-cost-total">
            per step ≈ <strong>20d² + 4T·d² + 4T·d</strong> ≈{' '}
            <strong>{fmtNum(perStepNoCache, 'FLOP')}</strong>
            <em> · the 4T·d² term is the redundant K, V projections</em>
          </div>
          <div className="viz-cost-cumulative">
            over N = {N.toLocaleString()} tokens:{' '}
            <strong>{fmtNum(cumNoCache, 'FLOP')}</strong>
          </div>
        </div>

        <div className="viz-cost-col viz-cost-with-cache">
          <div className="viz-cost-title">
            <span className="viz-cost-badge viz-cost-badge-yes">✓</span>
            With cache · per decode step
          </div>
          <div className="viz-cost-row">
            <span>Project K for the <strong>1 new token</strong>, append to cache</span>
            <span className="viz-cost-mag">2d²</span>
          </div>
          <div className="viz-cost-row">
            <span>Project V for the <strong>1 new token</strong>, append to cache</span>
            <span className="viz-cost-mag">2d²</span>
          </div>
          <div className="viz-cost-row">
            <span>Project Q for the 1 new token</span>
            <span className="viz-cost-mag">2d²</span>
          </div>
          <div className="viz-cost-row">
            <span>Project O for the 1 new token</span>
            <span className="viz-cost-mag">2d²</span>
          </div>
          <div className="viz-cost-row">
            <span>Attention scores  q · Kᵀ <em>(read whole cache)</em></span>
            <span className="viz-cost-mag">2T · d</span>
          </div>
          <div className="viz-cost-row">
            <span>Weighted sum  A · V <em>(read whole cache)</em></span>
            <span className="viz-cost-mag">2T · d</span>
          </div>
          <div className="viz-cost-row">
            <span>FFN up + down  (8d² each)</span>
            <span className="viz-cost-mag">16d²</span>
          </div>
          <div className="viz-cost-total">
            per step ≈ <strong>24d² + 4T·d</strong> ≈{' '}
            <strong>{fmtNum(perStepCache, 'FLOP')}</strong>
            <em> · the T-factor on K, V projections is gone</em>
          </div>
          <div className="viz-cost-cumulative">
            over N = {N.toLocaleString()} tokens:{' '}
            <strong>{fmtNum(cumCache, 'FLOP')}</strong>
          </div>
        </div>
      </div>

      <div className="viz-cost-sliders">
        <label className="viz-cost-slider">
          <span className="viz-cost-slider-label">
            d <em>(d_model — model width)</em>: <strong>{d.toLocaleString()}</strong>
          </span>
          <input
            type="range"
            min={0}
            max={COST_D_OPTIONS.length - 1}
            step={1}
            value={dIdx}
            onChange={(e) => setDIdx(+e.target.value)}
          />
        </label>
        <label className="viz-cost-slider">
          <span className="viz-cost-slider-label">
            T <em>(context length at decode time — tokens already in the cache)</em>: <strong>{T.toLocaleString()}</strong>
          </span>
          <input
            type="range"
            min={0}
            max={COST_T_OPTIONS.length - 1}
            step={1}
            value={tIdx}
            onChange={(e) => setTIdx(+e.target.value)}
          />
        </label>
        <label className="viz-cost-slider">
          <span className="viz-cost-slider-label">
            N <em>(decode steps to integrate over — new tokens to generate)</em>: <strong>{N.toLocaleString()}</strong>
          </span>
          <input
            type="range"
            min={0}
            max={COST_N_OPTIONS.length - 1}
            step={1}
            value={nIdx}
            onChange={(e) => setNIdx(+e.target.value)}
          />
        </label>
      </div>
      <div className="viz-cost-tn-note">
        <strong>T vs N:</strong> <Katex tex="T" /> is the context already sitting
        in the cache when you start a decode step.{' '}
        <Katex tex="N" /> is how many <em>more</em> tokens you plan to generate.
        The per-step rows use <Katex tex="T" />; the "over N tokens" line just
        multiplies that by <Katex tex="N" />. Strictly, <Katex tex="T" /> grows
        by 1 each step — we treat it as constant here, which is a clean
        approximation when <Katex tex="N \ll T" />.
      </div>

      <div className="viz-cost-readouts">
        <div className="viz-cost-readout viz-cost-readout-good">
          <div className="viz-cost-readout-label">Compute saved</div>
          <div className="viz-cost-readout-value">
            {ratio < 10
              ? ratio.toFixed(2)
              : ratio < 100
              ? ratio.toFixed(1)
              : Math.round(ratio).toLocaleString()}× fewer FLOPs/step
          </div>
          <div className="viz-cost-readout-sub">
            cumulative over N: {fmtNum(cumNoCache, 'FLOP')} → {fmtNum(cumCache, 'FLOP')}
          </div>
        </div>
        <div className="viz-cost-readout viz-cost-readout-bad">
          <div className="viz-cost-readout-label">Memory it costs</div>
          <div className="viz-cost-readout-value">
            {fmtNum(cacheBytesPerLayer, 'B')} per layer
            <span className="viz-cost-readout-arrow"> → </span>
            {fmtNum(cacheBytesPerLayerAfterN, 'B')}{' '}
            <span className="viz-cost-readout-after">after N more</span>
          </div>
          <div className="viz-cost-readout-sub">
            ≈ {fmtNum(cacheBytesAllLayers, 'B')} →{' '}
            {fmtNum(cacheBytesAllLayersAfterN, 'B')} across {N_LAYERS_REF} layers · fp16 · per sequence.
            <br />
            Cache grows by <strong>2·d_kv·dtype</strong> bytes per layer per
            decode step (one row of K and V).
          </div>
        </div>
      </div>

      <p className="viz-caption" style={{ marginTop: 14 }}>
        The cache moves the <strong>T factor</strong> off the K and V
        projection rows (the big-d² ones) and onto attention only (the
        small-d rows). Drag T up and watch the left column blow up while the
        right column barely twitches — but the orange "Memory it costs"
        panel climbs in lockstep. That's the trade we keep paying: more
        compute saved → more memory consumed. §7 (GQA) and §8 (PagedAttention)
        are how the field keeps that memory bill bounded. <em>(MHA assumed
        here for clean numbers — d_kv = d. GQA shrinks K and V row costs by{' '}
        n_q / n_kv.)</em>
      </p>
    </div>
  );
}

/* =========================================================
   Info box — reusable callout for asides
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
   §3 — Interactive B: Arithmetic Intensity Calculator
   ========================================================= */

// H100 SXM reference numbers — Nvidia spec sheet (dense tensor-core math).
const H100_HBM_BYTES_PER_S = 3.35e12;          // 3.35 TB/s HBM3 bandwidth
const H100_TFLOPS_FP16 = 989;                  // dense fp16 / bf16 TFLOPs
const H100_RIDGE_FP16 = (H100_TFLOPS_FP16 * 1e12) / H100_HBM_BYTES_PER_S; // ≈ 295

const SEQ_LENS    = [64, 256, 1024, 4096, 16384, 65536];
const BATCH_SIZES = [1, 4, 16, 64, 256];
const D_MODELS    = [768, 1024, 2048, 4096, 8192, 12288];
const N_LAYERS    = [12, 24, 32, 48, 80, 96];
const N_Q_HEADS_FOR_DM = { 768: 12, 1024: 16, 2048: 32, 4096: 32, 8192: 64, 12288: 96 };
const PRECISIONS  = [
  // peakFlops: dense tensor-core throughput on H100 at that precision (FLOPs/s).
  { id: 'fp16', label: 'fp16',  bytes: 2,   peakFlops: 989e12 },
  { id: 'int8', label: 'int8',  bytes: 1,   peakFlops: 1979e12 },
  { id: 'fp4',  label: 'fp4',   bytes: 0.5, peakFlops: 3958e12 },
];

function fmtTime(s) {
  if (s >= 1)    return `${s.toFixed(2)} s`;
  if (s >= 1e-3) return `${(s * 1e3).toFixed(2)} ms`;
  if (s >= 1e-6) return `${(s * 1e6).toFixed(1)} µs`;
  return `${(s * 1e9).toFixed(0)} ns`;
}

// Tooltip wrapper: hover or tap-focus reveals the formula breakdown.
function Tip({ children, label }) {
  return (
    <span className="viz-tip" tabIndex={0}>
      {children}
      <span className="viz-tip-content" role="tooltip">
        {label}
      </span>
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

function fmtNum(x, unit) {
  if (!Number.isFinite(x) || x === 0) return `0 ${unit}`;
  const abs = Math.abs(x);
  if (abs >= 1e12) return `${(x / 1e12).toFixed(2)} T${unit}`;
  if (abs >= 1e9)  return `${(x / 1e9).toFixed(2)} G${unit}`;
  if (abs >= 1e6)  return `${(x / 1e6).toFixed(2)} M${unit}`;
  if (abs >= 1e3)  return `${(x / 1e3).toFixed(2)} k${unit}`;
  return `${x.toFixed(2)} ${unit}`;
}

// ---------- Shared compute helper: roofline numbers from a config ----------
function computeSimMetrics({ mode, T, B, d_model, n_layers, n_kv, prec }) {
  const n_q_heads = N_Q_HEADS_FOR_DM[d_model] || 32;
  const kv = Math.min(n_kv, n_q_heads);
  const d_head = d_model / n_q_heads;
  const d_kv_total = kv * d_head;

  // FLOPs per layer per token  (matmul = 2·M·N·P)
  //   Q + O = 4d²,  K + V (cached) = 4·d·d_kv,  FFN = 16d²,  attention = 4·T·d
  const flopsPerToken =
    4 * d_model * d_model +
    4 * d_model * d_kv_total +
    16 * d_model * d_model +
    4 * T * d_model;

  // tokens flowing through ALL layers this step:
  //   decode   → 1 new token per request   → B tokens
  //   training → T tokens in parallel       → B · T tokens
  const tokensPerStep = mode === 'training' ? B * T : B;
  const flopsPerStep = tokensPerStep * n_layers * flopsPerToken;

  // Bytes moved per step
  const weightParamsPerLayer = 10 * d_model * d_model + 2 * d_model * d_kv_total;
  const weightBytes = n_layers * weightParamsPerLayer * prec.bytes;
  // KV cache read only happens in decode (we read prior K, V). In training we
  // build the activations on the fly and there's no cache to read.
  const kvBytes = mode === 'decode'
    ? B * n_layers * 2 * d_kv_total * T * prec.bytes
    : 0;
  const totalBytes = weightBytes + kvBytes;

  const ai = flopsPerStep / totalBytes;
  const ridge = prec.peakFlops / H100_HBM_BYTES_PER_S;
  const isComputeBound = ai >= ridge;

  const stepTime = Math.max(
    flopsPerStep / prec.peakFlops,
    totalBytes / H100_HBM_BYTES_PER_S
  );
  // Throughput is total tokens produced per second.
  //   decode → B tokens per step
  //   training → B · T tokens per forward pass
  const tokensProduced = mode === 'training' ? B * T : B;
  const tokensPerSec = tokensProduced / stepTime;

  return {
    flopsPerToken, flopsPerStep, weightBytes, kvBytes, totalBytes,
    ai, ridge, isComputeBound, stepTime, tokensPerSec,
    d_kv_total, n_q_heads, kv,
  };
}

// ---------- Presets ----------
// Each preset corresponds to a known model + workload. The tok/s on the
// button is computed live with the same formula as the rest of the widget.
const SIM_PRESETS = [
  {
    id: 'llama7b-decode-1',
    label: 'Llama-7B · decode · B=1',
    sub: 'single-user latency',
    cfg: { mode: 'decode', seqIdx: 3, batchIdx: 0, dmIdx: 3, layIdx: 2, kv: 32, precId: 'fp16' },
  },
  {
    id: 'llama7b-decode-64',
    label: 'Llama-7B · decode · B=64',
    sub: 'small-scale serving',
    cfg: { mode: 'decode', seqIdx: 3, batchIdx: 3, dmIdx: 3, layIdx: 2, kv: 32, precId: 'fp16' },
  },
  {
    id: 'llama70b-gqa8',
    label: 'Llama-70B · GQA-8 · B=4',
    sub: 'big model, real GQA',
    cfg: { mode: 'decode', seqIdx: 3, batchIdx: 1, dmIdx: 4, layIdx: 4, kv: 8, precId: 'fp16' },
  },
  {
    id: 'mistral-7b',
    label: 'Mistral-7B · GQA-8 · B=64',
    sub: 'efficient serving',
    cfg: { mode: 'decode', seqIdx: 3, batchIdx: 3, dmIdx: 3, layIdx: 2, kv: 8, precId: 'fp16' },
  },
  {
    id: 'llama7b-training',
    label: 'Llama-7B · training · B=1',
    sub: 'forward pass over T tokens',
    cfg: { mode: 'training', seqIdx: 3, batchIdx: 0, dmIdx: 3, layIdx: 2, kv: 32, precId: 'fp16' },
  },
];

function presetToConfig(p) {
  return {
    mode: p.cfg.mode,
    T: SEQ_LENS[p.cfg.seqIdx],
    B: BATCH_SIZES[p.cfg.batchIdx],
    d_model: D_MODELS[p.cfg.dmIdx],
    n_layers: N_LAYERS[p.cfg.layIdx],
    n_kv: p.cfg.kv,
    prec: PRECISIONS.find((x) => x.id === p.cfg.precId),
  };
}

function SectionAICalc() {
  const [mode, setMode]           = useState('decode');
  const [seqIdx, setSeqIdx]       = useState(3);  // 2048 (matches §1 reference)
  const [batchIdx, setBatchIdx]   = useState(0);  // 1
  const [dmIdx, setDmIdx]         = useState(3);  // 4096
  const [layIdx, setLayIdx]       = useState(2);  // 32
  const [precId, setPrecId]       = useState('fp16');
  const [kvHeads, setKvHeads]     = useState(32); // MHA by default

  const T        = SEQ_LENS[seqIdx];
  const B        = BATCH_SIZES[batchIdx];
  const d_model  = D_MODELS[dmIdx];
  const n_layers = N_LAYERS[layIdx];
  const prec     = PRECISIONS.find((p) => p.id === precId);
  const n_q_heads = N_Q_HEADS_FOR_DM[d_model] || 32;
  const n_kv = Math.min(kvHeads, n_q_heads);
  const d_head = d_model / n_q_heads;
  const d_kv_total = n_kv * d_head;

  // Apply a preset to all sliders.
  const applyPreset = (p) => {
    setMode(p.cfg.mode);
    setSeqIdx(p.cfg.seqIdx);
    setBatchIdx(p.cfg.batchIdx);
    setDmIdx(p.cfg.dmIdx);
    setLayIdx(p.cfg.layIdx);
    setKvHeads(p.cfg.kv);
    setPrecId(p.cfg.precId);
  };

  // Live metrics for the active config + each preset (for the button labels).
  const m = computeSimMetrics({ mode, T, B, d_model, n_layers, n_kv, prec });
  const presetMetrics = SIM_PRESETS.map((p) => ({
    p,
    metrics: computeSimMetrics(presetToConfig(p)),
  }));

  // Is the active config exactly equal to a preset?
  const activePresetId = (() => {
    for (const { p } of presetMetrics) {
      if (
        p.cfg.mode === mode &&
        p.cfg.seqIdx === seqIdx &&
        p.cfg.batchIdx === batchIdx &&
        p.cfg.dmIdx === dmIdx &&
        p.cfg.layIdx === layIdx &&
        p.cfg.kv === n_kv &&
        p.cfg.precId === precId
      ) return p.id;
    }
    return null;
  })();

  // Pull the active metrics out of the helper for use in the JSX.
  const {
    flopsPerStep, weightBytes, kvBytes, totalBytes, ai, ridge,
    isComputeBound, stepTime, tokensPerSec,
  } = m;

  // Position the marker on the log-scale meter (0.1 ... 100k FLOPs/B).
  const meterLo = Math.log10(0.1);
  const meterHi = Math.log10(100000);
  const meterPct = (v) => {
    const x = Math.max(meterLo, Math.min(meterHi, Math.log10(Math.max(0.1, v))));
    return ((x - meterLo) / (meterHi - meterLo)) * 100;
  };
  const markerPct = meterPct(ai);
  const ridgePct  = meterPct(ridge);

  const isTraining = mode === 'training';
  // The label for "per-step throughput": decode produces B tokens/step,
  // training produces B·T tokens per forward pass.
  const throughputDenom = isTraining ? `${B} × ${T.toLocaleString()}` : `${B}`;

  // Attention-type quick selector. The three canonical regimes set n_kv to:
  //   MHA: n_kv = n_q (full multi-head, biggest cache)
  //   GQA: n_kv = 8 (or n_q/4 if smaller) — Llama-2-70B / Mistral style
  //   MQA: n_kv = 1 (one shared K/V head)
  const attentionType =
    n_kv === n_q_heads ? 'MHA' :
    n_kv === 1         ? 'MQA' :
                         'GQA';
  const setAttentionType = (t) => {
    if (t === 'MHA') setKvHeads(n_q_heads);
    else if (t === 'MQA') setKvHeads(1);
    else /* GQA */ {
      // Pick a sensible group size: target 8, but clamp to be smaller than n_q
      // and at least 2 (otherwise it'd be MQA).
      const target = Math.min(8, Math.max(2, Math.floor(n_q_heads / 4)));
      setKvHeads(target);
    }
  };

  return (
    <div className="viz-panel">
      <div className="viz-sim-controls-grid">
        <div className="viz-sim-control-col">
          <div className="viz-sim-col-label">Workload</div>
          <label>
            <span>context T</span>
            <select value={seqIdx} onChange={(e) => setSeqIdx(+e.target.value)}>
              {SEQ_LENS.map((v, i) => <option key={i} value={i}>{v}</option>)}
            </select>
          </label>
          <label>
            <span>batch B</span>
            <select value={batchIdx} onChange={(e) => setBatchIdx(+e.target.value)}>
              {BATCH_SIZES.map((v, i) => <option key={i} value={i}>{v}</option>)}
            </select>
          </label>
          <label>
            <span>precision</span>
            <select value={precId} onChange={(e) => setPrecId(e.target.value)}>
              {PRECISIONS.map((p) => <option key={p.id} value={p.id}>{p.label}</option>)}
            </select>
          </label>
        </div>

        <div className="viz-sim-control-col">
          <div className="viz-sim-col-label">Model</div>
          <label>
            <span>d_model</span>
            <select
              value={dmIdx}
              onChange={(e) => {
                setDmIdx(+e.target.value);
                setKvHeads(N_Q_HEADS_FOR_DM[D_MODELS[+e.target.value]] || 32);
              }}
            >
              {D_MODELS.map((v, i) => <option key={i} value={i}>{v}</option>)}
            </select>
          </label>
          <label>
            <span>n_layers</span>
            <select value={layIdx} onChange={(e) => setLayIdx(+e.target.value)}>
              {N_LAYERS.map((v, i) => <option key={i} value={i}>{v}</option>)}
            </select>
          </label>
        </div>

        <div className="viz-sim-control-col">
          <div className="viz-sim-col-label">Attention</div>
          <label>
            <span>type</span>
            <div className="viz-tabs viz-sim-attn-tabs" role="tablist">
              {['MHA', 'GQA', 'MQA'].map((t) => (
                <button
                  key={t}
                  role="tab"
                  className={`viz-tab ${attentionType === t ? 'active' : ''}`}
                  onClick={() => setAttentionType(t)}
                  title={
                    t === 'MHA' ? `n_kv = n_q = ${n_q_heads}` :
                    t === 'GQA' ? `n_kv ∈ (1, n_q) — sets n_kv = ${Math.min(8, Math.max(2, Math.floor(n_q_heads / 4)))}` :
                                  'n_kv = 1 (one shared K/V head)'
                  }
                >
                  {t}
                </button>
              ))}
            </div>
          </label>
          <label>
            <span>n_kv_heads</span>
            <select value={n_kv} onChange={(e) => setKvHeads(+e.target.value)}>
              {[1, 2, 4, 8, 16, 32, 64, 96].filter((h) => h <= n_q_heads).map((h) => (
                <option key={h} value={h}>{h}</option>
              ))}
            </select>
          </label>
        </div>
      </div>

      {/* Preset buttons — each shows live tok/s */}
      <div className="viz-sim-presets">
        {presetMetrics.map(({ p, metrics }) => (
          <button
            key={p.id}
            className={`viz-sim-preset${activePresetId === p.id ? ' active' : ''}`}
            onClick={() => applyPreset(p)}
            title={p.sub}
          >
            <div className="viz-sim-preset-label">{p.label}</div>
            <div className="viz-sim-preset-sub">{p.sub}</div>
            <div className="viz-sim-preset-tps">
              {Math.round(metrics.tokensPerSec).toLocaleString()} tok/s
            </div>
          </button>
        ))}
      </div>

      {/* Mode toggle */}
      <div className="viz-controls" style={{ marginTop: 14 }}>
        <span style={{ color: 'var(--ink-soft)', fontSize: '0.9rem' }}>Mode:</span>
        <div className="viz-tabs" role="tablist">
          {[
            { id: 'decode',   label: 'Inference (decode)' },
            { id: 'training', label: 'Training (forward)' },
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

      {/* Single-column readout */}
      <div className="viz-sim-readout">
        <div className="viz-ai-row">
          <span>FLOPs per step</span>
          <strong>
            <Tip label={tipBody(
              isTraining
                ? '(B × T) × L × (4d² + 4d·d_kv + 16d² + 4T·d)'
                : 'B × L × (4d² + 4d·d_kv + 16d² + 4T·d)',
              `${isTraining ? `${B}·${T.toLocaleString()}` : B} × ${n_layers} × (${fmtNum(4*d_model*d_model,'FLOP')} + ${fmtNum(4*d_model*d_kv_total,'FLOP')} + ${fmtNum(16*d_model*d_model,'FLOP')} + ${fmtNum(4*T*d_model,'FLOP')})`,
              fmtNum(flopsPerStep, 'FLOP') + (isTraining ? ' per forward pass' : ' per decode step')
            )}>
              {fmtNum(flopsPerStep, 'FLOP')}
            </Tip>
          </strong>
        </div>
        <div className="viz-ai-row">
          <span>Bytes — weights (shared across batch)</span>
          <strong>
            <Tip label={tipBody(
              'L × (10d² + 2d·d_kv) × dtype',
              `${n_layers} × (${fmtNum(10*d_model*d_model,'B')} + ${fmtNum(2*d_model*d_kv_total,'B')}) × ${prec.bytes} B`,
              fmtNum(weightBytes, 'B')
            )}>
              {fmtNum(weightBytes, 'B')}
            </Tip>
          </strong>
        </div>
        {!isTraining && (
          <div className="viz-ai-row">
            <span>Bytes — KV cache · per request</span>
            <strong>
              <Tip label={tipBody(
                'B × L × 2·d_kv·T × dtype',
                `${B} × ${n_layers} × 2·${d_kv_total.toLocaleString()}·${T.toLocaleString()} × ${prec.bytes} B`,
                fmtNum(kvBytes, 'B')
              )}>
                {fmtNum(kvBytes, 'B')}
              </Tip>
            </strong>
          </div>
        )}
        {isTraining && (
          <div className="viz-ai-row" style={{ color: 'var(--ink-faint)', fontStyle: 'italic' }}>
            <span>Bytes — KV cache</span>
            <strong>not used in training (no cache to read)</strong>
          </div>
        )}
        <div className="viz-ai-row viz-ai-row-total">
          <span>Arithmetic intensity</span>
          <strong>
            <Tip label={tipBody(
              'FLOPs / total bytes',
              `${fmtNum(flopsPerStep,'FLOP')} / ${fmtNum(totalBytes,'B')}`,
              `${ai >= 100 ? Math.round(ai).toLocaleString() : ai.toFixed(2)} FLOPs/B`
            )}>
              {ai >= 100 ? Math.round(ai).toLocaleString() : ai.toFixed(2)} FLOPs/B
            </Tip>
          </strong>
        </div>
        <div className="viz-ai-row">
          <span>{isTraining ? 'Forward-pass time' : 'Per-token latency'} · max(compute, memory)</span>
          <strong>
            <Tip label={tipBody(
              'max(FLOPs / peak compute, bytes / HBM)',
              `max(${fmtNum(flopsPerStep,'FLOP')} / ${fmtNum(prec.peakFlops,'FLOP')}/s, ${fmtNum(totalBytes,'B')} / 3.35 TB/s)`,
              `max(${fmtTime(flopsPerStep/prec.peakFlops)}, ${fmtTime(totalBytes/H100_HBM_BYTES_PER_S)}) = ${fmtTime(stepTime)}`
            )}>
              {fmtTime(stepTime)}
            </Tip>
          </strong>
        </div>
        <div className="viz-ai-row viz-ai-row-throughput">
          <span>Throughput</span>
          <strong>
            <Tip label={tipBody(
              isTraining ? '(B × T) / step time' : 'B / step time',
              `${throughputDenom} / ${fmtTime(stepTime)}`,
              `${Math.round(tokensPerSec).toLocaleString()} tok/s`
            )}>
              {Math.round(tokensPerSec).toLocaleString()} tok/s
            </Tip>
          </strong>
        </div>
      </div>

      <div className="viz-ai-assumptions">
        <strong>Assumptions for the tok/s numbers above:</strong>{' '}
        a single NVIDIA H100 SXM (3.35 TB/s HBM3, peak dense{' '}
        {prec.label === 'fp16' ? '989 fp16' : prec.label === 'int8' ? '1,979 int8' : '3,958 fp4'}{' '}
        TFLOPs/s), step time = max(FLOPs / peak compute, bytes / HBM bandwidth)
        — the theoretical roofline. Real-world utilization is typically{' '}
        60–80% of these numbers (kernel launch overhead, non-FFN/attention ops,
        scheduling). Treat the values as a "what's achievable in principle"
        bound, not a benchmark.
      </div>

      <div style={{ marginTop: 24 }}>
        <div className="viz-ai-meter-header">
          <span className="viz-ai-zone viz-ai-zone-mem">← memory-bandwidth-bound (HBM is the limit)</span>
          <span className="viz-ai-meter-title">arithmetic intensity (log scale)</span>
          <span className="viz-ai-zone viz-ai-zone-cmp">compute-bound (math units are the limit) →</span>
        </div>
        <div className="viz-ai-meter">
          <div className="viz-ai-ridge" style={{ left: `${ridgePct}%` }} />
          <div className="viz-ai-ridge-label" style={{ left: `${ridgePct}%` }}>
            H100 ridge ≈ {ridge.toFixed(0)} FLOPs/B ({prec.label})
          </div>
          <div
            className={`viz-ai-marker ${isComputeBound ? 'viz-ai-marker-yes' : 'viz-ai-marker-no'}`}
            style={{ left: `calc(${markerPct}% - 2px)` }}
            title={`AI: ${ai.toFixed(2)} FLOPs/B`}
          />
        </div>
        <div className="viz-ai-axis">
          <span>0.1</span><span>1</span><span>10</span><span>100</span><span>1k</span><span>10k</span><span>100k</span>
        </div>
        <div className="viz-ai-marker-legend">
          <span className="viz-ai-ridge-legend">
            <span className="viz-ai-ridge-swatch" /> The black line is the H100 ridge point — the FLOPs/byte
            you need to keep the tensor cores fed at full bandwidth. Right of it: math-bound. Left of it:
            memory-bound.
          </span>
        </div>
      </div>

      <div className={`viz-ai-verdict ${isComputeBound ? 'viz-ai-verdict-compute' : 'viz-ai-verdict-memory'}`} style={{ marginTop: 14 }}>
        <strong>{isTraining ? 'Training (forward):' : 'Inference (decode):'}</strong>{' '}
        {isComputeBound ? (
          <>compute-bound — AI sits {Math.round(ai / ridge)}× above the ridge. The math units
          are the limit; HBM has spare bandwidth.</>
        ) : (
          <>memory-bandwidth-bound — AI sits {(ridge / ai).toFixed(0)}× below the ridge.
          Most wall-clock time is the GPU waiting on HBM reads.</>
        )}
      </div>

      <div className="viz-ai-assumptions">
        <strong>Assumptions for the tok/s numbers above:</strong>{' '}
        a single NVIDIA H100 SXM (3.35 TB/s HBM3, peak dense{' '}
        {prec.label === 'fp16' ? '989 fp16' : prec.label === 'int8' ? '1,979 int8' : '3,958 fp4'}{' '}
        TFLOPs/s), step time = max(FLOPs / peak compute, bytes / HBM bandwidth)
        — the theoretical roofline. Real-world utilization is typically{' '}
        60–80% of these numbers. Training mode shows the forward pass only;
        the backward roughly doubles the FLOPs but keeps the AI verdict the
        same.
      </div>

      <pre className="viz-code">
{`# Matmul FLOPs = 2·M·N·P    decode = 1 new token / step,  training = T tokens parallel
# Per layer per token: 4d² + 4·d·d_kv + 16d² + 4·T·d
flops_per_token   = 4*d**2 + 4*d*d_kv + 16*d**2 + 4*T*d
tokens_per_step   = B if mode == "decode" else B * T
flops_per_step    = tokens_per_step * n_layers * flops_per_token

# Bytes
bytes_weight = n_layers * (10*d**2 + 2*d*d_kv) * dtype
bytes_kv     = B * n_layers * 2 * d_kv * T * dtype  if mode == "decode" else 0
bytes_total  = bytes_weight + bytes_kv

AI         = flops_per_step / bytes_total
step_time  = max(flops_per_step / peak_flops, bytes_total / HBM_bw)
throughput = tokens_per_step / step_time     # tokens / second`}
      </pre>
    </div>
  );
}

/* =========================================================
   §4 — Interactive C: MHA → GQA → MQA
   ========================================================= */

const KV_HEAD_OPTIONS = [1, 2, 4, 8]; // with n_q_heads = 8
const N_Q_HEADS = 8;

const REAL_MODELS_GQA = [
  { name: 'Llama 2 7B (MHA)',   q: 32, kv: 32 },
  { name: 'Llama 2 70B (GQA-8)',q: 64, kv: 8 },
  { name: 'Mistral 7B (GQA-8)', q: 32, kv: 8 },
  { name: 'MQA (Shazeer 2019)', q: '*', kv: 1 },
  { name: 'DeepSeek-V2 (MLA)',  q: '*', kv: 'latent' },
];

function SectionGQA() {
  const [n_kv, setNkv] = useState(8); // MHA by default

  // Map each Q head to a KV head group (round-robin).
  const groupOf = (qHead) => Math.floor(qHead * n_kv / N_Q_HEADS);

  // Bytes saved relative to MHA, for one reference config:
  // Llama-style: 32 layers, d_head=128, T=8192, fp16.
  const refLayers = 32, refDHead = 128, refSeqLen = 8192, refBytes = 2;
  const cacheBytes = (kv) =>
    refLayers * 2 * (kv * refDHead) * refSeqLen * refBytes;

  const fmtGB = (x) => `${(x / 1e9).toFixed(2)} GB`;

  return (
    <div className="viz-panel">
      <div className="viz-controls">
        <label>
          KV heads (Q heads fixed at {N_Q_HEADS}):
        </label>
        <div className="viz-tabs" role="tablist">
          {KV_HEAD_OPTIONS.map((h) => (
            <button
              key={h}
              role="tab"
              className={`viz-tab ${n_kv === h ? 'active' : ''}`}
              onClick={() => setNkv(h)}
            >
              {h === 1 ? 'MQA · 1' : h === N_Q_HEADS ? `MHA · ${h}` : `GQA · ${h}`}
            </button>
          ))}
        </div>
      </div>

      <div className="viz-heads-row">
        <div className="viz-heads-stack">
          <div className="viz-mat-label">Q heads (×{N_Q_HEADS}, one per attention head)</div>
          <div className="viz-heads-bar" style={{ gridTemplateColumns: `repeat(${N_Q_HEADS}, 1fr)` }}>
            {Array.from({ length: N_Q_HEADS }, (_, i) => (
              <div key={i} className="viz-heads-cell q">Q<sub>{i}</sub></div>
            ))}
          </div>

          <div className="viz-mat-label" style={{ marginTop: 14 }}>
            K heads — <strong>{n_kv}</strong> total, each shared by {N_Q_HEADS / n_kv} Q head{N_Q_HEADS / n_kv === 1 ? '' : 's'}
          </div>
          <div className="viz-heads-bar" style={{ gridTemplateColumns: `repeat(${N_Q_HEADS}, 1fr)` }}>
            {Array.from({ length: N_Q_HEADS }, (_, i) => {
              const g = groupOf(i);
              const first = groupOf(i - 1) !== g;
              const last  = groupOf(i + 1) !== g;
              return (
                <div
                  key={i}
                  className="viz-heads-cell k"
                  style={{
                    borderTopLeftRadius: first ? 6 : 0,
                    borderBottomLeftRadius: first ? 6 : 0,
                    borderTopRightRadius: last ? 6 : 0,
                    borderBottomRightRadius: last ? 6 : 0,
                    opacity: first ? 1 : 0.55,
                  }}
                >
                  K<sub>{g}</sub>
                </div>
              );
            })}
          </div>

          <div className="viz-mat-label" style={{ marginTop: 6 }}>V heads</div>
          <div className="viz-heads-bar" style={{ gridTemplateColumns: `repeat(${N_Q_HEADS}, 1fr)` }}>
            {Array.from({ length: N_Q_HEADS }, (_, i) => {
              const g = groupOf(i);
              const first = groupOf(i - 1) !== g;
              const last  = groupOf(i + 1) !== g;
              return (
                <div
                  key={i}
                  className="viz-heads-cell v"
                  style={{
                    borderTopLeftRadius: first ? 6 : 0,
                    borderBottomLeftRadius: first ? 6 : 0,
                    borderTopRightRadius: last ? 6 : 0,
                    borderBottomRightRadius: last ? 6 : 0,
                    opacity: first ? 1 : 0.55,
                  }}
                >
                  V<sub>{g}</sub>
                </div>
              );
            })}
          </div>

          <div className="viz-heads-legend">
            Faded cells share the K/V row of the leftmost cell in their group — that's
            what "sharing K and V across Q heads" means in code.
          </div>
        </div>

        <div className="viz-cache-table">
          <h5>KV cache size</h5>
          <div className="viz-cache-table-row">
            <span>config</span><strong>{n_kv === 1 ? 'MQA' : n_kv === N_Q_HEADS ? 'MHA' : `GQA-${n_kv}`}</strong>
          </div>
          <div className="viz-cache-table-row">
            <span>per layer per token</span>
            <strong>{n_kv * refDHead * 2 * refBytes} B</strong>
          </div>
          <div className="viz-cache-table-row">
            <span>per sequence</span>
            <strong>{fmtGB(cacheBytes(n_kv))}</strong>
          </div>
          <div className="viz-cache-table-row">
            <span>vs MHA</span>
            <strong>{n_kv === N_Q_HEADS ? '1.0×' : `${(N_Q_HEADS / n_kv).toFixed(0)}× smaller`}</strong>
          </div>
          <div className="viz-cache-table-row" style={{ marginTop: 8, color: 'var(--ink-faint)', fontStyle: 'italic' }}>
            ref: 32 layers · d_head 128 · T=8192 · fp16
          </div>
        </div>
      </div>

      <div style={{ marginTop: 18, padding: '10px 12px', background: 'hsl(var(--accent-h), 30%, 97%)', borderRadius: 8, fontSize: '0.9rem', color: 'var(--ink-soft)' }}>
        <strong>Used in the wild:</strong>{' '}
        {REAL_MODELS_GQA.map((m, i) => (
          <span key={i}>
            {i > 0 && <> · </>}
            {m.name}
          </span>
        ))}.
      </div>

      <p className="viz-caption">
        MQA was the first big lever (Shazeer 2019) — but pushing every Q head
        through one K/V hurt quality. GQA (Ainslie et al. 2023) keeps a handful
        of K/V heads and groups Q heads around them: near-MHA quality at a
        fraction of the cache. Most frontier open-weight models now ship with
        GQA-8.
      </p>

      <pre className="viz-code">
{`# GQA — expand n_kv heads back up to n_q heads using repeat_interleave
K = K.repeat_interleave(n_q // n_kv, dim=-3)   # (B, n_q, T, d_head)
V = V.repeat_interleave(n_q // n_kv, dim=-3)
# Then attention proceeds exactly like MHA.`}
      </pre>
    </div>
  );
}

/* =========================================================
   §5 — Interactive D: PagedAttention block allocation
   ========================================================= */

const PAGED_BLOCK_SIZE = 4;
const PAGED_TOTAL_SLOTS = 64;
const PAGED_MAX_SEQ_PER_REQ = 24; // for the "contiguous" side, the up-front reservation
const PAGED_SCRIPT = [
  // [delta_a, delta_b]  — how many new tokens each request adds this step.
  // negative means "finishes and releases".
  { a: 5, b: 0,  caption: 'Request A arrives with a 5-token prompt.' },
  { a: 1, b: 7,  caption: 'A decodes one token; B arrives with a 7-token prompt.' },
  { a: 1, b: 1,  caption: 'Both decode one more token.' },
  { a: 3, b: 1,  caption: 'A keeps generating; B decodes one.' },
  { a: 0, b: -1, caption: 'B finishes and is released.' },
  { a: 2, b: 0,  caption: 'A keeps going. (Notice what happens to B\'s slots.)' },
];

function SectionPaged() {
  const [step, setStep] = useState(0);

  // Compute occupancy at this step
  const tokens = (() => {
    let a = 0, b = 0, bAlive = true;
    for (let i = 0; i <= step && i < PAGED_SCRIPT.length; i++) {
      const ev = PAGED_SCRIPT[i];
      if (ev.b < 0) bAlive = false;
      else if (bAlive) b += ev.b;
      if (ev.a >= 0) a += ev.a;
    }
    return { a, b, bAlive };
  })();

  // ---- Contiguous side ----
  // Reserve PAGED_MAX_SEQ_PER_REQ slots for each request, contiguously, up front.
  const contig = Array(PAGED_TOTAL_SLOTS).fill('free');
  for (let i = 0; i < PAGED_MAX_SEQ_PER_REQ; i++) {
    contig[i] = i < tokens.a ? 'a' : 'reserved-a';
  }
  if (tokens.b > 0 || !tokens.bAlive) {
    for (let i = 0; i < PAGED_MAX_SEQ_PER_REQ; i++) {
      const idx = PAGED_MAX_SEQ_PER_REQ + i;
      if (tokens.bAlive) {
        contig[idx] = i < tokens.b ? 'b' : 'reserved-b';
      } else {
        // B finished but its block stays reserved — fragmentation
        contig[idx] = 'reserved-b';
      }
    }
  }

  // ---- Paged side ----
  // Block size 4. A and B each get blocks allocated on demand.
  // We'll track block_index -> owner.
  const paged = Array(PAGED_TOTAL_SLOTS).fill('free');
  const blocksA = Math.ceil(tokens.a / PAGED_BLOCK_SIZE);
  const blocksB = tokens.bAlive ? Math.ceil(tokens.b / PAGED_BLOCK_SIZE) : 0;
  // Allocate A's blocks at positions 0, 2, 5, 7 (illustrative non-contiguous, but we'll just go contiguous from block 0 for clarity)
  // For pedagogy: interleave them
  const blockAssignments = [];
  let bi = 0;
  for (let k = 0; k < blocksA; k++) blockAssignments.push({ block: bi++, owner: 'a', n: k + 1 });
  for (let k = 0; k < blocksB; k++) blockAssignments.push({ block: bi++, owner: 'b', n: k + 1 });
  for (const { block, owner } of blockAssignments) {
    const start = block * PAGED_BLOCK_SIZE;
    for (let j = 0; j < PAGED_BLOCK_SIZE; j++) paged[start + j] = owner;
  }
  // Mark only the slots actually used (within the last block of each request)
  // Slots beyond used-token-count in a block become "reserved-x" (small internal frag)
  // For simplicity we mark all block slots as the owner color — the user reads "the block is taken".

  const wastedContig = contig.filter((s) => s === 'reserved-a' || s === 'reserved-b').length;
  const wastedPaged = paged.filter((s) => s === 'free').length;
  // ^ "wasted on the left" includes still-reserved slots (won't be used by another request); on the right, free slots are available.

  // Block-table strings
  const blockTableA = Array.from({ length: blocksA }, (_, k) => k);
  const blockTableB = blocksB > 0 ? Array.from({ length: blocksB }, (_, k) => blocksA + k) : [];

  return (
    <div className="viz-panel">
      <div className="viz-controls">
        <button
          className="active"
          onClick={() => setStep((s) => Math.min(PAGED_SCRIPT.length - 1, s + 1))}
          disabled={step >= PAGED_SCRIPT.length - 1}
        >
          Next step
        </button>
        <button onClick={() => setStep(0)}>Reset</button>
        <span style={{ color: 'var(--ink-soft)', fontSize: '0.9rem' }}>
          step <strong>{step + 1}</strong> of {PAGED_SCRIPT.length} — {PAGED_SCRIPT[step].caption}
        </span>
      </div>

      <div className="viz-paged-pair">
        <div className="viz-paged-side">
          <h5>Contiguous allocation (the old way)</h5>
          <div className="viz-paged-grid">
            {contig.map((kind, i) => (
              <div key={i} className={`viz-slot ${kind === 'a' ? 'req-a' : kind === 'b' ? 'req-b' : kind === 'reserved-a' ? 'reserved-a' : kind === 'reserved-b' ? 'reserved-b' : ''}`}>
                {kind === 'a' ? 'A' : kind === 'b' ? 'B' : ''}
              </div>
            ))}
          </div>
          <div className="viz-block-table">
            <div className="viz-block-table-row">
              Each request reserves <strong>{PAGED_MAX_SEQ_PER_REQ} slots</strong> up front
              ({(PAGED_MAX_SEQ_PER_REQ / PAGED_BLOCK_SIZE).toFixed(0)} blocks),
              shaded lighter until actually filled.
            </div>
            <div className="viz-block-table-row" style={{ color: 'var(--neg)' }}>
              <strong>Wasted slots:</strong> {wastedContig} (reserved but never used yet, and once B finishes its slots stay reserved).
            </div>
          </div>
        </div>

        <div className="viz-paged-side">
          <h5>PagedAttention (vLLM)</h5>
          <div className="viz-paged-grid">
            {paged.map((kind, i) => (
              <div key={i} className={`viz-slot ${kind === 'a' ? 'req-a' : kind === 'b' ? 'req-b' : ''}`}>
                {kind === 'a' ? 'A' : kind === 'b' ? 'B' : ''}
              </div>
            ))}
          </div>
          <div className="viz-block-table">
            <div className="viz-block-table-row">
              <strong>Block size:</strong> {PAGED_BLOCK_SIZE} slots. Allocate on demand.
            </div>
            <div className="viz-block-table-row">
              <strong>Block table A:</strong> [{blockTableA.join(', ')}]
            </div>
            <div className="viz-block-table-row">
              <strong>Block table B:</strong> {tokens.bAlive ? `[${blockTableB.join(', ')}]` : <em>released — blocks free</em>}
            </div>
            <div className="viz-block-table-row" style={{ color: 'var(--pos)' }}>
              <strong>Free slots:</strong> {wastedPaged} (available to the next request).
            </div>
          </div>
        </div>
      </div>

      <p className="viz-caption">
        Same scenario, two allocation strategies. Contiguous mirrors{' '}
        <code>malloc</code> with a fixed max-length reservation per request:
        most slots sit empty, and freeing a request leaves its slots stranded
        until the request struct is torn down. PagedAttention mirrors the OS
        page table — fixed-size physical blocks, a per-request logical→physical
        map, allocate on demand, free when done. Roughly 4× more concurrent
        requests on the same GPU in production deployments.
      </p>

      <pre className="viz-code">
{`# Per-request logical→physical block map (vLLM, simplified)
block_table = {req_id: []}             # list of physical block ids
on_new_token(req):
    if len(req.tokens) % BLOCK == 1:    # need a new block
        block_table[req].append(allocator.alloc())
    # write the new K, V into the last block at the right offset
on_finish(req):
    for b in block_table.pop(req):
        allocator.free(b)`}
      </pre>
    </div>
  );
}

/* =========================================================
   §6 — Interactive E: Continuous batching Gantt
   ========================================================= */

const GANTT_T = 14;

// Request specs: A and B both arrive at step 0 (for simplicity)
// A: prompt 4 tokens, generates 8 tokens
// B: prompt 2 tokens, generates 4 tokens
const REQ_A = { promptLen: 4, outLen: 8 };
const REQ_B = { promptLen: 2, outLen: 4 };

function staticBatchingTimeline(T) {
  // Both prompts processed at step 0 (pad B to A's prompt length)
  // Then both generate together for max(A.out, B.out) steps
  // B finishes after its outLen but pads until A is done
  const rowA = new Array(T).fill('idle');
  const rowB = new Array(T).fill('idle');

  // Step 0: prompt processed in parallel (pad B by promptLen_A - promptLen_B "pad" cells visualized inside the prompt slot)
  rowA[0] = 'a-prompt';
  rowB[0] = 'b-prompt'; // pad cells will be rendered inside

  const totalDecode = Math.max(REQ_A.outLen, REQ_B.outLen);
  for (let t = 1; t <= totalDecode && t < T; t++) {
    rowA[t] = t <= REQ_A.outLen ? 'a-decode' : 'pad';
    rowB[t] = t <= REQ_B.outLen ? 'b-decode' : 'pad'; // B keeps the slot until batch finishes
  }
  return { rowA, rowB };
}

function continuousBatchingTimeline(T) {
  const rowA = new Array(T).fill('idle');
  const rowB = new Array(T).fill('idle');

  rowA[0] = 'a-prompt';
  rowB[0] = 'b-prompt';

  // Both decode independently; B's slot is freed the moment it finishes
  for (let t = 1; t <= REQ_A.outLen && t < T; t++) rowA[t] = 'a-decode';
  for (let t = 1; t <= REQ_B.outLen && t < T; t++) rowB[t] = 'b-decode';
  // After B finishes, its slot is genuinely free — could host a new request
  return { rowA, rowB };
}

function GanttRow({ row, label, T }) {
  return (
    <div className="viz-gantt-row" style={{ gridTemplateColumns: `90px 1fr` }}>
      <div className="viz-gantt-label">{label}</div>
      <div className="viz-gantt-cells" style={{ gridTemplateColumns: `repeat(${T}, 1fr)` }}>
        {row.map((kind, i) => (
          <div key={i} className={`viz-gantt-cell ${kind}`}>
            {kind === 'a-prompt' || kind === 'b-prompt' ? 'P' :
             kind === 'a-decode' || kind === 'b-decode' ? '·' :
             kind === 'pad' ? '░' : ''}
          </div>
        ))}
      </div>
    </div>
  );
}

function SectionContinuousBatching() {
  const stat = staticBatchingTimeline(GANTT_T);
  const cont = continuousBatchingTimeline(GANTT_T);

  const isUseful  = (k) => k !== 'idle' && k !== 'pad';
  const isWasted  = (k) => k === 'pad';
  const isFree    = (k) => k === 'idle';
  const sum = (rows, pred) => rows.reduce((n, row) => n + row.filter(pred).length, 0);

  const total = 2 * GANTT_T;
  const usefulStat = sum([stat.rowA, stat.rowB], isUseful);
  const wastedStat = sum([stat.rowA, stat.rowB], isWasted);
  const freeStat   = sum([stat.rowA, stat.rowB], isFree);
  const usefulCont = sum([cont.rowA, cont.rowB], isUseful);
  const wastedCont = sum([cont.rowA, cont.rowB], isWasted);
  const freeCont   = sum([cont.rowA, cont.rowB], isFree);

  return (
    <div className="viz-panel">
      <div className="viz-mat-label" style={{ marginBottom: 6 }}>
        Static batching — both requests share one batched step; short request waits
      </div>
      <div className="viz-gantt">
        <GanttRow row={stat.rowA} label="Request A" T={GANTT_T} />
        <GanttRow row={stat.rowB} label="Request B" T={GANTT_T} />
      </div>
      <div className="viz-gantt-axis" style={{ gridTemplateColumns: `90px repeat(${GANTT_T}, 1fr)` }}>
        <span />
        {Array.from({ length: GANTT_T }, (_, i) => <span key={i}>{i}</span>)}
      </div>
      <div className="viz-gantt-legend">
        Useful: <strong>{usefulStat}</strong> · Wasted on padding: <strong style={{ color: 'var(--neg)' }}>{wastedStat}</strong> ·
        Free for new requests: <strong>{freeStat}</strong> &nbsp;/&nbsp; {total} cells.
        B is forced to pad until A finishes — those slots belong to B but produce nothing.
      </div>

      <div className="viz-mat-label" style={{ marginBottom: 6, marginTop: 22 }}>
        Continuous batching (vLLM, Orca) — each request leaves the batch the moment it finishes
      </div>
      <div className="viz-gantt">
        <GanttRow row={cont.rowA} label="Request A" T={GANTT_T} />
        <GanttRow row={cont.rowB} label="Request B" T={GANTT_T} />
      </div>
      <div className="viz-gantt-axis" style={{ gridTemplateColumns: `90px repeat(${GANTT_T}, 1fr)` }}>
        <span />
        {Array.from({ length: GANTT_T }, (_, i) => <span key={i}>{i}</span>)}
      </div>
      <div className="viz-gantt-legend">
        Useful: <strong>{usefulCont}</strong> · Wasted on padding: <strong style={{ color: 'var(--pos)' }}>{wastedCont}</strong> ·
        Free for new requests: <strong>{freeCont}</strong> &nbsp;/&nbsp; {total} cells.
        Once B finishes at step {REQ_B.outLen + 1}, its slot is genuinely free — a new request can land in it.
      </div>

      <div className="viz-gantt-legend" style={{ marginTop: 16 }}>
        <span><span className="viz-gantt-swatch" style={{ background: 'hsl(218, 55%, 75%)' }} /> A prompt</span>
        <span><span className="viz-gantt-swatch" style={{ background: 'hsl(218, 55%, 88%)' }} /> A decode</span>
        <span><span className="viz-gantt-swatch" style={{ background: 'hsl(28, 65%, 72%)' }} /> B prompt</span>
        <span><span className="viz-gantt-swatch" style={{ background: 'hsl(28, 65%, 88%)' }} /> B decode</span>
        <span><span className="viz-gantt-swatch" style={{ background: 'repeating-linear-gradient(45deg, #f3eee2, #f3eee2 3px, #e9e3d3 3px, #e9e3d3 6px)' }} /> padding (wasted)</span>
      </div>

      <p className="viz-caption">
        Static batching is what naive frameworks do — pad to the longest sequence
        and run the batch as a unit. vLLM's continuous batching reschedules the
        batch every step: a request that just finished leaves immediately, a new
        request arriving mid-flight joins on the next step. In practice this
        roughly doubles throughput on shared workloads.
      </p>

      <pre className="viz-code">
{`# Continuous batching scheduler (simplified)
while running_requests:
    batch = pick_ready(running_requests)        # requests that need a step now
    run_one_decode_step(batch)
    for req in batch:
        if req.done:
            release(req)                        # slot is free this very step
    admit_new_requests_if_possible()`}
      </pre>
    </div>
  );
}

/* =========================================================
   The post
   ========================================================= */

export default function VisualizingKVCache() {
  useEffect(() => {
    const prevTitle = document.title;
    document.title = 'Visualizing the KV Cache: Prefill, Decode, and Why Inference Is Bandwidth-Bound — Bernhard Walser';

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
      'An interactive walk through the KV cache: prefill vs decode, why decode is memory-bandwidth-bound, MHA → GQA → MQA, PagedAttention, and continuous batching.';

    setMeta('description', description);
    setMeta('og:title', 'Visualizing the KV Cache', 'property');
    setMeta('og:description', description, 'property');
    setMeta('og:type', 'article', 'property');
    setMeta('twitter:card', 'summary_large_image');
    setMeta('twitter:title', 'Visualizing the KV Cache');
    setMeta('twitter:description', description);

    return () => { document.title = prevTitle; };
  }, []);

  return (
    <article className="viz-post">
      <div className="viz-wide">
        <div className="viz-series-tag">Visualizing ML · Part 2</div>
        <h1>Visualizing the KV Cache: Prefill, Decode, and Why Inference Is Bandwidth-Bound</h1>
        <p className="viz-lede">
          You hit Enter on a 4000-token prompt. The first token comes back in
          ~500 ms; the rest stream in at ~20 ms each. Why are they so different?
          Because they aren't doing the same work — and understanding why is
          the foundation of every inference optimization at the frontier.
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
          This is post #2 of the <em>Visualizing ML</em> series. If you haven't
          read{' '}
          <a className="viz-link" href="#/blog/visualizing-attention">Visualizing Attention</a>,
          start there — this post reuses the same toy world
          (<Katex tex="d_{model}=8" />, <Katex tex="d_k=4" />, the same sentences) and picks
          up exactly where that one left off.
        </p>

        <h2>1. What does one transformer layer cost?</h2>
        <p>
          Every interesting trade-off later in this post — caching, GQA,
          paging, training vs inference — bottoms out in the same question:
          how much math and how many bytes does a single forward pass through
          one transformer layer actually take? The answer depends on only a
          handful of dimensions and one accounting rule, so let's get them
          on the table first.
        </p>
        <p>
          A transformer layer has two parts. The <strong>attention block</strong>
          {' '}projects the input into queries, keys, values, runs attention,
          then projects the result back with an output projection (the four
          matrices <Katex tex="W_Q" />, <Katex tex="W_K" />, <Katex tex="W_V" />,
          <Katex tex="W_O" />). The <strong>FFN block</strong> applies two
          linear layers with a 4× hidden expansion in between
          (<Katex tex="W_{up}" />, <Katex tex="W_{down}" />). Inputs and
          outputs are the same shape — <Katex tex="(T, d_{model})" /> — so
          layers can be stacked.
        </p>
        <p>
          Two dimensions matter throughout: <Katex tex="d_{model}" /> (often
          shortened to <Katex tex="d" />) is the residual stream width, and{' '}
          <Katex tex="d_{kv}" /> is the output width of K and V — that is,
          <Katex tex="n_{kv\,heads} \cdot d_{head}" />. In classic
          <strong> multi-head attention (MHA)</strong> there's one K/V head per
          query head, so <Katex tex="d_{kv} = d_{model}" /> and the K and V
          projections look identical in shape to Q. That's the default in
          §1's diagram. In <strong>GQA</strong> and <strong>MQA</strong>{' '}
          (§7), the number of K/V heads is reduced — same Q heads, fewer K/V
          heads — and <Katex tex="d_{kv}" /> shrinks accordingly. Keeping{' '}
          <Katex tex="d_{kv}" /> in the formulas as a separate variable is
          what lets §7's slider have something to actually move.
        </p>
        <p>
          Every operation in the layer is a matrix multiply. The standard
          accounting: a matmul of shape{' '}
          <Katex tex="(M \times N) \cdot (N \times P)" /> costs{' '}
          <strong>2·M·N·P FLOPs</strong> (each of the <Katex tex="M \cdot P" />{' '}
          output cells is <Katex tex="N" /> multiplies plus{' '}
          <Katex tex="N" /> adds). Apply that rule to each operation in the
          layer, for one new token, and you get the cost shown below.
        </p>
        <TransformerLayerDiagram />
        <p>
          Sum the boxes: per-layer, per-token cost ≈{' '}
          <Katex tex="4d^2 + 4 d \cdot d_{kv} + 16d^2 + 4T \cdot d" /> on the
          compute side, and{' '}
          <Katex tex="(10d^2 + 2 d \cdot d_{kv}) \cdot \mathrm{dtype}" /> on
          the memory side. For a Llama-7B-shaped model
          (<Katex tex="d=4096" />, <Katex tex="d_{kv}=4096" /> for MHA,{' '}
          <Katex tex="T=2048" />) that's <strong>~436 MFLOPs</strong> and{' '}
          <strong>~403 MB</strong> per layer per token, both at fp16.
        </p>
        <p>
          One thing worth pausing on: <strong>each block's FLOPs number
          equals its bytes number</strong> (33.55 MFLOP = 33.55 MB; 134.22
          MFLOP = 134.22 MB). That isn't a typo, it's the matmul rule meeting
          fp16. For one input row through a <Katex tex="(d \times d)" />{' '}
          weight matrix, FLOPs = <Katex tex="2d^2" /> and bytes ={' '}
          <Katex tex="d^2 \cdot \mathrm{dtype}" /> = <Katex tex="2d^2" /> for
          fp16 — literally the same number. So arithmetic intensity per linear
          layer is exactly <strong>1 FLOP/byte</strong> at batch 1, fp16,
          independent of <Katex tex="d" />. That's already orders of magnitude
          below an H100's ridge of ~295. Bigger batches multiply FLOPs but
          leave weight reads fixed, so AI grows linearly with{' '}
          <Katex tex="B" />. That single observation — <em>batches buy AI</em>{' '}
          — is what §6 and §11 keep coming back to.
        </p>

        <h2>2. Training vs inference</h2>
        <p>
          There are two ways to run a transformer, and they have completely
          different compute profiles.
        </p>
        <p>
          <strong>Training</strong> processes whole sequences in parallel.
          You feed in a <Katex tex="(B, T)" /> batch of tokens, run one
          forward pass through every layer (with all <Katex tex="T" /> tokens
          contributing to the matmuls), then backprop. Each linear projection
          becomes a real matmul over <Katex tex="T" /> rows at once, not
          <Katex tex="T" /> separate vector-matrix products. The GPU's tensor
          cores love this — lots of work per byte of weights read.
        </p>
        <p>
          <strong>Inference</strong> can't do that for the part that matters.
          You don't know token <em>n+1</em> until the model has produced
          token <em>n</em>, so you can't lay them out for one big parallel
          matmul. The model must run once per output token, sequentially.
          That sequential, one-token-at-a-time pattern is what makes
          inference expensive in a different way from training — and it's
          where every interesting optimization in this post lives.
        </p>

        <h2>3. Inference: prefill and decode</h2>
        <p>
          Inference itself has two phases. <strong>Prefill</strong> happens
          once, at the start: the user's prompt is already given, so its{' '}
          <Katex tex="T_{prompt}" /> tokens are processed in parallel just
          like training — one big forward pass, exactly the per-layer
          computation from §1 but with <Katex tex="T" /> tokens at once.
          <strong> Decode</strong> happens once per generated token: one new
          token's worth of work per step.
        </p>
        <p>
          The widget below makes the difference concrete. Pick a prompt, run
          prefill, and watch the K/V cache fill up in one shot. Then decode
          one step at a time and notice that <em>only one new row</em> is
          added each time — but the new token's query reads the <em>entire</em>{' '}
          cache to compute attention.
        </p>
        <SectionPrefillDecode />

        <h2>4. What caching saves</h2>
        <p>
          Look back at the layer diagram. The <Katex tex="K" /> and{' '}
          <Katex tex="V" /> boxes are <Katex tex="2 \cdot d \cdot d_{kv}" />{' '}
          FLOPs <em>per token</em>. At decode time you only generate one new
          token per step, so naively those two boxes cost{' '}
          <Katex tex="2 \cdot d \cdot d_{kv}" /> each. The trouble is the
          attention compute below them: it needs K and V for <em>every</em>{' '}
          token in the context, all <Katex tex="T" /> of them, to score the
          new query against them.
        </p>
        <p>
          Without a cache, you'd have to re-project <Katex tex="K" /> and{' '}
          <Katex tex="V" /> for all <Katex tex="T" /> previous tokens too,
          paying <Katex tex="2 T \cdot d \cdot d_{kv}" /> FLOPs <em>per
          step</em> for each of K and V. That's the redundant work the cache
          avoids: project K and V exactly once when a token first appears,
          then keep them around for every later step.
        </p>
        <p>
          Here's what changes in the layer diagram during a decode step{' '}
          <em>with</em> the cache:
        </p>
        <TransformerLayerDiagram mode="decode" />
        <p>
          The K and V boxes shrink (one token's projection, not{' '}
          <Katex tex="T" />); the attention compute stays the same (still
          reads <Katex tex="T" /> rows from the cache); a new orange{' '}
          <strong>KV cache</strong> block appears, holding{' '}
          <Katex tex="2 \cdot d_{kv} \cdot T" /> values per layer per request.
          That last block is the new cost: memory. The cache trades
          recomputation for storage.
        </p>
        <p>
          Below is the same trade laid out as a side-by-side receipt: every
          operation in the layer, with and without the cache. Drag the
          sliders to see how the totals scale with model dimensions and
          context length — the redundant <Katex tex="2T \cdot d \cdot d_{kv}" />{' '}
          term in the no-cache column is what blows up.
        </p>
        <CostBreakdown />

        <h2>5. The total cost — and a number worth memorising</h2>
        <p>
          Add up the boxes in the cached decode diagram. A single decode step
          on one Llama-7B-shaped layer costs about <strong>~440 MFLOPs</strong>{' '}
          of math and reads about <strong>~400 MB</strong> of weights from HBM.
          Across 32 layers that's <strong>~14 GFLOPs</strong> of compute and{' '}
          <strong>~13 GB</strong> of reads per token at batch 1.
        </p>
        <p>
          The ratio of those two is the number that runs everything: the{' '}
          <strong>arithmetic intensity</strong>{' '}
          (<a className="viz-link" href="https://horace.io/brrr_intro.html" target="_blank" rel="noreferrer">Horace He</a>{' '}
          has the canonical write-up). FLOPs divided by bytes. For decode at
          batch 1, AI ≈ 1 FLOP per byte moved. Every byte you fetch from HBM
          gets used in roughly one arithmetic operation before it's discarded.
        </p>
        <p>
          To know whether that's good or bad, you compare against the
          hardware's own ratio: peak compute divided by peak bandwidth. For
          an H100 in fp16 that's <strong>~295 FLOPs per byte</strong>. So
          for decode at batch 1, the GPU could in principle do 295× more
          arithmetic per byte read — but the model only asks for 1×. The
          remaining 294× of math capacity is wasted. The math units are
          fed bytes far slower than they can chew them. The next section
          says what to do about it.
        </p>

        <h2>6. Why decode is bandwidth-bound, not compute-bound</h2>
        <p>
          So decode lives at AI ≈ 1 against a ridge of ≈ 295. That's not
          close. Most of the wall-clock time of a decode step is the GPU
          waiting on <strong>HBM</strong> reads — model weights (gigabytes)
          plus the KV cache (more gigabytes at long context). The math is so
          much faster that it's nearly free in absolute terms.
        </p>
        <p>
          This has two consequences. <strong>First</strong>, the levers that
          actually buy throughput are the ones that reduce bytes moved per
          token — smaller cache (§7), better memory allocation (§8),
          interleaving requests so the same byte feeds more arithmetic
          (§9). <strong>Second</strong>, latency at batch 1 is basically a
          function of model size: roughly <em>weight bytes / HBM bandwidth</em>.
          A 13 GB model on an H100 won't generate tokens faster than ~3.9 ms
          each, no matter how clever your kernels are.
        </p>

        <InfoBox title="What is HBM? (and where do I find one?)">
          <p>
            <strong>HBM = High-Bandwidth Memory.</strong> It's the chunk of RAM
            physically stacked on (or right next to) an AI accelerator. An H100
            ships with 80 GB of HBM3 at <strong>3.35 TB/s</strong>; an MI300X
            ships with 192 GB at 5.3 TB/s. For comparison: DDR5 in a desktop
            tops out around <strong>60 GB/s</strong> — fifty-plus times slower.
          </p>
          <p>
            Two trade-offs make HBM special. It is{' '}
            <strong>wide</strong>: physically wider memory buses (1024-bit
            stacks vs 64-bit for DDR), packed close to the compute die so the
            wires can be short and parallel. And it is{' '}
            <strong>expensive</strong>: small capacities, only economical when
            you really need the bandwidth. You will not find HBM on a normal
            PC, laptop, or phone — only on GPU accelerators (Nvidia
            H100/H200/B100/B200, AMD MI300/MI325, Google TPU v5p, Trainium
            etc.). The price tag follows the bandwidth: an H100 costs roughly
            $30 K, with much of that going to HBM3 packaging.
          </p>
          <p>
            <strong>Why it matters here:</strong> the entire KV cache lives in
            HBM, and every decode step has to <em>read all of it</em>. If your
            cache is 8 GB and the GPU sustains 3.35 TB/s, the read alone takes
            about <Katex tex="8/3350 \approx 2.4 \text{ ms}" /> — which is most
            of a typical per-token latency budget.
          </p>
        </InfoBox>

        <h2>7. Shrinking the cache: MHA → GQA → MQA</h2>
        <p>
          The KV cache size is dictated by the architecture, not the user.
          Per layer, per token, per request it is{' '}
          <Katex tex="2 \cdot n_{kv} \cdot d_{head} \cdot \mathrm{dtype}" />{' '}
          bytes — the factor of 2 covers K and V. For a Llama-7B-style model
          with <Katex tex="n_{kv}=32" /> heads, <Katex tex="d_{head}=128" />,
          32 layers, fp16, an 8k context, that's roughly 4 GB <em>per request</em>.
          On an 80 GB H100 you fit maybe 10 concurrent users before the cache
          alone fills the memory.
        </p>
        <p>
          The fix is architectural: shrink <Katex tex="n_{kv}" />. Multi-Query
          Attention (Shazeer 2019) keeps a single K and V head, shared across
          all <Katex tex="n_q" /> query heads. Grouped-Query Attention
          (Ainslie et al. 2023) is the middle ground — <Katex tex="n_{kv} < n_q" />,
          with Q heads partitioned into groups that share. Same attention math,
          {' '}<Katex tex="n_q / n_{kv}" />× less cache. Quality stays close to
          full MHA; cache size collapses by 4–8×.
        </p>
        <SectionGQA />

        <h2>8. PagedAttention: stop fragmenting the cache</h2>
        <p>
          GQA shrinks the cache. PagedAttention squeezes the inefficiency out
          of how it's allocated. Naive serving systems reserve a contiguous
          chunk of GPU memory for each request, sized for the worst case
          (max_seq_len). Most requests never hit that ceiling — so most of
          that memory sits empty. Worse, even after a request finishes its
          slot stays bound to that request until the connection tears down.
        </p>
        <p>
          The{' '}
          <a className="viz-link" href="https://arxiv.org/abs/2309.06180" target="_blank" rel="noreferrer">
            vLLM paper
          </a>{' '}
          (Kwon et al. 2023) borrows the trick from operating-system virtual
          memory: chop the cache into fixed-size <strong>blocks</strong> (16
          tokens is typical), maintain a per-request <strong>block table</strong>{' '}
          mapping logical positions → physical blocks, allocate blocks on
          demand, free them on request completion. It's <code>malloc</code> →{' '}
          <code>mmap</code>. The effect on serving throughput is dramatic
          (≈ 4×) because you stop wasting GPU memory on never-used slots.
        </p>
        <SectionPaged />

        <h2>9. Continuous batching</h2>
        <p>
          One last source of waste. Even with a perfectly-allocated cache,
          you still need to <em>batch</em> requests through the model.
          Traditional ("static") batching freezes a batch at the start —
          short requests get padded to the longest, slots stay bound until
          the whole batch finishes. Continuous batching, introduced by Orca
          and popularised by vLLM, reschedules the batch <em>every decode
          step</em>: finished requests release their slot immediately, new
          requests join on the next iteration.
        </p>
        <SectionContinuousBatching />

        <h2>10. What about training?</h2>
        <p>
          Training is the other side of the bandwidth-bound coin. The
          per-layer arithmetic is the same as §1, but two things change.
          First, there's no decode loop — you process the whole training
          sequence in one shot, like a giant prefill. Every linear
          projection becomes a real <Katex tex="(B \cdot T, d) \cdot (d, d)" />{' '}
          matmul instead of <Katex tex="B \cdot T" /> sequential vector
          ops. Second, there's no KV cache to read; you're consuming the
          activations as you go.
        </p>
        <p>
          The arithmetic intensity that follows is enormous. At a typical
          training config — say <Katex tex="B = 64" />,{' '}
          <Katex tex="T = 4096" /> — the effective number of queries per
          weight read is <Katex tex="B \cdot T" /> ≈ a quarter of a million.
          AI rockets above 1000 FLOPs/byte, well past any GPU's ridge.
          Training is solidly <strong>compute-bound</strong> — the math units
          are the bottleneck, exactly opposite to decode.
        </p>
        <p>
          (For honesty: a full training step also includes the backward pass,
          which roughly doubles the FLOPs of the forward pass, and the
          optimizer update. I'm leaving those out of the §11 simulator to
          keep the numbers comparable to inference — but the AI verdict
          doesn't change because backward scales bytes and FLOPs roughly
          the same way.)
        </p>
        <p>
          The practical consequence: training clusters are sized for FLOPs
          (raw compute), serving clusters are sized for HBM and bandwidth
          (memory). Same hardware, two different bottlenecks, two different
          economics.
        </p>

        <h2>11. Simulator</h2>
        <p>
          Putting it all together. Pick a preset to load a known config —
          each button shows its tokens/sec right on the chip — then tune the
          sliders to see how each lever moves the throughput. The mode
          toggle switches between an inference decode step and a training
          forward pass; you should see the AI verdict flip with it.
        </p>
        <SectionAICalc />

        <h2>12. Back to the question</h2>
        <p>
          We started with a puzzle. Hit Enter on a 4000-token prompt, the
          first token comes back in ~500 ms, the rest stream at ~20 ms each.
          Why so different? Here's the answer assembled from every section
          above.
        </p>
        <p>
          The 500 ms is <strong>prefill</strong> (§3). The model processes
          all 4000 prompt tokens in one parallel forward pass — same
          per-layer arithmetic as one token (§1), but 4000 times as much of
          it. That's about <Katex tex="4000 \times 14 \mathrm{\,GFLOPs} \approx 56 \mathrm{\,TFLOPs}" />{' '}
          of math at fp16. An H100 sustains nearly 1 PFLOP/s, so the math
          itself takes ~60 ms; the rest of the half-second is weight reads,
          kernel launches, and the non-attention bits. Arithmetic intensity
          is in the thousands of FLOPs/byte (§10) — the GPU's math units are
          the bottleneck. Prefill is <strong>compute-bound</strong>.
        </p>
        <p>
          The 20 ms is <strong>decode</strong> (§3). Each step processes one
          new token: about <Katex tex="14 \mathrm{\,GFLOPs}" /> total, a
          thousand times less than prefill, because the model only does its
          forward pass <em>once</em> per token, not 4000 times. The KV cache
          (§4) is what makes that possible — without it you'd re-project K
          and V for the entire context every step and decode would cost as
          much per token as prefill. But to score the new query against the
          context, the GPU must read the entire cache plus all the weights
          from HBM. At 3.35 TB/s, reading ~13 GB of state takes 3.9 ms, and
          real-world overhead brings it to about 20 ms per token. Arithmetic
          intensity is ~1 FLOP/byte. Decode is{' '}
          <strong>memory-bandwidth-bound</strong> (§6).
        </p>
        <p>
          That asymmetry is what every optimization later in the post is
          attacking: <strong>GQA / MQA</strong> (§7) shrinks the bytes per
          decode step by squeezing the cache; <strong>PagedAttention</strong>{' '}
          (§8) packs more sequences into the same HBM so the bandwidth covers
          more users; <strong>continuous batching</strong> (§9) reschedules
          so each weight read feeds the maximum number of requests. None of
          them deliver more FLOPs/s — they all deliver more decode tokens
          per byte moved. Different lever, same goal.
        </p>
        <p>
          The shape stays the same across model sizes and architectures.
          Prefill is one big matmul, done once, compute-bound. Decode is a
          long sequence of vector-matrix products against a growing cache,
          done once per output token, memory-bound. The 500 ms / 20 ms gap
          isn't a quirk of any particular implementation — it's the geometry
          of the problem. Everything you build on top of an LLM inherits it.
        </p>

        <h2>What comes next</h2>
        <p>
          <strong>MLA — multi-latent attention (DeepSeek-V2).</strong> GQA
          shares K and V across heads. MLA goes further: store a low-rank{' '}
          <em>latent</em> projection and reconstruct K, V on the fly. Cache
          shrinks by another order of magnitude, with comparable quality.{' '}
          <em>Future post in this series.</em>
        </p>
        <p>
          <strong>StreamingLLM — attention sinks.</strong>{' '}
          <a className="viz-link" href="https://arxiv.org/abs/2309.17453" target="_blank" rel="noreferrer">
            Xiao et al. 2023
          </a>{' '}
          show that keeping just the first few tokens (the "attention sinks")
          plus a sliding window over recent tokens gives near-full-context
          quality at bounded cache size. Crucial for very long contexts on
          fixed memory. <em>Future post.</em>
        </p>
        <p>
          <strong>Speculative decoding.</strong> If decode is bandwidth-bound,
          maybe we shouldn't decode one token at a time. Have a small "draft"
          model propose <em>K</em> tokens, then verify them all in parallel
          with the big model — same compute as one big step, but multiple
          tokens accepted per round. <em>Post #7 in this series.</em>
        </p>

        <footer className="viz-footer">
          <p>
            <strong>Part 2 of Visualizing ML</strong> · Previous:{' '}
            <a className="viz-link" href="#/blog/visualizing-attention">Visualizing Attention</a>
            . Next: <em>RoPE — Rotary Position Embeddings</em>.
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
