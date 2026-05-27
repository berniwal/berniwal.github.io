// src/posts/InferenceCost.jsx
// Transformer Costs · Part 3 of 4 — Inference Cost of a Transformer.
// Ported from the original VisualizingKVCache post; prose preserved, chrome
// updated to the post-2026 redesign. Anchors on prefill/decode + the
// arithmetic-intensity (roofline) calculator.
//
// TAGS FOR REGISTRATION: ['transformers', 'inference', 'cost']
// EXCERPT: When you chat with an LLM, the first token takes ~500 ms but the rest stream at ~20 ms each. That asymmetry is prefill vs decode — and the reason inference is memory-bound while training is compute-bound.
import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneLight } from 'react-syntax-highlighter/dist/esm/styles/prism';
import './PostChrome.css';
import './InferenceCost.css';

/* =========================================================
   Tiny matmul + KaTeX wrapper
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
   Toy world — tokens / embeddings / weights for the prefill-decode widget
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

const CONTINUATIONS = {
  she:  ['the', 'cat', 'sat', 'on'],
  cat:  ['she', 'gave', 'him', 'a'],
  fox:  ['the', 'mat', 'on', 'book'],
  code: ['a', 'fox', 'jumps', 'on'],
};

const buildX = (tokens) => tokens.map((t) => EMB[t]);

/* =========================================================
   MatrixGrid primitive
   ========================================================= */
function MatrixGrid({
  data, label, rowLabels, colLabels,
  decimals = 2, cellWidth = 46, freshRow,
}) {
  if (!data || data.length === 0) {
    return (
      <div className="viz-mat">
        {label && <div className="viz-mat-label">{label}</div>}
        <div className="viz-mat-empty">empty</div>
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
              const isFresh = i === freshRow;
              const display = Number.isFinite(v) ? v.toFixed(decimals) : '·';
              return (
                <div
                  key={`c-${i}-${j}`}
                  className={['viz-cell', isFresh ? 'viz-row-hi' : ''].filter(Boolean).join(' ')}
                  style={{ width: cellWidth }}
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
   Widget 1 — Prefill → Decode step-through
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
    const isDecode = step >= 2;
    if (isDecode) {
      const lastRow = X[X.length - 1];
      const Qlast = matmul([lastRow], W_Q);
      return { K, V, currentQ: Qlast, freshRowIdx: K.length - 1 };
    }
    const Q = matmul(X, W_Q);
    return { K, V, currentQ: Q, freshRowIdx: -1 };
  }, [allTokens, step]);

  const phase = step === 0 ? 'idle' : step === 1 ? 'prefill' : 'decode';
  const cacheBytes = K.length === 0 ? 0 : K.length * 4 * 2 * 2;
  const newRowsThisStep = step === 1 ? T_prompt : step >= 2 ? 1 : 0;
  const cacheBytesReadThisStep = step >= 2 ? K.length * 4 * 2 * 2 : 0;

  const canDecode = step >= 1 && decodedSoFar < maxDecodeSteps;
  const canPrefill = step === 0;

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
          <span className="viz-prompt-hint">
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
          colLabels={['k0','k1','k2','k3']}
          cellWidth={46}
          freshRow={freshRowIdx}
        />
        <div className="viz-matrix-op">·</div>
        <MatrixGrid
          data={V}
          label={`V cache  (${V.length} × d_k)`}
          rowLabels={allTokens}
          colLabels={['v0','v1','v2','v3']}
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
    </div>
  );
}

/* =========================================================
   Roofline + arithmetic-intensity calculator
   ========================================================= */
const H100_HBM_BYTES_PER_S = 3.35e12;
const H100_TFLOPS_FP16 = 989;
// eslint-disable-next-line no-unused-vars
const H100_RIDGE_FP16 = (H100_TFLOPS_FP16 * 1e12) / H100_HBM_BYTES_PER_S; // ≈ 295

const SEQ_LENS    = [64, 256, 1024, 4096, 16384, 65536];
const BATCH_SIZES = [1, 4, 16, 64, 256];
const D_MODELS    = [768, 1024, 2048, 4096, 8192, 12288];
const N_LAYERS    = [12, 24, 32, 48, 80, 96];
const N_Q_HEADS_FOR_DM = { 768: 12, 1024: 16, 2048: 32, 4096: 32, 8192: 64, 12288: 96 };
const PRECISIONS  = [
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

function computeSimMetrics({ mode, T, B, d_model, n_layers, n_kv, prec }) {
  const n_q_heads = N_Q_HEADS_FOR_DM[d_model] || 32;
  const kv = Math.min(n_kv, n_q_heads);
  const d_head = d_model / n_q_heads;
  const d_kv_total = kv * d_head;

  const flopsPerToken =
    4 * d_model * d_model +
    4 * d_model * d_kv_total +
    16 * d_model * d_model +
    4 * T * d_model;

  const tokensPerStep = mode === 'training' ? B * T : B;
  const flopsPerStep = tokensPerStep * n_layers * flopsPerToken;

  const weightParamsPerLayer = 10 * d_model * d_model + 2 * d_model * d_kv_total;
  const weightBytes = n_layers * weightParamsPerLayer * prec.bytes;
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
  const tokensProduced = mode === 'training' ? B * T : B;
  const tokensPerSec = tokensProduced / stepTime;

  return {
    flopsPerToken, flopsPerStep, weightBytes, kvBytes, totalBytes,
    ai, ridge, isComputeBound, stepTime, tokensPerSec,
    d_kv_total, n_q_heads, kv,
  };
}

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
  const [seqIdx, setSeqIdx]       = useState(3);
  const [batchIdx, setBatchIdx]   = useState(0);
  const [dmIdx, setDmIdx]         = useState(3);
  const [layIdx, setLayIdx]       = useState(2);
  const [precId, setPrecId]       = useState('fp16');
  const [kvHeads, setKvHeads]     = useState(32);

  const T        = SEQ_LENS[seqIdx];
  const B        = BATCH_SIZES[batchIdx];
  const d_model  = D_MODELS[dmIdx];
  const n_layers = N_LAYERS[layIdx];
  const prec     = PRECISIONS.find((p) => p.id === precId);
  const n_q_heads = N_Q_HEADS_FOR_DM[d_model] || 32;
  const n_kv = Math.min(kvHeads, n_q_heads);
  const d_head = d_model / n_q_heads;
  const d_kv_total = n_kv * d_head;

  const applyPreset = (p) => {
    setMode(p.cfg.mode);
    setSeqIdx(p.cfg.seqIdx);
    setBatchIdx(p.cfg.batchIdx);
    setDmIdx(p.cfg.dmIdx);
    setLayIdx(p.cfg.layIdx);
    setKvHeads(p.cfg.kv);
    setPrecId(p.cfg.precId);
  };

  const m = computeSimMetrics({ mode, T, B, d_model, n_layers, n_kv, prec });
  const presetMetrics = SIM_PRESETS.map((p) => ({
    p,
    metrics: computeSimMetrics(presetToConfig(p)),
  }));

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

  const {
    flopsPerStep, weightBytes, kvBytes, totalBytes, ai, ridge,
    isComputeBound, stepTime, tokensPerSec,
  } = m;

  const meterLo = Math.log10(0.1);
  const meterHi = Math.log10(100000);
  const meterPct = (v) => {
    const x = Math.max(meterLo, Math.min(meterHi, Math.log10(Math.max(0.1, v))));
    return ((x - meterLo) / (meterHi - meterLo)) * 100;
  };
  const markerPct = meterPct(ai);
  const ridgePct  = meterPct(ridge);

  const isTraining = mode === 'training';
  const throughputDenom = isTraining ? `${B} × ${T.toLocaleString()}` : `${B}`;

  const attentionType =
    n_kv === n_q_heads ? 'MHA' :
    n_kv === 1         ? 'MQA' :
                         'GQA';
  const setAttentionType = (t) => {
    if (t === 'MHA') setKvHeads(n_q_heads);
    else if (t === 'MQA') setKvHeads(1);
    else {
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

      <div className="viz-controls" style={{ marginTop: 14 }}>
        <span className="viz-mode-label">Mode:</span>
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
          <div className="viz-ai-row viz-ai-row-dim">
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
    </div>
  );
}

/* =========================================================
   Optional code block — the KV cache pseudocode
   ========================================================= */
const CODE = `# Prefill — single parallel matmul over the whole prompt
K_cache = X_prompt @ W_K        # (T_prompt, d_k)
V_cache = X_prompt @ W_V

# Decode — one new token per step, append to cache, re-read all of it
for _ in range(max_new_tokens):
    q  = x_new @ W_Q                              # (1, d_k)
    k  = x_new @ W_K;  v  = x_new @ W_V
    K_cache = torch.cat([K_cache, k], dim=0)      # grow by 1 row
    V_cache = torch.cat([V_cache, v], dim=0)
    A  = (q @ K_cache.T / d_k**0.5).softmax(-1)   # read ALL of K
    y  = A @ V_cache                              # read ALL of V
`;

function CodeBlock() {
  return (
    <details className="post-code-details">
      <summary>
        <span className="post-code-summary-icon" aria-hidden="true">›</span>
        <span>Show the KV-cache loop in Python</span>
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
export default function InferenceCost() {
  useEffect(() => {
    const prev = document.title;
    document.title = 'Inference Cost of a Transformer — Bernhard Walser';
    return () => { document.title = prev; };
  }, []);

  return (
    <article className="post-2026 ic-post">
      <div className="post-wrap">
        <div className="post-hero">
          <div className="post-series-tag">
            <span className="post-live-dot" aria-hidden="true" />
            Transformer Costs · Part 3 of 4
          </div>
          <h1>Inference Cost of a Transformer</h1>
          <p className="post-lede">
            When you chat with an LLM, the first token takes around half a second
            but the rest stream at twenty milliseconds each. Why the asymmetry?
            That's the prefill / decode split, and it's the heart of why inference
            cost looks nothing like training cost.
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

        <h2 className="reveal">Training vs inference</h2>
        <p>
          There are two ways to run a transformer, and they have completely
          different compute profiles.
        </p>
        <p>
          <strong>Training</strong> processes whole sequences in parallel.
          You feed in a <Katex tex="(B, T)" /> batch of tokens, run one
          forward pass through every layer (with all <Katex tex="T" /> tokens
          contributing to the matmuls), then backprop. Each linear projection
          becomes a real matmul over <Katex tex="T" /> rows at once, not{' '}
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
          what the rest of this post is about.
        </p>

        <h2 className="reveal">Prefill and decode</h2>
        <p>
          Inference itself has two phases. <strong>Prefill</strong> happens
          once, at the start: the user's prompt is already given, so its{' '}
          <Katex tex="T_{prompt}" /> tokens are processed in parallel just
          like training — one big forward pass.{' '}
          <strong>Decode</strong> happens once per generated token: one new
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

        <h2 className="reveal">The KV cache: what caching saves</h2>
        <p>
          Look at what happens during decode. The new token's query{' '}
          <Katex tex="q" /> needs to attend over <em>every</em> previous
          token — all <Katex tex="T" /> of them. To do that, the model needs
          their keys and values. Without a cache, you'd have to re-project
          K and V for every previous token from scratch, every single decode
          step. That's <Katex tex="O(L^2)" /> cumulative work to generate{' '}
          <Katex tex="L" /> tokens — the same K, V projections paid over
          and over.
        </p>
        <p>
          With a cache, you project K and V exactly <em>once</em> per token,
          when it first appears, then keep them around. Each decode step
          adds one new row to K and one to V; the previous rows are reused
          for free. Cumulative work drops to <Katex tex="O(L)" />. The cost:
          the cache itself lives in HBM, growing with the context length —
          memory traded for compute.
        </p>
        <p>
          The pseudocode below is the whole trick in ten lines. Prefill is
          one matmul; decode is a loop that appends one row to each cache
          and re-reads all of it.
        </p>
        <CodeBlock />

        <h2 className="reveal">Compute or memory bound?</h2>
        <p>
          Add up the work in a cached decode step. On one Llama-7B-shaped
          layer it's about <strong>~440 MFLOPs</strong> of math and reads
          about <strong>~400 MB</strong> of weights from HBM. Across 32
          layers that's <strong>~14 GFLOPs</strong> of compute and{' '}
          <strong>~13 GB</strong> of reads per token at batch 1.
        </p>
        <p>
          The ratio of those two is the number that runs everything: the{' '}
          <strong>arithmetic intensity</strong>{' '}
          (<a className="post-link" href="https://horace.io/brrr_intro.html" target="_blank" rel="noreferrer">Horace He</a>{' '}
          has the canonical write-up). FLOPs divided by bytes. For decode
          at batch 1, AI ≈ 1 FLOP per byte moved. Every byte you fetch from
          HBM gets used in roughly one arithmetic operation before it's
          discarded.
        </p>
        <p>
          To know whether that's good or bad, you compare against the
          hardware's own ratio: peak compute divided by peak bandwidth.
          This is the <strong>roofline model</strong> (Williams 2009). For
          an A100 in fp16 it's about{' '}
          <a className="post-link" href="https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf" target="_blank" rel="noreferrer">78 FLOPs per byte</a>;
          for an H100 it's about{' '}
          <a className="post-link" href="https://resources.nvidia.com/en-us-tensor-core/nvidia-tensor-core-gpu-datasheet" target="_blank" rel="noreferrer">295 FLOPs per byte</a>{' '}
          (both approximate — peak dense tensor-core math divided by HBM
          bandwidth). If your workload's AI sits below that ridge, you're
          memory-bound: the math units idle while HBM is the bottleneck.
          Above it: compute-bound.
        </p>
        <p>
          Drop a workload into the calculator below. Notice the same model
          flips from compute-bound (training, AI well past the ridge) to
          memory-bound (decode at batch 1, AI ≈ 1) — and what happens when
          you crank the batch up. The prefill phase, with all{' '}
          <Katex tex="T_{prompt}" /> tokens running in parallel, behaves
          like training: high AI, compute-bound. Decode at batch 1 is the
          opposite extreme.
        </p>
        <SectionAICalc />

        <p>
          So decode at batch 1 lives at AI ≈ 1 against a ridge of ≈ 295 on
          an H100. The math is so much faster than HBM that it's nearly
          free in absolute terms; most of the wall-clock time of a decode
          step is the GPU waiting on memory. <strong>Decode is
          memory-bound.</strong> That single fact is the lever every
          inference optimization pulls on — and it's where Part 4 picks up.
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
            <div className="ref-note">The transformer paper. Same architecture, two very different cost regimes.</div>
          </div>
          <div className="ref-link"><a href="https://arxiv.org/abs/1706.03762" target="_blank" rel="noreferrer">arxiv.org</a></div>

          <div className="ref-cite">Pope et al. 2022</div>
          <div>
            <div className="ref-title">
              <a href="https://arxiv.org/abs/2211.05102" target="_blank" rel="noreferrer">Efficiently Scaling Transformer Inference</a>
            </div>
            <div className="ref-note">The canonical analysis of prefill/decode arithmetic and where bandwidth limits bite.</div>
          </div>
          <div className="ref-link"><a href="https://arxiv.org/abs/2211.05102" target="_blank" rel="noreferrer">arxiv.org</a></div>

          <div className="ref-cite">Williams et al. 2009</div>
          <div>
            <div className="ref-title">
              <a href="https://dl.acm.org/doi/10.1145/1498765.1498785" target="_blank" rel="noreferrer">Roofline: An Insightful Visual Performance Model</a>
            </div>
            <div className="ref-note">Where the ridge-point / AI framing comes from.</div>
          </div>
          <div className="ref-link"><a href="https://dl.acm.org/doi/10.1145/1498765.1498785" target="_blank" rel="noreferrer">acm.org</a></div>

          <div className="ref-cite">Horace He 2022</div>
          <div>
            <div className="ref-title">
              <a href="https://horace.io/brrr_intro.html" target="_blank" rel="noreferrer">Making Deep Learning Go Brrrr From First Principles</a>
            </div>
            <div className="ref-note">The plain-English version of compute-bound vs memory-bound.</div>
          </div>
          <div className="ref-link"><a href="https://horace.io/brrr_intro.html" target="_blank" rel="noreferrer">horace.io</a></div>

          <div className="ref-cite">NVIDIA A100</div>
          <div>
            <div className="ref-title">
              <a href="https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf" target="_blank" rel="noreferrer">A100 Tensor Core GPU datasheet</a>
            </div>
            <div className="ref-note">312 TFLOPs fp16 dense / 2.04 TB/s HBM2e ≈ 78 FLOPs/byte ridge point.</div>
          </div>
          <div className="ref-link"><a href="https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf" target="_blank" rel="noreferrer">nvidia.com</a></div>

          <div className="ref-cite">NVIDIA H100</div>
          <div>
            <div className="ref-title">
              <a href="https://resources.nvidia.com/en-us-tensor-core/nvidia-tensor-core-gpu-datasheet" target="_blank" rel="noreferrer">H100 Tensor Core GPU datasheet</a>
            </div>
            <div className="ref-note">989 TFLOPs fp16 dense / 3.35 TB/s HBM3 ≈ 295 FLOPs/byte ridge point.</div>
          </div>
          <div className="ref-link"><a href="https://resources.nvidia.com/en-us-tensor-core/nvidia-tensor-core-gpu-datasheet" target="_blank" rel="noreferrer">nvidia.com</a></div>
        </div>

        <footer className="post-footer">
          <p>
            <strong>Part 3 of Transformer Costs</strong> · Previous:{' '}
            <a className="post-link" href="#/blog/training-cost-optimization">Optimizing Training Cost</a>
            {' · '}Next:{' '}
            <a className="post-link" href="#/blog/inference-cost-optimization">Optimizing Inference Cost</a>.
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
