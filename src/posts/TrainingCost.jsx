// src/posts/TrainingCost.jsx
// Transformer Costs · Part 1 of 4 — Training Cost of a Transformer.
// Ported from VisualizingKVCache (§1 + §10 prose, TransformerLayerDiagram, CostBreakdown).
// New widget: TrainingMemoryFootprint — interactive stacked bar of params, gradients,
// optimizer state, and activations.
//
// TAGS FOR REGISTRATION: ['transformers', 'training', 'cost']
// EXCERPT: Training Llama-2-7B took ~1.7M GPU-hours. To know why, count what one transformer layer actually does — forward and backward — and what has to be held in memory between them.

import React, { useEffect, useMemo, useRef, useState } from 'react';
import usePageMeta from '../usePageMeta';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneLight } from 'react-syntax-highlighter/dist/esm/styles/prism';
import './PostChrome.css';
import './TrainingCost.css';

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
          window.katex.render(tex, ref.current, { displayMode: block, throwOnError: false });
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
      className={`${block ? 'tc-math-block' : 'tc-math-inline'} ${className}`}
    />
  );
}

/* =========================================================
   Number formatting
   ========================================================= */
function fmtNum(x, unit) {
  if (!Number.isFinite(x) || x === 0) return `0 ${unit}`;
  const abs = Math.abs(x);
  if (abs >= 1e12) return `${(x / 1e12).toFixed(2)} T${unit}`;
  if (abs >= 1e9)  return `${(x / 1e9).toFixed(2)} G${unit}`;
  if (abs >= 1e6)  return `${(x / 1e6).toFixed(2)} M${unit}`;
  if (abs >= 1e3)  return `${(x / 1e3).toFixed(2)} k${unit}`;
  return `${x.toFixed(2)} ${unit}`;
}

/* =========================================================
   Widget 1 — TransformerLayerDiagram (forward / backward)
   Ported from VisualizingKVCache; backward mode is new and surfaces
   gradient flow + the ~2× FLOP cost of backprop.
   ========================================================= */

const ARCH_D     = 4096;
const ARCH_D_KV  = 4096;
const ARCH_T     = 2048;
const ARCH_DTYPE = 2; // bf16

function ArchBlock({
  name, shape, flops, numeric, bytes, bytesNumeric,
  kind = 'op', highlight, badge,
}) {
  return (
    <div className={`tc-arch-block tc-arch-block-${kind}${highlight ? ' tc-arch-block-saved' : ''}`}>
      {badge && <div className="tc-arch-badge">{badge}</div>}
      <div className="tc-arch-name">{name}</div>
      {shape && <div className="tc-arch-shape">{shape}</div>}
      {flops && (
        <div className="tc-arch-flops">
          <span className="tc-arch-formula">{flops}</span>
          {numeric && <span className="tc-arch-numeric">{numeric}</span>}
        </div>
      )}
      {bytes && (
        <div className="tc-arch-flops tc-arch-flops-mem">
          <span className="tc-arch-formula">{bytes}</span>
          {bytesNumeric && <span className="tc-arch-numeric">{bytesNumeric}</span>}
        </div>
      )}
    </div>
  );
}

function ArchArrow({ direction = 'down' }) {
  return (
    <div className={`tc-arch-arrow tc-arch-arrow-${direction}`} aria-hidden="true">
      {direction === 'up' ? '▲' : '▼'}
    </div>
  );
}

function TransformerLayerDiagram() {
  const [mode, setMode] = useState('forward');
  const d = ARCH_D, dkv = ARCH_D_KV, T = ARCH_T, dtype = ARCH_DTYPE;
  const num = (f) => fmtNum(f, 'FLOP');
  const numB = (b) => fmtNum(b, 'B');
  const backward = mode === 'backward';

  // Forward per-token FLOPs (matmul = 2·M·N·P, training treats T tokens at once
  // but we annotate per-token here so the numbers match KVCache §1).
  const flopsQ   = 2 * d * d;
  const flopsK   = 2 * d * dkv;
  const flopsV   = 2 * d * dkv;
  const flopsAtt = 4 * T * d;
  const flopsO   = 2 * d * d;
  const flopsUp  = 8 * d * d;
  const flopsDn  = 8 * d * d;
  const fwdFlops = flopsQ + flopsK + flopsV + flopsAtt + flopsO + flopsUp + flopsDn;
  // Backward roughly doubles the forward FLOPs (input-grad + weight-grad
  // matmuls for each linear). The Chinchilla "6·N·D" accounting uses 6 FLOPs
  // per parameter per token: 2 forward + 4 backward.
  const bwdFlops = 2 * fwdFlops;

  const bytesQ   = d * d * dtype;
  const bytesK   = d * dkv * dtype;
  const bytesV   = d * dkv * dtype;
  const bytesO   = d * d * dtype;
  const bytesUp  = 4 * d * d * dtype;
  const bytesDn  = 4 * d * d * dtype;
  const totalBytes = bytesQ + bytesK + bytesV + bytesO + bytesUp + bytesDn;

  return (
    <div className="viz-panel tc-arch-diagram">
      <div className="tc-arch-tabs" role="tablist" aria-label="Forward or backward pass">
        <button
          type="button"
          role="tab"
          aria-selected={!backward}
          className={`tc-arch-tab${!backward ? ' active' : ''}`}
          onClick={() => setMode('forward')}
        >
          Forward
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={backward}
          className={`tc-arch-tab${backward ? ' active' : ''}`}
          onClick={() => setMode('backward')}
        >
          Backward
        </button>
      </div>

      {backward
        ? <ArchBlock name="∂L / ∂Output" shape="(1, d) — grad in" kind="io" />
        : <ArchBlock name="Input X" shape="(1, d)" kind="io" />}
      <ArchArrow direction={backward ? 'up' : 'down'} />

      <div className="tc-arch-section">
        <div className="tc-arch-section-label">
          {backward ? 'FFN block — backward' : 'Attention block — forward'}
        </div>
        {backward ? (
          <>
            <ArchBlock
              name="FFN down ᵀ"
              shape="(1, d) · (d, 4d)"
              flops="16d²" numeric={num(2 * flopsDn)}
              bytes="4d²·dtype" bytesNumeric={numB(bytesDn)}
              highlight
            />
            <ArchArrow direction="up" />
            <ArchBlock
              name="FFN up ᵀ"
              shape="(1, 4d) · (4d, d)"
              flops="16d²" numeric={num(2 * flopsUp)}
              bytes="4d²·dtype" bytesNumeric={numB(bytesUp)}
              highlight
            />
          </>
        ) : (
          <div className="tc-arch-parallel">
            <ArchBlock name="Q proj" shape="(1, d) · (d, d)"
              flops="2d²" numeric={num(flopsQ)}
              bytes="d²·dtype" bytesNumeric={numB(bytesQ)} />
            <ArchBlock name="K proj" shape="(1, d) · (d, d_kv)"
              flops="2·d·d_kv" numeric={num(flopsK)}
              bytes="d·d_kv·dtype" bytesNumeric={numB(bytesK)} />
            <ArchBlock name="V proj" shape="(1, d) · (d, d_kv)"
              flops="2·d·d_kv" numeric={num(flopsV)}
              bytes="d·d_kv·dtype" bytesNumeric={numB(bytesV)} />
          </div>
        )}
        <ArchArrow direction={backward ? 'up' : 'down'} />
        {backward ? null : (
          <>
            <ArchBlock name="Attention" shape="q · Kᵀ  +  A · V"
              flops="4T·d" numeric={num(flopsAtt)} />
            <ArchArrow direction="down" />
            <ArchBlock name="O proj" shape="(1, d) · (d, d)"
              flops="2d²" numeric={num(flopsO)}
              bytes="d²·dtype" bytesNumeric={numB(bytesO)} />
          </>
        )}
      </div>

      <ArchArrow direction={backward ? 'up' : 'down'} />

      <div className="tc-arch-section">
        <div className="tc-arch-section-label">
          {backward ? 'Attention block — backward' : 'FFN block — forward'}
        </div>
        {backward ? (
          <>
            <ArchBlock name="O proj ᵀ" shape="(1, d) · (d, d)"
              flops="4d²" numeric={num(2 * flopsO)}
              bytes="d²·dtype" bytesNumeric={numB(bytesO)} highlight />
            <ArchArrow direction="up" />
            <ArchBlock name="Attention ᵀ" shape="∂A, ∂Q, ∂K, ∂V"
              flops="8T·d" numeric={num(2 * flopsAtt)} highlight />
            <ArchArrow direction="up" />
            <div className="tc-arch-parallel">
              <ArchBlock name="Q proj ᵀ" shape="(1, d) · (d, d)"
                flops="4d²" numeric={num(2 * flopsQ)}
                bytes="d²·dtype" bytesNumeric={numB(bytesQ)} highlight />
              <ArchBlock name="K proj ᵀ" shape="(1, d) · (d, d_kv)"
                flops="4·d·d_kv" numeric={num(2 * flopsK)}
                bytes="d·d_kv·dtype" bytesNumeric={numB(bytesK)} highlight />
              <ArchBlock name="V proj ᵀ" shape="(1, d) · (d, d_kv)"
                flops="4·d·d_kv" numeric={num(2 * flopsV)}
                bytes="d·d_kv·dtype" bytesNumeric={numB(bytesV)} highlight />
            </div>
          </>
        ) : (
          <>
            <ArchBlock name="FFN up" shape="(1, d) · (d, 4d)"
              flops="8d²" numeric={num(flopsUp)}
              bytes="4d²·dtype" bytesNumeric={numB(bytesUp)} />
            <ArchArrow direction="down" />
            <ArchBlock name="FFN down" shape="(1, 4d) · (4d, d)"
              flops="8d²" numeric={num(flopsDn)}
              bytes="4d²·dtype" bytesNumeric={numB(bytesDn)} />
          </>
        )}
      </div>

      <ArchArrow direction={backward ? 'up' : 'down'} />
      {backward
        ? <ArchBlock name="∂L / ∂Input" shape="(1, d) — grad out" kind="io" />
        : <ArchBlock name="Output" shape="(1, d)" kind="io" />}

      <div className="tc-arch-total">
        <div className="tc-arch-total-line">
          <span className="tc-arch-total-key">Compute</span>
          <span className="tc-arch-total-formula">
            {backward
              ? '≈ 2 × forward (input-grad + weight-grad matmuls)'
              : '4d² + 4·d·d_kv + 16d² + 4T·d'}
          </span>
          <span className="tc-arch-total-value">
            ≈ <strong>{num(backward ? bwdFlops : fwdFlops)}</strong>
          </span>
        </div>
        <div className="tc-arch-total-line tc-arch-total-mem">
          <span className="tc-arch-total-key">Memory (weights)</span>
          <span className="tc-arch-total-formula">(10d² + 2·d·d_kv) · dtype</span>
          <span className="tc-arch-total-value">≈ <strong>{numB(totalBytes)}</strong></span>
        </div>
        <span className="tc-arch-total-note">
          per layer per token · d = {d}, d_kv = {dkv}, T = {T} · MHA · bf16
        </span>
      </div>

      <p className="viz-caption" style={{ marginTop: 14 }}>
        {backward
          ? 'Each linear in the backward pass costs two matmuls — one for the gradient of the input, one for the gradient of the weight. That\'s why total backward FLOPs are roughly 2× the forward, and total training FLOPs ≈ 3× forward. To do any of this, every activation from the forward pass must still be in memory.'
          : 'Two FLOP categories dominate: attention scales as O(L²·d) — it dominates at long sequences. The MLP scales as O(L·d²) — it dominates per token at moderate length. Flip to Backward to see what gradient flow adds.'}
      </p>
    </div>
  );
}

/* =========================================================
   Widget 2 — CostBreakdown (ported; training framing)
   Per-token / per-step training FLOPs as you vary d and T.
   ========================================================= */
const COST_D_OPTIONS = [256, 512, 1024, 2048, 4096, 8192, 12288];
const COST_T_OPTIONS = [128, 512, 1024, 2048, 4096, 8192, 16384, 32768];
const COST_B_OPTIONS = [1, 4, 16, 64, 256, 1024];

function CostBreakdown() {
  const [dIdx, setDIdx] = useState(4); // 4096
  const [tIdx, setTIdx] = useState(3); // 2048
  const [bIdx, setBIdx] = useState(2); // 16

  const d = COST_D_OPTIONS[dIdx];
  const T = COST_T_OPTIONS[tIdx];
  const B = COST_B_OPTIONS[bIdx];

  // Per-token (one token in the sequence) forward FLOPs, MHA assumed (d_kv = d).
  //   Q + K + V + O  = 8d²
  //   FFN up + down  = 16d²
  //   attention      = 4T·d
  const fwdPerToken = 8 * d * d + 16 * d * d + 4 * T * d; // = 24d² + 4T·d
  const bwdPerToken = 2 * fwdPerToken;
  const totalPerToken = fwdPerToken + bwdPerToken; // ≈ 6 · N_params * 1 token-equivalent
  // Per training step: B·T tokens through every (one) layer.
  const tokensPerStep = B * T;
  const fwdPerStep = tokensPerStep * fwdPerToken;
  const totalPerStep = tokensPerStep * totalPerToken;

  return (
    <div className="viz-panel">
      <div className="tc-cost-pair">
        <div className="tc-cost-col tc-cost-fwd">
          <div className="tc-cost-title">
            <span className="tc-cost-badge tc-cost-badge-fwd">F</span>
            Forward pass · per token
          </div>
          <div className="tc-cost-row">
            <span>Q, K, V, O projections</span>
            <span className="tc-cost-mag">8d²</span>
          </div>
          <div className="tc-cost-row">
            <span>FFN up + down (8d² each)</span>
            <span className="tc-cost-mag">16d²</span>
          </div>
          <div className="tc-cost-row">
            <span>Attention  q · Kᵀ + A · V</span>
            <span className="tc-cost-mag">4T · d</span>
          </div>
          <div className="tc-cost-total">
            per token ≈ <strong>24d² + 4T·d</strong> ≈{' '}
            <strong>{fmtNum(fwdPerToken, 'FLOP')}</strong>
            <em> · the FFN/projections scale as d², attention as T·d</em>
          </div>
        </div>

        <div className="tc-cost-col tc-cost-bwd">
          <div className="tc-cost-title">
            <span className="tc-cost-badge tc-cost-badge-bwd">B</span>
            Backward pass · per token
          </div>
          <div className="tc-cost-row">
            <span>Input-grad matmuls  <em>(every linear)</em></span>
            <span className="tc-cost-mag">≈ forward</span>
          </div>
          <div className="tc-cost-row">
            <span>Weight-grad matmuls <em>(every linear)</em></span>
            <span className="tc-cost-mag">≈ forward</span>
          </div>
          <div className="tc-cost-row">
            <span>Attention backward (∂A, ∂Q, ∂K, ∂V)</span>
            <span className="tc-cost-mag">≈ 2 × forward</span>
          </div>
          <div className="tc-cost-total">
            per token ≈ <strong>2 × forward</strong> ≈{' '}
            <strong>{fmtNum(bwdPerToken, 'FLOP')}</strong>
            <em> · so total training ≈ 3 × forward per token</em>
          </div>
        </div>
      </div>

      <div className="tc-cost-sliders">
        <label className="tc-cost-slider">
          <span className="tc-cost-slider-label">
            d <em>(model width)</em>: <strong>{d.toLocaleString()}</strong>
          </span>
          <input type="range" min={0} max={COST_D_OPTIONS.length - 1} step={1}
            value={dIdx} onChange={(e) => setDIdx(+e.target.value)} />
        </label>
        <label className="tc-cost-slider">
          <span className="tc-cost-slider-label">
            T <em>(sequence length)</em>: <strong>{T.toLocaleString()}</strong>
          </span>
          <input type="range" min={0} max={COST_T_OPTIONS.length - 1} step={1}
            value={tIdx} onChange={(e) => setTIdx(+e.target.value)} />
        </label>
        <label className="tc-cost-slider">
          <span className="tc-cost-slider-label">
            B <em>(batch size)</em>: <strong>{B.toLocaleString()}</strong>
          </span>
          <input type="range" min={0} max={COST_B_OPTIONS.length - 1} step={1}
            value={bIdx} onChange={(e) => setBIdx(+e.target.value)} />
        </label>
      </div>

      <div className="tc-cost-readouts">
        <div className="tc-cost-readout">
          <div className="tc-cost-readout-label">Forward / step</div>
          <div className="tc-cost-readout-value">{fmtNum(fwdPerStep, 'FLOP')}</div>
          <div className="tc-cost-readout-sub">
            B · T · per-token = {B.toLocaleString()} × {T.toLocaleString()} ×{' '}
            {fmtNum(fwdPerToken, 'FLOP')}
          </div>
        </div>
        <div className="tc-cost-readout">
          <div className="tc-cost-readout-label">Total (fwd + bwd) / step</div>
          <div className="tc-cost-readout-value">{fmtNum(totalPerStep, 'FLOP')}</div>
          <div className="tc-cost-readout-sub">
            roughly 3 × forward · one transformer layer
          </div>
        </div>
      </div>

      <p className="viz-caption" style={{ marginTop: 14 }}>
        Slide <strong>T</strong> up: the attention term 4T·d eventually catches and overtakes the
        24d² FFN/projection cost — that's the quadratic-in-sequence-length blow-up. Slide{' '}
        <strong>B</strong> up: per-step FLOPs grow linearly, but weight reads stay fixed, so
        arithmetic intensity grows. That's why training is compute-bound at realistic batch
        sizes.
      </p>
    </div>
  );
}

/* =========================================================
   Widget 3 — TrainingMemoryFootprint (new)
   Stacked bar of params / gradients / optimizer state / activations.
   Activations are the lever: slider for sequence length and number of layers.
   ========================================================= */

const PARAMS_OPTIONS = [
  { label: '1B',  N: 1e9 },
  { label: '7B',  N: 7e9 },
  { label: '13B', N: 13e9 },
  { label: '70B', N: 70e9 },
  { label: '175B', N: 175e9 },
];
const SEQ_OPTIONS = [512, 2048, 4096, 8192, 16384, 32768];

function TrainingMemoryFootprint() {
  const [pIdx, setPIdx] = useState(1); // 7B
  const [tIdx, setTIdx] = useState(1); // 2048

  // Standard mixed-precision Adam accounting:
  //   parameters       2 bytes/param  (bf16)
  //   gradients        2 bytes/param  (bf16)
  //   optimizer state  8 bytes/param  (Adam: fp32 momentum + fp32 variance)
  //                                   + fp32 master copy of weights is often
  //                                   counted separately as another 4 B/param.
  //   activations      ≈ batch · layers · hidden · seq · constant  (bf16)
  // We use the Megatron-style constant per layer per token at bf16 as a
  // floor; users should read this as "the right *direction*" not gospel,
  // hence we lean on the slider and a caption disclaimer.

  const N = PARAMS_OPTIONS[pIdx].N;
  const T = SEQ_OPTIONS[tIdx];

  // Match each model size to a representative hidden width / layer count.
  // (Approximate — chosen so the model has roughly N parameters at ~12·L·d²
  //  the standard transformer rule.)
  const ARCH = {
    1e9:   { d: 2048, L: 16 },
    7e9:   { d: 4096, L: 32 },
    13e9:  { d: 5120, L: 40 },
    70e9:  { d: 8192, L: 80 },
    175e9: { d: 12288, L: 96 },
  }[N];

  const paramsBytes      = N * 2;
  const gradientsBytes   = N * 2;
  const optimizerBytes   = N * 8;  // Adam: fp32 m + fp32 v
  const masterBytes      = N * 4;  // fp32 master copy of weights
  // Activations: ~ B · L · d · T · k bytes at bf16. With activation checkpointing
  // turned off and a batch of 1, the floor is roughly 2 · L · d · T · c bytes
  // where c counts the saved tensors per layer (~16-20 in a standard transformer).
  const ACT_PER_TOKEN_PER_LAYER = 20 * 2; // 20 saved tensors · 2 bytes
  const activationsBytes = ARCH.L * ARCH.d * T * ACT_PER_TOKEN_PER_LAYER;

  const total = paramsBytes + gradientsBytes + optimizerBytes + masterBytes + activationsBytes;

  const segments = [
    { key: 'params', label: 'Parameters (bf16)',     bytes: paramsBytes,     color: '#2563eb' },
    { key: 'grads',  label: 'Gradients (bf16)',      bytes: gradientsBytes,  color: '#60a5fa' },
    { key: 'opt',    label: 'Optimizer state (Adam, fp32 m + v)', bytes: optimizerBytes, color: '#0ea5e9' },
    { key: 'master', label: 'fp32 master weights',   bytes: masterBytes,     color: '#0891b2' },
    { key: 'acts',   label: 'Activations (bf16, batch = 1)', bytes: activationsBytes, color: '#f59e0b' },
  ];

  return (
    <div className="viz-panel">
      <div className="tc-mem-controls">
        <label className="tc-cost-slider">
          <span className="tc-cost-slider-label">
            Model size: <strong>{PARAMS_OPTIONS[pIdx].label}</strong>{' '}
            <em>(d = {ARCH.d}, layers = {ARCH.L})</em>
          </span>
          <input type="range" min={0} max={PARAMS_OPTIONS.length - 1} step={1}
            value={pIdx} onChange={(e) => setPIdx(+e.target.value)} />
        </label>
        <label className="tc-cost-slider">
          <span className="tc-cost-slider-label">
            Sequence length T: <strong>{T.toLocaleString()}</strong>
          </span>
          <input type="range" min={0} max={SEQ_OPTIONS.length - 1} step={1}
            value={tIdx} onChange={(e) => setTIdx(+e.target.value)} />
        </label>
      </div>

      <div className="tc-mem-bar" role="img" aria-label="Training memory footprint stacked bar">
        {segments.map((s) => {
          const pct = (s.bytes / total) * 100;
          if (pct < 0.4) return null;
          return (
            <div
              key={s.key}
              className="tc-mem-seg"
              style={{ width: `${pct}%`, background: s.color }}
              title={`${s.label}: ${fmtNum(s.bytes, 'B')}`}
            >
              {pct >= 6 && (
                <span className="tc-mem-seg-label">
                  {fmtNum(s.bytes, 'B')}
                </span>
              )}
            </div>
          );
        })}
      </div>

      <div className="tc-mem-legend">
        {segments.map((s) => (
          <div key={s.key} className="tc-mem-legend-row">
            <span className="tc-mem-legend-swatch" style={{ background: s.color }} />
            <span className="tc-mem-legend-label">{s.label}</span>
            <span className="tc-mem-legend-value">{fmtNum(s.bytes, 'B')}</span>
          </div>
        ))}
        <div className="tc-mem-legend-row tc-mem-legend-total">
          <span className="tc-mem-legend-swatch" style={{ background: 'transparent' }} />
          <span className="tc-mem-legend-label"><strong>Total</strong></span>
          <span className="tc-mem-legend-value"><strong>{fmtNum(total, 'B')}</strong></span>
        </div>
      </div>

      <p className="viz-caption" style={{ marginTop: 14 }}>
        At short sequences, parameters + gradients + optimizer state dominate — that's
        the famous <strong>16 bytes per parameter</strong> footprint of mixed-precision
        Adam (2 + 2 + 8 + 4). Slide T up: activations grow linearly and overtake everything
        else. Activations dominate at long sequences — that's why every training
        optimization in Part 2 (gradient checkpointing, ZeRO, FlashAttention) targets them.
        <em>{' '}Numbers shown assume batch = 1 and no activation checkpointing — the
        direction matters more than the exact constant.</em>
      </p>
    </div>
  );
}

/* =========================================================
   Optional 6·N·D code expander
   ========================================================= */
const CODE = `# Chinchilla-style rule of thumb: total training FLOPs
# Hoffmann et al. 2022, eq. 2.
# N = number of parameters, D = number of training tokens.
def training_flops(N, D):
    return 6 * N * D   # 2 forward + 4 backward FLOPs per param per token

# Llama-2-7B: 7e9 params trained on 2e12 tokens
print(training_flops(7e9, 2e12))  # 8.4e22 FLOPs
`;

function CodeBlock() {
  return (
    <details className="post-code-details">
      <summary>
        <span className="post-code-summary-icon" aria-hidden="true">›</span>
        <span>The 6·N·D rule of thumb in four lines</span>
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
export default function TrainingCost() {
  usePageMeta({
    title: 'Training Cost of a Transformer',
    description: 'What one transformer layer actually does — forward and backward — and what has to be held in memory between them.',
    slug: 'training-cost',
    publishedDate: '2026-02-19',
    keywords: ['training cost', 'transformer', 'FLOPs', 'memory', 'Llama'],
  });

  return (
    <article className="post-2026 tc-post">
      <div className="post-wrap">
        <div className="post-hero">
          <div className="post-series-tag">
            <span className="post-live-dot" aria-hidden="true" />
            Transformer Costs · Part 1 of 4
          </div>
          <h1>Training Cost of a Transformer</h1>
          <p className="post-lede">
            Training Llama-2-7B took approximately <a className="post-link" href="https://arxiv.org/abs/2307.09288" target="_blank" rel="noreferrer">1.7 million A100-GPU-hours</a>;
            GPT-3 reportedly cost on the order of <a className="post-link" href="https://lambdalabs.com/blog/demystifying-gpt-3" target="_blank" rel="noreferrer">a few million dollars</a> in
            compute alone. To know why, you need to count what one transformer layer
            actually does — forward and backward — and what has to be held in memory
            between them.
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

        <h2 className="reveal">What one transformer layer costs</h2>
        <p>
          Every interesting trade-off in this series — training cost, inference cost,
          KV caching, paging — bottoms out in the same question: how much math and how
          many bytes does a single forward pass through one transformer layer actually
          take? The answer depends on only a handful of dimensions and one accounting
          rule, so let's get them on the table.
        </p>
        <p>
          A transformer layer has two parts. The <strong>attention block</strong>{' '}
          projects the input into queries, keys, values, runs attention, then projects
          the result back (the four matrices <Katex tex="W_Q" />, <Katex tex="W_K" />,{' '}
          <Katex tex="W_V" />, <Katex tex="W_O" />). The <strong>FFN block</strong>{' '}
          applies two linear layers with a 4× hidden expansion in between
          (<Katex tex="W_{up}" />, <Katex tex="W_{down}" />). Inputs and outputs are
          the same shape, so layers can be stacked.
        </p>
        <p>
          Two dimensions matter: <Katex tex="d_{model}" /> (often shortened to{' '}
          <Katex tex="d" />) is the residual stream width, and <Katex tex="T" /> is the
          sequence length. Every operation is a matrix multiply, and the standard
          accounting is: a matmul of shape{' '}
          <Katex tex="(M \times N) \cdot (N \times P)" /> costs{' '}
          <strong>2·M·N·P FLOPs</strong>. Apply that rule to each block.
        </p>

        <TransformerLayerDiagram />

        <p>
          Two FLOP categories dominate. The MLP and the four projections all scale as{' '}
          <Katex tex="O(L \cdot d^2)" /> per token — they're the cost per token at
          moderate sequence length. Attention itself scales as{' '}
          <Katex tex="O(L^2 \cdot d)" /> across the whole sequence — small at short{' '}
          <Katex tex="T" />, but eventually quadratic and dominant.
        </p>

        <h2 className="reveal">The backward pass</h2>
        <p>
          Training adds a backward pass. For each linear layer, backprop computes two
          things: the gradient with respect to the input (so it can flow into the
          previous layer) and the gradient with respect to the weights (so the
          optimizer has something to update). Each of those is a matmul of roughly the
          same size as the forward.
        </p>
        <p>
          The accepted rule of thumb is{' '}
          <Katex tex="\text{training FLOPs} \approx 6 \cdot N \cdot D" />, where{' '}
          <Katex tex="N" /> is the number of parameters and <Katex tex="D" /> the
          number of training tokens. The decomposition: 2 FLOPs per parameter per
          token for the forward pass, and 4 for the backward — see{' '}
          <a className="post-link" href="https://arxiv.org/abs/2001.08361" target="_blank" rel="noreferrer">Kaplan
          et al. 2020</a> and{' '}
          <a className="post-link" href="https://arxiv.org/abs/2203.15556" target="_blank" rel="noreferrer">Hoffmann
          et al. 2022</a>. Flip the diagram above to <em>Backward</em> and you can
          see where the doubling comes from.
        </p>

        <CodeBlock />

        <p>
          Vary the dimensions in the next widget to see how the per-token cost moves.
          Push <Katex tex="T" /> high enough and the attention term — flat compared to
          the d² terms at first — overtakes everything else.
        </p>

        <CostBreakdown />

        <h2 className="reveal">What needs to be stored on the forward pass</h2>
        <p>
          FLOPs are only half the cost. Training also has to hold the model itself in
          GPU memory, plus everything needed to run backprop. There are four buckets:
        </p>
        <ul className="tc-list">
          <li>
            <strong>Parameters.</strong> The weights, at 2 bytes per parameter for
            bf16 — the standard mixed-precision setup.
          </li>
          <li>
            <strong>Gradients.</strong> Same shape as the parameters, also at 2 bytes
            each.
          </li>
          <li>
            <strong>Optimizer state.</strong> Adam stores momentum and variance, both
            kept in fp32. That's 2 × 4 = 8 bytes per parameter, plus a fp32 master
            copy of the weights (another 4 bytes). This is the famous "16 bytes per
            parameter" footprint of mixed-precision Adam.
          </li>
          <li>
            <strong>Activations.</strong> Every intermediate tensor from the forward
            pass that backprop needs to compute gradients. Roughly proportional to{' '}
            <Katex tex="B \cdot L \cdot d \cdot T" /> — batch, layers, width,
            sequence length. The big one.
          </li>
        </ul>

        <TrainingMemoryFootprint />

        <h2 className="reveal">Compute or memory bound?</h2>
        <p>
          Decode-time inference is famously memory-bandwidth-bound: one new token, one
          read of all the weights, almost no arithmetic per byte. Training is the
          opposite story.
        </p>
        <p>
          A training step processes <Katex tex="B \cdot T" /> tokens through every
          weight at once. The effective number of FLOPs per byte of weights read — the{' '}
          <em>arithmetic intensity</em> — is in the thousands. An A100 sits at roughly{' '}
          <a className="post-link" href="https://www.nvidia.com/en-us/data-center/a100/" target="_blank" rel="noreferrer">78
          FLOPs/byte</a> at fp16, an H100 at <Katex tex="\approx 295" />. Training
          flies past both. The math units are the bottleneck.
        </p>
        <p>
          The practical consequence: training clusters are sized for FLOPs; serving
          clusters are sized for HBM bandwidth. Same hardware, two different
          economics. Part 3 picks up the inference side of that asymmetry — see{' '}
          <a className="post-link" href="/blog/inference-cost">Inference Cost of a Transformer</a>.
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
            <div className="ref-note">Original transformer paper — defines the layer we're costing.</div>
          </div>
          <div className="ref-link"><a href="https://arxiv.org/abs/1706.03762" target="_blank" rel="noreferrer">arxiv.org</a></div>

          <div className="ref-cite">Kaplan et al. 2020</div>
          <div>
            <div className="ref-title">
              <a href="https://arxiv.org/abs/2001.08361" target="_blank" rel="noreferrer">Scaling Laws for Neural Language Models</a>
            </div>
            <div className="ref-note">Source of the 6·N·D FLOP accounting (Appendix B).</div>
          </div>
          <div className="ref-link"><a href="https://arxiv.org/abs/2001.08361" target="_blank" rel="noreferrer">arxiv.org</a></div>

          <div className="ref-cite">Hoffmann et al. 2022</div>
          <div>
            <div className="ref-title">
              <a href="https://arxiv.org/abs/2203.15556" target="_blank" rel="noreferrer">Training Compute-Optimal Large Language Models (Chinchilla)</a>
            </div>
            <div className="ref-note">Refines the FLOPs-per-token-per-parameter accounting and the data/compute trade-off.</div>
          </div>
          <div className="ref-link"><a href="https://arxiv.org/abs/2203.15556" target="_blank" rel="noreferrer">arxiv.org</a></div>

          <div className="ref-cite">Touvron et al. 2023</div>
          <div>
            <div className="ref-title">
              <a href="https://arxiv.org/abs/2307.09288" target="_blank" rel="noreferrer">Llama 2: Open Foundation and Fine-Tuned Chat Models</a>
            </div>
            <div className="ref-note">Reports ~1.7M A100-hours for the 7B model (Table 2).</div>
          </div>
          <div className="ref-link"><a href="https://arxiv.org/abs/2307.09288" target="_blank" rel="noreferrer">arxiv.org</a></div>

          <div className="ref-cite">NVIDIA</div>
          <div>
            <div className="ref-title">
              <a href="https://resources.nvidia.com/en-us-tensor-core/nvidia-tensor-core-gpu-datasheet" target="_blank" rel="noreferrer">A100 / H100 datasheets</a>
            </div>
            <div className="ref-note">Peak FLOPs and HBM bandwidth used for the ridge-point numbers.</div>
          </div>
          <div className="ref-link"><a href="https://resources.nvidia.com/en-us-tensor-core/nvidia-tensor-core-gpu-datasheet" target="_blank" rel="noreferrer">nvidia.com</a></div>

          <div className="ref-cite">Rajbhandari et al. 2020</div>
          <div>
            <div className="ref-title">
              <a href="https://arxiv.org/abs/1910.02054" target="_blank" rel="noreferrer">ZeRO: Memory Optimizations Toward Training Trillion Parameter Models</a>
            </div>
            <div className="ref-note">Where the 16-bytes-per-param mixed-precision Adam accounting is laid out cleanly (Table 1).</div>
          </div>
          <div className="ref-link"><a href="https://arxiv.org/abs/1910.02054" target="_blank" rel="noreferrer">arxiv.org</a></div>
        </div>

        <footer className="post-footer">
          <p>
            <strong>Part 1 of Transformer Costs</strong> · Next:{' '}
            <a className="post-link" href="/blog/training-cost-optimization">Optimizing Training Cost</a>
            {' · '}Continues with:{' '}
            <a className="post-link" href="/blog/inference-cost">Inference Cost of a Transformer</a>.
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
