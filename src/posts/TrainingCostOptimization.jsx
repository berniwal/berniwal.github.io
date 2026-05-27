// src/posts/TrainingCostOptimization.jsx
// Transformer Costs · Part 2 of 4 — Optimizing Training Cost.
// TAGS FOR REGISTRATION: ['transformers', 'training', 'optimization']
// EXCERPT: Part 1 showed activations dominate training memory. Part 2 walks through the four tricks that fight back — gradient accumulation, gradient checkpointing, mixed precision, and ZeRO/FSDP sharding — and where Flash Attention fits in.
import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneLight } from 'react-syntax-highlighter/dist/esm/styles/prism';
import './PostChrome.css';
import './TrainingCostOptimization.css';

/* ============================================================
   KaTeX wrapper — uses the CDN global window.katex
   ============================================================ */
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
  return block
    ? <div ref={ref} className={`tco-katex-block ${className}`} />
    : <span ref={ref} className={`tco-katex-inline ${className}`} />;
}

/* ============================================================
   Widget 1 — Gradient accumulation
   N micro-batches → accumulator → one optimizer step.
   Slider sets N; playback animates the loop.
   ============================================================ */
function GradientAccumulation() {
  const STEPS = [1, 2, 4, 8, 16];
  const [n, setN] = useState(4);
  const [tick, setTick] = useState(0);
  const [playing, setPlaying] = useState(true);
  const timerRef = useRef(null);

  // Animation: each micro-batch fades in over its slot, then the optimizer step pops.
  const TOTAL = n + 1; // n micro-batches + 1 optimizer step
  useEffect(() => {
    if (!playing) return;
    timerRef.current = setInterval(() => {
      setTick((t) => (t + 1) % (TOTAL + 1));
    }, 650);
    return () => clearInterval(timerRef.current);
  }, [playing, TOTAL]);

  // When n changes, restart at zero so animation reads cleanly.
  useEffect(() => { setTick(0); }, [n]);

  const W = 760, H = 220;
  const padX = 56, padY = 40;
  const laneTop = padY + 18;
  const laneBot = H - padY - 18;
  const accY = (laneTop + laneBot) / 2;
  const slotW = (W - 2 * padX - 110) / n; // leave room for accumulator on the right
  const accX = W - padX - 60;

  return (
    <div className="viz-panel tco-ga">
      <div className="tco-ga-controls">
        <label className="tco-slider">
          <span>Accumulation steps <strong>N</strong></span>
          <input
            type="range"
            min={0}
            max={STEPS.length - 1}
            step={1}
            value={STEPS.indexOf(n)}
            onChange={(e) => setN(STEPS[+e.target.value])}
          />
          <span className="tco-val">{n}</span>
        </label>
        <button
          type="button"
          className="tco-action"
          onClick={() => setPlaying((p) => !p)}
        >
          {playing ? '❚❚ Pause' : '▶ Play'}
        </button>
      </div>

      <svg viewBox={`0 0 ${W} ${H}`} width="100%" preserveAspectRatio="xMidYMid meet">
        {/* Micro-batch lane */}
        <text x={padX} y={laneTop - 12} className="tco-ga-lane-label">micro-batches</text>
        {Array.from({ length: n }).map((_, i) => {
          const x = padX + i * slotW + 4;
          const filled = tick > i;
          return (
            <g key={`mb-${i}`} className={`tco-ga-mb${filled ? ' on' : ''}`}>
              <rect x={x} y={laneTop} width={slotW - 8} height={36} rx={6} />
              <text x={x + (slotW - 8) / 2} y={laneTop + 22} className="tco-ga-mb-label">
                {`b${i + 1}`}
              </text>
              {/* arrow into accumulator */}
              <line
                x1={x + (slotW - 8) / 2}
                y1={laneTop + 36}
                x2={accX + 30}
                y2={accY - 4}
                className={`tco-ga-arrow${filled ? ' on' : ''}`}
                markerEnd="url(#tco-arrow)"
              />
            </g>
          );
        })}

        {/* Accumulator */}
        <g className={`tco-ga-acc${tick >= n && tick > 0 ? ' full' : tick > 0 ? ' filling' : ''}`}>
          <rect x={accX} y={accY - 40} width={70} height={80} rx={10} />
          <text x={accX + 35} y={accY - 12} className="tco-ga-acc-title">acc</text>
          <text x={accX + 35} y={accY + 10} className="tco-ga-acc-count">
            {Math.min(tick, n)}/{n}
          </text>
          <text x={accX + 35} y={accY + 28} className="tco-ga-acc-sub">grads</text>
        </g>

        {/* Optimizer step */}
        <g className={`tco-ga-opt${tick === TOTAL ? ' on' : ''}`}>
          <line x1={accX + 35} y1={accY + 40} x2={accX + 35} y2={laneBot - 6}
            className={`tco-ga-arrow${tick === TOTAL ? ' on' : ''}`}
            markerEnd="url(#tco-arrow)" />
          <rect x={accX - 35} y={laneBot - 6} width={140} height={36} rx={8} />
          <text x={accX + 35} y={laneBot + 16} className="tco-ga-opt-label">
            optimizer.step()
          </text>
        </g>

        <defs>
          <marker id="tco-arrow" viewBox="0 0 10 10" refX="9" refY="5"
            markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M0,0 L10,5 L0,10 z" fill="currentColor" />
          </marker>
        </defs>
      </svg>

      <div className="tco-ga-readout">
        <div className="tco-ga-stat">
          <div className="tco-ga-stat-label">effective batch</div>
          <div className="tco-ga-stat-value">N × micro = <strong>{n}×</strong></div>
        </div>
        <div className="tco-ga-stat">
          <div className="tco-ga-stat-label">peak activation memory</div>
          <div className="tco-ga-stat-value">one micro-batch <strong>(unchanged)</strong></div>
        </div>
      </div>
      <p className="viz-caption">
        Run <em>N</em> forward+backward passes, add the gradients into one buffer, then take a single
        optimizer step. The optimizer sees the average over N micro-batches; memory only ever holds one.
      </p>
    </div>
  );
}

/* ============================================================
   Widget 2 — Gradient checkpointing
   Row of layer boxes. Toggle: full vs sqrt-checkpointed.
   ============================================================ */
function GradientCheckpointing() {
  const [mode, setMode] = useState('full'); // 'full' | 'ckpt'
  const L = 16;
  const k = Math.round(Math.sqrt(L)); // checkpoint every k-th
  const W = 760, H = 200;
  const padX = 40, padY = 60;
  const slot = (W - 2 * padX) / L;

  const stored = mode === 'full'
    ? Array.from({ length: L }, (_, i) => i)
    : Array.from({ length: L }, (_, i) => i).filter((i) => i % k === 0);
  const recompute = mode === 'ckpt'
    ? Array.from({ length: L }, (_, i) => i).filter((i) => i % k !== 0)
    : [];

  const storedPct = Math.round((stored.length / L) * 100);
  const extraCompute = mode === 'ckpt' ? '+1 forward pass' : '0';

  return (
    <div className="viz-panel tco-gc">
      <div className="tco-tabs">
        <button
          type="button"
          className={`tco-tab${mode === 'full' ? ' active' : ''}`}
          onClick={() => setMode('full')}
        >
          Full activations
        </button>
        <button
          type="button"
          className={`tco-tab${mode === 'ckpt' ? ' active' : ''}`}
          onClick={() => setMode('ckpt')}
        >
          √L checkpointing
        </button>
      </div>

      <svg viewBox={`0 0 ${W} ${H}`} width="100%" preserveAspectRatio="xMidYMid meet">
        <text x={padX} y={padY - 14} className="tco-gc-lane-label">layers (forward →)</text>
        {Array.from({ length: L }).map((_, i) => {
          const x = padX + i * slot + 2;
          const isStored = stored.includes(i);
          const isRecomp = recompute.includes(i);
          return (
            <g key={`l-${i}`}
               className={`tco-gc-layer${isStored ? ' stored' : ''}${isRecomp ? ' recomp' : ''}`}>
              <rect x={x} y={padY} width={slot - 4} height={48} rx={5} />
              <text x={x + (slot - 4) / 2} y={padY + 30} className="tco-gc-layer-label">
                {i + 1}
              </text>
            </g>
          );
        })}

        {/* Legend */}
        <g transform={`translate(${padX}, ${H - 32})`}>
          <rect x={0} y={0} width={14} height={14} rx={3} className="tco-gc-legend stored" />
          <text x={20} y={11} className="tco-gc-legend-label">activation stored</text>
          <rect x={170} y={0} width={14} height={14} rx={3} className="tco-gc-legend recomp" />
          <text x={190} y={11} className="tco-gc-legend-label">recomputed on backward</text>
        </g>
      </svg>

      <div className="tco-gc-readout">
        <div className="tco-ga-stat">
          <div className="tco-ga-stat-label">activations stored</div>
          <div className="tco-ga-stat-value"><strong>{storedPct}%</strong> ({stored.length} of {L})</div>
        </div>
        <div className="tco-ga-stat">
          <div className="tco-ga-stat-label">extra compute</div>
          <div className="tco-ga-stat-value"><strong>{extraCompute}</strong></div>
        </div>
      </div>
      <p className="viz-caption">
        Store only every <Katex tex={'\\sqrt{L}'} />-th layer's activations. When the backward pass needs the rest,
        rerun the forward from the nearest checkpoint. One extra forward, roughly <Katex tex={'\\sqrt{L}'} /> memory.
      </p>
    </div>
  );
}

/* ============================================================
   Widget 3 — Mixed precision comparison
   Two-row bar chart: fp32 vs bf16 bytes per parameter.
   ============================================================ */
function MixedPrecisionCompare() {
  // Adam-style breakdown per parameter, including master copy:
  // fp32: 4 (params) + 4 (grads) + 8 (Adam m,v) = 16
  // mixed: 2 (bf16 params) + 2 (bf16 grads) + 4 (fp32 master) + 8 (fp32 Adam m,v) = 16
  // Activations are what really shrink (~2x). Show params+grads as the simple story.
  const rows = [
    { label: 'fp32',  parts: [{ k: 'params', v: 4 }, { k: 'grads', v: 4 }], total: 8 },
    { label: 'bf16',  parts: [{ k: 'params', v: 2 }, { k: 'grads', v: 2 }], total: 4 },
  ];
  const W = 760, H = 170;
  const padX = 80, padY = 24, rowH = 50, rowGap = 18;
  // 8 bytes is widest; leave ~150 px on the right for the "= N bytes/param" label
  const scale = (W - padX - 150) / 8;

  return (
    <div className="viz-panel tco-mp">
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" preserveAspectRatio="xMidYMid meet">
        {rows.map((row, ri) => {
          const y = padY + ri * (rowH + rowGap);
          let acc = 0;
          return (
            <g key={`mp-${ri}`}>
              <text x={padX - 12} y={y + rowH / 2 + 5} className="tco-mp-row-label">
                {row.label}
              </text>
              {row.parts.map((p, pi) => {
                const x = padX + acc * scale;
                const w = p.v * scale;
                acc += p.v;
                return (
                  <g key={`p-${ri}-${pi}`} className={`tco-mp-seg tco-mp-seg-${p.k}`}>
                    <rect x={x} y={y} width={w - 2} height={rowH} rx={4} />
                    <text x={x + w / 2} y={y + rowH / 2 + 5} className="tco-mp-seg-label">
                      {p.k} {p.v}B
                    </text>
                  </g>
                );
              })}
              <text x={padX + row.total * scale + 8} y={y + rowH / 2 + 5}
                    className="tco-mp-total">
                = {row.total} bytes/param
              </text>
            </g>
          );
        })}
      </svg>
      <p className="viz-caption">
        For the live tensors. Activations in fp32 vs bf16 follow the same 2× shrink — and activations
        are the dominant bucket at long sequences. The optimizer state (Adam <Katex tex={'m, v'} />,
        plus the fp32 master weights) stays in fp32 for numerical stability.
      </p>
    </div>
  );
}

/* ============================================================
   Widget 4 — ZeRO / FSDP sharding diagram
   One model partitioned across 4 GPUs at three stages.
   ============================================================ */
function ZeROShardingDiagram() {
  const STAGES = [
    {
      key: 'dp',
      name: 'Plain DP',
      desc: 'Every GPU holds full params, grads, optimizer state. Wasteful.',
      shards: { params: false, grads: false, optim: false },
    },
    {
      key: 'z1',
      name: 'ZeRO-1',
      desc: 'Shard the optimizer state across ranks.',
      shards: { params: false, grads: false, optim: true },
    },
    {
      key: 'z2',
      name: 'ZeRO-2',
      desc: 'Also shard the gradients.',
      shards: { params: false, grads: true, optim: true },
    },
    {
      key: 'z3',
      name: 'ZeRO-3 / FSDP',
      desc: 'Shard everything including parameters; gather just-in-time.',
      shards: { params: true, grads: true, optim: true },
    },
  ];
  const [stageIdx, setStageIdx] = useState(3);
  const stage = STAGES[stageIdx];

  const GPUS = 4;
  const W = 760, H = 230;
  // padX needs to fit the longest row label ("Optimizer state") at 13px mono-bold ≈ 120px.
  const padX = 150, padY = 26;
  const colW = (W - padX - 24) / GPUS;
  const rowDefs = [
    { key: 'params', label: 'Params' },
    { key: 'grads',  label: 'Grads'  },
    { key: 'optim',  label: 'Optimizer state' },
  ];
  const rowH = 44;
  const rowGap = 8;

  return (
    <div className="viz-panel tco-zero">
      <div className="tco-tabs">
        {STAGES.map((s, i) => (
          <button
            key={s.key}
            type="button"
            className={`tco-tab${stageIdx === i ? ' active' : ''}`}
            onClick={() => setStageIdx(i)}
          >
            {s.name}
          </button>
        ))}
      </div>

      <svg viewBox={`0 0 ${W} ${H}`} width="100%" preserveAspectRatio="xMidYMid meet">
        {/* GPU column headers */}
        {Array.from({ length: GPUS }).map((_, g) => (
          <text key={`h-${g}`}
                x={padX + g * colW + colW / 2}
                y={padY - 4}
                className="tco-zero-col-head">
            GPU {g}
          </text>
        ))}

        {rowDefs.map((row, ri) => {
          const y = padY + ri * (rowH + rowGap) + 6;
          const sharded = stage.shards[row.key];
          return (
            <g key={`r-${ri}`}>
              <text x={padX - 14} y={y + rowH / 2 + 4}
                    className="tco-zero-row-label">
                {row.label}
              </text>
              {sharded
                ? Array.from({ length: GPUS }).map((_, g) => (
                    <rect key={`s-${ri}-${g}`}
                          x={padX + g * colW + 4}
                          y={y}
                          width={colW - 8}
                          height={rowH}
                          rx={6}
                          className={`tco-zero-cell shard shard-${g}`} />
                  ))
                : Array.from({ length: GPUS }).map((_, g) => (
                    <rect key={`f-${ri}-${g}`}
                          x={padX + g * colW + 4}
                          y={y}
                          width={colW - 8}
                          height={rowH}
                          rx={6}
                          className="tco-zero-cell full" />
                  ))}
            </g>
          );
        })}
      </svg>

      <div className="tco-zero-desc">{stage.desc}</div>
      <p className="viz-caption">
        Each color = a different rank's slice. Sharded rows hold only <Katex tex={'1/N'} /> of the
        bytes on each GPU; full rows replicate the whole thing on every rank. ZeRO-3 (≈ FSDP)
        gathers the slice it needs the moment the layer fires, then drops it.
      </p>
    </div>
  );
}

/* ============================================================
   Code snippet
   ============================================================ */
const CODE = `# Gradient accumulation + checkpointing + mixed precision (PyTorch sketch)
import torch
from torch.utils.checkpoint import checkpoint

scaler = torch.cuda.amp.GradScaler()  # for fp16; bf16 needs no scaler
optimizer.zero_grad()

for step, batch in enumerate(loader):
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        # checkpoint() reruns the forward on backward instead of storing activations
        h = checkpoint(transformer_block, batch.x, use_reentrant=False)
        loss = loss_fn(h, batch.y) / ACCUM_STEPS  # scale so the average is right

    loss.backward()  # gradients accumulate into .grad

    if (step + 1) % ACCUM_STEPS == 0:
        optimizer.step()
        optimizer.zero_grad()
`;

function CodeBlock() {
  return (
    <details className="post-code-details">
      <summary>
        <span className="post-code-summary-icon" aria-hidden="true">›</span>
        <span>Show PyTorch sketch</span>
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

/* ============================================================
   Page
   ============================================================ */
export default function TrainingCostOptimization() {
  useEffect(() => {
    const prev = document.title;
    document.title = 'Optimizing Training Cost — Bernhard Walser';
    return () => { document.title = prev; };
  }, []);

  return (
    <article className="post-2026 tco-post">
      <div className="post-wrap">
        <div className="post-hero">
          <div className="post-series-tag">
            <span className="post-live-dot" aria-hidden="true" />
            Transformer Costs · Part 2 of 4
          </div>
          <h1>Optimizing Training Cost</h1>
          <p className="post-lede">
            Part 1 ended with an uncomfortable fact: at long sequences, the activations a
            transformer keeps for the backward pass dominate memory. This post is about
            the small set of tricks the field uses to fight back — each one targeting a
            different bucket of the memory budget.
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

        <h2 className="reveal">The four buckets</h2>
        <p>
          Training memory has four big tenants: <strong>parameters</strong>,{' '}
          <strong>gradients</strong>, <strong>optimizer state</strong>, and{' '}
          <strong>activations</strong>. Adam, the standard optimizer, keeps two
          fp32 moments per parameter, so the optimizer state alone is twice the size of
          the model. Activations scale with batch size, sequence length, and depth — and
          they're what blows up first.
        </p>
        <p>
          Each trick below targets exactly one of those buckets. None of them is a
          silver bullet; in a real training run you stack all four.
        </p>

        <h2 className="reveal">Gradient accumulation — large batch, small memory</h2>
        <p>
          A small batch gives a noisy gradient estimate. Doubling the batch halves the
          noise but doubles the activations you have to hold. Gradient accumulation buys
          the large-batch signal without paying the large-batch memory cost: run several
          forward+backward passes on smaller micro-batches, sum the gradients into a
          single buffer, then take one optimizer step.
        </p>
        <GradientAccumulation />
        <p>
          The optimizer sees the average over <em>N</em> micro-batches as if they were
          one big batch. Peak activation memory is the size of one micro-batch — the rest
          of the gradients have already been freed by the time the next pass starts.
          It's the cheapest of the four tricks and almost free to implement.
        </p>

        <h2 className="reveal">Gradient checkpointing — recompute instead of store</h2>
        <p>
          Activations are stored on the forward pass because the backward pass needs them
          to compute gradients. Gradient checkpointing throws most of them away and
          recomputes them on demand. The standard strategy — checkpoint every{' '}
          <Katex tex={'\\sqrt{L}'} />-th layer — drops memory from <Katex tex={'O(L)'} /> to{' '}
          <Katex tex={'O(\\sqrt{L})'} /> in exchange for roughly one extra forward pass.{' '}
          <a className="post-link" href="https://arxiv.org/abs/1604.06174" target="_blank" rel="noreferrer">Chen et al. 2016</a>{' '}
          worked out the bookkeeping.
        </p>
        <GradientCheckpointing />
        <p>
          The cost is real but contained: one extra forward against three full passes
          (forward, backward, optimizer) — call it on the order of 30% more compute for a
          substantial memory cut. In practice you wrap each transformer block in{' '}
          <code>torch.utils.checkpoint.checkpoint(...)</code> and call it a day.
        </p>

        <h2 className="reveal">Mixed precision — half the bytes, same model</h2>
        <p>
          Modern GPUs run bf16 (or fp16) tensor-core math at roughly twice the throughput
          of fp32 and use half the bytes. Store params, grads, and activations in 16-bit;
          keep a fp32 <em>master copy</em> of the weights and the Adam moments for
          stability. <a className="post-link" href="https://arxiv.org/abs/1710.03740" target="_blank" rel="noreferrer">Micikevicius et al. 2018</a>{' '}
          introduced the recipe; bf16 (with its wider exponent) has since become the
          default because it doesn't need loss scaling.
        </p>
        <MixedPrecisionCompare />
        <p>
          The visible win is the live tensors: every activation and every gradient halves.
          The optimizer state stays in fp32, so the savings aren't quite 2× end-to-end —
          but for activation-dominated workloads it's close.
        </p>

        <h2 className="reveal">ZeRO / FSDP — slice the model across GPUs</h2>
        <p>
          The previous three tricks shrink memory on one GPU. ZeRO goes further: when even
          a single layer's footprint doesn't fit, partition the state across data-parallel
          ranks. <a className="post-link" href="https://arxiv.org/abs/1910.02054" target="_blank" rel="noreferrer">Rajbhandari et al. 2020</a>{' '}
          define three stages — shard the optimizer state (ZeRO-1), then the gradients
          (ZeRO-2), then the parameters themselves (ZeRO-3). PyTorch's FSDP is
          essentially ZeRO-3 with a different API.
        </p>
        <ZeROShardingDiagram />
        <p>
          The trade is more network traffic: when a layer fires, every rank gathers the
          slice it doesn't own (an <em>all-gather</em>), runs the layer, then drops the
          gathered weights. The communication overlaps with compute on a fast
          interconnect, so the slowdown is often modest given how much memory you free.
        </p>

        <h2 className="reveal">A separate front: Flash Attention</h2>
        <p>
          Everything above attacks the four memory buckets directly. There's a parallel
          line of work that attacks the <em>attention operation itself</em>:{' '}
          <a className="post-link" href="https://arxiv.org/abs/2205.14135" target="_blank" rel="noreferrer">Dao et al. 2022</a>{' '}
          (Flash Attention) reorganizes the softmax so the L×L attention matrix is
          never materialized in HBM — it's tiled in on-chip SRAM. That's a different
          flavour of optimization (IO-aware kernel design rather than memory budgeting)
          and a separate topic from the four tricks above. In a modern stack you use both.
        </p>

        <h2 className="reveal">Putting it together</h2>
        <p>
          A real training run uses all four at once. Llama 2{' '}
          (<a className="post-link" href="https://arxiv.org/abs/2307.09288" target="_blank" rel="noreferrer">Touvron et al. 2023</a>){' '}
          and most open recipes since combine bf16 + FSDP/ZeRO-3 + gradient checkpointing
          + gradient accumulation + Flash Attention. Each one fights a different
          bottleneck: precision shrinks every live tensor, sharding spreads what's left
          across ranks, checkpointing trims activations on the rank that holds them, and
          accumulation lets the effective batch size keep growing.
        </p>
        <CodeBlock />

        <h2 className="reveal">References</h2>
        <div className="post-refs">
          <div className="head ref-cite">Cite</div>
          <div className="head ref-title">Title</div>
          <div className="head ref-link">Link</div>

          <div className="ref-cite">Chen et al. 2016</div>
          <div>
            <div className="ref-title">
              <a href="https://arxiv.org/abs/1604.06174" target="_blank" rel="noreferrer">
                Training Deep Nets with Sublinear Memory Cost
              </a>
            </div>
            <div className="ref-note">The original √L gradient-checkpointing analysis.</div>
          </div>
          <div className="ref-link">
            <a href="https://arxiv.org/abs/1604.06174" target="_blank" rel="noreferrer">arxiv.org</a>
          </div>

          <div className="ref-cite">Micikevicius et al. 2018</div>
          <div>
            <div className="ref-title">
              <a href="https://arxiv.org/abs/1710.03740" target="_blank" rel="noreferrer">
                Mixed Precision Training
              </a>
            </div>
            <div className="ref-note">fp16 training with a fp32 master copy and loss scaling.</div>
          </div>
          <div className="ref-link">
            <a href="https://arxiv.org/abs/1710.03740" target="_blank" rel="noreferrer">arxiv.org</a>
          </div>

          <div className="ref-cite">Rajbhandari et al. 2020</div>
          <div>
            <div className="ref-title">
              <a href="https://arxiv.org/abs/1910.02054" target="_blank" rel="noreferrer">
                ZeRO: Memory Optimizations Toward Training Trillion Parameter Models
              </a>
            </div>
            <div className="ref-note">Stages 1–3: shard optimizer state, gradients, parameters.</div>
          </div>
          <div className="ref-link">
            <a href="https://arxiv.org/abs/1910.02054" target="_blank" rel="noreferrer">arxiv.org</a>
          </div>

          <div className="ref-cite">Dao et al. 2022</div>
          <div>
            <div className="ref-title">
              <a href="https://arxiv.org/abs/2205.14135" target="_blank" rel="noreferrer">
                FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness
              </a>
            </div>
            <div className="ref-note">Tiled, IO-aware attention kernel; no L×L matrix in HBM.</div>
          </div>
          <div className="ref-link">
            <a href="https://arxiv.org/abs/2205.14135" target="_blank" rel="noreferrer">arxiv.org</a>
          </div>

          <div className="ref-cite">Touvron et al. 2023</div>
          <div>
            <div className="ref-title">
              <a href="https://arxiv.org/abs/2307.09288" target="_blank" rel="noreferrer">
                Llama 2: Open Foundation and Fine-Tuned Chat Models
              </a>
            </div>
            <div className="ref-note">A practical stack that combines all of the above.</div>
          </div>
          <div className="ref-link">
            <a href="https://arxiv.org/abs/2307.09288" target="_blank" rel="noreferrer">arxiv.org</a>
          </div>
        </div>

        <footer className="post-footer">
          <p>
            <strong>Part 2 of Transformer Costs</strong> · Previous:{' '}
            <a className="post-link" href="#/blog/training-cost">Training Cost of a Transformer</a>
            {' · '}Next:{' '}
            <a className="post-link" href="#/blog/inference-cost">Inference Cost of a Transformer</a>.
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
