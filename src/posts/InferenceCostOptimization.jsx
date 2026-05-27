// src/posts/InferenceCostOptimization.jsx
// Transformer Costs · Part 4 of 4 — Optimizing Inference Cost.
// Ported from the legacy VisualizingKVCache post; prose preserved verbatim
// where it carries over, visuals re-skinned with the post-2026 chrome.
//
// TAGS FOR REGISTRATION: ['transformers', 'inference', 'optimization']
// EXCERPT: How modern inference engines claw back throughput from a memory-bound
//          decode: shrinking the KV cache (GQA/MQA), paging it like virtual memory,
//          and batching requests at the iteration level.
import React, { useEffect, useRef, useState } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneLight } from 'react-syntax-highlighter/dist/esm/styles/prism';
import './PostChrome.css';
import './InferenceCostOptimization.css';

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
   Widget 1 — GQA / MQA head schematic
   ========================================================= */
const N_Q_HEADS = 8;
const KV_HEAD_OPTIONS = [1, 2, 4, 8]; // with n_q_heads = 8
const REAL_MODELS_GQA = [
  { name: 'Llama 2 7B (MHA)' },
  { name: 'Llama 2 70B (GQA-8)' },
  { name: 'Mistral 7B (GQA-8)' },
  { name: 'MQA (Shazeer 2019)' },
  { name: 'DeepSeek-V2 (MLA)' },
];

function SectionGQA() {
  const [n_kv, setNkv] = useState(8);
  const groupOf = (qHead) => Math.floor(qHead * n_kv / N_Q_HEADS);

  const refLayers = 32, refDHead = 128, refSeqLen = 8192, refBytes = 2;
  const cacheBytes = (kv) => refLayers * 2 * (kv * refDHead) * refSeqLen * refBytes;
  const fmtGB = (x) => `${(x / 1e9).toFixed(2)} GB`;

  return (
    <div className="viz-panel ico-gqa">
      <div className="viz-controls">
        <label>KV heads (Q heads fixed at {N_Q_HEADS}):</label>
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
          <div className="viz-cache-table-row" style={{ marginTop: 8, color: 'var(--post-text-fnt)', fontStyle: 'italic' }}>
            ref: 32 layers · d_head 128 · T=8192 · fp16
          </div>
        </div>
      </div>

      <div className="ico-in-wild">
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
   Widget 2 — PagedAttention block allocation
   ========================================================= */
const PAGED_BLOCK_SIZE = 4;
const PAGED_TOTAL_SLOTS = 64;
const PAGED_MAX_SEQ_PER_REQ = 24;
const PAGED_SCRIPT = [
  { a: 5, b: 0,  caption: 'Request A arrives with a 5-token prompt.' },
  { a: 1, b: 7,  caption: 'A decodes one token; B arrives with a 7-token prompt.' },
  { a: 1, b: 1,  caption: 'Both decode one more token.' },
  { a: 3, b: 1,  caption: 'A keeps generating; B decodes one.' },
  { a: 0, b: -1, caption: 'B finishes and is released.' },
  { a: 2, b: 0,  caption: 'A keeps going. (Notice what happens to B\'s slots.)' },
];

function SectionPaged() {
  const [step, setStep] = useState(0);

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
        contig[idx] = 'reserved-b';
      }
    }
  }

  const paged = Array(PAGED_TOTAL_SLOTS).fill('free');
  const blocksA = Math.ceil(tokens.a / PAGED_BLOCK_SIZE);
  const blocksB = tokens.bAlive ? Math.ceil(tokens.b / PAGED_BLOCK_SIZE) : 0;
  const blockAssignments = [];
  let bi = 0;
  for (let k = 0; k < blocksA; k++) blockAssignments.push({ block: bi++, owner: 'a', n: k + 1 });
  for (let k = 0; k < blocksB; k++) blockAssignments.push({ block: bi++, owner: 'b', n: k + 1 });
  for (const { block, owner } of blockAssignments) {
    const start = block * PAGED_BLOCK_SIZE;
    for (let j = 0; j < PAGED_BLOCK_SIZE; j++) paged[start + j] = owner;
  }

  const wastedContig = contig.filter((s) => s === 'reserved-a' || s === 'reserved-b').length;
  const wastedPaged = paged.filter((s) => s === 'free').length;

  const blockTableA = Array.from({ length: blocksA }, (_, k) => k);
  const blockTableB = blocksB > 0 ? Array.from({ length: blocksB }, (_, k) => blocksA + k) : [];

  return (
    <div className="viz-panel ico-paged">
      <div className="viz-controls">
        <button
          className="active"
          onClick={() => setStep((s) => Math.min(PAGED_SCRIPT.length - 1, s + 1))}
          disabled={step >= PAGED_SCRIPT.length - 1}
        >
          Next step
        </button>
        <button onClick={() => setStep(0)}>Reset</button>
        <span className="ico-paged-step">
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
            <div className="viz-block-table-row ico-neg">
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
            <div className="viz-block-table-row ico-pos">
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
    </div>
  );
}

/* =========================================================
   Widget 3 — Continuous batching Gantt
   ========================================================= */
const GANTT_T = 14;
const REQ_A = { promptLen: 4, outLen: 8 };
const REQ_B = { promptLen: 2, outLen: 4 };

function staticBatchingTimeline(T) {
  const rowA = new Array(T).fill('idle');
  const rowB = new Array(T).fill('idle');
  rowA[0] = 'a-prompt';
  rowB[0] = 'b-prompt';
  const totalDecode = Math.max(REQ_A.outLen, REQ_B.outLen);
  for (let t = 1; t <= totalDecode && t < T; t++) {
    rowA[t] = t <= REQ_A.outLen ? 'a-decode' : 'pad';
    rowB[t] = t <= REQ_B.outLen ? 'b-decode' : 'pad';
  }
  return { rowA, rowB };
}

function continuousBatchingTimeline(T) {
  const rowA = new Array(T).fill('idle');
  const rowB = new Array(T).fill('idle');
  rowA[0] = 'a-prompt';
  rowB[0] = 'b-prompt';
  for (let t = 1; t <= REQ_A.outLen && t < T; t++) rowA[t] = 'a-decode';
  for (let t = 1; t <= REQ_B.outLen && t < T; t++) rowB[t] = 'b-decode';
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

  const isUseful = (k) => k !== 'idle' && k !== 'pad';
  const isWasted = (k) => k === 'pad';
  const isFree   = (k) => k === 'idle';
  const sum = (rows, pred) => rows.reduce((n, row) => n + row.filter(pred).length, 0);

  const total = 2 * GANTT_T;
  const usefulStat = sum([stat.rowA, stat.rowB], isUseful);
  const wastedStat = sum([stat.rowA, stat.rowB], isWasted);
  const freeStat   = sum([stat.rowA, stat.rowB], isFree);
  const usefulCont = sum([cont.rowA, cont.rowB], isUseful);
  const wastedCont = sum([cont.rowA, cont.rowB], isWasted);
  const freeCont   = sum([cont.rowA, cont.rowB], isFree);

  return (
    <div className="viz-panel ico-gantt">
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
        Useful: <strong>{usefulStat}</strong> · Wasted on padding: <strong className="ico-neg">{wastedStat}</strong> ·
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
        Useful: <strong>{usefulCont}</strong> · Wasted on padding: <strong className="ico-pos">{wastedCont}</strong> ·
        Free for new requests: <strong>{freeCont}</strong> &nbsp;/&nbsp; {total} cells.
        Once B finishes at step {REQ_B.outLen + 1}, its slot is genuinely free — a new request can land in it.
      </div>

      <div className="viz-gantt-legend" style={{ marginTop: 16 }}>
        <span><span className="viz-gantt-swatch" style={{ background: 'hsl(218, 55%, 75%)' }} /> A prompt</span>
        <span><span className="viz-gantt-swatch" style={{ background: 'hsl(218, 55%, 88%)' }} /> A decode</span>
        <span><span className="viz-gantt-swatch" style={{ background: 'hsl(28, 65%, 72%)' }} /> B prompt</span>
        <span><span className="viz-gantt-swatch" style={{ background: 'hsl(28, 65%, 88%)' }} /> B decode</span>
        <span><span className="viz-gantt-swatch ico-swatch-pad" /> padding (wasted)</span>
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
   Optional code expander — PagedAttention block-table sketch
   ========================================================= */
const PAGED_CODE = `# Per-request logical -> physical block map (vLLM, simplified)
block_table = {req_id: []}              # list of physical block ids per request

def on_new_token(req):
    if len(req.tokens) % BLOCK == 1:    # need a new block
        block_table[req].append(allocator.alloc())
    # write the new K, V into the last block at the right offset

def on_finish(req):
    for b in block_table.pop(req):
        allocator.free(b)               # blocks return to the pool
`;

function PagedCodeBlock() {
  return (
    <details className="post-code-details ico-code-details">
      <summary>
        <span className="post-code-summary-icon" aria-hidden="true">›</span>
        <span>Show the page-table indirection in Python</span>
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
        {PAGED_CODE}
      </SyntaxHighlighter>
    </details>
  );
}

/* =========================================================
   Page
   ========================================================= */
export default function InferenceCostOptimization() {
  useEffect(() => {
    const prev = document.title;
    document.title = 'Optimizing Inference Cost — Bernhard Walser';
    return () => { document.title = prev; };
  }, []);

  return (
    <article className="post-2026 ico-post">
      <div className="post-wrap">
        <div className="post-hero">
          <div className="post-series-tag">
            <span className="post-live-dot" aria-hidden="true" />Transformer Costs · Part 4 of 4
          </div>
          <h1>Optimizing Inference Cost</h1>
          <p className="post-lede">
            Decode is memory-bound: the GPU spends most of its time reading the KV
            cache out of HBM, not doing math. Every optimization that follows
            fights that one bottleneck — by shrinking the cache, by allocating it
            more carefully, or by batching more work behind each byte.
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

        <h2 className="reveal">Why decode is bandwidth-bound, not compute-bound</h2>
        <p>
          <a className="post-link" href="#/blog/inference-cost">Part 3</a> ended on a
          number: a decode step on a 7B model with an 8k context sits at an{' '}
          <em>arithmetic intensity</em> — useful FLOPs per byte read from HBM —
          of roughly 1. The H100's "ridge point" between memory-bound and
          compute-bound is about 295. Decode isn't close to compute-bound; it
          isn't even in the same room.
        </p>
        <p>
          Most of the wall-clock time of a decode step is the GPU waiting on{' '}
          <strong>HBM</strong> reads — model weights (gigabytes) plus the KV
          cache (more gigabytes at long context). The math is so much faster
          than the memory that it's nearly free in absolute terms.
        </p>
        <p>
          This has two consequences. <strong>First</strong>, the levers that
          actually buy throughput are the ones that reduce bytes moved per
          token — smaller cache, better allocation, interleaving requests so
          the same byte feeds more arithmetic. <strong>Second</strong>, latency
          at batch 1 is basically a function of model size: roughly{' '}
          <em>weight bytes / HBM bandwidth</em>. A 13 GB model on an H100 won't
          generate tokens faster than ~3.9 ms each, no matter how clever your
          kernels are.
        </p>
        <p>
          The rest of this post is three techniques modern inference engines
          combine to fight that bottleneck. Each one tackles a different kind
          of waste.
        </p>

        <h2 className="reveal">Shrinking the cache: MHA → GQA → MQA</h2>
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
          The fix is architectural: shrink <Katex tex="n_{kv}" />.{' '}
          <strong>Multi-Query Attention</strong> (Shazeer 2019) keeps a single
          K and V head, shared across all <Katex tex="n_q" /> query heads —
          the extreme. <strong>Grouped-Query Attention</strong> (Ainslie et al.
          2023) is the middle ground: <Katex tex="n_{kv} < n_q" />, with Q heads
          partitioned into groups that share. Same attention math,{' '}
          <Katex tex="n_q / n_{kv}" />× less cache. Quality stays close to full
          MHA; cache size collapses by 4–8×.
        </p>
        <p>
          In practice: Llama-2-7B uses 32 query heads and 32 KV heads — that's
          plain MHA, no compression. Llama-2-70B drops to 8 KV heads against
          64 query heads — an 8× smaller cache, with the
          {' '}<a className="post-link" href="https://arxiv.org/abs/2307.09288" target="_blank" rel="noreferrer">Llama-2 paper</a>{' '}
          showing roughly comparable downstream quality. Most frontier
          open-weight models now ship with some flavour of GQA.
        </p>
        <SectionGQA />

        <h2 className="reveal">PagedAttention: stop fragmenting the cache</h2>
        <p>
          GQA shrinks the cache. PagedAttention squeezes the inefficiency out
          of how it's allocated. Naive serving systems reserve a contiguous
          chunk of GPU memory for each request, sized for the worst case
          (<code>max_seq_len</code>). Most requests never hit that ceiling — so
          most of that memory sits empty. Worse, even after a request finishes
          its slot stays bound to that request until the connection tears down.
        </p>
        <p>
          The{' '}
          <a className="post-link" href="https://arxiv.org/abs/2309.06180" target="_blank" rel="noreferrer">
            vLLM paper
          </a>{' '}
          (Kwon et al. 2023) borrows the trick from operating-system virtual
          memory: chop the cache into fixed-size <strong>blocks</strong> (16
          tokens is typical), maintain a per-request <strong>block table</strong>{' '}
          mapping logical positions → physical blocks, allocate blocks on
          demand, free them on request completion. It's <code>malloc</code> →{' '}
          <code>mmap</code>.
        </p>
        <p>
          The paper reports KV-cache memory utilization jumping from the 20–40%
          range under contiguous allocation to ~96% under paging, and serving
          throughput improving by roughly 2–4× over prior systems on standard
          workloads. The exact number depends on the workload mix; the
          direction is unambiguous.
        </p>
        <SectionPaged />
        <PagedCodeBlock />

        <h2 className="reveal">Continuous batching</h2>
        <p>
          One last source of waste. Even with a perfectly-allocated cache,
          you still need to <em>batch</em> requests through the model.
          Traditional ("static") batching freezes a batch at the start —
          short requests get padded to the longest, slots stay bound until
          the whole batch finishes.
        </p>
        <p>
          <strong>Continuous batching</strong>, introduced by{' '}
          <a className="post-link" href="https://www.usenix.org/conference/osdi22/presentation/yu" target="_blank" rel="noreferrer">Orca</a>{' '}
          (Yu et al. 2022) and popularised by vLLM, reschedules the batch{' '}
          <em>every decode step</em>: finished requests release their slot
          immediately, new requests join on the next iteration. No padding,
          no waiting on stragglers. On mixed-length workloads this roughly
          doubles throughput on its own.
        </p>
        <SectionContinuousBatching />

        <h2 className="reveal">Putting it together</h2>
        <p>
          Modern inference engines — <a className="post-link" href="https://github.com/vllm-project/vllm" target="_blank" rel="noreferrer">vLLM</a>,{' '}
          <a className="post-link" href="https://github.com/NVIDIA/TensorRT-LLM" target="_blank" rel="noreferrer">TensorRT-LLM</a>,{' '}
          <a className="post-link" href="https://github.com/sgl-project/sglang" target="_blank" rel="noreferrer">SGLang</a> — combine
          all three: GQA-shaped models, PagedAttention for cache memory,
          continuous batching for the scheduler. They also layer in tricks
          beyond this post (speculative decoding, quantization, fused
          attention kernels). Compounded, the gap between a naive serving
          loop and a production engine is roughly an order of magnitude in
          throughput at the same latency.
        </p>
        <p>
          The story of inference optimization is the story of one bottleneck.
          Decode is memory-bound; every technique here moves fewer bytes per
          generated token, or extracts more useful tokens per byte moved. Once
          you see the roofline, the design choices stop looking like a grab
          bag and start looking like a single idea applied in three places.
        </p>

        <h2 className="reveal">References</h2>
        <div className="post-refs">
          <div className="head ref-cite">Cite</div>
          <div className="head ref-title">Title</div>
          <div className="head ref-link">Link</div>

          <div className="ref-cite">Shazeer 2019</div>
          <div>
            <div className="ref-title">
              <a href="https://arxiv.org/abs/1911.02150" target="_blank" rel="noreferrer">Fast Transformer Decoding: One Write-Head is All You Need</a>
            </div>
            <div className="ref-note">The original Multi-Query Attention paper.</div>
          </div>
          <div className="ref-link"><a href="https://arxiv.org/abs/1911.02150" target="_blank" rel="noreferrer">arxiv.org</a></div>

          <div className="ref-cite">Ainslie et al. 2023</div>
          <div>
            <div className="ref-title">
              <a href="https://arxiv.org/abs/2305.13245" target="_blank" rel="noreferrer">GQA: Training Generalized Multi-Query Transformer Models</a>
            </div>
            <div className="ref-note">The middle ground between MHA and MQA.</div>
          </div>
          <div className="ref-link"><a href="https://arxiv.org/abs/2305.13245" target="_blank" rel="noreferrer">arxiv.org</a></div>

          <div className="ref-cite">Touvron et al. 2023</div>
          <div>
            <div className="ref-title">
              <a href="https://arxiv.org/abs/2307.09288" target="_blank" rel="noreferrer">Llama 2: Open Foundation and Fine-Tuned Chat Models</a>
            </div>
            <div className="ref-note">Llama-2-70B uses GQA-8 in practice.</div>
          </div>
          <div className="ref-link"><a href="https://arxiv.org/abs/2307.09288" target="_blank" rel="noreferrer">arxiv.org</a></div>

          <div className="ref-cite">Kwon et al. 2023</div>
          <div>
            <div className="ref-title">
              <a href="https://arxiv.org/abs/2309.06180" target="_blank" rel="noreferrer">Efficient Memory Management for LLM Serving with PagedAttention</a>
            </div>
            <div className="ref-note">The vLLM paper. Source for the cache-utilization and throughput numbers.</div>
          </div>
          <div className="ref-link"><a href="https://arxiv.org/abs/2309.06180" target="_blank" rel="noreferrer">arxiv.org</a></div>

          <div className="ref-cite">Yu et al. 2022</div>
          <div>
            <div className="ref-title">
              <a href="https://www.usenix.org/conference/osdi22/presentation/yu" target="_blank" rel="noreferrer">Orca: A Distributed Serving System for Transformer-Based Generative Models</a>
            </div>
            <div className="ref-note">Iteration-level (continuous) batching, the predecessor to vLLM's scheduler.</div>
          </div>
          <div className="ref-link"><a href="https://www.usenix.org/conference/osdi22/presentation/yu" target="_blank" rel="noreferrer">usenix.org</a></div>

          <div className="ref-cite">Earlier posts</div>
          <div>
            <div className="ref-title">
              <a href="#/blog/inference-cost" target="_blank" rel="noreferrer">Inference Cost of a Transformer</a>
            </div>
            <div className="ref-note">Part 3 — where the bandwidth-bound diagnosis came from.</div>
          </div>
          <div className="ref-link"><a href="#/blog/inference-cost" target="_blank" rel="noreferrer">this site</a></div>
        </div>

        <footer className="post-footer">
          <p>
            <strong>Part 4 of Transformer Costs</strong> · Previous:{' '}
            <a className="post-link" href="#/blog/inference-cost">Inference Cost of a Transformer</a>
            {' · '}Start of the series:{' '}
            <a className="post-link" href="#/blog/training-cost">Training Cost of a Transformer</a>.
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
