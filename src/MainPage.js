// src/MainPage.js
import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { blogPosts } from './blogPosts';
import './MainPage.css';

/* ============================================================
   Hero — animated attention constellation (port of hero-bg.js)
   ============================================================ */
function useHeroConstellation(hostRef) {
  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;
    if (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;

    const NS = 'http://www.w3.org/2000/svg';
    const svg = document.createElementNS(NS, 'svg');
    svg.setAttribute('preserveAspectRatio', 'xMidYMid slice');
    svg.style.cssText = 'position:absolute;inset:0;width:100%;height:100%;pointer-events:none;';
    host.appendChild(svg);

    let W = 1400, H = 700;
    function resize() {
      const r = host.getBoundingClientRect();
      W = Math.max(800, r.width);
      H = Math.max(400, r.height);
      svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    }
    resize();
    window.addEventListener('resize', resize);

    function seededRandom(seed) {
      let s = seed;
      return () => { s = (s * 9301 + 49297) % 233280; return s / 233280; };
    }
    const rng = seededRandom(7);
    const NODE_COUNT = 28;
    const nodes = [];
    for (let i = 0; i < NODE_COUNT * 5 && nodes.length < NODE_COUNT; i++) {
      const x = 0.04 + rng() * 0.92;
      const y = 0.06 + rng() * 0.88;
      let ok = true;
      for (const n of nodes) {
        const dx = (x - n.x), dy = (y - n.y);
        if (dx * dx + dy * dy < 0.014) { ok = false; break; }
      }
      if (ok) nodes.push({ x, y, baseR: 2 + rng() * 1.6, phase: rng() * Math.PI * 2 });
    }

    const dotsLayer = document.createElementNS(NS, 'g');
    svg.appendChild(dotsLayer);
    const dotEls = nodes.map((n) => {
      const c = document.createElementNS(NS, 'circle');
      c.setAttribute('r', n.baseR);
      c.setAttribute('fill', '#cbd5e1');
      c.setAttribute('opacity', '0.55');
      dotsLayer.appendChild(c);
      return c;
    });

    const edgesLayer = document.createElementNS(NS, 'g');
    svg.appendChild(edgesLayer);
    const ringsLayer = document.createElementNS(NS, 'g');
    svg.appendChild(ringsLayer);

    function place() {
      nodes.forEach((n, i) => {
        dotEls[i].setAttribute('cx', n.x * W);
        dotEls[i].setAttribute('cy', n.y * H);
      });
    }
    place();
    window.addEventListener('resize', place);

    let bursts = [];
    function fireBurst() {
      const qi = Math.floor(rng() * nodes.length);
      const q = nodes[qi];
      const others = nodes
        .map((n, i) => ({ i, d: (n.x - q.x) ** 2 + (n.y - q.y) ** 2 }))
        .filter((o) => o.i !== qi)
        .sort((a, b) => a.d - b.d)
        .slice(0, 10);
      for (let i = others.length - 1; i > 0; i--) {
        const j = Math.floor(rng() * (i + 1));
        [others[i], others[j]] = [others[j], others[i]];
      }
      const targets = others.slice(0, 3 + Math.floor(rng() * 3));

      const ring = document.createElementNS(NS, 'circle');
      ring.setAttribute('cx', q.x * W);
      ring.setAttribute('cy', q.y * H);
      ring.setAttribute('r', 3);
      ring.setAttribute('fill', 'none');
      ring.setAttribute('stroke', 'var(--accent)');
      ring.setAttribute('stroke-width', '1.4');
      ring.setAttribute('opacity', '0.55');
      ringsLayer.appendChild(ring);

      const lines = targets.map((t, idx) => {
        const n = nodes[t.i];
        const ln = document.createElementNS(NS, 'line');
        ln.setAttribute('x1', q.x * W);
        ln.setAttribute('y1', q.y * H);
        ln.setAttribute('x2', q.x * W);
        ln.setAttribute('y2', q.y * H);
        ln.setAttribute('stroke', 'var(--accent)');
        ln.setAttribute('stroke-width', '1');
        ln.setAttribute('opacity', '0');
        ln.setAttribute('stroke-linecap', 'round');
        edgesLayer.appendChild(ln);
        return { el: ln, q, n, delay: idx * 90, weight: 0.18 + rng() * 0.32 };
      });

      bursts.push({ ring, lines, t0: performance.now(), dur: 2600 });
    }

    const t1 = setTimeout(fireBurst, 400);
    const t2 = setTimeout(fireBurst, 1100);
    const iv = setInterval(fireBurst, 1600);

    let raf;
    function frame(now) {
      nodes.forEach((n, i) => {
        const t = now / 1200 + n.phase;
        const op = 0.4 + 0.25 * Math.sin(t);
        dotEls[i].setAttribute('opacity', op.toFixed(3));
      });
      bursts = bursts.filter((b) => {
        const elapsed = now - b.t0;
        if (elapsed > b.dur) {
          b.lines.forEach((l) => l.el.remove());
          b.ring.remove();
          return false;
        }
        const rp = Math.min(1, elapsed / 900);
        const r = 3 + rp * 18;
        b.ring.setAttribute('r', r.toFixed(2));
        b.ring.setAttribute('opacity', (0.55 * (1 - rp)).toFixed(3));
        b.lines.forEach((l) => {
          const lt = elapsed - l.delay;
          if (lt < 0) return;
          const drawDur = 380, holdDur = 900, fadeDur = 1100;
          if (lt < drawDur) {
            const p = lt / drawDur;
            const e = 1 - Math.pow(1 - p, 3);
            l.el.setAttribute('x2', (l.q.x * W + (l.n.x - l.q.x) * W * e).toFixed(2));
            l.el.setAttribute('y2', (l.q.y * H + (l.n.y - l.q.y) * H * e).toFixed(2));
            l.el.setAttribute('opacity', (l.weight * p).toFixed(3));
          } else if (lt < drawDur + holdDur) {
            l.el.setAttribute('opacity', l.weight.toFixed(3));
          } else if (lt < drawDur + holdDur + fadeDur) {
            const p = (lt - drawDur - holdDur) / fadeDur;
            l.el.setAttribute('opacity', (l.weight * (1 - p)).toFixed(3));
          } else {
            l.el.setAttribute('opacity', '0');
          }
        });
        return true;
      });
      raf = requestAnimationFrame(frame);
    }
    raf = requestAnimationFrame(frame);

    return () => {
      window.removeEventListener('resize', resize);
      window.removeEventListener('resize', place);
      clearTimeout(t1); clearTimeout(t2); clearInterval(iv);
      cancelAnimationFrame(raf);
      svg.remove();
    };
  }, [hostRef]);
}

/* ============================================================
   Reveal-on-scroll observer
   ============================================================ */
function useRevealObserver() {
  useEffect(() => {
    if (!('IntersectionObserver' in window)) {
      document.querySelectorAll('.reveal, .reveal-stagger').forEach(el => el.classList.add('in'));
      return;
    }
    const io = new IntersectionObserver((entries) => {
      entries.forEach(e => {
        if (e.isIntersecting) {
          e.target.classList.add('in');
          io.unobserve(e.target);
        }
      });
    }, { rootMargin: '0px 0px -10% 0px', threshold: 0.05 });
    document.querySelectorAll('.reveal, .reveal-stagger').forEach(el => io.observe(el));
    return () => io.disconnect();
  }, []);
}

/* ============================================================
   Static data — writing list (dates approximate, ordered newest first)
   ============================================================ */
const writingEntries = [
  {
    slug: 'symbolic-regression-llm-transfer',
    date: '2026 · May',
    title: 'Does the Ranking Transfer to an LLM?',
    desc: 'Swap the proposer for an LLM and ask whether the same Layer-0 ranking survives. What the model sees, how the search budget is shared, and what the transfer experiment found.',
    tags: ['symbolic-regression', 'llm', 'self-improvement'],
  },
  {
    slug: 'symbolic-regression-arena',
    date: '2026 · May',
    title: 'Racing Search Algorithms on Symbolic Regression',
    desc: 'Symbolic regression as a self-improvement testbed: evolution, greedy RL, and risk-seeking RL race to recover a hidden equation under one shared compute budget.',
    tags: ['symbolic-regression', 'rl', 'self-improvement'],
  },
  {
    slug: 'measuring-self-improvement',
    date: '2026 · May',
    title: 'Measuring Self-Improvement',
    desc: "MLE-Bench, RE-Bench, PaperBench, PostTrainBench, METR time horizons — plus Anthropic's Automated Alignment Researcher. How we tell whether the self-improvement claims actually mean anything.",
    tags: ['self-improvement', 'benchmarks', 'alignment'],
  },
  {
    slug: 'test-time-training',
    date: '2026 · May',
    title: 'Test-Time Training: When the Model Updates Mid-Inference',
    desc: "TTT-Discover updates weights mid-problem and resets between problems. An entropic peak-vs-average objective, PUCT-guided mutation selection, and a real answer to 'why doesn't this collapse?'",
    tags: ['self-improvement', 'ttt', 'test-time-training'],
  },
  {
    slug: 'evolutionary-search',
    date: '2026 · May',
    title: 'Evolutionary Search: FunSearch, AlphaEvolve, ThetaEvolve',
    desc: 'When the LM is a mutation operator, not the learner. FunSearch and AlphaEvolve freeze the LM and evolve a program population; ThetaEvolve breaks that rule.',
    tags: ['self-improvement', 'evolutionary', 'funsearch'],
  },
  {
    slug: 'internal-verifiers',
    date: '2026 · May',
    title: 'Internal Verifiers: When the Model Judges Its Own Work',
    desc: 'R-Zero, Agent0, and G-Zero generate their own training signal: one model proposes, another solves, reward comes from self-consistency or behavioral shift — no external grader.',
    tags: ['self-improvement', 'r-zero', 'agent0'],
  },
  {
    slug: 'external-verifiers',
    date: '2026 · May',
    title: 'External Verifiers: STaR and ReST/RLVR',
    desc: "When ground truth exists outside the model — STaR's filter-and-fine-tune loop and ReST/RLVR's RL framing of the same trick.",
    tags: ['self-improvement', 'star', 'rlvr'],
  },
  {
    slug: 'alphago-to-alphazero',
    date: '2026 · May',
    title: 'AlphaGo to AlphaZero',
    desc: 'The 2-axis map of self-improvement methods, and how AlphaGo → AlphaGo Zero → AlphaZero set the template: search + a learned policy/value, self-play, no human labels.',
    tags: ['self-improvement', 'alphazero', 'mcts'],
  },
  {
    slug: 'beyond-human-feedback',
    date: '2026 · Apr',
    title: 'Beyond Human Feedback: RLAIF, Process Rewards, RLVR',
    desc: 'Human labels are expensive and slow. RLAIF replaces the labeller with a model; process reward models densify the signal; RLVR removes the learned reward entirely in verifiable domains.',
    tags: ['rlhf', 'rlaif', 'rlvr'],
  },
  {
    slug: 'beyond-ppo',
    date: '2026 · Apr',
    title: 'Beyond PPO: DPO, GRPO, DAPO',
    desc: "PPO works but it needs four networks in memory and is hard to tune. DPO drops the reward model, GRPO drops the value net, and DAPO patches GRPO's failure modes — the post-PPO landscape side by side.",
    tags: ['rlhf', 'dpo', 'grpo'],
  },
  {
    slug: 'rlhf-and-ppo',
    date: '2026 · Apr',
    title: 'RLHF and PPO',
    desc: 'How a base language model becomes an instruction-following assistant: SFT, reward modelling from preferences, and PPO with its clipped surrogate.',
    tags: ['rlhf', 'training', 'transformers'],
  },
  {
    slug: 'rope',
    date: '2026 · Mar',
    title: 'Rotary Positional Embeddings (RoPE)',
    desc: 'RoPE encodes token position by rotating each query and key vector — the dot product naturally depends on relative offset, with zero new parameters and a clean path to long context.',
    tags: ['transformers', 'positions', 'rope'],
  },
  {
    slug: 'positional-encodings',
    date: '2026 · Mar',
    title: 'Positional Encodings: A Tour',
    desc: 'Self-attention is permutation-invariant. Five additive position-encoding schemes — absolute, Shaw, T5, Swin, ALiBi — and where each one injects position.',
    tags: ['transformers', 'positions', 'history'],
  },
  {
    slug: 'inference-cost-optimization',
    date: '2026 · Feb',
    title: 'Optimizing Inference Cost',
    desc: 'Decode is bandwidth-bound, so every optimization shrinks the KV cache or packs the GPU better: GQA / MQA, PagedAttention, and continuous batching.',
    tags: ['transformers', 'inference', 'optimization'],
  },
  {
    slug: 'inference-cost',
    date: '2026 · Feb',
    title: 'Inference Cost of a Transformer',
    desc: 'Why the first token takes 500 ms and the rest stream at 20 ms each — the prefill/decode split, the KV cache, and why decode is memory-bound.',
    tags: ['transformers', 'inference', 'cost'],
  },
  {
    slug: 'training-cost-optimization',
    date: '2026 · Feb',
    title: 'Optimizing Training Cost',
    desc: 'Gradient accumulation, gradient checkpointing, mixed precision, ZeRO/FSDP — four optimizations that each target a different memory bucket in the training stack.',
    tags: ['transformers', 'training', 'optimization'],
  },
  {
    slug: 'training-cost',
    date: '2026 · Feb',
    title: 'Training Cost of a Transformer',
    desc: 'Training Llama-2-7B took ~1.7M GPU-hours. Count what one transformer layer actually does — forward and backward — and what has to be held in memory between them.',
    tags: ['transformers', 'training', 'cost'],
  },
  {
    slug: 'from-attention-to-transformer',
    date: '2026 · Jan',
    title: 'From Attention to Transformer',
    desc: "A single attention operation isn't yet a transformer. Add multiple heads, an MLP per token, a causal mask, and positional information — and you have the block that powers every modern language model.",
    tags: ['transformers', 'architecture'],
  },
  {
    slug: 'self-attention',
    date: '2026 · Jan',
    title: 'Inside Self-Attention: Q, K, V, and Softmax',
    desc: 'How a single self-attention layer works — three projections (Q, K, V), a scaled dot-product, a softmax, and a weighted sum. Every number on the page is live.',
    tags: ['attention', 'transformers'],
  },
  {
    slug: 'nlp-history',
    date: '2026 · Jan',
    title: 'From TF-IDF to Attention',
    desc: 'How NLP got from counting words to attention — tokens, TF-IDF, embeddings, and the RNN/CNN limits that attention was built to escape.',
    tags: ['nlp', 'history', 'transformers'],
  },
  {
    slug: 'monte-carlo',
    date: '2026 · Jan',
    title: 'Monte Carlo Tree Search',
    desc: 'Simulation-based search: random rollouts replace exhaustive evaluation. Selection, expansion, rollout, backpropagation — the four steps that beat humans at Go.',
    tags: ['chess', 'simulation'],
  },
  {
    slug: 'alpha-beta',
    date: '2026 · Jan',
    title: 'Alpha-Beta Pruning',
    desc: 'An enhancement to minimax that skips branches it can prove can\'t change the answer — same chosen move, far fewer nodes evaluated.',
    tags: ['chess', 'search', 'pruning'],
  },
  {
    slug: 'mini-max',
    date: '2026 · Jan',
    title: 'Minimax Search',
    desc: 'Simple decision-making algorithm for two-player games — walk every move sequence, assume the opponent plays optimally, pick the move that minimizes your worst case.',
    tags: ['chess', 'search'],
  },
];

/* ============================================================
   Experience timeline — from LinkedIn
   ============================================================ */
const experience = [
  {
    current: true,
    meta: ['Aug 2024 — Present', 'Zurich'],
    title: 'Senior Machine Learning Engineer',
    org: 'Digitec Galaxus AG',
    url: 'https://www.digitec.ch/',
    desc: 'Improving automated purchasing processes and increasing product data quality.',
  },
  {
    meta: ['Mar 2022 — Jul 2024', 'Zurich'],
    title: 'Machine Learning Engineer',
    org: 'Digitec Galaxus AG',
    url: 'https://www.digitec.ch/',
    desc: 'Developed and deployed ML models for document processing, pricing, and product data quality. Maintained CI/CD/CT pipelines on GCP and Azure — Kubernetes (GKE), Airflow (Cloud Composer), Azure DevOps, Terraform.',
  },
  {
    meta: ['Jun 2021 — Feb 2022', 'Zurich'],
    title: 'Junior Machine Learning Engineer',
    org: 'Digitec Galaxus AG',
    url: 'https://www.digitec.ch/',
    desc: 'First steps in production ML at Switzerland’s largest online retailer.',
  },
  {
    meta: ['Mar 2020 — Aug 2020', 'Zurich'],
    title: 'Computer Vision & Machine Learning Intern',
    org: 'Advertima AG',
    url: 'https://advertima.com/',
    desc: 'Neural network architectures for face recognition, object detection, and multitask learning using PyTorch, TensorFlow, scikit-learn, MLflow, and PyTorch Lightning.',
  },
  {
    meta: ['2019 — 2021', 'Zurich'],
    title: 'M.Sc. in Computer Science',
    org: 'ETH Zürich',
    url: 'https://ethz.ch/',
    desc: 'Thesis: “On the similarities between Vision Transformers and Residual Networks for Weather Prediction.” Major in machine learning and visual computing.',
  },
  {
    meta: ['2016 — 2020', 'Zurich'],
    title: 'B.Sc. in Computer Science',
    org: 'ETH Zürich',
    url: 'https://ethz.ch/',
    desc: 'Foundations in algorithms, systems, statistics, machine learning, and information security.',
  },
];

/* ============================================================
   Ticker — short rotating facts
   ============================================================ */
function TickerGroup() {
  return (
    <div className="ticker-group">
      <span className="label accent">// CURRENTLY</span>
      <span className="bullet">●</span>
      <span className="label">SENIOR ML ENGINEER @ DIGITEC GALAXUS</span>
      <span className="bullet">●</span>
      <span className="label">TRANSFORMERS</span>
      <span className="bullet">●</span>
      <span className="label">SELF-IMPROVING AI</span>
      <span className="bullet">●</span>
      <span className="label">ETH ZÜRICH ALUM</span>
      <span className="bullet">●</span>
      <span className="label accent">ZURICH · CET</span>
      <span className="bullet">●</span>
      <span className="label">M.SC. COMPUTER SCIENCE</span>
      <span className="bullet">●</span>
      <span className="label">CFA LEVEL I</span>
      <span className="bullet">●</span>
      <span className="label">PRODUCTION ML</span>
      <span className="bullet">●</span>
    </div>
  );
}

/* ============================================================
   Writing section — list + tag filter
   ============================================================ */
function WritingSection() {
  const [activeTag, setActiveTag] = useState(null);

  // Tag frequency map (sorted by count desc, alphabetical tiebreak)
  const tagOrder = useMemo(() => {
    const counts = {};
    writingEntries.forEach((e) => e.tags.forEach((t) => { counts[t] = (counts[t] || 0) + 1; }));
    return Object.entries(counts)
      .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
      .map(([t]) => t);
  }, []);

  const filtered = useMemo(
    () => (activeTag ? writingEntries.filter((e) => e.tags.includes(activeTag)) : writingEntries),
    [activeTag],
  );

  return (
    <section className="page section reveal" id="writing">
      <div className="section-kicker">Writing</div>
      <h2 className="section-title">Notes on transformers and ML internals</h2>
      <p className="section-lede">Posts are built around interactive widgets — verify the math live, then read the prose.</p>

      <div className="tag-filter" role="group" aria-label="Filter posts by tag">
        <button
          type="button"
          className={`tag-filter-btn${activeTag === null ? ' active' : ''}`}
          onClick={() => setActiveTag(null)}
        >
          All <span className="tag-filter-count">{writingEntries.length}</span>
        </button>
        {tagOrder.map((t) => {
          const count = writingEntries.filter((e) => e.tags.includes(t)).length;
          return (
            <button
              key={t}
              type="button"
              className={`tag-filter-btn${activeTag === t ? ' active' : ''}`}
              onClick={() => setActiveTag(activeTag === t ? null : t)}
            >
              {t} <span className="tag-filter-count">{count}</span>
            </button>
          );
        })}
      </div>

      <div className="writing-list reveal-stagger">
        {filtered.length === 0 ? (
          <div className="writing-empty">No posts match that tag yet.</div>
        ) : (
          filtered.map((p) => (
            <Link to={`/blog/${p.slug}`} key={p.slug} className="writing-item">
              <div className="writing-date">{p.date}</div>
              <div>
                <h3 className="writing-title">{p.title}</h3>
                <p className="writing-desc">{p.desc}</p>
              </div>
              <div className="writing-tags">
                {p.tags.map((t) => <span className="tag" key={t}>{t}</span>)}
              </div>
            </Link>
          ))
        )}
      </div>
    </section>
  );
}

/* ============================================================
   Page
   ============================================================ */
const MainPage = () => {
  const heroBgRef = useRef(null);
  useHeroConstellation(heroBgRef);
  useRevealObserver();

  return (
    <main>
      <section className="hero" id="top">
        <div id="hero-bg" ref={heroBgRef}></div>
        <div className="page hero-anim">
          <div className="hero-status">
            <span className="dot"></span>
            <span>Currently in Zurich · CET</span>
          </div>

          <h1>
            ML engineer in Zurich, visualizing how <span className="accent">language models</span> actually learn.
          </h1>

          <p className="hero-lede">
            Most ML ideas are easier to grasp than the equations make them look — and almost all of them feel intuitive once you can <em>see</em> them. Every projection, softmax and mask on this site is computed live in your browser, so the picture is the proof.
          </p>

          <div className="hero-meta">
            <a href="https://www.linkedin.com/in/bernhardwalser/" target="_blank" rel="noopener noreferrer">
              <svg className="icon" viewBox="0 0 24 24"><path d="M16 8a6 6 0 0 1 6 6v7h-4v-7a2 2 0 0 0-4 0v7h-4v-7a6 6 0 0 1 6-6z"/><rect x="2" y="9" width="4" height="12"/><circle cx="4" cy="4" r="2"/></svg>
              LinkedIn
            </a>
            <span className="hero-meta-sep">·</span>
            <a href="https://github.com/berniwal" target="_blank" rel="noopener noreferrer">
              <svg className="icon" viewBox="0 0 24 24"><path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22"/></svg>
              GitHub
            </a>
          </div>
        </div>
      </section>

      <div className="ticker" aria-hidden="true">
        <div className="ticker-track">
          <TickerGroup />
          <TickerGroup />
        </div>
      </div>

      <WritingSection />

      <section className="page section reveal" id="experience">
        <div className="section-kicker">Experience</div>
        <h2 className="section-title">Currently building ML at Digitec Galaxus</h2>
        <p className="section-lede">Five years moving between research, computer vision, and production ML in Zurich.</p>

        <div className="timeline reveal-stagger">
          {experience.map((e, i) => (
            <div className={`tl-item${e.current ? ' current' : ''}`} key={i}>
              <div className="tl-marker"></div>
              <div className="tl-content">
                <div className="tl-meta">
                  <span>{e.meta[0]}</span>
                  <span className="sep">·</span>
                  <span>{e.meta[1]}</span>
                  {e.current && <span className="tl-current-tag">Now</span>}
                </div>
                <h3 className="tl-title">{e.title}</h3>
                <p className="tl-org">
                  <a href={e.url} target="_blank" rel="noopener noreferrer">{e.org}</a>
                </p>
                <p className="tl-desc">{e.desc}</p>
              </div>
            </div>
          ))}
        </div>
      </section>

      <section className="page section reveal" id="contact">
        <div className="section-kicker">Contact</div>
        <h2 className="section-title">Say hello</h2>
        <p className="section-lede">Happy to chat about anything in here — send me a message on LinkedIn, it's the fastest way to reach me.</p>

        <div className="contact-grid">
          <div>
            <p className="contact-lede">I'm interested in talking about self-learning dynamics, GPU inference / training optimization, and any project where the right visualization unlocks an idea.</p>
            <p className="contact-sub">If you've found a bug in one of the widgets, please reach out.</p>
            <div className="contact-actions">
              <a className="btn btn-primary" href="https://www.linkedin.com/in/bernhardwalser/" target="_blank" rel="noopener noreferrer">
                <svg className="icon" viewBox="0 0 24 24" style={{ stroke: 'currentColor' }}><path d="M16 8a6 6 0 0 1 6 6v7h-4v-7a2 2 0 0 0-4 0v7h-4v-7a6 6 0 0 1 6-6z"/><rect x="2" y="9" width="4" height="12"/><circle cx="4" cy="4" r="2"/></svg>
                Message on LinkedIn
              </a>
              <a className="btn" href="https://github.com/berniwal" target="_blank" rel="noopener noreferrer">
                <svg className="icon" viewBox="0 0 24 24"><path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22"/></svg>
                GitHub
              </a>
            </div>
          </div>

          <div className="contact-list">
            <a className="contact-row" href="https://www.linkedin.com/in/bernhardwalser/" target="_blank" rel="noopener noreferrer">
              <span className="contact-label">LinkedIn</span>
              <span className="contact-value">in/bernhardwalser</span>
              <svg className="icon" viewBox="0 0 24 24"><path d="M7 17 17 7"/><path d="M7 7h10v10"/></svg>
            </a>
            <a className="contact-row" href="https://github.com/berniwal" target="_blank" rel="noopener noreferrer">
              <span className="contact-label">GitHub</span>
              <span className="contact-value">@berniwal</span>
              <svg className="icon" viewBox="0 0 24 24"><path d="M7 17 17 7"/><path d="M7 7h10v10"/></svg>
            </a>
            <div className="contact-row" style={{ cursor: 'default' }}>
              <span className="contact-label">Location</span>
              <span className="contact-value">Zurich, Switzerland · CET</span>
              <svg className="icon" viewBox="0 0 24 24"><path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"/><circle cx="12" cy="10" r="3"/></svg>
            </div>
          </div>
        </div>
      </section>
    </main>
  );
};

export default MainPage;
