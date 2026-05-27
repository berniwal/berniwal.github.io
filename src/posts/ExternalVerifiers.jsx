// src/posts/ExternalVerifiers.jsx
// Self-Improvement · Part 2 of 6 — External Verifiers: STaR and ReST/RLVR.
// Ported from §3 + §4 of the original VisualizingSelfImprovement post; prose
// preserved, chrome redesigned to match MinimaxSearch.
//
// TAGS FOR REGISTRATION: ['self-improvement', 'star', 'rlvr']
// EXCERPT: AlphaZero had a built-in verifier — the game told it who won. Most useful tasks don't. STaR and ReST/RLVR keep the verifier outside the model and train on what it accepts.

import React, { useEffect, useRef, useState } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneLight } from 'react-syntax-highlighter/dist/esm/styles/prism';
import './PostChrome.css';
import './ExternalVerifiers.css';

/* =========================================================
   KaTeX wrapper — CDN-loaded global window.katex
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

function Counter({ label, value, sub }) {
  return (
    <div className="ev-counter">
      <div className="ev-counter-label">{label}</div>
      <div className="ev-counter-value">{value}</div>
      {sub && <div className="ev-counter-sub">{sub}</div>}
    </div>
  );
}

/* =========================================================
   §1 — STaR widget
   ========================================================= */

const STAR_PROMPT = 'A train travels 60 km in 1.5 hours. How far does it go in 4 hours at the same speed?';

const STAR_ROUNDS = [
  {
    round: 1,
    rationales: [
      { id: 'a', text: 'Speed = 60/1.5 = 40 km/h.\n40 × 4 = 160 km.',          finalAnswer: 160, correct: true  },
      { id: 'b', text: 'Speed = 60/1.5 = 30 km/h.\n30 × 4 = 120 km.',          finalAnswer: 120, correct: false },
      { id: 'c', text: 'In 1.5 h, 60 km.\nIn 4 h: 60 × 4 = 240 km.',           finalAnswer: 240, correct: false },
      { id: 'd', text: 'Rate: 60 km / 1.5 h = 40.\nDistance = 40 × 4 = 160.',  finalAnswer: 160, correct: true  },
      { id: 'e', text: 'Try ratios: 4/1.5 ≈ 2.67.\n60 × 2.67 ≈ 160 km.',       finalAnswer: 160, correct: true  },
    ],
  },
  {
    round: 2,
    rationales: [
      { id: 'a', text: 'Speed = 60/1.5 = 40 km/h.\n40 × 4 = 160 km.',          finalAnswer: 160, correct: true },
      { id: 'b', text: '60 ÷ 1.5 = 40 km/h.\nDistance = 40 × 4 = 160 km.',     finalAnswer: 160, correct: true },
      { id: 'c', text: 'Rate = 60 km / 1.5 h = 40 km/h.\n40 × 4 = 160 km.',    finalAnswer: 160, correct: true },
      { id: 'd', text: '60 / 1.5 = 40.\nThen 40 × 4 = 160 km.',                finalAnswer: 160, correct: true },
      { id: 'e', text: '40 km/h × 4 h = 160 km.',                              finalAnswer: 160, correct: true },
    ],
  },
];

function SectionStaR() {
  const [round, setRound] = useState(0);
  const data = STAR_ROUNDS[round];
  const kept = data.rationales.filter((r) => r.correct);

  const advance = () => setRound((r) => Math.min(r + 1, STAR_ROUNDS.length - 1));
  const reset = () => setRound(0);

  const corpus = [];
  for (let i = 0; i <= round; i++) {
    for (const r of STAR_ROUNDS[i].rationales) {
      if (r.correct) {
        corpus.push({
          round: STAR_ROUNDS[i].round,
          summary: r.text.split('\n')[0],
          ans: r.finalAnswer,
        });
      }
    }
  }

  return (
    <div className="viz-panel ev-star">
      <div className="ev-controls">
        <button onClick={advance} className={round < STAR_ROUNDS.length - 1 ? 'active' : ''}>
          {round < STAR_ROUNDS.length - 1 ? 'Run next round' : 'Final round'}
        </button>
        <button onClick={reset}>Reset</button>
        <span className="ev-controls-note">
          round <strong>{data.round}</strong> of {STAR_ROUNDS.length}{' '}
          · prompt below, sample {data.rationales.length} chains-of-thought
        </span>
      </div>

      <div className="ev-prompt">{STAR_PROMPT}</div>

      <div className="ev-star-grid">
        {data.rationales.map((r) => (
          <div key={r.id} className={`ev-star-card ${r.correct ? 'kept' : 'dropped'}`}>
            <div className="ev-star-card-tag">Sample {r.id.toUpperCase()}</div>
            <div className="ev-star-card-text">{r.text}</div>
            <div className="ev-star-card-badge">
              answer: {r.finalAnswer} km ·{' '}
              {r.correct ? '✓ verifier kept' : '✗ verifier dropped'}
            </div>
          </div>
        ))}
      </div>

      <div className="ev-star-corpus">
        <div className="ev-star-corpus-title">
          Training corpus so far · {corpus.length} kept rationale{corpus.length === 1 ? '' : 's'}
        </div>
        {corpus.length === 0 ? (
          <div className="ev-star-corpus-row">empty — sample first</div>
        ) : (
          corpus.map((c, i) => (
            <div key={i} className="ev-star-corpus-row">
              [round {c.round}] {c.summary} → {c.ans} km
            </div>
          ))
        )}
      </div>

      <div className="ev-counters">
        <Counter label="Round" value={data.round} />
        <Counter label="Sampled this round" value={data.rationales.length} />
        <Counter
          label="Verifier kept"
          value={`${kept.length} / ${data.rationales.length}`}
          sub={kept.length === data.rationales.length ? 'all reach 160 km' : 'wrong-answer ones dropped'}
        />
        <Counter label="Corpus size" value={corpus.length} sub="cumulative" />
      </div>

      <p className="viz-caption">
        Round 1: five rationales sampled from the base model; three reach the
        correct answer (160 km), two don't. Only the correct ones flow into
        the training corpus — the verifier <em>is</em> the filter. After a
        fine-tune on the kept set, the model is more likely to sample
        correct rationales next round; click <em>Run next round</em> to see
        what the post-fine-tune samples look like.
      </p>
    </div>
  );
}

/* =========================================================
   §2 — ReST / RLVR slim widget
   ========================================================= */

function SectionReST() {
  const prompt = 'Factor:  x² + 5x + 6.';
  const rollouts = [
    { id: 1, text: '(x + 2)(x + 3)',                correct: true,  rmScore: 0.78 },
    { id: 2, text: '(x − 2)(x − 3)',                correct: false, rmScore: 0.60 },
    { id: 3, text: '(x + 1)(x + 6)',                correct: false, rmScore: 0.55 },
    { id: 4, text: '(x + 2)(x + 3) — by inspection', correct: true, rmScore: 0.65 },
  ];
  const rewards = rollouts.map((r) => (r.correct ? 1.0 : 0.0));
  const mean = rewards.reduce((s, x) => s + x, 0) / rewards.length;
  const variance = rewards.reduce((s, x) => s + (x - mean) ** 2, 0) / rewards.length;
  const std = Math.sqrt(variance) || 1e-9;
  const advs = rewards.map((r) => (r - mean) / std);

  return (
    <div className="viz-panel ev-rest">
      <div className="ev-prompt">{prompt}</div>

      <div className="ev-rest-grid">
        {rollouts.map((r, i) => {
          const a = advs[i];
          return (
            <div key={r.id} className={`ev-rest-card ${r.correct ? 'pass' : 'fail'}`}>
              <div className="ev-rest-card-tag">Rollout {r.id}</div>
              <div className="ev-rest-card-text">{r.text}</div>
              <div className="ev-rest-card-badge">
                {r.correct ? '✓ verifier PASS' : '✗ verifier FAIL'} · r = {rewards[i].toFixed(0)} · Â = {a >= 0 ? '+' : ''}{a.toFixed(2)}
              </div>
            </div>
          );
        })}
      </div>

      <div className="ev-counters">
        <Counter label="Group size G" value={rollouts.length} />
        <Counter label="μ (group mean)" value={mean.toFixed(2)} />
        <Counter label="σ (group std)" value={std.toFixed(2)} />
        <Counter
          label="Advantages"
          value="z-scored"
          sub="standardised within the group"
        />
      </div>

      <p className="viz-caption">
        Four rollouts on the same prompt; the verifier marks each correct or
        incorrect; rewards are standardised within the group to get
        advantages. STaR (above) and ReST/RLVR are the <em>same fundamental
        move</em>: keep what passes the verifier, push the policy toward it.
        The only difference is whether the "keep" step is a supervised
        fine-tune (STaR) or a policy-gradient update with group-relative
        advantages (ReST/RLVR).
      </p>
    </div>
  );
}

/* =========================================================
   STaR pseudocode block (collapsible)
   ========================================================= */

const STAR_CODE = `# STaR (one round)
for problem in dataset:
    samples = [model(problem) for _ in range(K)]
    kept    = [s for s in samples if verifier(problem, s)]
    corpus += [(problem, s) for s in kept]
model = finetune(model, corpus)`;

function CodeBlock() {
  return (
    <details className="post-code-details">
      <summary>
        <span className="post-code-summary-icon" aria-hidden="true">›</span>
        <span>Show STaR pseudocode</span>
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
        {STAR_CODE}
      </SyntaxHighlighter>
    </details>
  );
}

/* =========================================================
   Page
   ========================================================= */

export default function ExternalVerifiers() {
  useEffect(() => {
    const prev = document.title;
    document.title = 'External Verifiers: STaR and ReST/RLVR — Bernhard Walser';
    return () => { document.title = prev; };
  }, []);

  return (
    <article className="post-2026 ev-post">
      <div className="post-wrap">
        <div className="post-hero">
          <div className="post-series-tag"><span className="post-live-dot" aria-hidden="true" />Self-Improvement · Part 2 of 6</div>
          <h1>External Verifiers: STaR and ReST/RLVR</h1>
          <p className="post-lede">
            AlphaZero had a built-in verifier — the rules of the game told it
            who had won. Most useful tasks don't come with a referee. The
            first answer to that problem: keep the verifier <em>outside</em>{' '}
            the model — a unit test, an answer key, a labeller — and train
            on what it accepts.
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

        <h2 className="reveal">STaR — bootstrap rationales against a verifier</h2>
        <p>
          The most natural port of AlphaGo Zero's recipe to language:{' '}
          <a className="post-link" href="https://arxiv.org/abs/2203.14465" target="_blank" rel="noreferrer">STaR</a>{' '}
          (Zelikman et al., 2022). Sample many chains-of-thought from the
          model on a problem with a known answer; keep the ones that reach
          the correct answer; fine-tune on the kept set; repeat. The
          "verifier" is whatever programmatic check certifies the final
          answer. The model gradually concentrates probability mass on
          reasoning paths the verifier accepts.
        </p>

        <CodeBlock />

        <SectionStaR />

        <p>
          Step it forward. Round 1's samples include three correct and two
          wrong rationales; only the correct ones survive into the training
          corpus. After fine-tuning, round 2 samples are <em>more
          likely</em> to be correct — and the corpus keeps growing. This is
          exactly AlphaGo Zero's self-improvement loop:{' '}
          <em>self-play (sampling) → verifier (rules / answer-check) →
          training-target generation → policy improvement → better
          self-play.</em> The only difference is that the "game tree" is
          replaced by token-level generation.
        </p>

        <h2 className="reveal">ReST / RLVR — the same trick, written as RL</h2>
        <p>
          STaR's <em>"keep the correct ones, fine-tune"</em> filter has an
          obvious RL reading: the verifier is a binary reward,{' '}
          <Katex tex="r \in \{0, 1\}" />, and we want to push policy mass
          toward responses with <Katex tex="r = 1" />. That's RL with a
          verifiable reward. DeepMind's{' '}
          <a className="post-link" href="https://arxiv.org/abs/2308.08998" target="_blank" rel="noreferrer">ReST</a>{' '}
          and{' '}
          <a className="post-link" href="https://arxiv.org/abs/2312.06585" target="_blank" rel="noreferrer">ReST-EM</a>{' '}
          formalise the move as an outer expectation-maximisation loop —
          sample, filter, fine-tune — that recovers STaR when the filter is
          binary. The mature form is the GRPO + verifier recipe from{' '}
          <a className="post-link" href="https://arxiv.org/abs/2402.03300" target="_blank" rel="noreferrer">DeepSeekMath</a>{' '}
          (Shao et al., 2024): rollouts are scored by a verifier, advantages
          are computed group-relatively, no learned reward model is needed.
        </p>

        <SectionReST />

        <p>
          The conceptual unification is the important bit. <strong>STaR
          (supervised filter then fine-tune) and ReST/RLVR (group-relative
          policy gradient) are the same fundamental move</strong> — keep
          what passes the verifier, push the policy toward it. STaR is the
          supervised version (cross-entropy on the kept rationales, treating
          them as fixed targets); RLVR is the policy-gradient version (a
          clipped surrogate that explicitly accounts for the current
          policy's distribution via the importance ratio{' '}
          <Katex tex="\rho_t" />). Both are AlphaGo Zero's loop projected
          onto language.
        </p>

        <h2 className="reveal">What these methods share</h2>
        <p>
          STaR, ReST, and RLVR all assume an <strong>external
          verifier</strong>: the rules of the game, an answer key, a unit
          test, a labeller. The verifier sits outside the model and decides
          which samples count. The model improves because the loop is biased
          — every fine-tune step is taken on data the verifier has already
          waved through.
        </p>
        <p>
          That assumption is the whole game. It's also the whole limitation.
          For math problems with known answers, code with unit tests, or
          chess positions with a rule book, an external verifier is cheap.
          For open-ended writing, novel research, or any task where you{' '}
          <em>don't already know</em> the right answer, there is no
          verifier to call.
        </p>
        <p>
          What happens when the verifier has to come from <em>inside</em>{' '}
          the model? That's{' '}
          <a className="post-link" href="#/blog/internal-verifiers">Part 3 — Internal Verifiers</a>.
        </p>

        <h2 className="reveal">References</h2>
        <div className="post-refs">
          <div className="head ref-cite">Cite</div>
          <div className="head ref-title">Title</div>
          <div className="head ref-link">Link</div>

          <div className="ref-cite">Zelikman et al., 2022</div>
          <div>
            <div className="ref-title">
              <a href="https://arxiv.org/abs/2203.14465" target="_blank" rel="noreferrer">STaR: Bootstrapping Reasoning With Reasoning</a>
            </div>
            <div className="ref-note">The original sample-filter-finetune loop on chain-of-thought reasoning.</div>
          </div>
          <div className="ref-link"><a href="https://arxiv.org/abs/2203.14465" target="_blank" rel="noreferrer">arxiv.org</a></div>

          <div className="ref-cite">Singh et al., 2024</div>
          <div>
            <div className="ref-title">
              <a href="https://arxiv.org/abs/2312.06585" target="_blank" rel="noreferrer">Beyond Human Data: Scaling Self-Training for Problem-Solving with Language Models (ReST-EM)</a>
            </div>
            <div className="ref-note">STaR cast as an outer expectation-maximisation loop, with experiments at scale.</div>
          </div>
          <div className="ref-link"><a href="https://arxiv.org/abs/2312.06585" target="_blank" rel="noreferrer">arxiv.org</a></div>

          <div className="ref-cite">Shao et al., 2024</div>
          <div>
            <div className="ref-title">
              <a href="https://arxiv.org/abs/2402.03300" target="_blank" rel="noreferrer">DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models</a>
            </div>
            <div className="ref-note">Introduces GRPO; the canonical RLVR baseline — group-relative policy gradient with a verifier and no learned reward model.</div>
          </div>
          <div className="ref-link"><a href="https://arxiv.org/abs/2402.03300" target="_blank" rel="noreferrer">arxiv.org</a></div>

          <div className="ref-cite">Gulcehre et al., 2023</div>
          <div>
            <div className="ref-title">
              <a href="https://arxiv.org/abs/2308.08998" target="_blank" rel="noreferrer">Reinforced Self-Training (ReST) for Language Modeling</a>
            </div>
            <div className="ref-note">The original ReST formulation as alternating Grow / Improve steps.</div>
          </div>
          <div className="ref-link"><a href="https://arxiv.org/abs/2308.08998" target="_blank" rel="noreferrer">arxiv.org</a></div>
        </div>

        <footer className="post-footer">
          <p>
            <strong>Part 2 of Self-Improvement</strong> · Previous:{' '}
            <a className="post-link" href="#/blog/alphago-to-alphazero">AlphaGo to AlphaZero</a>
            {' · '}Next:{' '}
            <a className="post-link" href="#/blog/internal-verifiers">Internal Verifiers (R-Zero, Agent0, G-Zero)</a>.
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
