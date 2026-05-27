// src/posts/InternalVerifiers.jsx
// Self-Improvement · Part 3 of 6 — Internal Verifiers.
// New content. Loosely informed by the "When the verifier is self-generated"
// section of the legacy VisualizingSelfImprovement post, but the prose,
// structure, and widget are written from scratch for the post-2026 chrome.
//
// TAGS FOR REGISTRATION: ['self-improvement', 'r-zero', 'agent0']
// EXCERPT: When ground truth runs out, can a model judge its own work? Three recent papers — R-Zero, Agent0, G-Zero — try, each dropping a different assumption about what counts as verifiable.

import React, { useEffect, useRef, useState } from 'react';
import './PostChrome.css';
import './InternalVerifiers.css';

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
      className={`${block ? 'iv-math-block' : 'iv-math-inline'} ${className}`}
    />
  );
}

/* =========================================================
   Widget — Compare three self-verifier designs
   Three columns: who proposes, who scores, what reward flows.
   Hover a column to highlight its reward arrow + reveal a caption.
   ========================================================= */
const METHODS = [
  {
    id: 'rzero',
    name: 'R-Zero',
    arxiv: '2508.05004',
    proposer: 'Challenger',
    solver: 'Solver',
    scorer: 'Majority vote',
    scorerNote: 'over Solver samples',
    rewardP: 'hard-but-solvable',
    rewardS: 'matches consensus',
    blurb: 'Two copies of the base model split into roles. The Challenger gets a high reward for problems where the Solver hovers near 50% accuracy. The Solver is rewarded for matching a pseudo-label produced by majority vote across its own samples.',
  },
  {
    id: 'agent0',
    name: 'Agent0',
    arxiv: '2511.16043',
    proposer: 'Curriculum',
    solver: 'Executor + tools',
    scorer: 'Self-consistency',
    scorerNote: 'with Python sandbox',
    rewardP: 'frontier tasks',
    rewardS: 'consistent answers',
    blurb: 'Same two-agent shape as R-Zero, but the Executor can call a Python interpreter. Code execution does not directly score the answer — the reward is still internal — but grounding intermediate steps in deterministic execution makes the self-consistency signal less noisy.',
  },
  {
    id: 'gzero',
    name: 'G-Zero',
    arxiv: '2605.09959',
    proposer: 'Proposer',
    solver: 'Generator',
    scorer: 'Hint-δ',
    scorerNote: 'distributional shift',
    rewardP: 'hints that matter',
    rewardS: 'internalises hint',
    blurb: 'No vote, no checker. The Proposer writes a hint and the reward is how much the Generator\'s response shifts when the hint is appended. A useful hint changes the answer; a generic one ("think carefully") doesn\'t. The Generator is then trained — via DPO — to prefer the hint-conditioned response.',
  },
];

function ComparisonDiagram() {
  const [active, setActive] = useState('rzero');
  const W = 760, H = 320;
  const colW = W / 3;

  const colCenters = METHODS.map((_, i) => colW * (i + 0.5));

  return (
    <div className="viz-panel iv-compare">
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" preserveAspectRatio="xMidYMid meet">
        {/* Column dividers */}
        <line x1={colW} y1={26} x2={colW} y2={H - 20}
          stroke="var(--post-border)" strokeDasharray="3 4" />
        <line x1={2 * colW} y1={26} x2={2 * colW} y2={H - 20}
          stroke="var(--post-border)" strokeDasharray="3 4" />

        {METHODS.map((m, i) => {
          const cx = colCenters[i];
          const isActive = active === m.id;
          const propY = 70;
          const solvY = 170;
          const scorY = 260;
          return (
            <g
              key={m.id}
              className={`iv-col${isActive ? ' is-active' : ''}`}
              onMouseEnter={() => setActive(m.id)}
              onFocus={() => setActive(m.id)}
              tabIndex={0}
              role="button"
              aria-label={`Show ${m.name} flow`}
            >
              {/* Header */}
              <text x={cx} y={20} className="iv-col-title" textAnchor="middle">
                {m.name}
              </text>

              {/* Proposer node */}
              <rect x={cx - 78} y={propY - 22} width={156} height={44} rx={10}
                className="iv-node iv-node-prop" />
              <text x={cx} y={propY - 4} className="iv-node-role" textAnchor="middle">
                Proposes
              </text>
              <text x={cx} y={propY + 13} className="iv-node-name" textAnchor="middle">
                {m.proposer}
              </text>

              {/* Arrow: Proposer -> Solver (problem) */}
              <line x1={cx} y1={propY + 22} x2={cx} y2={solvY - 22}
                className="iv-arrow iv-arrow-task" markerEnd="url(#iv-head-task)" />
              <text x={cx + 6} y={(propY + solvY) / 2 + 4} className="iv-arrow-label">
                problem
              </text>

              {/* Solver node */}
              <rect x={cx - 78} y={solvY - 22} width={156} height={44} rx={10}
                className="iv-node iv-node-solv" />
              <text x={cx} y={solvY - 4} className="iv-node-role" textAnchor="middle">
                Solves
              </text>
              <text x={cx} y={solvY + 13} className="iv-node-name" textAnchor="middle">
                {m.solver}
              </text>

              {/* Arrow: Solver -> Scorer (answers) */}
              <line x1={cx} y1={solvY + 22} x2={cx} y2={scorY - 22}
                className="iv-arrow iv-arrow-task" markerEnd="url(#iv-head-task)" />
              <text x={cx + 6} y={(solvY + scorY) / 2 + 4} className="iv-arrow-label">
                answers
              </text>

              {/* Scorer node */}
              <rect x={cx - 88} y={scorY - 22} width={176} height={48} rx={10}
                className="iv-node iv-node-score" />
              <text x={cx} y={scorY - 4} className="iv-node-role" textAnchor="middle">
                Internal verifier
              </text>
              <text x={cx} y={scorY + 13} className="iv-node-name" textAnchor="middle">
                {m.scorer}
              </text>

              {/* Reward arrows curving back to Proposer & Solver */}
              <path
                d={`M ${cx + 88} ${scorY - 6} C ${cx + 120} ${(scorY + propY) / 2}, ${cx + 120} ${propY}, ${cx + 78} ${propY + 6}`}
                className="iv-arrow iv-arrow-reward"
                fill="none"
                markerEnd="url(#iv-head-reward)"
              />
              <path
                d={`M ${cx - 88} ${scorY - 6} C ${cx - 120} ${(scorY + solvY) / 2}, ${cx - 120} ${solvY}, ${cx - 78} ${solvY + 6}`}
                className="iv-arrow iv-arrow-reward"
                fill="none"
                markerEnd="url(#iv-head-reward)"
              />
            </g>
          );
        })}

        <defs>
          <marker id="iv-head-task" viewBox="0 0 10 10" refX="8" refY="5"
            markerWidth="6" markerHeight="6" orient="auto">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="var(--post-border-s)" />
          </marker>
          <marker id="iv-head-reward" viewBox="0 0 10 10" refX="8" refY="5"
            markerWidth="6" markerHeight="6" orient="auto">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="var(--post-accent)" />
          </marker>
        </defs>
      </svg>

      <div className="iv-compare-caption">
        <div className="iv-compare-tabs" role="tablist">
          {METHODS.map((m) => (
            <button
              key={m.id}
              type="button"
              role="tab"
              aria-selected={active === m.id}
              className={`iv-compare-tab${active === m.id ? ' active' : ''}`}
              onClick={() => setActive(m.id)}
            >
              {m.name}
            </button>
          ))}
        </div>
        <p className="iv-compare-blurb">
          {METHODS.find((m) => m.id === active).blurb}
        </p>
        <p className="viz-caption">
          Grey arrows carry problems and answers down each column; blue arrows
          carry the reward signal back up. In all three designs the reward
          source lives inside the dashed column — no external grader.
        </p>
      </div>
    </div>
  );
}

/* =========================================================
   Page
   ========================================================= */
export default function InternalVerifiers() {
  useEffect(() => {
    const prev = document.title;
    document.title = 'Internal Verifiers — Bernhard Walser';
    return () => { document.title = prev; };
  }, []);

  return (
    <article className="post-2026 iv-post">
      <div className="post-wrap">
        <div className="post-hero">
          <div className="post-series-tag">
            <span className="post-live-dot" aria-hidden="true" />
            Self-Improvement · Part 3 of 6
          </div>
          <h1>Internal Verifiers: When the Model Judges Its Own Work</h1>
          <p className="post-lede">
            External verifiers — unit tests, answer keys, game rules — are
            the engine behind most current self-improvement results. They
            also pin the trick to a narrow strip of problems. What happens
            on everything else? Three recent papers try to let the model
            grade itself, and each drops a different assumption about what
            "verifiable" means.
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

        <h2 className="reveal">Where external verifiers stop</h2>
        <p>
          <a className="post-link" href="#/blog/external-verifiers">Part 2</a>{' '}
          covered the methods that quietly carry most of the field:
          AlphaZero's win/loss signal, STaR and ReST/RLVR's
          answer-checkers, RLHF's reward model. Their power comes from a
          single shared ingredient — a function you can point at a
          candidate response and trust the label that comes back.
        </p>
        <p>
          That ingredient is the bottleneck. Code with unit tests, math
          with closed-form answers, and games with rules give you one for
          free. Most useful work doesn't. Writing a careful argument,
          designing an experiment, debugging an unfamiliar system — none
          of these have a cheap external grader, and hiring human
          annotators at the scale RL needs is an expensive way to find
          out.
        </p>
        <p>
          So the question, for the past year of self-improvement research:
          <em> can the model itself produce the verification signal?</em>
        </p>

        <h2 className="reveal">The basic shape</h2>
        <p>
          An "internal verifier" is anything that scores the model's
          output without consulting ground truth. Two patterns recur:
        </p>
        <ul className="iv-bullets">
          <li>
            <strong>A critic that scores.</strong> A separate head or
            model that estimates correctness, usually trained on
            preference data or self-generated labels.
          </li>
          <li>
            <strong>Consistency across samples.</strong> Generate the
            same answer many ways — different seeds, different prompts —
            and trust what the samples agree on. Majority voting and
            self-consistency live here.
          </li>
        </ul>
        <p>
          Both patterns face the same hazard: if the verifier is wrong in
          the same direction as the generator, training will amplify the
          shared error. The signal is real only when the verifier knows
          something the generator doesn't — or when the generator's
          mistakes are at least uncorrelated across samples.
        </p>

        <h2 className="reveal">R-Zero — Challenger vs. Solver</h2>
        <p>
          <a className="post-link" href="https://arxiv.org/abs/2508.05004" target="_blank" rel="noreferrer">R-Zero</a>{' '}
          (Huang et al. 2025, arXiv 2508.05004) starts from a single base
          LLM and forks it into two roles. A <em>Challenger</em> proposes
          problems; a <em>Solver</em> attempts them. The Challenger is
          rewarded when its problems land near the edge of the Solver's
          ability — hard, but not so hard the Solver fails every time.
          That keeps the curriculum tight against the Solver's frontier.
        </p>
        <p>
          The verifier is majority vote. The Solver samples many answers
          to the same problem, the most common one is treated as a
          pseudo-label, and the policy is trained — via GRPO — toward
          that pseudo-label. The whole loop runs without human-curated
          data. The assumption it bakes in: <em>tasks must be
          verifiable-in-principle</em>, so that sampling enough times
          would, in expectation, converge on the right answer. Math and
          symbolic reasoning fit; "write a good essay" doesn't.
        </p>

        <h2 className="reveal">Agent0 — same shape, with tools</h2>
        <p>
          <a className="post-link" href="https://arxiv.org/abs/2511.16043" target="_blank" rel="noreferrer">Agent0</a>{' '}
          (Xia et al. 2025, arXiv 2511.16043) keeps the two-agent
          structure — here called <em>Curriculum</em> and{' '}
          <em>Executor</em> — and adds a sandboxed Python interpreter
          the Executor can call mid-reasoning. The paper frames this as
          "tool-integrated reasoning": tools become part of the trace,
          not an external grader.
        </p>
        <p>
          The reward is still internal. The Curriculum is rewarded for
          producing tasks the Executor finds hard, and the Executor is
          trained on its own self-consistency, not on what the Python
          interpreter returns. But execution stabilises the
          self-consistency signal: intermediate steps that route through
          deterministic code are far less likely to drift apart across
          samples. We won't reproduce the full method here — see the
          paper for the exact reward composition — but the structural
          point is that <em>tools widen the verifiable region without
          handing the reward to an external grader</em>.
        </p>

        <h2 className="reveal">G-Zero — no verifier at all</h2>
        <p>
          <a className="post-link" href="https://arxiv.org/abs/2605.09959" target="_blank" rel="noreferrer">G-Zero</a>{' '}
          (Huang et al., arXiv 2605.09959) goes one step further and
          drops verifiability outright. Its signal is called{' '}
          <strong>Hint-δ</strong>: the gap between the Generator's
          unassisted answer and its answer when conditioned on a hint
          the model produced for itself.
        </p>
        <p>
          The intuition is short. A <em>useful</em> hint — say,{' '}
          <em>"try factoring"</em> appended to <Katex tex="x^2 + 5x + 6 = 0" /> —
          changes the response. A trivial hint — <em>"think
          carefully"</em> — doesn't. So the magnitude of the change
          measures how much information the hint actually carried. The
          Proposer is trained (via GRPO) to write hints that produce
          large δ — the model's blind spots. The Generator is trained
          (via DPO) to prefer the hint-conditioned response, folding the
          hint's content into its own behaviour so next time it doesn't
          need the hint. The paper describes it as a verifier-free
          framework, with supervision drawn from "internal
          distributional dynamics."
        </p>

        <h2 className="reveal">Comparing the three at a glance</h2>
        <p>
          The three designs share a skeleton — a proposer, a solver, a
          scoring step — but differ in what plays the role of the
          verifier and how much external structure they implicitly rely
          on. Hover a column for the details.
        </p>
        <ComparisonDiagram />
        <p>
          Read left to right and the verifier loosens its grip:
          majority vote requires that consensus tracks truth; tools
          require that the relevant work is computable; Hint-δ requires
          neither, and trades that freedom for a more abstract,
          harder-to-trust gradient.
        </p>

        <h2 className="reveal">Where this breaks</h2>
        <p>
          The same property that makes internal verifiers useful makes
          them fragile. If the verifier and the generator share a bias,
          training reinforces it — verifier-generator collusion. If the
          reward is gameable in any cheap way, the model finds the cheap
          way — reward hacking. And if the proposer keeps narrowing the
          task distribution toward what the solver already does well,
          the loop quietly turns into an <em>echo chamber</em> that
          mistakes self-agreement for progress. The papers above
          mitigate this with frontier-difficulty rewards, sample
          diversity bonuses, and coverage assumptions, but none of them
          claim to have solved it.
        </p>

        <h2 className="reveal">An open research front</h2>
        <p>
          Internal verifiers are the most promising route past the
          ground-truth bottleneck and also the easiest place to fool
          yourself. Whether a given loop is actually improving, or just
          looking improved through the same biased lens that trained it,
          turns out to be a surprisingly hard question. We'll come back
          to it in{' '}
          <a className="post-link" href="#/blog/measuring-self-improvement">Part 6 (Measuring Self-Improvement)</a>{' '}
          — what it would even mean to know.
        </p>

        <h2 className="reveal">References</h2>
        <div className="post-refs iv-refs">
          <div className="head ref-cite">Cite</div>
          <div className="head ref-title">Title</div>
          <div className="head ref-link">Link</div>

          <div className="ref-cite">Huang et al. 2025</div>
          <div>
            <div className="ref-title">
              <a href="https://arxiv.org/abs/2508.05004" target="_blank" rel="noreferrer">
                R-Zero: Self-Evolving Reasoning LLM from Zero Data
              </a>
            </div>
            <div className="ref-note">Challenger / Solver self-play with majority-vote pseudo-labels.</div>
          </div>
          <div className="ref-link"><a href="https://arxiv.org/abs/2508.05004" target="_blank" rel="noreferrer">arxiv 2508.05004</a></div>

          <div className="ref-cite">Xia et al. 2025</div>
          <div>
            <div className="ref-title">
              <a href="https://arxiv.org/abs/2511.16043" target="_blank" rel="noreferrer">
                Agent0: Tool-Integrated Self-Evolving Agents
              </a>
            </div>
            <div className="ref-note">Curriculum / Executor pair with a sandboxed Python interpreter.</div>
          </div>
          <div className="ref-link"><a href="https://arxiv.org/abs/2511.16043" target="_blank" rel="noreferrer">arxiv 2511.16043</a></div>

          <div className="ref-cite">Huang et al.</div>
          <div>
            <div className="ref-title">
              <a href="https://arxiv.org/abs/2605.09959" target="_blank" rel="noreferrer">
                G-Zero: Self-Play for Open-Ended Generation from Zero Data
              </a>
            </div>
            <div className="ref-note">Verifier-free framework; Hint-δ as the intrinsic reward.</div>
          </div>
          <div className="ref-link"><a href="https://arxiv.org/abs/2605.09959" target="_blank" rel="noreferrer">arxiv 2605.09959</a></div>
        </div>

        <footer className="post-footer">
          <p>
            <strong>Part 3 of Self-Improvement</strong> · Previous:{' '}
            <a className="post-link" href="#/blog/external-verifiers">External Verifiers</a>
            {' · '}Next:{' '}
            <a className="post-link" href="#/blog/evolutionary-search">Evolutionary Search (FunSearch, AlphaEvolve, ThetaEvolve)</a>.
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
