// src/posts/BeyondHumanFeedback.jsx
// Aligning LMs · Part 3 of 3 — Beyond Human Feedback: RLAIF, Process Rewards, RLVR.
// TAGS FOR REGISTRATION: ['rlhf', 'rlaif', 'rlvr']
// EXCERPT: PPO, DPO and GRPO fixed the algorithm side of alignment. The labels are still the bottleneck. Three answers — RLAIF, process rewards, RLVR — each replace the human in the loop with something cheaper, sharper, or deterministic.
import React, { useEffect, useRef, useState } from 'react';
import './PostChrome.css';
import './BeyondHumanFeedback.css';

/* ============================================================
   KaTeX wrapper — uses the CDN-loaded global window.katex
   ============================================================ */
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
      className={`${block ? 'bhf-math-block' : 'bhf-math-inline'} ${className}`}
    />
  );
}

/* ============================================================
   Counter — small KPI chip
   ============================================================ */
function Counter({ label, value, sub }) {
  return (
    <div className="bhf-counter">
      <div className="bhf-counter-label">{label}</div>
      <div className="bhf-counter-value">{value}</div>
      {sub && <div className="bhf-counter-sub">{sub}</div>}
    </div>
  );
}

/* ============================================================
   InfoBox — reusable aside callout
   ============================================================ */
function InfoBox({ title, children }) {
  return (
    <aside className="bhf-aside">
      <h4 className="bhf-aside-head">
        <span className="bhf-aside-tag">i</span>
        {title}
      </h4>
      <div className="bhf-aside-body">{children}</div>
    </aside>
  );
}

/* ============================================================
   §RLAIF widget — same two responses, different principle, different label
   ============================================================ */
const RLAIF_PROMPT =
  "I've been drinking a glass of wine most evenings to unwind from work. Should I cut back?";

const RLAIF_RESPONSES = {
  a: {
    label: 'Response A · short and direct',
    text:
      'Drinking less is generally good for your sleep and mood. Try swapping the evening glass for something else like tea or a walk and see if you feel better in a week.',
  },
  b: {
    label: 'Response B · longer, more careful',
    text:
      'Two things to separate: (1) the work stress itself, which you can address by talking to your manager, setting limits, or seeing a therapist; (2) the drinking, which can quietly escalate when used as a coping tool. If you\'re wondering whether it\'s a problem, the AUDIT-C is a short screening tool you can take, and your GP can help. Neither of these is something you have to figure out alone.',
  },
};

const RLAIF_PRINCIPLES = [
  {
    id: 'helpful',
    label: 'Be helpful',
    text: 'Be helpful and direct. Answer the actual question without hedging or stalling.',
    preferred: 'a',
    critique:
      'Response A directly answers the question with a concrete suggestion (swap the wine for tea or a walk). Response B is helpful but spends most of its length on framing, screening tools, and referrals — which the user didn\'t ask for. Prefer A.',
  },
  {
    id: 'harmless',
    label: 'Be honest about risks',
    text: 'Be honest about risks. When something the user describes could plausibly be a serious problem, name it rather than glossing over it.',
    preferred: 'b',
    critique:
      'The user is describing a coping pattern that can escalate. Response A treats it as a sleep/mood lifestyle tweak; Response B names the risk explicitly, gives a real screening tool (AUDIT-C), and points to professional help without alarmism. Prefer B.',
  },
  {
    id: 'autonomy',
    label: 'Respect autonomy',
    text: 'Respect autonomy. Don\'t pathologize ordinary behaviour or push the user toward professional services unless they ask for them.',
    preferred: 'a',
    critique:
      'The user asked a casual question. Response B reframes their behaviour as potentially a clinical issue and suggests a screening tool and GP visit unprompted. Response A trusts the user to manage their own life. Prefer A.',
  },
];

function SectionRLAIF() {
  const [principleId, setPrincipleId] = useState('helpful');
  const principle = RLAIF_PRINCIPLES.find((p) => p.id === principleId);

  return (
    <div className="viz-panel bhf-widget">
      <div className="bhf-tabs" role="tablist">
        {RLAIF_PRINCIPLES.map((p) => (
          <button
            key={p.id}
            type="button"
            role="tab"
            className={`bhf-tab ${principleId === p.id ? 'active' : ''}`}
            onClick={() => setPrincipleId(p.id)}
          >
            {p.label}
          </button>
        ))}
      </div>

      <div className="bhf-principle">
        <strong>Principle:</strong> {principle.text}
      </div>

      <div className="bhf-prompt">{RLAIF_PROMPT}</div>

      <div className="bhf-pair-pick">
        {['a', 'b'].map((id) => {
          const r = RLAIF_RESPONSES[id];
          const chosen = principle.preferred === id;
          return (
            <div
              key={id}
              className={`bhf-candidate ${chosen ? 'locked-chosen' : 'locked-rejected'}`}
            >
              <div className="bhf-candidate-tag">
                {r.label} · {chosen ? 'chosen' : 'rejected'} by AI
              </div>
              {r.text}
            </div>
          );
        })}
      </div>

      <div className="bhf-critique">
        <span className="label">AI critic — reasoning</span>
        {principle.critique}
      </div>

      <p className="viz-caption" style={{ marginTop: 14 }}>
        Same two responses, three different written principles, three different
        labels. The "human in RLHF" is now a written rule plus a critic LLM
        applying it. Cheap to scale, and the principle is auditable — you can
        read what the model is being asked to be.{' '}
        <a className="post-link" href="https://arxiv.org/abs/2212.08073" target="_blank" rel="noreferrer">Bai et al. 2022</a>.
      </p>
    </div>
  );
}

/* ============================================================
   §Process rewards — outcome vs per-step
   ============================================================ */
const COT_STEPS = [
  {
    n: 1,
    text: '14 × 17 = 14 × 10 + 14 × 7',
    process: { reward: +1.0 },
    critique: 'Distributive law applied correctly. The substeps that follow can be checked independently.',
  },
  {
    n: 2,
    text: '       = 140 + 96',
    process: { reward: -1.0 },
    critique: '14 × 7 = 98, not 96. The error is here, even though the final answer might still happen to be in the right ballpark.',
  },
  {
    n: 3,
    text: '       = 236',
    process: { reward: +0.4 },
    critique: '140 + 96 = 236 is correct given step 2 — the addition is right, but the input was wrong. PRMs typically still give partial credit for locally valid moves.',
  },
];

function SectionProcessRewards() {
  const [mode, setMode] = useState('process');
  const [activeStep, setActiveStep] = useState(null);

  const outcomeReward = -0.7;
  const totalProcess = COT_STEPS.reduce((s, x) => s + x.process.reward, 0);

  return (
    <div className="viz-panel bhf-widget">
      <div className="bhf-tabs" role="tablist">
        <button type="button" role="tab"
          className={`bhf-tab ${mode === 'outcome' ? 'active' : ''}`}
          onClick={() => setMode('outcome')}>
          Outcome reward (ORM)
        </button>
        <button type="button" role="tab"
          className={`bhf-tab ${mode === 'process' ? 'active' : ''}`}
          onClick={() => setMode('process')}>
          Process reward (PRM)
        </button>
      </div>

      <div className="bhf-prompt">What is 14 × 17?</div>

      <div className="bhf-chain">
        {COT_STEPS.map((s) => {
          const isLast = s.n === COT_STEPS.length;
          const shown = mode === 'process' || isLast;
          const reward = isLast && mode === 'outcome' ? outcomeReward : s.process.reward;
          const cls = shown ? (reward > 0 ? 'pos' : 'neg') : 'muted';
          return (
            <div
              key={s.n}
              className={`bhf-step ${activeStep === s.n ? 'active' : ''}`}
              onClick={() => setActiveStep(activeStep === s.n ? null : s.n)}
            >
              <div className="bhf-step-num">{s.n}</div>
              <div className="bhf-step-body">{s.text}</div>
              <div className={`bhf-step-reward ${cls}`}>
                {shown ? (reward > 0 ? `+${reward.toFixed(1)}` : reward.toFixed(1)) : '—'}
              </div>
            </div>
          );
        })}
      </div>

      {activeStep !== null && (
        <div className="bhf-critique-line">
          <strong>Step {activeStep}:</strong>{' '}
          {COT_STEPS.find((s) => s.n === activeStep).critique}
        </div>
      )}

      <div className="bhf-counters">
        <Counter
          label="Mode"
          value={mode === 'outcome' ? 'Outcome RM' : 'Process RM (PRM)'}
          sub={mode === 'outcome' ? 'one scalar, final answer' : 'one scalar per step'}
        />
        <Counter
          label="Total reward signal"
          value={mode === 'outcome' ? outcomeReward.toFixed(1) : totalProcess.toFixed(1)}
          sub={mode === 'outcome' ? 'wrong answer · whole trace penalised' : 'step 1 ✓ · step 2 ✗ · step 3 ✓ (mod)'}
        />
        <Counter
          label="Credit assignment"
          value={mode === 'outcome' ? 'blamed evenly' : 'pinned to step 2'}
          sub={mode === 'outcome' ? 'noisy gradient' : 'sharp gradient'}
        />
      </div>

      <p className="viz-caption" style={{ marginTop: 14 }}>
        Click any step for the PRM's critique. The same trace produces two very
        different training signals — outcome rewards punish the whole chain
        when the answer is wrong, while process rewards pin the gradient to
        the exact step that went off the rails.
      </p>
    </div>
  );
}

/* ============================================================
   §RLVR — verifier vs learned RM, GRPO advantages
   ============================================================ */
const RLVR_PROMPT = 'Solve for x:   2x + 5 = 13.';

const RLVR_ROLLOUTS = [
  {
    id: 1,
    text:
      'Subtract 5 from both sides:  2x = 8.\n' +
      'Divide by 2:  x = 4.',
    finalAnswer: 4, correct: true, rmScore: 0.75,
  },
  {
    id: 2,
    text:
      'Add 5 to both sides:  2x = 18.\n' +
      'Divide by 2:  x = 9.',
    finalAnswer: 9, correct: false, rmScore: 0.50,
  },
  {
    id: 3,
    text: '2x = 8\nx = 4.',
    finalAnswer: 4, correct: true, rmScore: 0.40,
  },
  {
    id: 4,
    text:
      'Reading the equation carefully, the answer is clearly x = 7. ' +
      'This follows from standard algebraic manipulation, and the ' +
      'reasoning is straightforward.',
    finalAnswer: 7, correct: false, rmScore: 0.65,
  },
];

function SectionRLVR() {
  const [mode, setMode] = useState('verifier');

  const rewards = RLVR_ROLLOUTS.map((r) =>
    mode === 'verifier' ? (r.correct ? 1.0 : 0.0) : r.rmScore
  );
  const mean = rewards.reduce((s, x) => s + x, 0) / rewards.length;
  const variance = rewards.reduce((s, x) => s + (x - mean) ** 2, 0) / rewards.length;
  const std = Math.sqrt(variance) || 1e-9;
  const advantages = rewards.map((r) => (r - mean) / std);
  const maxAbsA = Math.max(...advantages.map(Math.abs), 1e-9);

  const signsAgree = advantages.every((a, i) => (a >= 0) === RLVR_ROLLOUTS[i].correct);

  return (
    <div className="viz-panel bhf-widget">
      <div className="bhf-tabs" role="tablist">
        <button type="button" role="tab"
          className={`bhf-tab ${mode === 'verifier' ? 'active' : ''}`}
          onClick={() => setMode('verifier')}>
          Reward = Verifier (RLVR)
        </button>
        <button type="button" role="tab"
          className={`bhf-tab ${mode === 'rm' ? 'active' : ''}`}
          onClick={() => setMode('rm')}>
          Reward = Learned RM
        </button>
      </div>

      <div className="bhf-prompt">{RLVR_PROMPT}</div>

      <div className="bhf-rlvr-grid">
        {RLVR_ROLLOUTS.map((rollout, i) => {
          const r = rewards[i];
          const a = advantages[i];
          const passing = mode === 'verifier' && rollout.correct;
          const failing = mode === 'verifier' && !rollout.correct;
          const cardCls =
            'bhf-rlvr-card' +
            (passing ? ' pass' : '') +
            (failing ? ' fail' : '');
          return (
            <div key={rollout.id} className={cardCls}>
              <div className="bhf-rlvr-tag">
                Rollout {rollout.id}{' '}
                <span style={{ color: 'var(--post-text-mut)' }}>
                  · final x = {rollout.finalAnswer}
                </span>
              </div>
              <div className="bhf-rlvr-text">{rollout.text}</div>
              <div className="bhf-rlvr-badge">
                {mode === 'verifier'
                  ? rollout.correct ? '✓ verifier PASS' : '✗ verifier FAIL'
                  : `RM score: ${rollout.rmScore.toFixed(2)}`}
              </div>
              <div className="bhf-rlvr-meta">
                r = {r.toFixed(2)}{'  '}Â = {a >= 0 ? '+' : ''}{a.toFixed(2)}
              </div>
              <div className="bhf-rlvr-bar-track">
                <div className="center-line" />
                <div
                  className={`bhf-rlvr-bar-fill ${a >= 0 ? 'pos' : 'neg'}`}
                  style={{
                    width: `${(Math.abs(a) / maxAbsA) * 50}%`,
                    transform: a >= 0 ? 'none' : 'translateX(-100%)',
                  }}
                />
              </div>
            </div>
          );
        })}
      </div>

      <div className="bhf-counters">
        <Counter
          label="Reward source"
          value={mode === 'verifier' ? 'verifier' : 'learned RM'}
          sub={mode === 'verifier' ? 'r ∈ {0, 1}' : 'r ∈ [0, 1]'}
        />
        <Counter label="Group mean μ" value={mean.toFixed(2)} />
        <Counter label="Group std σ" value={std.toFixed(2)} />
        <Counter
          label="Sign(Â) = sign(correct)?"
          value={signsAgree ? 'yes' : 'NO'}
          sub={signsAgree ? 'gradients point right way' : 'reward hacking — gradients lie'}
        />
      </div>

      <p className="viz-caption" style={{ marginTop: 14 }}>
        {mode === 'verifier' ? (
          <>
            Rollouts <strong>1</strong> and <strong>3</strong> reach{' '}
            <Katex tex="x = 4" /> (verifier PASS) and get a positive
            advantage; the other two get a negative one. Notice rollout{' '}
            <strong>4</strong> — a confidently-stated wrong answer — is
            correctly penalised. Now switch to <em>Learned RM</em> above and
            watch the counter at the bottom flip.
          </>
        ) : (
          <>
            Look at <strong>rollout 3</strong> (terse but correct) — the RM
            scores it <em>lower</em> than rollout 4, which is wrong but
            sounds confident. Rollout 3 gets a <em>negative</em> advantage;
            rollout 4 gets a <em>positive</em> one. The gradient is now
            pulling the policy <em>away</em> from a correct answer and{' '}
            <em>toward</em> a wrong one. A verifier doesn't have this
            failure mode — it doesn't care how a rollout sounds, only
            whether it's right.
          </>
        )}
      </p>
    </div>
  );
}

/* ============================================================
   Page
   ============================================================ */
export default function BeyondHumanFeedback() {
  useEffect(() => {
    const prev = document.title;
    document.title = 'Beyond Human Feedback: RLAIF, Process Rewards, RLVR — Bernhard Walser';
    return () => { document.title = prev; };
  }, []);

  return (
    <article className="post-2026 bhf-post">
      <div className="post-wrap">
        <div className="post-hero">
          <div className="post-series-tag">
            <span className="post-live-dot" aria-hidden="true" />
            Aligning LMs · Part 3 of 3
          </div>
          <h1>Beyond Human Feedback: RLAIF, Process Rewards, RLVR</h1>
          <p className="post-lede">
            <a className="post-link" href="#/blog/rlhf-and-ppo">Part 1</a> and{' '}
            <a className="post-link" href="#/blog/beyond-ppo">Part 2</a> fixed
            the algorithm side of alignment — clipped surrogates, group
            baselines, closed-form policies. This post tackles a different
            bottleneck: <em>human labels are expensive, noisy, and slow</em>.
            Three answers, each replacing the human in the loop with
            something else.
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

        <p>
          PPO, DPO and GRPO all assume the same input: a dataset of human
          preferences. Two responses, a person picks the better one, repeat
          ten thousand times. That dataset is the most expensive thing in the
          alignment pipeline. It's also the most fragile — annotators
          disagree, standards drift, and every change to the policy you want
          (more honest, less sycophantic, better at math) means more labels.
          The three ideas below each replace humans with something cheaper,
          sharper, or deterministic.
        </p>

        <h2 className="reveal">RLAIF — when the labeler is a model</h2>
        <p>
          Human preference data is expensive. Worse, the bar for what counts
          as "good behaviour" varies between annotators, drifts over time,
          and has to be re-collected when you change what you want the model
          to do.{' '}
          <a className="post-link" href="https://arxiv.org/abs/2212.08073" target="_blank" rel="noreferrer">
            Anthropic's Constitutional AI
          </a>{' '}
          (Bai et al. 2022) replaces the human labeler with a stronger LLM
          and a written set of principles — the "constitution." The critic
          reads each candidate response and judges it against an explicit
          rule. The downstream Bradley-Terry loss is unchanged:
        </p>
        <div className="bhf-math-block">
          <Katex
            block
            tex={String.raw`\mathcal{L}_{\mathrm{RM}} \;=\; -\,\mathbb{E}\!\left[\,\log \sigma\!\big(r_\varphi(x, y_w) - r_\varphi(x, y_l)\big)\,\right]`}
          />
        </div>
        <p>
          Only the source of <Katex tex="(y_w, y_l)" /> changes — pairs come
          from an AI critic applying a written rule, not from a human picking
          between cards. Google's{' '}
          <a className="post-link" href="https://arxiv.org/abs/2309.00267" target="_blank" rel="noreferrer">
            RLAIF paper
          </a>{' '}
          (Lee et al. 2023) showed AI-labelled preferences are roughly on par
          with human ones for summarisation, dialogue and harmlessness.
        </p>
        <p>
          Toggle the principle below; the same two responses get a different
          label depending on which rule the critic is applying.
        </p>
        <SectionRLAIF />
        <p>
          Two properties make this interesting beyond cost. First, the
          principles are <em>auditable</em>: you can read what the model is
          being trained to value, in plain English, rather than inferring it
          from labeller-preference statistics. Second, the same data can be
          re-derived by re-running the critic with a new principle — when
          policies need to change, you don't have to re-collect labels. Most
          modern production stacks use a mix: some human-labelled data, some
          AI-labelled data, and the downstream RL step (PPO, DPO, or GRPO)
          doesn't care which is which.
        </p>

        <h2 className="reveal">Outcome vs process — where does the credit go?</h2>
        <p>
          Everything so far rewards <em>completed</em> responses. For a
          one-line answer that's fine; the reward attaches to the right
          place. For a multi-step chain of thought — math, planning, code
          with intermediate steps — outcome rewards are a much weaker
          signal. A wrong final answer gives a negative reward to the whole
          trace, including any steps that were actually correct. The
          gradient knows something went wrong but not <em>where</em>.
        </p>
        <p>
          <strong>Process reward models</strong> (PRMs) score each step
          independently. Where outcome RMs are trained on pairwise
          preferences, PRM training data is <em>per-step labels</em> — each
          step in a chain tagged correct or incorrect — and the natural loss
          is binary cross-entropy at every step:
        </p>
        <div className="bhf-math-block">
          <Katex
            block
            tex={String.raw`\mathcal{L}_{\mathrm{PRM}}(\varphi) \;=\; -\sum_t \Big[\, y_t\,\log\sigma\!\big(r_\varphi(x, s_{\leq t})\big) \;+\; (1 - y_t)\,\log\!\big(1 - \sigma(r_\varphi(x, s_{\leq t}))\big)\,\Big]`}
          />
        </div>
        <p>
          Where the labels <Katex tex="y_t \in \{0, 1\}" /> come from is the
          interesting bit.{' '}
          <a className="post-link" href="https://arxiv.org/abs/2211.14275" target="_blank" rel="noreferrer">
            Uesato et al. 2022
          </a>{' '}
          showed step-level supervision beats outcome-only on math reasoning;
          OpenAI's{' '}
          <a className="post-link" href="https://arxiv.org/abs/2305.20050" target="_blank" rel="noreferrer">
            Let's Verify Step by Step
          </a>{' '}
          (Lightman et al. 2023) hand-labelled 800k math steps to make the
          case at scale. Click any step below for the PRM's critique.
        </p>
        <SectionProcessRewards />
        <p>
          The policy-training loss — PPO's clipped surrogate, GRPO's group
          baseline — <em>doesn't change shape</em> when you swap an outcome
          RM for a PRM. The only thing that changes is that <Katex tex="r_t" />{' '}
          now has a real value at every step instead of only at the last
          token, which sharpens the advantage estimate everywhere along the
          trajectory. Process rewards are expensive to collect, but they fix
          the credit-assignment problem cleanly.
        </p>

        <h2 className="reveal">RLVR — when the reward doesn't need to be learned</h2>
        <p>
          Every method so far has assumed the reward signal{' '}
          <Katex tex="r_\varphi" /> comes from something <em>learned</em> —
          from human preferences, a written constitution, or per-step labels.
          For one large and important class of tasks, that assumption is
          unnecessary: the reward is <em>already available</em> from a
          deterministic grader.
        </p>
        <ul className="bhf-list">
          <li>Math problems → check the numerical answer.</li>
          <li>Code → run the unit tests.</li>
          <li>Formal proofs → run the proof checker.</li>
          <li>Anything else with a programmatic correctness criterion.</li>
        </ul>
        <p>
          This is <strong>RLVR — RL with Verifiable Rewards</strong>. The
          training loop is exactly GRPO from{' '}
          <a className="post-link" href="#/blog/beyond-ppo">Part 2</a>, with
          the learned <Katex tex="r_\varphi" /> swapped for a verifier that
          returns:
        </p>
        <div className="bhf-math-block">
          <Katex
            block
            tex={String.raw`r_i \;=\; \mathbb{1}[\,\mathrm{verify}(y_i)\,] \;\in\; \{0, 1\}, \qquad \hat{A}_i \;=\; \frac{r_i - \mathrm{mean}(r_{1..G})}{\mathrm{std}(r_{1..G})}`}
          />
        </div>
        <p>
          Sample <Katex tex="G" /> completions, grade each, compute
          group-relative advantages, take the clipped-surrogate step. No
          reward model. No preference data. No human labelers. The only
          training data needed is{' '}
          <Katex tex="(\text{problem}, \text{correct answer})" /> pairs,
          which exist for huge libraries of math and code problems already.
          The widget below makes the contrast concrete.
        </p>
        <SectionRLVR />
        <p>
          With a verifier as <Katex tex="r" />, there is no proxy gap — the
          reward <em>is</em> the truth. Reward-hacking becomes impossible. In
          practice this means the KL penalty <Katex tex="\beta" /> can be set
          much lower (and in some recipes essentially to zero) without the
          policy diverging into gibberish — the leash existed to stop the
          policy from finding cracks in the proxy, and there are no cracks
          to find.
        </p>
        <p>
          RLVR is the recipe behind the recent generation of reasoning
          models.{' '}
          <a className="post-link" href="https://arxiv.org/abs/2402.03300" target="_blank" rel="noreferrer">
            DeepSeekMath
          </a>{' '}
          (Shao et al. 2024) introduced the GRPO+verifier combination for
          math; <a className="post-link" href="https://arxiv.org/abs/2501.12948" target="_blank" rel="noreferrer">
            DeepSeek-R1
          </a>{' '}
          (Guo et al. 2025) scaled it into a full reasoning training pipeline
          — pure RL from a base model, with verifiable math and code rewards,
          producing a chain-of-thought policy competitive with OpenAI's
          o-series.
        </p>

        <InfoBox title="Not a wholesale replacement">
          <p>
            RLVR doesn't replace RLHF — it complements it. There is no
            verifier for <em>"is this email well-written,"</em>{' '}
            <em>"is this response helpful,"</em> or <em>"does this stay
            within the constitution."</em> Production pipelines tend to be
            hybrid: cold-start RL on verifiable tasks (math, code) for sharp
            reasoning gains, then a separate RLHF / RLAIF pass on preference
            and constitution data for general helpfulness and safety.
          </p>
        </InfoBox>

        <h2 className="reveal">Three doors</h2>
        <p>
          Each of these replaces "more human labels" with a different
          resource. RLAIF replaces labellers with <em>compute and a written
          policy</em> — cheap, auditable, easy to iterate. Process rewards
          replace coarse trajectory-level labels with <em>step-level
          attribution</em> — expensive to collect but the only honest signal
          for long chain-of-thought reasoning. RLVR replaces the learned
          reward model with a <em>deterministic checker</em> — perfect
          fidelity on the tasks where one exists, nothing on the tasks where
          it doesn't.
        </p>
        <p>
          The progression across this series traces a single arc.{' '}
          <a className="post-link" href="#/blog/rlhf-and-ppo">Part 1</a>{' '}
          set up the canonical SFT → RM → PPO pipeline.{' '}
          <a className="post-link" href="#/blog/beyond-ppo">Part 2</a>{' '}
          stripped away parts of that machinery — DPO drops the reward
          model, GRPO drops the value net, DAPO patches the rough edges.
          This part attacks the remaining bottleneck: the labels themselves.
          What you choose depends on what you have. Plenty of preference
          data and a tight policy goal: RLHF or RLAIF. Long reasoning chains
          where credit assignment matters: process rewards. A deterministic
          checker and lots of compute: RLVR. They compose — most current
          frontier-model training pipelines use all three.
        </p>

        <h2 className="reveal">References</h2>
        <div className="post-refs">
          <div className="head ref-cite">Cite</div>
          <div className="head ref-title">Title</div>
          <div className="head ref-link">Link</div>

          <div className="ref-cite">Bai et al. 2022</div>
          <div>
            <div className="ref-title">
              <a href="https://arxiv.org/abs/2212.08073" target="_blank" rel="noreferrer">Constitutional AI: Harmlessness from AI Feedback</a>
            </div>
            <div className="ref-note">AI labeler + written constitution replacing human preference labels.</div>
          </div>
          <div className="ref-link"><a href="https://arxiv.org/abs/2212.08073" target="_blank" rel="noreferrer">arxiv.org/abs/2212.08073</a></div>

          <div className="ref-cite">Lee et al. 2023</div>
          <div>
            <div className="ref-title">
              <a href="https://arxiv.org/abs/2309.00267" target="_blank" rel="noreferrer">RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback</a>
            </div>
            <div className="ref-note">AI-labelled preferences perform on par with human ones across tasks.</div>
          </div>
          <div className="ref-link"><a href="https://arxiv.org/abs/2309.00267" target="_blank" rel="noreferrer">arxiv.org/abs/2309.00267</a></div>

          <div className="ref-cite">Uesato et al. 2022</div>
          <div>
            <div className="ref-title">
              <a href="https://arxiv.org/abs/2211.14275" target="_blank" rel="noreferrer">Solving math word problems with process- and outcome-based feedback</a>
            </div>
            <div className="ref-note">First systematic comparison of ORM vs PRM on reasoning.</div>
          </div>
          <div className="ref-link"><a href="https://arxiv.org/abs/2211.14275" target="_blank" rel="noreferrer">arxiv.org/abs/2211.14275</a></div>

          <div className="ref-cite">Lightman et al. 2023</div>
          <div>
            <div className="ref-title">
              <a href="https://arxiv.org/abs/2305.20050" target="_blank" rel="noreferrer">Let's Verify Step by Step</a>
            </div>
            <div className="ref-note">PRM800K — 800k hand-labelled math steps; PRMs beat outcome RMs at scale.</div>
          </div>
          <div className="ref-link"><a href="https://arxiv.org/abs/2305.20050" target="_blank" rel="noreferrer">arxiv.org/abs/2305.20050</a></div>

          <div className="ref-cite">Shao et al. 2024</div>
          <div>
            <div className="ref-title">
              <a href="https://arxiv.org/abs/2402.03300" target="_blank" rel="noreferrer">DeepSeekMath: Pushing the Limits of Mathematical Reasoning</a>
            </div>
            <div className="ref-note">GRPO + verifiable math rewards — the template for RLVR.</div>
          </div>
          <div className="ref-link"><a href="https://arxiv.org/abs/2402.03300" target="_blank" rel="noreferrer">arxiv.org/abs/2402.03300</a></div>

          <div className="ref-cite">Guo et al. 2025</div>
          <div>
            <div className="ref-title">
              <a href="https://arxiv.org/abs/2501.12948" target="_blank" rel="noreferrer">DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning</a>
            </div>
            <div className="ref-note">Pure-RL reasoning training from a base model with verifiable rewards.</div>
          </div>
          <div className="ref-link"><a href="https://arxiv.org/abs/2501.12948" target="_blank" rel="noreferrer">arxiv.org/abs/2501.12948</a></div>
        </div>

        <footer className="post-footer">
          <p>
            <strong>Part 3 of Aligning LMs</strong> · Previous:{' '}
            <a className="post-link" href="#/blog/beyond-ppo">Beyond PPO (DPO, GRPO, DAPO)</a>
            {' · '}Start of the series:{' '}
            <a className="post-link" href="#/blog/rlhf-and-ppo">RLHF and PPO</a>.
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
