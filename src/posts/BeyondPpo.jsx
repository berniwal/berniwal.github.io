// src/posts/BeyondPpo.jsx
// Aligning LMs · Part 2 of 3 — Beyond PPO: DPO, GRPO, DAPO.
// TAGS FOR REGISTRATION: ['rlhf', 'dpo', 'grpo']
// EXCERPT: PPO works, but it needs four networks in memory and is hard to tune. DPO drops the reward model, GRPO drops the value net, and DAPO patches GRPO's failure modes — the post-PPO landscape, side by side.
import React, { useEffect, useRef, useState } from 'react';
import usePageMeta from '../usePageMeta';
import './PostChrome.css';
import './BeyondPpo.css';

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
      className={`${block ? 'bp-math-block' : 'bp-math-inline'} ${className}`}
    />
  );
}

/* ============================================================
   InfoBox — reusable aside callout
   ============================================================ */
function InfoBox({ title, children }) {
  return (
    <aside className="bp-aside">
      <h4 className="bp-aside-head">
        <span className="bp-aside-tag">i</span>
        {title}
      </h4>
      <div className="bp-aside-body">{children}</div>
    </aside>
  );
}

/* ============================================================
   §Comparison widget — PPO / DPO / GRPO side by side
   ============================================================ */
const METHOD_BOXES = [
  { id: 'policy',    name: 'Policy π_θ',     note: 'the model we ship' },
  { id: 'reference', name: 'Reference π_ref', note: 'frozen SFT model' },
  { id: 'reward',    name: 'Reward model r_φ', note: 'learned from prefs' },
  { id: 'value',     name: 'Value net V_ψ',    note: 'variance reduction' },
];

const METHODS = {
  ppo: {
    label: 'PPO (InstructGPT)',
    active: { policy: true, reference: true, reward: true, value: true },
    formula: String.raw`
      \mathcal{L}_{\mathrm{PPO}}(\theta)
      \;=\; -\,\mathbb{E}\!\left[\,
        \min\!\big(\,\rho_t\,\hat{A}_t,\; \mathrm{clip}(\rho_t,\,1\!-\!\epsilon,\,1\!+\!\epsilon)\,\hat{A}_t\,\big)
      \right]
      \;+\; \beta \cdot \mathrm{KL}\!\big(\pi_\theta\,\|\,\pi_{\mathrm{ref}}\big)
    `,
    formulaSub: String.raw`\rho_t \;=\; \tfrac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)}, \qquad \hat{A}_t \;\approx\; r_\varphi(x, y) - V_\psi(s_t)`,
    cite: { paper: 'InstructGPT (Ouyang et al. 2022)', url: 'https://arxiv.org/abs/2203.02155' },
    summary: (
      <>
        The classical pipeline. Sample completions, score them with{' '}
        <Katex tex="r_\varphi" />, advantage-weight the gradient using{' '}
        <Katex tex="V_\psi" /> for variance reduction, and clip the policy
        ratio so each update stays in a trust region. <strong>All four boxes
        live in GPU memory at once</strong> — expensive, but well-understood.
      </>
    ),
  },
  dpo: {
    label: 'DPO',
    active: { policy: true, reference: true, reward: false, value: false },
    formula: String.raw`
      \mathcal{L}_{\mathrm{DPO}}(\theta)
      \;=\; -\log \sigma\!\left(
        \beta\,\log \tfrac{\pi_\theta(y_w \mid x)}{\pi_{\mathrm{ref}}(y_w \mid x)}
        \;-\;
        \beta\,\log \tfrac{\pi_\theta(y_l \mid x)}{\pi_{\mathrm{ref}}(y_l \mid x)}
      \right)
    `,
    formulaSub: String.raw`(x, y_w, y_l) \sim \mathcal{D}_{\mathrm{pref}}, \quad y_w \succ y_l`,
    cite: { paper: 'DPO (Rafailov et al. 2023)', url: 'https://arxiv.org/abs/2305.18290' },
    summary: (
      <>
        PPO's optimum is closed-form in terms of the reward, and for a
        Bradley-Terry reward this gives a loss that depends only on log-prob
        ratios over preference pairs <Katex tex="(y_w \succ y_l)" />.{' '}
        <strong>No reward model, no rollouts, no value net</strong> — a single
        supervised-style fine-tune on preference data.
      </>
    ),
  },
  grpo: {
    label: 'GRPO (DeepSeek)',
    active: { policy: true, reference: true, reward: true, value: false },
    formula: String.raw`
      \mathcal{L}_{\mathrm{GRPO}}(\theta)
      \;=\; -\,\mathbb{E}\!\left[\, \tfrac{1}{G}\sum_{i=1}^{G}
        \min\!\big(\rho_i\,\hat{A}_i,\; \mathrm{clip}(\rho_i)\,\hat{A}_i\big)
      \right]
      \;+\; \beta \cdot \mathrm{KL}\!\big(\pi_\theta\,\|\,\pi_{\mathrm{ref}}\big)
    `,
    formulaSub: String.raw`\hat{A}_i \;=\; \tfrac{r_i \,-\, \mathrm{mean}(r_{1..G})}{\mathrm{std}(r_{1..G})}, \qquad \{y_1, \ldots, y_G\} \sim \pi_{\theta_{\mathrm{old}}}(\cdot\mid x)`,
    cite: { paper: 'DeepSeekMath / GRPO (Shao et al. 2024)', url: 'https://arxiv.org/abs/2402.03300' },
    summary: (
      <>
        Sample <Katex tex="G" /> completions per prompt and use the group's
        own statistics as the baseline — z-score each reward against its
        siblings instead of subtracting <Katex tex="V_\psi(s_t)" />.{' '}
        <strong>Drops the value net entirely</strong>, at the cost of needing{' '}
        <Katex tex="G" /> rollouts per prompt. The reward can be a learned RM
        or a verifiable scalar.
      </>
    ),
  },
};

function SectionMethodCompare() {
  const [method, setMethod] = useState('ppo');
  const m = METHODS[method];

  return (
    <div className="viz-panel bp-compare">
      <div className="bp-tabs" role="tablist">
        {Object.entries(METHODS).map(([id, info]) => (
          <button
            key={id}
            type="button"
            role="tab"
            className={`bp-tab ${method === id ? 'active' : ''}`}
            onClick={() => setMethod(id)}
          >
            {info.label}
          </button>
        ))}
      </div>

      <div className="bp-compare-caption">components in memory during training</div>
      <div className="bp-method-grid">
        {METHOD_BOXES.map((box) => {
          const isActive = m.active[box.id];
          return (
            <div
              key={box.id}
              className={`bp-method-box ${isActive ? 'active' : 'inactive'}`}
            >
              <div className="bp-box-tag">{isActive ? 'required' : 'not used'}</div>
              <div className="bp-box-name">{box.name}</div>
              <div className="bp-box-note">{box.note}</div>
            </div>
          );
        })}
      </div>

      <div className="bp-math-block">
        <Katex block tex={m.formula} />
      </div>
      <div className="bp-math-block bp-math-sub">
        <Katex block tex={m.formulaSub} />
      </div>

      <p className="bp-compare-summary">{m.summary}</p>
      <p className="bp-compare-source">
        Source:{' '}
        <a className="post-link" href={m.cite.url} target="_blank" rel="noreferrer">
          {m.cite.paper}
        </a>
      </p>
    </div>
  );
}

/* ============================================================
   Page
   ============================================================ */
export default function BeyondPpo() {
  usePageMeta({
    title: 'Beyond PPO: DPO, GRPO, DAPO',
    description: 'DPO drops the reward model, GRPO drops the value net, and DAPO patches GRPO\'s failure modes — the post-PPO landscape, side by side.',
    slug: 'beyond-ppo',
    publishedDate: '2026-04-09',
    keywords: ['DPO', 'GRPO', 'DAPO', 'RLHF', 'alignment'],
  });

  return (
    <article className="post-2026 bp-post">
      <div className="post-wrap">
        <div className="post-hero">
          <div className="post-series-tag">
            <span className="post-live-dot" aria-hidden="true" />
            Aligning LMs · Part 2 of 3
          </div>
          <h1>Beyond PPO: DPO, GRPO, DAPO</h1>
          <p className="post-lede">
            PPO works, but it keeps four networks in GPU memory and is famously
            sensitive to hyperparameters. The last two years of alignment
            research are mostly the same story told three ways: <em>do more
            with less</em>. Drop the reward model, drop the value net, patch
            the failure modes.
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
          A quick recap from{' '}
          <a className="post-link" href="/blog/rlhf-and-ppo">Part 1</a>: PPO
          aligns a language model by sampling completions from the policy,
          scoring them with a learned reward model, and pushing the policy
          toward higher reward inside a trust region. It works — InstructGPT,
          GPT-4, Claude, Gemini — but it carries four models in memory
          (policy, frozen reference, reward, value) and is notoriously fiddly
          to tune. Everything below is a way of keeping the alignment signal
          and throwing pieces of that machinery away.
        </p>

        <h2 className="reveal">DPO — drop the reward model</h2>
        <p>
          <a className="post-link" href="https://arxiv.org/abs/2305.18290" target="_blank" rel="noreferrer">DPO</a>{' '}
          (Rafailov et al. 2023) rests on a clean piece of math worth walking
          through. Start from the same PPO+KL objective from Part 1, but
          imagine maximising it <em>analytically</em> — solving for the
          policy <Katex tex="\pi" /> directly, rather than approximating with
          gradient descent. The optimal solution is closed-form:
        </p>
        <div className="bp-math-block">
          <Katex
            block
            tex={String.raw`
              \pi^*(y \mid x) \;=\; \tfrac{1}{Z(x)}\,\pi_{\mathrm{ref}}(y \mid x)\,\exp\!\big(r_\varphi(x, y)\,/\,\beta\big)
            `}
          />
        </div>
        <p>
          Read it as: <strong>start from the SFT prior, then tilt it by{' '}
          <Katex tex="\exp(r/\beta)" /></strong>. Responses with high reward
          get exponentially more probability mass;{' '}
          <Katex tex="Z(x)" /> is the partition function that normalises the
          result. Small <Katex tex="\beta" /> means strong tilt; large{' '}
          <Katex tex="\beta" /> means barely any tilt (optimum ≈ SFT).
        </p>
        <p>
          So far this is just a re-statement of PPO+KL. DPO's trick is to{' '}
          <em>invert</em> the relationship. Solving for <Katex tex="r" />:
        </p>
        <div className="bp-math-block">
          <Katex
            block
            tex={String.raw`
              r_\varphi(x, y) \;=\; \beta\,\log\!\tfrac{\pi^*(y \mid x)}{\pi_{\mathrm{ref}}(y \mid x)} \;+\; \beta\,\log Z(x)
            `}
          />
        </div>
        <p>
          Every reward function defines an optimal policy, and <em>every
          policy defines an implicit reward function</em>. Plug that
          expression into the Bradley-Terry preference loss — which only
          depends on <em>differences</em> of rewards on the same prompt — and{' '}
          <Katex tex="\beta\log Z(x)" /> cancels exactly. What's left is a
          loss expressed purely in policy log-ratios. <strong>No{' '}
          <Katex tex="r_\varphi" /> ever needs to be trained</strong>: the
          policy itself encodes the reward.
        </p>
        <p>
          The pipeline collapses to a single supervised-style fine-tune on
          preference pairs. Half the codebase, none of the RL-training
          stability headaches. The real catch: <strong>DPO is off-policy</strong>.
          It only sees the preference pairs <Katex tex="(y_w, y_l)" /> in the
          dataset, sampled by some other policy. PPO keeps re-rolling fresh
          completions from the live policy and grading them — if the policy
          drifts to a region the preference data doesn't cover, PPO keeps
          collecting new information; DPO is stuck. The community's response
          was <em>iterative DPO</em>: run DPO, sample from the new policy,
          collect fresh preferences over those samples, repeat. It recovers
          most of PPO's on-policy advantage without the value net.
        </p>

        <h2 className="reveal">GRPO — drop the value net</h2>
        <p>
          <a className="post-link" href="https://arxiv.org/abs/2402.03300" target="_blank" rel="noreferrer">GRPO</a>{' '}
          (Shao et al., DeepSeekMath 2024) takes the opposite tack: keep
          PPO's on-policy RL loop in full, but drop <Katex tex="V_\psi" />.
          The value net is the single most expensive piece of PPO — same
          size as the policy, fully trained, optimizer state and all. GRPO
          replaces it with a baseline that costs nothing to maintain. For
          each prompt, sample <Katex tex="G" /> completions in parallel,
          score them, and use the group's own mean and standard deviation
          as the baseline:
        </p>
        <div className="bp-math-block">
          <Katex
            block
            tex={String.raw`\hat{A}_i \;=\; \frac{r_i - \mathrm{mean}(r_{1..G})}{\mathrm{std}(r_{1..G})}, \qquad \{y_1, \ldots, y_G\} \sim \pi_{\theta_{\mathrm{old}}}(\cdot \mid x)`}
          />
        </div>
        <p>
          No learned value function — the variance reduction comes from{' '}
          <em>siblings on the same prompt</em>. The trade-off is a different
          flavour of compute: PPO does ~1 rollout per prompt and a
          forward+backward through <Katex tex="V_\psi" />; GRPO does{' '}
          <Katex tex="G" /> rollouts per prompt (typically 4–16) and no
          value net at all. On big models that asymmetry strongly favours
          GRPO — inference is cheaper and easier to parallelise across
          machines than the training stack.
        </p>
        <p>
          Two further consequences are why GRPO matters beyond memory
          savings. First, it plays cleanly with <em>verifiable</em> rewards
          — a unit-test pass-rate, a proof-checker, a math-grading script.
          Nothing in the objective requires <Katex tex="r_i" /> to come from
          a learned model; it can be 0 or 1 from a deterministic checker.
          Second, batched rollouts per prompt make the group a natural unit
          for "explore K branches in parallel" — the recipe behind
          DeepSeek-R1.
        </p>

        <h2 className="reveal">DAPO — production-scale refinements to GRPO</h2>
        <p>
          Once GRPO became the default for reasoning models, its failure
          modes showed up at scale. <a className="post-link" href="https://arxiv.org/abs/2503.14476" target="_blank" rel="noreferrer">DAPO</a>{' '}
          (Yu et al. 2025, ByteDance Seed) is the most influential of the
          patches — a recipe of four refinements that together stabilise
          long-CoT training on large open models, with a reported AIME-2024
          score of 50 on a 32B base using fully open data.
        </p>
        <p>
          The four refinements, each fixing a concrete pathology:
        </p>
        <ol className="bp-list">
          <li>
            <strong>Clip-Higher.</strong> PPO's symmetric clip{' '}
            <Katex tex="[1-\epsilon, 1+\epsilon]" /> caps how much a token's
            probability can <em>grow</em> as hard as how much it can shrink
            — which silently kills exploration once the policy gets
            confident. DAPO widens the upper bound to{' '}
            <Katex tex="1+\epsilon_{\mathrm{high}}" /> while keeping the
            lower bound, letting low-probability tokens be promoted more
            aggressively. Entropy stops collapsing.
          </li>
          <li>
            <strong>Dynamic Sampling.</strong> If every completion in a
            group is correct (or every one is wrong), the group has zero
            advantage variance and contributes no gradient — but still
            costs a full forward pass. DAPO oversamples prompts, filters
            out the all-correct and all-wrong groups, and only trains on
            the informative ones.
          </li>
          <li>
            <strong>Token-Level Policy Gradient Loss.</strong> Vanilla
            GRPO averages the loss per response, then per group. With long
            chain-of-thought outputs that means a 4000-token correct trace
            and a 200-token incorrect one contribute equally per token of
            actual content — long good reasoning gets under-rewarded.
            DAPO sums per token across the whole group, restoring
            proportional credit.
          </li>
          <li>
            <strong>Overlong Reward Shaping.</strong> A hard
            length-cutoff that zeroes the reward of overlong samples
            injects a lot of noise. DAPO replaces it with a smooth linear
            penalty that ramps up as the response approaches the limit,
            so the model has a usable gradient telling it to be shorter
            rather than a cliff.
          </li>
        </ol>
        <p>
          DAPO also drops the KL-to-reference term entirely — for
          long-horizon reasoning the policy is supposed to drift far from
          the SFT prior, and the KL penalty fights that on purpose. To be
          precise about lineage: DeepSeek-R1 itself uses GRPO. DAPO came
          later and is the recipe ByteDance used to train comparable
          reasoning models from open bases; several subsequent open-source
          reasoning training stacks adopt its refinements.
        </p>

        <h2 className="reveal">Comparison and tradeoffs</h2>
        <p>
          Toggle between the three below. The four boxes show which
          components each method actually needs in memory during training;
          the formula is the exact objective; the prose summarises the trick.
        </p>
        <SectionMethodCompare />

        <InfoBox title="Reading the table">
          <p>
            DPO is the cheapest at training time but most constrained by
            its dataset. GRPO is the most flexible — drop in any verifier,
            scale rollouts horizontally — at the cost of more inference
            compute. PPO is the safest default when you already have an
            RM you trust. DAPO is GRPO with the rough edges sanded off for
            long-CoT training.
          </p>
        </InfoBox>

        <h2 className="reveal">Where this leaves us</h2>
        <p>
          RL-from-something is now a mature space, not a single recipe.
          Each simplification trades one thing for another: DPO trades
          online exploration for stability and code simplicity; GRPO
          trades the value net for more rollout samples; DAPO patches
          GRPO's failure modes at the cost of more knobs. None of them
          changed where the <em>signal</em> comes from — they're all still
          trained on human (or human-proxy) preferences or on verifiable
          ground truth.
        </p>
        <p>
          That last point is the axis Part 3 attacks. If the bottleneck
          isn't the optimiser but the labels, what happens when you let an
          AI do the labelling, grade individual reasoning steps, or hand
          the whole thing over to a deterministic checker?
        </p>

        <h2 className="reveal">References</h2>
        <div className="post-refs">
          <div className="head ref-cite">Cite</div>
          <div className="head ref-title">Title</div>
          <div className="head ref-link">Link</div>

          <div className="ref-cite">Schulman et al. 2017</div>
          <div>
            <div className="ref-title">
              <a href="https://arxiv.org/abs/1707.06347" target="_blank" rel="noreferrer">Proximal Policy Optimization Algorithms</a>
            </div>
            <div className="ref-note">The PPO baseline that everything in this post simplifies.</div>
          </div>
          <div className="ref-link"><a href="https://arxiv.org/abs/1707.06347" target="_blank" rel="noreferrer">arxiv.org/abs/1707.06347</a></div>

          <div className="ref-cite">Rafailov et al. 2023</div>
          <div>
            <div className="ref-title">
              <a href="https://arxiv.org/abs/2305.18290" target="_blank" rel="noreferrer">Direct Preference Optimization</a>
            </div>
            <div className="ref-note">DPO — closed-form policy ⇒ no reward model, no rollouts.</div>
          </div>
          <div className="ref-link"><a href="https://arxiv.org/abs/2305.18290" target="_blank" rel="noreferrer">arxiv.org/abs/2305.18290</a></div>

          <div className="ref-cite">Shao et al. 2024</div>
          <div>
            <div className="ref-title">
              <a href="https://arxiv.org/abs/2402.03300" target="_blank" rel="noreferrer">DeepSeekMath: Pushing the Limits of Mathematical Reasoning</a>
            </div>
            <div className="ref-note">Introduces GRPO — group-mean baseline replaces the value net.</div>
          </div>
          <div className="ref-link"><a href="https://arxiv.org/abs/2402.03300" target="_blank" rel="noreferrer">arxiv.org/abs/2402.03300</a></div>

          <div className="ref-cite">Yu et al. 2025</div>
          <div>
            <div className="ref-title">
              <a href="https://arxiv.org/abs/2503.14476" target="_blank" rel="noreferrer">DAPO: An Open-Source LLM Reinforcement Learning System at Scale</a>
            </div>
            <div className="ref-note">Clip-Higher, Dynamic Sampling, Token-Level Loss, Overlong Reward Shaping.</div>
          </div>
          <div className="ref-link"><a href="https://arxiv.org/abs/2503.14476" target="_blank" rel="noreferrer">arxiv.org/abs/2503.14476</a></div>

          <div className="ref-cite">DeepSeek-AI 2025</div>
          <div>
            <div className="ref-title">
              <a href="https://arxiv.org/abs/2501.12948" target="_blank" rel="noreferrer">DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning</a>
            </div>
            <div className="ref-note">The reasoning-model recipe built on GRPO + verifiable rewards.</div>
          </div>
          <div className="ref-link"><a href="https://arxiv.org/abs/2501.12948" target="_blank" rel="noreferrer">arxiv.org/abs/2501.12948</a></div>
        </div>

        <footer className="post-footer">
          <p>
            <strong>Part 2 of Aligning LMs</strong> · Previous:{' '}
            <a className="post-link" href="/blog/rlhf-and-ppo">RLHF and PPO</a>
            {' · '}Next:{' '}
            <a className="post-link" href="/blog/beyond-human-feedback">Beyond Human Feedback (RLAIF, Process Rewards, RLVR)</a>.
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
