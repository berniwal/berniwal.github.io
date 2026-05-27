// src/posts/SymbolicRegressionLlmTransfer.jsx
// Symbolic Regression · Part 2 of 2 — does the ranking transfer to an LLM?
// Ported from VisualizingSymbolicRegression.jsx (LLM-transfer half) into the
// post-2026 chrome. Prose preserved where the source had it; only chrome
// migrated.
// TAGS FOR REGISTRATION: ['symbolic-regression', 'llm', 'self-improvement']
// EXCERPT: Swap the numpy RNN proposer for an LLM behind the same ask/tell seam. Same task, same verifier, same budget — does the Part-1 ranking survive?
import React, { useEffect, useRef, useState } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneLight } from 'react-syntax-highlighter/dist/esm/styles/prism';
import './PostChrome.css';
import './SymbolicRegressionLlmTransfer.css';

/* ============================ KaTeX (CDN global) ============================ */
function Katex({ tex, block = false }) {
  const ref = useRef(null);
  useEffect(() => {
    let cancelled = false;
    const render = () => {
      if (cancelled || !ref.current) return;
      if (window.katex) {
        try {
          window.katex.render(tex, ref.current, { displayMode: block, throwOnError: false });
        } catch { if (ref.current) ref.current.textContent = tex; }
      } else { setTimeout(render, 60); }
    };
    render();
    return () => { cancelled = true; };
  }, [tex, block]);
  return <span ref={ref} className={block ? 'srl-math-block' : 'srl-math-inline'} />;
}

/* ============================ data hook ============================ */
const DATA_BASE = `${process.env.PUBLIC_URL || ''}/data/symbolic-regression`;

function useJson(url) {
  const [data, setData] = useState(null);
  useEffect(() => {
    let alive = true;
    fetch(url)
      .then((r) => { if (!r.ok) throw new Error(`${r.status}`); return r.json(); })
      .then((d) => { if (alive) setData(d); })
      .catch(() => { if (alive) setData(false); });
    return () => { alive = false; };
  }, [url]);
  return data;
}

/* ============================ Layer-1 transfer widget ============================ */
const L1_ARM_ORDER = ['risk', 'best_of_n', 'entropic', 'evolution', 'greedy'];

function Layer1Transfer() {
  const data = useJson(`${DATA_BASE}/layer1.json`);
  const [t, setT] = useState('harder');
  if (data === false) {
    return <div className="viz-panel"><p className="srl-note">Could not load Layer-1 data.</p></div>;
  }
  if (!data) {
    return <div className="viz-panel"><p className="srl-note">Loading…</p></div>;
  }
  const targets = data.targets || {};
  const tgt = targets[t];
  const arms = (tgt && tgt.arms) || {};
  const order = L1_ARM_ORDER.filter((a) => arms[a]);
  const avail = ['easy', 'medium', 'harder'].filter((k) => targets[k]);

  return (
    <div className="viz-panel">
      <div className="srl-controls">
        <label className="srl-control-label">benchmark
          <div className="srl-tabs">
            {avail.map((k) => (
              <button
                key={k}
                type="button"
                className={`srl-tab${t === k ? ' active' : ''}`}
                onClick={() => setT(k)}
              >
                {k}
              </button>
            ))}
          </div>
        </label>
        <span className="srl-note srl-tag-right">Qwen2.5-0.5B · GRPO + LoRA</span>
      </div>
      <table className="srl-results">
        <thead>
          <tr><th>arm</th><th>numeric recovery</th><th>symbolic recovery</th><th>mean best</th></tr>
        </thead>
        <tbody>
          {order.map((a) => {
            const arm = arms[a];
            return (
              <tr key={a}>
                <td><span className="srl-swatch" style={{ background: arm.color }} /> {arm.label}</td>
                <td><b>{arm.numeric_solved}/{arm.seeds}</b></td>
                <td>{arm.symbolic_solved}/{arm.seeds}</td>
                <td>{arm.mean_best != null ? arm.mean_best.toFixed(3) : '—'}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="srl-note srl-note-block">
        {t === 'easy'
          ? 'On easy every arm recovers the exact closed form on every seed — x²+1 is in the model\'s prior, no objective needed.'
          : t === 'medium'
          ? 'On medium every arm recovers numerically; symbolic recovery is also near-perfect (the entropic arm misses one seed). The methods only separate on harder.'
          : 'On harder every arm gets the curve roughly right, but exact symbolic recovery nearly vanishes — best-of-N lands it once in five, the rest never.'}
      </p>
    </div>
  );
}

/* ============================ Run-it-yourself code blocks ============================ */
const INSTALL_CMD = `git clone https://github.com/berniwal/berniwal.github.io
cd berniwal.github.io/experiments/self-improvement-arena
pip install -e ".[app]"           # core + Streamlit visualiser (numpy only)
streamlit run streamlit_app.py
# (add ",layer1" to the extras on Apple Silicon for the MLX LoRA arms)`;

function CodeDetails({ label, code, lang = 'bash' }) {
  return (
    <details className="post-code-details">
      <summary>
        <span className="post-code-summary-icon" aria-hidden="true">›</span>
        <span>{label}</span>
        <span className="post-code-summary-hint">click to expand</span>
      </summary>
      <SyntaxHighlighter
        style={oneLight}
        language={lang}
        PreTag="div"
        customStyle={{
          borderRadius: 10, overflow: 'auto', padding: '16px 18px',
          background: '#f1f5f9', border: '1px solid #e2e8f0',
          fontSize: 13.5, lineHeight: 1.55, margin: '12px 0 0',
        }}
        codeTagProps={{ style: { background: 'transparent', textShadow: 'none' } }}
      >
        {code}
      </SyntaxHighlighter>
    </details>
  );
}

/* ============================ the post ============================ */
export default function SymbolicRegressionLlmTransfer() {
  useEffect(() => {
    const prev = document.title;
    document.title = 'Does the Ranking Transfer to an LLM? — Bernhard Walser';
    return () => { document.title = prev; };
  }, []);

  return (
    <article className="post-2026 srl-post">
      <div className="post-wrap">
        <div className="post-hero">
          <div className="post-series-tag">
            <span className="post-live-dot" aria-hidden="true" />
            Symbolic Regression · Part 2 of 2
          </div>
          <h1>Does the Ranking Transfer to an LLM?</h1>
          <p className="post-lede">
            In <a className="post-link" href="#/blog/symbolic-regression-arena">Part 1</a>{' '}
            we raced four proposers on Nguyen — evolution, greedy RL, risk-seeking RL, an
            entropic variant — under one shared verifier and one shared budget. A stable
            ranking emerged: risk-seeking on top, evolution close behind, greedy collapsing
            onto a single attractor. Now we swap the small numpy RNN for a real language
            model behind the same <code>ask()</code>/<code>tell()</code> seam, and ask
            whether the same ranking survives.
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

        <h2 className="reveal">The four proposers, side by side</h2>
        <p>
          The seam is small on purpose. The verifier scores any expression tree the same way;
          only what generates the tree differs. We line up four cards: two Layer-0 baselines
          (numpy GP, numpy RNN) and their Layer-1 LLM counterparts. The gradient-based row
          shares the exact <Katex tex="\sum_i w_i \log \pi(\tau_i)" /> objective — only the
          policy <Katex tex="\pi" /> and the way we tokenise <Katex tex="\tau_i" /> change.
          The evolutionary row shares the high-level idea (best-so-far drives the next batch)
          but the LLM version has no gradient step at all — exactly the contrast AlphaEvolve
          and FunSearch are built on.
        </p>

        <div className="viz-panel srl-fourcol">
          <div className="srl-col srl-col-l0">
            <div className="srl-col-h"><span className="srl-col-tag">Layer 0</span><strong>Evolution (GP)</strong></div>
            <p><strong>Proposer.</strong> Population of 200 expression trees, evolved with tournament selection + subtree crossover + subtree mutation.</p>
            <p><strong>Search space.</strong> Trees built from <Katex tex="\{x, +, -, \times, \div, \sin, \cos, \mathrm{const}\}" />.</p>
            <p><strong>Optimisation.</strong> None — selection pressure on the population, no gradients.</p>
          </div>
          <div className="srl-col srl-col-l1">
            <div className="srl-col-h"><span className="srl-col-tag">Layer 1</span><strong>Evolution (LLM)</strong></div>
            <p><strong>Proposer.</strong> Frozen LLM prompted each round; the K best formulas found so far are injected into the prompt as "do better than these" (AlphaEvolve / FunSearch).</p>
            <p><strong>Search space.</strong> The LLM's full vocabulary (≈150 k tokens) → free-form formula text → parsed back into the same Node tree the verifier accepts.</p>
            <p><strong>Optimisation.</strong> None — the LLM weights are frozen; selection happens implicitly through the archive in the prompt.</p>
          </div>
          <div className="srl-col srl-col-l0">
            <div className="srl-col-h"><span className="srl-col-tag">Layer 0</span><strong>RNN (policy gradient)</strong></div>
            <p><strong>Proposer.</strong> 32-hidden numpy RNN, samples expression tokens autoregressively under the four constraints; ~5 k parameters.</p>
            <p><strong>Search space.</strong> Same fixed grammar as GP, 10 tokens, structured prefix decoding.</p>
            <p><strong>Optimisation.</strong> Policy gradient — for each batch compute the advantage <Katex tex="w_i" /> (greedy / risk-seeking / entropic) and Adam-step on <Katex tex="\mathcal{J} = \sum_i w_i \log \pi(\tau_i)" />.</p>
          </div>
          <div className="srl-col srl-col-l1">
            <div className="srl-col-h"><span className="srl-col-tag">Layer 1</span><strong>LLM (LoRA + GRPO)</strong></div>
            <p><strong>Proposer.</strong> Qwen2.5-0.5B-Instruct + LoRA adapters on the attention projections. Samples free-form formula strings; we parse them back into a Node tree for the verifier.</p>
            <p><strong>Search space.</strong> Full LLM vocabulary (≈150 k tokens) instead of a 10-token grammar.</p>
            <p><strong>Optimisation.</strong> <em>The same advantage formula</em>: <code>greedy_weights</code>, <code>quantile_weights</code>, <code>entropic_weights</code> from <code>sia.objectives</code> are imported by both proposers. Adam steps on the LoRA params via PyTorch GRPO.</p>
          </div>
        </div>
        <p className="srl-caption">
          Budget stays counted in <em>verifier calls</em>, not in tokens or wall-clock. A
          large LLM proposer cannot win by simply evaluating more candidates — it gets the
          same number of seats at the verifier as the RNN.
        </p>

        <h2 className="reveal">What the LLM sees</h2>
        <p>
          For the gradient-based arms (greedy / risk / entropic / best-of-N) we hand the LLM
          a plain prompt and parse its single-line reply. For the medium target{' '}
          <Katex tex="x^2 + \sin(x)" /> with the const-placeholder mode on, the prompt is
          literally:
        </p>
        <div className="srl-prompt-wrap">
          <div className="srl-prompt-label">
            <span>prompt sent to the LLM</span>
            <span>medium · const-placeholder mode</span>
          </div>
          <pre className="srl-prompt">{`You are doing symbolic regression. Find a formula y = f(x) that fits the data.
Allowed: the variable x, operators + - * /, the functions sin and cos, and the
constant placeholder C. Write C in place of every numeric constant or coefficient
-- for example write C*x + C instead of 2.5*x + 1.3. Each C's value is chosen
automatically.
The data may be nonlinear or periodic -- consider terms like x*x, x*x*x, or
sin/cos, not just straight lines.
Reply with ONLY the formula for f(x) on a single line -- no words, no 'y =', no
code fences.

Data points:
  x = -0.900   y = -0.027
  x = -0.700   y = -0.155
  x = -0.500   y = -0.229
  x = -0.300   y = -0.205
  x = +0.000   y = +0.000
  x = +0.300   y = +0.385
  x = +0.500   y = +0.729
  x = +0.700   y = +1.134
  x = +0.900   y = +1.594

New formula for f(x):`}</pre>
        </div>
        <p>
          A correct reply is a single line like <code>x*x + sin(x)</code>, which the verifier
          parses into a Node tree and scores. With const-placeholder mode the LLM is
          encouraged to write <code>C*x*x + C*sin(x)</code> instead; BFGS then fits each{' '}
          <code>C</code> to minimise the error on the data, and the verifier scores the
          constant-substituted tree. The <strong>evolutionary arm</strong> appends one extra
          block before "New formula":
        </p>
        <div className="srl-prompt-wrap">
          <div className="srl-prompt-label">
            <span>extra block — evolution arm only</span>
            <span>archive injected each round</span>
          </div>
          <pre className="srl-prompt">{`Best formulas found so far -- propose a DIFFERENT formula that fits better than these:
  f(x) = sin(x)             (score 0.412)
  f(x) = x*x + C            (score 0.821)
  f(x) = C*x + sin(x)       (score 0.793)`}</pre>
        </div>
        <p>
          ... and that's the whole "AlphaEvolve-style" arm — frozen weights, archive in the
          prompt, do better. The gradient arms instead drop the archive block and train the
          LoRA adapters via GRPO on the same per-sample advantage <Katex tex="w_i" /> formulas
          the Layer-0 RNN uses. Every arm gets the same number of verifier calls per seed;
          the LLM's higher per-sample cost shows up in wall-clock, not in budget.
        </p>

        <h2 className="reveal">The result</h2>
        <p>
          Five seeds per arm, Qwen2.5-0.5B with LoRA on the attention projections, the same
          numeric budget as the Layer-0 sweep. Switch benchmarks below:
        </p>
        <Layer1Transfer />
        <p>
          The ranking <em>survives but only where it matters</em>. On <code>easy</code> every
          arm recovers the exact closed form on every seed — the LLM has <Katex tex="x^2 + 1" />{' '}
          in its prior and no objective is needed. On <code>medium</code> every arm still
          recovers numerically and nearly always symbolically. The methods only{' '}
          <em>separate</em> on <code>harder</code>, and there the Layer-0 order returns:
          risk-seeking is the most reliable (5/5 numeric), best-of-N and entropic are in the
          middle, greedy and evolution trail.
        </p>
        <p>
          But notice the ceiling — exact symbolic recovery on <code>harder</code> nearly
          disappears even though the curves fit. The RNN, whose vocabulary <em>is</em> the
          task grammar, hits the exact expression; the LLM samples from a 150 k-token
          vocabulary, so its search space is vastly wider and it settles for{' '}
          numerically-close. The objective dynamic transfers; the broad prior makes{' '}
          <em>exact</em> recovery harder, not easier. Five seeds is a small N — read this as
          a <em>direction</em>, not a verdict.
        </p>

        <h2 className="reveal">Run it yourself</h2>
        <p>
          Everything in both posts — the four Layer-0 proposers, the LoRA + GRPO Layer-1 LLM
          arm, the verifier, the configs, and the RunPod harness for the cloud sweeps — is in
          the same repo as this blog, under{' '}
          <a className="post-link" href="https://github.com/berniwal/berniwal.github.io/tree/main/experiments/self-improvement-arena" target="_blank" rel="noreferrer">
            experiments/self-improvement-arena
          </a>. The Layer-0 stack is pure numpy (runs anywhere); the LLM stack is PyTorch +
          transformers + peft (GPU); an on-device MLX version of the LLM arm sits alongside
          for Apple Silicon.
        </p>
        <p>
          A <strong>Streamlit visualiser</strong> ships in the same repo — same engine as the
          widgets, but with per-tab controls and the raw LLM prompt / responses on the side.
          Useful for poking at a specific seed, changing a hyperparameter live, or watching
          the policy entropy crash in real time:
        </p>
        <CodeDetails label="Install and launch the visualiser" code={INSTALL_CMD} lang="bash" />
        <p>
          Everything is seeded; reproducing any number across the two posts is one config and
          one command. The full result tables live under <code>results/</code>; the canonical{' '}
          <code>replay.json</code> bakes that power these widgets are committed.
        </p>

        <h2 className="reveal">Wrap-up</h2>
        <p>
          The point of the two-layer split: the <em>objective</em> — how a batch of rewards
          becomes a per-sample weight — is what carries the ranking. Swap the proposer's
          architecture and the order largely holds. Swap the objective and it doesn't. That's
          the knob worth tuning when you scale up to a real LLM.
        </p>

        <h2 className="reveal">References</h2>
        <div className="post-refs">
          <div className="head ref-cite">Cite</div>
          <div className="head ref-title">Title</div>
          <div className="head ref-link">Link</div>

          <div className="ref-cite">Shao et al. 2024</div>
          <div>
            <div className="ref-title">
              <a href="https://arxiv.org/abs/2402.03300" target="_blank" rel="noreferrer">DeepSeekMath / GRPO</a>
            </div>
            <div className="ref-note">Group-relative policy optimisation — the GRPO update used for the LoRA arm.</div>
          </div>
          <div className="ref-link"><a href="https://arxiv.org/abs/2402.03300" target="_blank" rel="noreferrer">arxiv 2402.03300</a></div>

          <div className="ref-cite">Petersen et al. 2021</div>
          <div>
            <div className="ref-title">
              <a href="https://arxiv.org/abs/1912.04871" target="_blank" rel="noreferrer">Deep Symbolic Regression</a>
            </div>
            <div className="ref-note">Risk-seeking policy gradient: the Layer-0 reference and the source of the const-placeholder trick.</div>
          </div>
          <div className="ref-link"><a href="https://arxiv.org/abs/1912.04871" target="_blank" rel="noreferrer">arxiv 1912.04871</a></div>

          <div className="ref-cite">Novikov et al. 2025</div>
          <div>
            <div className="ref-title">
              <a href="https://arxiv.org/abs/2506.13131" target="_blank" rel="noreferrer">AlphaEvolve — evolutionary coding with LLMs</a>
            </div>
            <div className="ref-note">In-context program-database evolution: the template for our LLM evolution arm.</div>
          </div>
          <div className="ref-link"><a href="https://arxiv.org/abs/2506.13131" target="_blank" rel="noreferrer">arxiv 2506.13131</a></div>

          <div className="ref-cite">Romera-Paredes et al. 2023</div>
          <div>
            <div className="ref-title">
              <a href="https://www.nature.com/articles/s41586-023-06924-6" target="_blank" rel="noreferrer">FunSearch — mathematical discoveries with LLMs</a>
            </div>
            <div className="ref-note">Frozen-LLM search with an archive of best programs in the prompt.</div>
          </div>
          <div className="ref-link"><a href="https://www.nature.com/articles/s41586-023-06924-6" target="_blank" rel="noreferrer">nature</a></div>

          <div className="ref-cite">Hu et al. 2021</div>
          <div>
            <div className="ref-title">
              <a href="https://arxiv.org/abs/2106.09685" target="_blank" rel="noreferrer">LoRA — low-rank adaptation of large language models</a>
            </div>
            <div className="ref-note">The cheap-fine-tune trick that makes per-arm Qwen training fit on a single GPU.</div>
          </div>
          <div className="ref-link"><a href="https://arxiv.org/abs/2106.09685" target="_blank" rel="noreferrer">arxiv 2106.09685</a></div>

          <div className="ref-cite">Brown et al. 2020</div>
          <div>
            <div className="ref-title">
              <a href="https://arxiv.org/abs/2005.14165" target="_blank" rel="noreferrer">Language Models are Few-Shot Learners</a>
            </div>
            <div className="ref-note">The original few-shot prompting work — relevant to how the data points are placed inline in the prompt.</div>
          </div>
          <div className="ref-link"><a href="https://arxiv.org/abs/2005.14165" target="_blank" rel="noreferrer">arxiv 2005.14165</a></div>

          <div className="ref-cite">this post</div>
          <div>
            <div className="ref-title">
              <a href="https://github.com/berniwal/berniwal.github.io/tree/main/experiments/self-improvement-arena" target="_blank" rel="noreferrer">berniwal.github.io / experiments / self-improvement-arena</a>
            </div>
            <div className="ref-note">All proposers, configs, RunPod harness, Streamlit visualiser.</div>
          </div>
          <div className="ref-link"><a href="https://github.com/berniwal/berniwal.github.io/tree/main/experiments/self-improvement-arena" target="_blank" rel="noreferrer">github</a></div>

          <div className="ref-cite">Earlier posts</div>
          <div>
            <div className="ref-title">
              <a href="#/blog/symbolic-regression-arena">Racing Search Algorithms on Symbolic Regression</a>
            </div>
            <div className="ref-note">Part 1 of this series — the Layer-0 arena and the Nguyen result.</div>
          </div>
          <div className="ref-link"><a href="#/blog/symbolic-regression-arena">post</a></div>
        </div>

        <footer className="post-footer">
          <p>
            <strong>Part 2 of Symbolic Regression</strong> · Previous:{' '}
            <a className="post-link" href="#/blog/symbolic-regression-arena">Racing Search Algorithms on Symbolic Regression</a>
            {' · '}Start of the series:{' '}
            <a className="post-link" href="#/blog/symbolic-regression-arena">Racing Search Algorithms on Symbolic Regression</a>.
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
