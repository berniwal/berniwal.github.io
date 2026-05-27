// src/posts/TestTimeTraining.jsx
// Self-Improvement · Part 5 of 6 — Test-Time Training: When the Model Updates Mid-Inference.
// Ported from VisualizingSelfImprovement.jsx §6.
// TAGS FOR REGISTRATION: ['self-improvement', 'ttt', 'test-time-training']
// EXCERPT: TTT-Discover updates the model's weights mid-inference, accumulates them within one problem, then resets — a peak-targeting entropic objective with PUCT search.
import React, { useEffect, useMemo, useRef, useState } from 'react';
import './PostChrome.css';
import './TestTimeTraining.css';

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
      className={`${block ? 'ttt-math-block' : 'ttt-math-inline'} ${className}`}
    />
  );
}

function InfoBox({ title, children }) {
  return (
    <aside className="ttt-aside">
      <h4 className="ttt-subhead">
        <span className="ttt-sub-tag" aria-hidden="true">i</span>
        {title}
      </h4>
      <div className="ttt-aside-body">{children}</div>
    </aside>
  );
}

function Counter({ label, value, sub }) {
  return (
    <div className="ttt-counter">
      <div className="ttt-counter-label">{label}</div>
      <div className="ttt-counter-value">{value}</div>
      {sub && <div className="ttt-counter-sub">{sub}</div>}
    </div>
  );
}

/* =========================================================
   PUCT tree helpers (binary tree, depth 3)
   ========================================================= */

function buildBinaryTree(depth = 3) {
  const nodes = [];
  const edges = [];
  let nextId = 0;
  const root = { id: nextId++, parent: null, depth: 0, children: [] };
  nodes.push(root);
  let layer = [root];
  for (let d = 1; d <= depth; d++) {
    const nextLayer = [];
    for (const p of layer) {
      for (let k = 0; k < 2; k++) {
        const child = { id: nextId++, parent: p.id, depth: d, children: [] };
        p.children.push(child.id);
        nodes.push(child);
        edges.push({ from: p.id, to: child.id });
        nextLayer.push(child);
      }
    }
    layer = nextLayer;
  }
  const leafIds = nodes.filter((n) => n.depth === depth).map((n) => n.id);
  return { nodes, edges, leafIds };
}

function layoutTree(treeStruct, width, height, padding = 24) {
  const depthGroups = {};
  for (const n of treeStruct.nodes) {
    if (!depthGroups[n.depth]) depthGroups[n.depth] = [];
    depthGroups[n.depth].push(n.id);
  }
  const layout = {};
  const maxDepth = Math.max(...treeStruct.nodes.map((n) => n.depth));
  for (let d = 0; d <= maxDepth; d++) {
    const ids = depthGroups[d];
    ids.forEach((id, i) => {
      const x = padding + ((i + 0.5) / ids.length) * (width - 2 * padding);
      const y = padding + (d / maxDepth) * (height - 2 * padding);
      layout[id] = { x, y };
    });
  }
  return layout;
}

function initMCTSState(treeStruct, leafRewards) {
  const stats = {};
  for (const n of treeStruct.nodes) {
    stats[n.id] = { n: 0, w: 0, q: 0 };
  }
  const P = {};
  for (const n of treeStruct.nodes) {
    if (n.children.length === 0) continue;
    n.children.forEach((cid) => { P[cid] = 1 / n.children.length; });
  }
  return { stats, P, iteration: 0, lastPath: [], leafRewards };
}

function puctScore(parentId, childId, state, c) {
  const parent = state.stats[parentId];
  const child = state.stats[childId];
  const N = parent.n;
  const p = state.P[childId];
  return child.q + c * p * Math.sqrt(1 + N) / (1 + child.n);
}

function bestChildByPUCT(parentId, treeStruct, state, c) {
  const parent = treeStruct.nodes[parentId];
  const scores = parent.children.map((cid) => ({
    id: cid,
    s: puctScore(parentId, cid, state, c),
  }));
  scores.sort((a, b) => b.s - a.s);
  return scores[0].id;
}

function mctsStep(state, treeStruct, c = 1.4) {
  const newStats = {};
  for (const k of Object.keys(state.stats)) newStats[k] = { ...state.stats[k] };
  const path = [];
  let cur = 0;
  path.push(cur);
  while (treeStruct.nodes[cur].children.length > 0) {
    cur = bestChildByPUCT(cur, treeStruct, { ...state, stats: newStats }, c);
    path.push(cur);
  }
  const v = state.leafRewards[cur];
  for (const id of path) {
    newStats[id].n += 1;
    newStats[id].w += v;
    newStats[id].q = newStats[id].w / newStats[id].n;
  }
  return { ...state, stats: newStats, iteration: state.iteration + 1, lastPath: path };
}

function PUCTTreeSVG({ treeStruct, layout, state, width, height, leafLabelFn }) {
  const pathSet = new Set(state.lastPath);
  let bestLeaf = 0;
  let bestVal = -Infinity;
  for (const lid of treeStruct.leafIds) {
    if (state.stats[lid].n > 0 && state.stats[lid].q > bestVal) {
      bestVal = state.stats[lid].q;
      bestLeaf = lid;
    }
  }
  return (
    <svg viewBox={`0 0 ${width} ${height}`} width="100%" style={{ maxWidth: width }}>
      {treeStruct.edges.map((e) => {
        const a = layout[e.from];
        const b = layout[e.to];
        const onPath = pathSet.has(e.from) && pathSet.has(e.to);
        return (
          <line key={`e-${e.from}-${e.to}`}
                x1={a.x} y1={a.y} x2={b.x} y2={b.y}
                className={`ttt-tree-edge ${onPath ? 'path' : ''}`} />
        );
      })}
      {treeStruct.nodes.map((n) => {
        const p = layout[n.id];
        const isLeaf = n.children.length === 0;
        const isOnPath = pathSet.has(n.id);
        const isBest = isLeaf && state.stats[n.id].n > 0 && n.id === bestLeaf;
        const visited = state.stats[n.id].n > 0;
        const cls =
          (isBest ? 'best ' : '') +
          (isOnPath ? 'path ' : '') +
          (visited && !isOnPath && !isBest ? 'visited' : '');
        const r = isLeaf ? 16 : 14;
        return (
          <g key={`n-${n.id}`}>
            <circle cx={p.x} cy={p.y} r={r}
                    className={`ttt-tree-node-bg ${cls.trim()}`} />
            <text x={p.x} y={p.y} className="ttt-tree-node-text">
              {isLeaf ? '' : `n=${state.stats[n.id].n}`}
            </text>
            {isLeaf && leafLabelFn && (
              <text x={p.x} y={p.y + r + 12}
                    className="ttt-tree-leaf-reward"
                    textAnchor="middle">
                {leafLabelFn(n.id)}
              </text>
            )}
          </g>
        );
      })}
    </svg>
  );
}

/* =========================================================
   The anchor widget — TTT-Discover, end-to-end
   ========================================================= */

const TTT_REWARDS = [1.00, 1.15, 1.30, 1.45, 1.55, 1.70, 1.90, 2.10];

function SectionTTTDiscover() {
  const [sliderPos, setSliderPos] = useState(0.30);
  const beta = 0.01 * Math.pow(5000, sliderPos);

  const weights = useMemo(() => {
    const exps = TTT_REWARDS.map((r) => Math.exp(beta * r));
    const Z = exps.reduce((s, x) => s + x, 0);
    return exps.map((e) => e / Z);
  }, [beta]);

  const meanR = TTT_REWARDS.reduce((s, x) => s + x, 0) / TTT_REWARDS.length;
  const maxR = Math.max(...TTT_REWARDS);
  const logZ = Math.log(TTT_REWARDS.reduce((s, r) => s + Math.exp(beta * r), 0) / TTT_REWARDS.length);
  const entropic = logZ / beta;

  const N = TTT_REWARDS.length;
  const H = -weights.reduce((s, w) => s + (w > 1e-12 ? w * Math.log(w) : 0), 0);
  const KL = Math.log(N) - H;

  const treeStruct = useMemo(() => buildBinaryTree(3), []);
  const layoutTreeMemo = useMemo(() => layoutTree(treeStruct, 600, 280, 28), [treeStruct]);
  const leafArr = useMemo(() => {
    const kernelLeaves = [1.00, 1.20, 1.35, 1.50, 1.45, 1.70, 1.95, 2.10];
    const arr = new Array(15).fill(0);
    treeStruct.leafIds.forEach((id, i) => { arr[id] = kernelLeaves[i]; });
    return arr;
  }, [treeStruct]);
  const [treeState, setTreeState] = useState(() => initMCTSState(treeStruct, leafArr));
  const treeStep = () => setTreeState((s) => mctsStep(s, treeStruct, 1.4));
  const treeRun = () => setTreeState((s) => {
    let cur = s;
    for (let i = 0; i < 10; i++) cur = mctsStep(cur, treeStruct, 1.4);
    return cur;
  });
  const treeReset = () => setTreeState(initMCTSState(treeStruct, leafArr));

  return (
    <div className="viz-panel">
      <h3 className="ttt-subhead-plain">The entropic peak-vs-average objective</h3>

      <div className="ttt-controls">
        <label style={{ minWidth: 200 }}>
          β = <strong className="ttt-mono-accent">
            {beta.toFixed(beta < 1 ? 3 : 2)}
          </strong>
        </label>
        <input
          type="range" min="0" max="1" step="0.01"
          value={sliderPos}
          onChange={(e) => setSliderPos(parseFloat(e.target.value))}
          style={{ flex: 1, minWidth: 220, accentColor: 'var(--post-accent)' }}
        />
      </div>

      <div className="ttt-beta-row">
        <div className="ttt-beta-bars">
          <div className="ttt-beta-bar-row ttt-beta-bar-head">
            <div className="label">sample</div>
            <div className="label">reward R<sub>i</sub></div>
            <div className="label">weight w<sub>i</sub>(β)</div>
            <div className="label" style={{ textAlign: 'right' }}>w</div>
          </div>
          {TTT_REWARDS.map((r, i) => (
            <div key={i} className="ttt-beta-bar-row">
              <div>#{i + 1}</div>
              <div className="ttt-beta-track">
                <div className="reward-fill" style={{ width: `${(r / 2.5) * 100}%` }} />
              </div>
              <div className="ttt-beta-track">
                <div className="weight-fill" style={{ width: `${weights[i] * 100}%` }} />
              </div>
              <div style={{ textAlign: 'right' }}>{weights[i].toFixed(2)}</div>
            </div>
          ))}
        </div>

        <div className="ttt-beta-formula-callout">
          <div style={{ marginBottom: 8 }}>
            <Katex block tex={String.raw`
              \mathcal{J}_\beta \;=\; \tfrac{1}{\beta}\,\log\,\mathbb{E}\!\left[\,e^{\beta R}\,\right]
            `} />
          </div>
          <div style={{ marginTop: 6 }}>
            β → 0  :  <Katex tex="\mathcal{J}_\beta \to \mathbb{E}[R]" /> (average)
          </div>
          <div>
            β → ∞ :  <Katex tex="\mathcal{J}_\beta \to \max R" /> (peak)
          </div>
          <div className="ttt-beta-formula-rule">
            adaptive β set per state by{' '}
            <Katex tex="\mathrm{KL}(q_\beta \,\|\, \pi_\theta) = \gamma = \ln 2 \approx 0.69" />,
            via bisection
          </div>
        </div>
      </div>

      <div className="ttt-counters">
        <Counter label="E[R] (mean)" value={meanR.toFixed(3)} sub="β→0 limit" />
        <Counter label={`Jβ at β=${beta.toFixed(beta < 1 ? 2 : 1)}`} value={entropic.toFixed(3)} sub="entropic objective" />
        <Counter label="max R" value={maxR.toFixed(2)} sub="β→∞ limit" />
        <Counter
          label="KL(qβ ‖ π_θ)"
          value={KL.toFixed(3)}
          sub={KL > Math.log(2) ? 'past γ — would stop' : 'within γ = ln 2 budget'}
        />
      </div>

      <p className="viz-caption" style={{ marginTop: 12 }}>
        Eight sample rewards from one inner-loop iteration (think: eight
        candidate kernels, each timed to give a speedup factor). The
        β-weighted distribution{' '}
        <Katex tex="w_i = e^{\beta R_i} / \sum_j e^{\beta R_j}" />{' '}
        controls what the policy is trained to look like. β → 0 spreads
        weight uniformly (target = average reward); β → ∞ collapses all
        weight onto the best sample (target = peak reward). The KL counter
        on the right is what TTT-Discover uses to set β automatically — it
        bisects until <Katex tex="\mathrm{KL}(q_\beta \,\|\, \pi_\theta) = \ln 2" />.
      </p>

      <h3 className="ttt-subhead-plain ttt-subhead-spaced">θ accumulates within a problem, resets across problems</h3>
      <div className="ttt-theta-diagram">
        <div className="ttt-theta-cell">
          <div className="h">within one problem</div>
          <div>
            <Katex tex="\theta_0 \to \theta_1 \to \theta_2 \to \cdots \to \theta_K" />
          </div>
          <div className="b">weights accumulate</div>
          <div className="ttt-theta-cell-note">
            each iteration refines θ on this one task
          </div>
        </div>
        <div className="ttt-theta-cell">
          <div className="h">across problems</div>
          <div>
            <Katex tex="\theta_K \;\longrightarrow\; \theta_0" />
          </div>
          <div className="b">reset to checkpoint</div>
          <div className="ttt-theta-cell-note">
            no cross-problem transfer — by design
          </div>
        </div>
      </div>

      <h3 className="ttt-subhead-plain ttt-subhead-spaced">PUCT — now selecting kernel mutations</h3>
      <div className="ttt-tree-controls">
        <button className="active" onClick={treeStep}>Run 1 iteration</button>
        <button onClick={treeRun}>Run 10 more</button>
        <button onClick={treeReset}>Reset</button>
        <span className="ttt-tree-controls-status">
          iteration <strong>{treeState.iteration}</strong>{' '}
          · leaves = simulated kernel speedups (×)
        </span>
      </div>
      <div className="ttt-tree-wrap">
        <PUCTTreeSVG
          treeStruct={treeStruct}
          layout={layoutTreeMemo}
          state={treeState}
          width={600}
          height={280}
          leafLabelFn={(id) => `${treeState.leafRewards[id].toFixed(2)}×`}
        />
      </div>

      <p className="viz-caption" style={{ marginTop: 12 }}>
        Same PUCT formula as classic MCTS; different "game." Root: a
        starting prompt plus a seed program. Children: variations proposed
        by the (now weight-mutating) model. Leaves: simulated speedups
        from actually running the kernel. <Katex tex="Q(s)" /> uses the{' '}
        <strong>max</strong> child reward instead of the mean — because we
        care about the <em>best</em> kernel found, not the average.
      </p>
    </div>
  );
}

/* =========================================================
   Page
   ========================================================= */
export default function TestTimeTraining() {
  useEffect(() => {
    const prev = document.title;
    document.title = 'Test-Time Training — Bernhard Walser';
    return () => { document.title = prev; };
  }, []);

  return (
    <article className="post-2026 ttt-post">
      <div className="post-wrap">
        <div className="post-hero">
          <div className="post-series-tag">
            <span className="post-live-dot" aria-hidden="true" />
            Self-Improvement · Part 5 of 6
          </div>
          <h1>Test-Time Training: When the Model Updates Mid-Inference</h1>
          <p className="post-lede">
            Every method we've covered so far has held the model's weights
            still while solving a problem. FunSearch and AlphaEvolve never
            touch them. STaR and ReST update them offline, between training
            runs. TTT-Discover does something stranger: it updates weights{' '}
            <em>while</em> solving one problem, then throws those updates
            away before the next one arrives.
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

        <h2 className="reveal">A different kind of update</h2>
        <p>
          Think of the methods so far as living in two camps. The
          training-time camp (STaR, ReST, AlphaZero) updates the model in
          long offline loops, then ships a fixed checkpoint that solves
          everything you throw at it. The search-only camp (FunSearch,
          AlphaEvolve) leaves the model untouched and lets the{' '}
          <em>population</em> of programs do the learning instead.
        </p>
        <p>
          TTT-Discover sits in a quadrant neither camp occupies. The model
          adapts to <em>this one problem</em> by taking gradient steps mid-inference,
          and those updates persist as the search tree grows.
          When the next problem arrives, the weights reset to the original
          checkpoint. There is no cross-problem transfer — on purpose.
        </p>

        <h2 className="reveal">TTT-Discover, end-to-end</h2>
        <p>
          What if you combined both worlds? Search a tree of program
          mutations like FunSearch, but also update the model's weights
          while you're doing it — specifically toward this <em>one</em>{' '}
          problem? That's the move{' '}
          <a className="post-link" href="https://arxiv.org/abs/2601.16175" target="_blank" rel="noreferrer">TTT-Discover</a>{' '}
          (Stanford / NVIDIA / Together AI, 2026) makes. Two design
          choices distinguish it from everything above:
        </p>
        <ul>
          <li>
            <strong>Weights θ update online, during inference</strong>, and
            accumulate across iterations <em>within one problem</em>. When
            the next problem arrives, θ resets to the original checkpoint{' '}
            <Katex tex="\theta_0" />. There is no cross-problem transfer —
            on purpose.
          </li>
          <li>
            <strong>The training objective targets peak, not average.</strong>{' '}
            Instead of maximising <Katex tex="\mathbb{E}[R]" /> over
            sampled candidates, TTT-Discover maximises the{' '}
            <em>entropic</em> objective{' '}
            <Katex tex="\mathcal{J}_\beta = \tfrac{1}{\beta}\log\,\mathbb{E}[e^{\beta R}]" /> —
            which smoothly interpolates between mean (β → 0) and max
            (β → ∞) reward. β is set adaptively per state by a fixed
            KL budget <Katex tex="\mathrm{KL}(q_\beta \,\|\, \pi_\theta) = \gamma = \ln 2" />,
            via bisection.
          </li>
        </ul>
        <p>
          Move the slider below to see the entropic objective in action.
          Eight sample rewards from one inner-loop iteration; the
          per-sample weights <Katex tex="w_i = e^{\beta R_i} / \sum_j e^{\beta R_j}" />{' '}
          are what the policy is asked to imitate. Watch the weight bars
          concentrate onto the highest reward as β grows.
        </p>
        <SectionTTTDiscover />
        <p>
          Tying both halves together: TTT-Discover runs a{' '}
          <strong>PUCT tree search over candidate programs</strong> — the
          same PUCT as classic MCTS, but now the "moves" are mutations of
          the prompt + seed kernel, and the leaf "values" are simulated
          runtimes from actually executing the kernel.{' '}
          <Katex tex="Q(s)" /> is the <em>max</em> of child rewards rather
          than the mean (peak, not average). At promising leaves, the
          weights <Katex tex="\theta" /> are updated by a single gradient
          step using the entropic objective — and those updates persist as
          the tree expands further.
        </p>
        <InfoBox title="The headline result">
          <p>
            Using the open-weight gpt-oss-120b as the proposer,
            TTT-Discover reportedly achieves a <strong>~2× speedup</strong>{' '}
            over the best human-written version of AlphaFold's TriMul GPU
            kernel on the GPUMode leaderboard. The combination that gets
            them there — inference-time weight updates + a peak-targeting
            entropic objective + PUCT tree search — is the climax of
            every progression in this series.
          </p>
        </InfoBox>

        <h2 className="reveal">Wait — doesn't RL collapse?</h2>
        <p>
          One uncomfortable fact deserves a callout. The dominant failure
          mode of training-time RL on long-form generation is{' '}
          <strong>diversity collapse</strong>: gradient updates push
          average reward up, but the upper bound stays flat. The policy
          converges on the safe, simple thing that maximises expected
          reward, abandoning the bold-but-broken candidates that might
          have led to a real discovery. The pattern shows up clearly in
          recent execution-grounded benchmarks (e.g.{' '}
          <a className="post-link" href="https://arxiv.org/abs/2601.14525" target="_blank" rel="noreferrer">arxiv 2601.14525</a>{' '}
          on automated ML research: RL improved average reward but the
          upper bound didn't — pure population-based evolution preserved
          diversity where RL didn't).
        </p>
        <p>
          The structural diagnosis is sharp:{' '}
          <strong>execution-as-reward is biased against novelty</strong>. A
          complex idea needs more reasoning steps → more places to fail →
          lower execution rate → lower reward. Bold ideas get filtered out
          before they're even debugged. The verifier itself penalises
          complexity. So how does TTT-Discover beat human-written kernels
          with test-time RL when the same loop collapses elsewhere? Three
          differences, each lining up with a feature TTT-Discover happens
          to have:
        </p>

        <div className="ttt-mini-compare">
          <div className="head"></div>
          <div className="head">naive on-policy RL</div>
          <div className="head">TTT-Discover</div>

          <div className="axis">Objective</div>
          <div>
            <Katex tex="\mathbb{E}[R]" /> — average reward.
            Greedy gradients prune complexity.
          </div>
          <div>
            <Katex tex="\tfrac{1}{\beta}\log\mathbb{E}[e^{\beta R}]" /> —
            peak, with adaptive <Katex tex="\beta" />. Upweights the best
            samples, designed not to collapse to the safe average.
          </div>

          <div className="axis">Search</div>
          <div>
            none — pure policy-gradient updates.
          </div>
          <div>
            PUCT tree <em>alongside</em> the weight updates. The tree
            preserves diversity (bold-but-broken candidates stay alive to
            mutate).
          </div>

          <div className="axis">Reward signal</div>
          <div>
            open-ended ("did the research idea run?") — complex ideas
            penalised by failure rate.
          </div>
          <div>
            measured performance on a fixed, well-scoped task ("did the
            kernel run faster?"). Complexity isn't a structural
            disadvantage.
          </div>

          <div className="axis">Problem scope</div>
          <div>open-ended; novelty matters.</div>
          <div>narrow; one specific kernel.</div>
        </div>

        <p>
          The field has split into several distinct fixes for collapse.{' '}
          <strong>Structural diversity</strong> — island populations
          (FunSearch, AlphaEvolve, ThetaEvolve), tree search (AlphaZero,
          TTT-Discover) — physically keeps bold-but-broken candidates
          alive. <strong>Entropic / peak objectives</strong>{' '}
          (TTT-Discover's <Katex tex="\log\mathbb{E}[e^{\beta R}]" /> with
          a KL budget) prevent collapse onto a single output.{' '}
          <strong>Difficulty-targeting curricula</strong> (R-Zero
          challenger aiming at ~50% solver success; Agent0 proposer
          rewarded for executor failure) kill collapse at the source by
          penalising trivial problems. <strong>Stagnation penalties</strong>{' '}
          (ThetaEvolve's "lazy penalty" term). <strong>Prompt-seed
          diversity</strong> (AAR's nine parallel agents with varied
          starts — works in practice, doesn't fix the underlying bias).
          The textbook RL entropy bonus on its own has been tried in
          several settings and consistently underperforms; the methods
          that actually hold up <em>stack several of these techniques</em>.
        </p>
        <p>
          TTT-Discover is the hybrid that lands: an entropic peak
          objective combined with a search structure that preserves
          diverse candidates, applied to a well-scoped problem where
          complexity penalties aren't structural. The lesson isn't{' '}
          <em>"RL works"</em> or <em>"RL fails"</em> — it's that{' '}
          <strong>test-time RL works exactly where the problem is narrow
          and the objective targets the peak, and fails where the problem
          is open-ended and the objective targets the average</strong>.
        </p>

        <h2 className="reveal">Where this sits</h2>
        <p>
          On the 2-axis map of the series — when do weights change × what
          does the objective target — TTT-Discover takes a quadrant nothing
          else does. Weights change <em>mid-inference</em>. The objective
          targets the <em>peak</em>. The next post asks the hard question
          that all of this leaves open: how do we know any of this actually
          works?
        </p>

        <h2 className="reveal">References</h2>
        <div className="post-refs">
          <div className="head ref-cite">Cite</div>
          <div className="head ref-title">Title</div>
          <div className="head ref-link">Link</div>

          <div className="ref-cite">Stanford / NVIDIA / Together AI, 2026</div>
          <div>
            <div className="ref-title">
              <a href="https://arxiv.org/abs/2601.16175" target="_blank" rel="noreferrer">TTT-Discover</a>
            </div>
            <div className="ref-note">Test-time training with an entropic peak objective and PUCT search; reports ~2× speedup over the best human-written TriMul kernel.</div>
          </div>
          <div className="ref-link"><a href="https://arxiv.org/abs/2601.16175" target="_blank" rel="noreferrer">arxiv.org</a></div>

          <div className="ref-cite">Wang et al., 2025</div>
          <div>
            <div className="ref-title">
              <a href="https://arxiv.org/abs/2511.23473" target="_blank" rel="noreferrer">ThetaEvolve</a>
            </div>
            <div className="ref-note">Bridge case — AlphaEvolve-style island evolution with test-time RL on the LLM's weights.</div>
          </div>
          <div className="ref-link"><a href="https://arxiv.org/abs/2511.23473" target="_blank" rel="noreferrer">arxiv.org</a></div>

          <div className="ref-cite">arxiv 2601.14525</div>
          <div>
            <div className="ref-title">
              <a href="https://arxiv.org/abs/2601.14525" target="_blank" rel="noreferrer">Execution-grounded RL for automated ML research</a>
            </div>
            <div className="ref-note">Documents the collapse pattern: RL improves average reward but the upper bound stays flat; pure population-based evolution preserves diversity.</div>
          </div>
          <div className="ref-link"><a href="https://arxiv.org/abs/2601.14525" target="_blank" rel="noreferrer">arxiv.org</a></div>
        </div>

        <footer className="post-footer">
          <p>
            <strong>Part 5 of Self-Improvement</strong> · Previous:{' '}
            <a className="post-link" href="#/blog/evolutionary-search">Evolutionary Search</a>
            {' · '}Next:{' '}
            <a className="post-link" href="#/blog/measuring-self-improvement">Measuring Self-Improvement</a>.
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
