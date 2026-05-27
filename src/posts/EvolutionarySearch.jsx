// src/posts/EvolutionarySearch.jsx
// Self-Improvement · Part 4 of 6 — Evolutionary Search.
// Ported from VisualizingSelfImprovement §5; the FunSearch / AlphaEvolve prose
// is preserved verbatim from the source post. ThetaEvolve is new content for
// this standalone post, based on arxiv 2511.23473.
//
// TAGS FOR REGISTRATION: ['self-improvement', 'evolutionary', 'funsearch']
// EXCERPT: Freeze the weights, evolve the programs. FunSearch, AlphaEvolve, and ThetaEvolve put the LLM in the mutation slot and let an evolutionary loop do the learning.

import React, { useEffect, useRef, useState } from 'react';
import './PostChrome.css';
import './EvolutionarySearch.css';

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
      className={`${block ? 'es-math-block' : 'es-math-inline'} ${className}`}
    />
  );
}

/* =========================================================
   Program-evolution widget (ported from §5)
   ========================================================= */

const EVO_GENERATIONS = [
  {
    gen: 0,
    cards: [
      { id: 'A', code: 'return greedy(x)',             fitness: 0.42 },
      { id: 'B', code: 'return random_local(x)',        fitness: 0.31 },
      { id: 'C', code: 'return greedy(x, k=2)',         fitness: 0.45 },
      { id: 'D', code: 'return swap_first(x)',          fitness: 0.28 },
      { id: 'E', code: 'return reverse(x)',             fitness: 0.30 },
      { id: 'F', code: 'return greedy(x, lookahead=1)', fitness: 0.48 },
      { id: 'G', code: 'return random(x)',              fitness: 0.20 },
      { id: 'H', code: 'return identity(x)',            fitness: 0.18 },
    ],
  },
  {
    gen: 1,
    cards: [
      { id: 'A', code: 'return greedy(x)',                 fitness: 0.42, parent: null   },
      { id: 'F', code: 'return greedy(x, lookahead=1)',    fitness: 0.48, parent: null   },
      { id: 'C', code: 'return greedy(x, k=2)',            fitness: 0.45, parent: null   },
      { id: 'I', code: 'return greedy(x, lookahead=2)',    fitness: 0.56, parent: 'F', born: true },
      { id: 'J', code: 'return greedy(x, k=3)',            fitness: 0.51, parent: 'C', born: true },
      { id: 'K', code: 'return greedy(x, lookahead=1)\n  if x>0 else swap(x)', fitness: 0.53, parent: 'F', born: true },
      { id: 'L', code: 'return beam(x, k=2)',              fitness: 0.50, parent: 'C', born: true },
      { id: 'B', code: 'return random_local(x)',           fitness: 0.31, parent: null   },
    ],
  },
  {
    gen: 2,
    cards: [
      { id: 'I', code: 'return greedy(x, lookahead=2)',                fitness: 0.56, parent: null },
      { id: 'K', code: 'return greedy(x, lookahead=1)\n  if x>0 else swap(x)', fitness: 0.53, parent: null },
      { id: 'M', code: 'return greedy(x, lookahead=3)',                fitness: 0.62, parent: 'I', born: true },
      { id: 'N', code: 'return beam(x, k=3, lookahead=2)',             fitness: 0.67, parent: 'I', born: true },
      { id: 'L', code: 'return beam(x, k=2)',                          fitness: 0.50, parent: null },
      { id: 'O', code: 'return beam(x, k=2, prune=true)',              fitness: 0.59, parent: 'L', born: true },
      { id: 'J', code: 'return greedy(x, k=3)',                        fitness: 0.51, parent: null },
      { id: 'P', code: 'return hybrid(greedy, beam)',                  fitness: 0.64, parent: 'I', born: true },
    ],
  },
  {
    gen: 3,
    cards: [
      { id: 'N', code: 'return beam(x, k=3, lookahead=2)',                fitness: 0.67, parent: null },
      { id: 'P', code: 'return hybrid(greedy, beam)',                     fitness: 0.64, parent: null },
      { id: 'Q', code: 'return beam(x, k=4, lookahead=3,\n             prune=adaptive)', fitness: 0.78, parent: 'N', born: true },
      { id: 'R', code: 'return mcts(x, sims=64)',                         fitness: 0.81, parent: 'N', born: true },
      { id: 'M', code: 'return greedy(x, lookahead=3)',                   fitness: 0.62, parent: null },
      { id: 'S', code: 'return mcts(x, sims=64,\n             prior=greedy)', fitness: 0.86, parent: 'R', born: true },
      { id: 'O', code: 'return beam(x, k=2, prune=true)',                 fitness: 0.59, parent: null },
      { id: 'T', code: 'return mcts(x, sims=128,\n             prior=hybrid)', fitness: 0.89, parent: 'R', born: true },
    ],
  },
];

function SectionEvolution() {
  const [gen, setGen] = useState(0);
  const data = EVO_GENERATIONS[gen];

  const advance = () => setGen((g) => Math.min(g + 1, EVO_GENERATIONS.length - 1));
  const reset = () => setGen(0);

  const bestFitnessByGen = EVO_GENERATIONS.map((g) =>
    Math.max(...g.cards.map((c) => c.fitness))
  );
  const sparkW = 220, sparkH = 60;
  const sparkPath = bestFitnessByGen.map((f, i) => {
    const x = 10 + (i / (EVO_GENERATIONS.length - 1)) * (sparkW - 20);
    const y = sparkH - 8 - (f - 0.4) * (sparkH - 16) / 0.5;
    return `${i === 0 ? 'M' : 'L'} ${x.toFixed(1)} ${y.toFixed(1)}`;
  }).join(' ');
  const curX = 10 + (gen / (EVO_GENERATIONS.length - 1)) * (sparkW - 20);
  const curF = bestFitnessByGen[gen];
  const curY = sparkH - 8 - (curF - 0.4) * (sparkH - 16) / 0.5;

  return (
    <div className="viz-panel">
      <div className="es-evo-banner">
        <span aria-hidden="true">🔒</span>&nbsp;
        <Katex tex="\theta_{\mathrm{LLM}}" /> <strong>unchanged</strong>.
        The model is a <em>frozen proposer</em> of mutations. Only the
        population improves over generations.
      </div>

      <div className="es-controls">
        <button
          className={gen < EVO_GENERATIONS.length - 1 ? 'active' : ''}
          onClick={advance}
        >
          {gen < EVO_GENERATIONS.length - 1 ? 'Evolve one generation' : 'Final generation'}
        </button>
        <button onClick={reset}>Reset</button>
        <span className="es-controls-meta">
          generation <strong>{gen}</strong> · best fitness{' '}
          <strong className="es-controls-fit">{curF.toFixed(2)}</strong>
        </span>
      </div>

      <div className="es-evo-grid">
        {data.cards.map((c) => (
          <div
            key={c.id}
            className={`es-evo-card ${c.born ? 'born' : 'elder'}`}
          >
            <div className="es-evo-card-tag">
              {c.id}{c.parent ? ` ← ${c.parent}` : ''} · {c.born ? 'new' : 'kept'}
            </div>
            <div className="es-evo-card-code">{c.code}</div>
            <div className="es-evo-card-fitness">fitness = {c.fitness.toFixed(2)}</div>
          </div>
        ))}
      </div>

      <div className="es-evo-spark">
        <div className="es-evo-spark-label">best fitness per generation</div>
        <svg viewBox={`0 0 ${sparkW} ${sparkH}`} width="100%" style={{ maxWidth: sparkW }}>
          <line x1={10} y1={sparkH - 8} x2={sparkW - 10} y2={sparkH - 8}
                stroke="#e2e8f0" />
          <path d={sparkPath} fill="none" stroke="#16a34a" strokeWidth="2" />
          <circle cx={curX} cy={curY} r="4" fill="#16a34a" stroke="#fff" strokeWidth="1.5" />
          <text x={sparkW - 10} y={sparkH - 2} textAnchor="end" fontSize="9"
                fill="#94a3b8" fontFamily="var(--post-mono)">
            gen {EVO_GENERATIONS.length - 1}
          </text>
        </svg>
      </div>

      <p className="viz-caption">
        A toy population of program candidates evolves against a fixed
        verifier (here: a stand-in "fitness" function). Each generation the
        LLM proposes mutations of the top-fitness parents; new programs
        enter the pool, lowest-fitness ones drop out. The bottom sparkline
        shows best fitness climbing across generations. Crucially: the LLM
        itself is never fine-tuned — the artefact of this method is the
        <em> found program</em>, not a better model.
      </p>
    </div>
  );
}

/* =========================================================
   Page
   ========================================================= */
export default function EvolutionarySearch() {
  useEffect(() => {
    const prev = document.title;
    document.title = 'Evolutionary Search — Bernhard Walser';
    return () => { document.title = prev; };
  }, []);

  return (
    <article className="post-2026 es-post">
      <div className="post-wrap">
        <div className="post-hero">
          <div className="post-series-tag">
            <span className="post-live-dot" aria-hidden="true" />
            Self-Improvement · Part 4 of 6
          </div>
          <h1>Evolutionary Search: FunSearch, AlphaEvolve, ThetaEvolve</h1>
          <p className="post-lede">
            The last two posts had the language model itself be the source of
            variation: STaR sampled rationales, ReST sampled rollouts, and an
            external verifier picked the winners that became training data.
            Evolutionary methods make the variation step explicit. An outer
            loop maintains a <em>population</em> of programs; an LLM proposes
            mutations of the best ones; a verifier scores them. The weights
            never move.
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

        <h2 className="reveal">When the weights freeze</h2>
        <p>
          Up to here, every method in the series <em>updated the model's
          weights</em>: STaR fine-tuned on filtered rationales, ReST / RLVR
          did gradient steps on verified rollouts, internal-verifier methods
          like R-Zero ran self-play with GRPO. The next family takes the
          opposite stance: <strong>don't update the model at all</strong>.
          Use it as a frozen proposer of <em>programs</em>, and run an
          evolutionary search loop over the proposals, scored by a
          programmatic verifier.
        </p>
        <p>
          What's improving in this regime is the population of candidate
          programs, not the proposer. The artefact of a successful run is a{' '}
          <strong>found program</strong> — a heuristic, a construction, a
          GPU kernel — not a better model. If you froze the weights of a
          decent open model today, the loop could keep producing better
          programs for months.
        </p>

        <h2 className="reveal">FunSearch</h2>
        <p>
          <a className="post-link" href="https://www.nature.com/articles/s41586-023-06924-6" target="_blank" rel="noreferrer">FunSearch</a>{' '}
          (DeepMind,{' '}
          <a className="post-link" href="https://arxiv.org/abs/2411.19744" target="_blank" rel="noreferrer">Romera-Paredes et al., 2024</a>)
          was the first to demonstrate this at scale — finding new
          mathematical constructions (improved cap-set lower bounds, better
          bin-packing heuristics) by having a frozen LLM propose Python
          functions, scoring them against an evaluator, and using an
          island-style evolutionary loop to breed better candidates from the
          top performers.
        </p>
        <p>
          The loop is small enough to fit in one paragraph. Sample a few
          high-fitness parents from the program database. Prompt the LLM
          with their source code and ask for a "better version." Run the
          proposed program through a sandboxed evaluator and record its
          fitness. Insert it back into the database, evicting the worst
          program in its island. Repeat. The widget below runs four
          generations of that loop on a toy population of search
          heuristics — watch the new (highlighted) cards displace the
          weakest elders each round.
        </p>
        <SectionEvolution />
        <p>
          The LLM is treated like a stochastic mutator with a built-in
          language model of "what plausible code looks like." That prior is
          the whole point: random edits to Python wouldn't compile, let
          alone score well, but the LLM's edits stay in the manifold of
          syntactically valid, semantically plausible programs. The
          evaluator handles the rest.
        </p>

        <h2 className="reveal">AlphaEvolve</h2>
        <p>
          <a className="post-link" href="https://arxiv.org/abs/2506.13131" target="_blank" rel="noreferrer">AlphaEvolve</a>{' '}
          (DeepMind, 2025) extends the same recipe to general algorithm
          discovery beyond the cap-set problem. The loop is still
          population-based evolution with a frozen LLM as the mutator, but
          the mutations are richer: rather than rewriting a single function,
          an agentic LLM proposes edits across multiple files, can leave
          comments explaining its intent, and can chain edits over several
          steps before the verifier runs. The reported results include
          improvements on a handful of long-standing open problems
          (matrix-multiplication tensor decompositions, kissing-number
          bounds, data-centre scheduling heuristics).
        </p>
        <p>
          What stays the same: <strong>weights never move</strong>. What
          changes: the unit of evolution is now closer to a small codebase
          than a single function, and the LLM gets more room to plan inside
          each mutation. Both FunSearch and AlphaEvolve sit cleanly in the
          search-only quadrant of the series map.
        </p>

        <h2 className="reveal">ThetaEvolve</h2>
        <p>
          <a className="post-link" href="https://arxiv.org/abs/2511.23473" target="_blank" rel="noreferrer">ThetaEvolve</a>{' '}
          (2025) is a recent variation that's worth flagging because it
          sits halfway between this post and the next one. It keeps
          AlphaEvolve's island-evolution scaffolding — a large program
          database, batched sampling, lazy penalties to escape stagnation —
          but swaps out two things. First, it runs on a single open-source
          model (DeepSeek-R1-0528-Qwen3-8B in the paper) instead of an
          ensemble of frontier LLMs. Second, it adds{' '}
          <strong>test-time reinforcement learning on the proposer's
          weights</strong>: as the population improves, the LLM that
          generates mutations is itself updated against the same fitness
          signal. The paper's targets are the circle-packing and
          first-autocorrelation-inequality problems that AlphaEvolve
          previously tackled, and the headline is that the small open
          model — once it's allowed to learn at test time — pushes new
          best-known bounds on both.
        </p>
        <p>
          Schematically, you can read ThetaEvolve as:
        </p>
        <p>
          <Katex block tex={String.raw`
            \underbrace{\text{program database}}_{\text{AlphaEvolve loop}}
            \;\;\xrightleftharpoons[\text{RL update}]{\text{sample + score}}\;\;
            \underbrace{\theta_{\mathrm{LLM}}}_{\text{no longer frozen}}
          `} />
        </p>
        <p>
          That arrow on the right is the difference. In FunSearch and
          AlphaEvolve the proposer is a fixed function; in ThetaEvolve it
          co-evolves with the population it's mutating. Which makes it not
          quite a search-only method — and a clean bridge into the next
          post, where the weights move <em>during inference</em> on
          purpose.
        </p>

        <h2 className="reveal">Where this sits</h2>
        <p>
          Evolutionary search is the <strong>"weights stay frozen"</strong>{' '}
          corner of the self-improvement map: the model is a mutation
          operator, not the learner. Improvement lives in the population,
          and the artefact you ship is the best program found, not a better
          checkpoint. ThetaEvolve breaks that purity by letting the
          proposer learn online — which is exactly the move the next post
          takes to its conclusion.
        </p>
        <p>
          <strong>Next:</strong>{' '}
          <a className="post-link" href="#/blog/test-time-training">Test-Time Training</a>{' '}
          flips the quadrant: weights update <em>during inference</em>,
          accumulating within a single problem and resetting between
          problems. Same population idea, but the proposer is now the
          learner too.
        </p>

        <h2 className="reveal">References</h2>
        <div className="post-refs">
          <div className="head ref-cite">Cite</div>
          <div className="head ref-title">Title</div>
          <div className="head ref-link">Link</div>

          <div className="ref-cite">Romera-Paredes et al., 2024</div>
          <div>
            <div className="ref-title">
              <a href="https://arxiv.org/abs/2411.19744" target="_blank" rel="noreferrer">FunSearch — Mathematical discoveries from program search with large language models</a>
            </div>
            <div className="ref-note">Cap-set and bin-packing results; island-evolution loop with a frozen LLM mutator.</div>
          </div>
          <div className="ref-link"><a href="https://arxiv.org/abs/2411.19744" target="_blank" rel="noreferrer">arxiv 2411.19744</a></div>

          <div className="ref-cite">Novikov et al., 2025</div>
          <div>
            <div className="ref-title">
              <a href="https://arxiv.org/abs/2506.13131" target="_blank" rel="noreferrer">AlphaEvolve — A coding agent for scientific and algorithmic discovery</a>
            </div>
            <div className="ref-note">Agentic LLM editor across files; reported improvements on matrix-multiplication and other open problems.</div>
          </div>
          <div className="ref-link"><a href="https://arxiv.org/abs/2506.13131" target="_blank" rel="noreferrer">arxiv 2506.13131</a></div>

          <div className="ref-cite">Wang et al., 2025</div>
          <div>
            <div className="ref-title">
              <a href="https://arxiv.org/abs/2511.23473" target="_blank" rel="noreferrer">ThetaEvolve — Test-time learning on open problems</a>
            </div>
            <div className="ref-note">Single open model + test-time RL on the proposer's weights; new best-known bounds on circle packing and the first auto-correlation inequality.</div>
          </div>
          <div className="ref-link"><a href="https://arxiv.org/abs/2511.23473" target="_blank" rel="noreferrer">arxiv 2511.23473</a></div>
        </div>

        <footer className="post-footer">
          <p>
            <strong>Part 4 of Self-Improvement</strong> · Previous:{' '}
            <a className="post-link" href="#/blog/internal-verifiers">Internal Verifiers</a>
            {' · '}Next:{' '}
            <a className="post-link" href="#/blog/test-time-training">Test-Time Training</a>.
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
