# berniwal.github.io

Personal blog + reproducible ML experiments. Live at **<https://berniwal.github.io>**.

## Blog

Long-form interactive explainers under the *Visualizing ML* series. Each post is a
self-contained React component with inline-SVG widgets, KaTeX math, and (where it
applies) data baked from the experiment folder it belongs to.

- [**Watching Search Algorithms Discover a Formula**](https://berniwal.github.io/#/blog/visualizing-symbolic-regression)
  — symbolic regression as a self-improvement testbed: evolution, greedy RL, and
  risk-seeking RL race to recover a hidden formula under one shared budget, then we
  swap the policy for an LLM and ask whether the ranking survives. Code in
  [`experiments/self-improvement-arena/`](experiments/self-improvement-arena/).
- [Visualizing Self-Improving AI: From AlphaZero to TTT-Discover](https://berniwal.github.io/#/blog/visualizing-self-improvement)
  — a map of the family (AlphaZero, STaR, RLVR, FunSearch, AlphaEvolve,
  TTT-Discover) on two axes: when do the weights change, and does the objective
  target peak or average?
- [Visualizing RLHF: PPO, DPO, GRPO](https://berniwal.github.io/#/blog/visualizing-rlhf)
  — how a base LM becomes instruction-following, with the RL math made geometric.
- [Visualizing RoPE](https://berniwal.github.io/#/blog/visualizing-rope) —
  rotary positional embeddings as 2D rotations.
- [Visualizing the KV Cache](https://berniwal.github.io/#/blog/visualizing-kv-cache)
  — prefill vs decode, GQA, PagedAttention, continuous batching.
- [Visualizing Attention](https://berniwal.github.io/#/blog/visualizing-attention)
  — Q/K/V, multi-head, causal masking from token embeddings up.
- A few earlier chess-engine notes (Minimax / Alpha-Beta / Monte Carlo Tree
  Search) under the **Algorithm** category.

Posts live in [`src/posts/*.jsx`](src/posts/), registered in
[`src/blogPosts.js`](src/blogPosts.js). The house style for new interactive
explainers — design tokens, components, content arc, references / footer
conventions, registration mechanics — is in [`CLAUDE.md`](CLAUDE.md).

## Experiments

Reproducible-research code lives in [`experiments/`](experiments/):

- [**`experiments/self-improvement-arena/`**](experiments/self-improvement-arena/)
  — the lab bench for the symbolic-regression post. Four search methods (random,
  GP, greedy RL, risk-seeking RL) on a shared task / verifier / budget; on the
  Layer-1 side a Qwen2.5-0.5B + LoRA + GRPO proposer behind the same
  `ask()` / `tell()` seam. Includes the DSR-own-code cross-check, the Nguyen 1–8
  sweep, a Streamlit step-through visualiser, and 49 unit tests. See
  [its README](experiments/self-improvement-arena/README.md).
- [`experiments/runpod/`](experiments/runpod/) — generic RunPod cloud harness
  (GCS-backed result sync, auto-terminating pods, `--max-price` guard, GPU/CPU
  availability query). Reusable across any future experiment folder.

Each experiment folder has its own README with quick-start commands, data layout,
findings, and honesty notes.

## Running the site locally

```bash
npm install
npm start                  # dev server at http://localhost:3000
npm run build              # production bundle into build/
npm run deploy             # publishes build/ to the gh-pages branch
```

Hosted on GitHub Pages at `berniwal.github.io` (custom domain via
[`CNAME`](CNAME)), built with Create React App + `react-router-dom` HashRouter.
KaTeX is loaded globally from a CDN in [`public/index.html`](public/index.html).

## Credits

By **Bernhard Walser** — ML Engineer (Digitec Galaxus), ETH Computer Science.
[LinkedIn](https://www.linkedin.com/in/bernhardwalser/) ·
[GitHub](https://github.com/berniwal).

Many of the recent posts and experiments are co-authored with
[Claude](https://www.anthropic.com/claude) (Anthropic) — initial drafts,
interactive scaffolding, and experiment code by Claude; refined and co-designed
by Bernhard.
