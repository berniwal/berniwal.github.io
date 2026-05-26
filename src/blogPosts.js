// src/blogPosts.js
export const blogPosts = [
  {
    slug: 'visualizing-symbolic-regression',
    title: 'Watching Search Algorithms Discover a Formula',
    category: 'Visualizing ML',
    excerpt: 'Symbolic regression as a self-improvement testbed: evolution, greedy RL, and risk-seeking RL race to recover a hidden equation under one shared budget — then we swap the search policy for an LLM and ask whether the ranking survives. Interactive replays included.',
    component: 'VisualizingSymbolicRegression'
  },
  {
    slug: 'visualizing-self-improvement',
    title: 'Visualizing Self-Improving AI: From AlphaZero to TTT-Discover',
    category: 'Visualizing ML',
    excerpt: 'A map of the self-improvement family — AlphaZero, STaR, RLVR, FunSearch, AlphaEvolve, and the frontier where the model rewrites its own weights mid-inference. Two axes: when do the weights change, and does the objective target peak or average?',
    component: 'VisualizingSelfImprovement'
  },
  {
    slug: 'visualizing-rlhf',
    title: 'Visualizing RLHF: From Next-Token Prediction to Following Instructions',
    category: 'Visualizing ML',
    excerpt: 'How a base language model becomes an instruction-following assistant — SFT, reward models, PPO, and the modern simplifications (DPO, GRPO) side-by-side, with the RL math made geometric.',
    component: 'VisualizingRLHF'
  },
  {
    slug: 'visualizing-rope',
    title: 'Visualizing RoPE: Rotary Positional Embeddings, Geometrically',
    category: 'Visualizing ML',
    excerpt: 'The geometric story of why Llama, Mistral, Qwen, and DeepSeek all rotate Q and K in 2D pairs — and how relative position falls out of the dot product for free.',
    component: 'VisualizingRoPE'
  },
  {
    slug: 'visualizing-kv-cache',
    title: 'Visualizing the KV Cache: Prefill, Decode, and Why Inference Is Bandwidth-Bound',
    category: 'Visualizing ML',
    excerpt: 'Why the first token takes 500 ms and the rest stream at 20 ms each — the KV cache, arithmetic intensity, GQA, PagedAttention, and continuous batching, every number small enough to read.',
    component: 'VisualizingKVCache'
  },
  {
    slug: 'visualizing-attention',
    title: 'Visualizing Attention: Q/K/V, Multi-Head, and Causal Masking',
    category: 'Visualizing ML',
    excerpt: 'An interactive walk through self-attention — from token embeddings to causal masking — with every matrix small enough to read by eye.',
    component: 'VisualizingAttention'
  },
  {
    slug: 'mini-max',
    title: 'Minimax Search',
    category: 'Algorithm',
    excerpt: 'Simple Decision Making Algorithm for two-player games.',
    file: 'chess/minimax.md'
  },
  {
    slug: 'alpha-beta',
    title: 'Alpha-Beta Pruning',
    category: 'Algorithm',
    excerpt: 'Decision Making Algorithm for two-player games.',
    file: 'chess/alphabeta.md'
  },
  {
    slug: 'monte-carlo',
    title: 'Monte Carlo Tree Search',
    category: 'Algorithm',
    excerpt: 'Decision Making Algorithm for two-player games.',
    file: 'chess/montecarlo.md'
  }
];
