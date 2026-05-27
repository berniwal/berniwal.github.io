// src/blogPosts.js
export const blogPosts = [
  {
    slug: 'symbolic-regression-arena',
    title: 'Racing Search Algorithms on Symbolic Regression',
    category: 'Symbolic Regression',
    excerpt: 'Symbolic regression as a self-improvement testbed: evolution, greedy RL, and risk-seeking RL race to recover a hidden equation under one shared compute budget. Interactive replays on Layer-0 and Nguyen 1–8.',
    component: 'SymbolicRegressionArena'
  },
  {
    slug: 'symbolic-regression-llm-transfer',
    title: 'Does the Ranking Transfer to an LLM?',
    category: 'Symbolic Regression',
    excerpt: 'Swap the proposer for an LLM and ask whether the same Layer-0 ranking survives — what the model sees, how the search budget is shared, and what the transfer experiment found.',
    component: 'SymbolicRegressionLlmTransfer'
  },
  {
    slug: 'alphago-to-alphazero',
    title: 'AlphaGo to AlphaZero',
    category: 'Self-Improvement',
    excerpt: 'The 2-axis map of self-improvement methods, and how AlphaGo → AlphaGo Zero → AlphaZero set the template: search + a learned policy/value, self-play, no human labels.',
    component: 'AlphaGoToAlphaZero'
  },
  {
    slug: 'external-verifiers',
    title: 'External Verifiers: STaR and ReST/RLVR',
    category: 'Self-Improvement',
    excerpt: "When ground truth exists outside the model (math answer keys, unit tests), bootstrap rationales against it. STaR's filter-and-fine-tune loop and ReST/RLVR's RL framing of the same trick.",
    component: 'ExternalVerifiers'
  },
  {
    slug: 'internal-verifiers',
    title: 'Internal Verifiers: When the Model Judges Its Own Work',
    category: 'Self-Improvement',
    excerpt: 'R-Zero, Agent0, and G-Zero generate their own training signal: one model proposes, another solves, and the reward comes from self-consistency or behavioral shift — no external grader.',
    component: 'InternalVerifiers'
  },
  {
    slug: 'evolutionary-search',
    title: 'Evolutionary Search: FunSearch, AlphaEvolve, ThetaEvolve',
    category: 'Self-Improvement',
    excerpt: 'When the model is a mutation operator, not the learner. FunSearch and AlphaEvolve freeze the LM and evolve a program population around it; ThetaEvolve breaks that rule.',
    component: 'EvolutionarySearch'
  },
  {
    slug: 'test-time-training',
    title: 'Test-Time Training: When the Model Updates Mid-Inference',
    category: 'Self-Improvement',
    excerpt: "TTT-Discover updates the model's weights while solving one problem, then resets before the next. An entropic peak-vs-average objective, PUCT-guided mutation selection, and a real answer to 'why doesn't this collapse?'",
    component: 'TestTimeTraining'
  },
  {
    slug: 'measuring-self-improvement',
    title: 'Measuring Self-Improvement',
    category: 'Self-Improvement',
    excerpt: "MLE-Bench, RE-Bench, PaperBench, PostTrainBench, and METR's time-horizon curves. Plus Anthropic's Automated Alignment Researcher — the closest thing to an AI doing AI research today.",
    component: 'MeasuringSelfImprovement'
  },
  {
    slug: 'rlhf-and-ppo',
    title: 'RLHF and PPO',
    category: 'Aligning LMs',
    excerpt: 'How a base language model becomes an instruction-following assistant: SFT, reward modelling from preferences, and PPO with its clipped surrogate.',
    component: 'RlhfAndPpo'
  },
  {
    slug: 'beyond-ppo',
    title: 'Beyond PPO: DPO, GRPO, DAPO',
    category: 'Aligning LMs',
    excerpt: "PPO works but it needs four networks in memory and is hard to tune. DPO drops the reward model, GRPO drops the value net, and DAPO patches GRPO's failure modes — the post-PPO landscape, side by side.",
    component: 'BeyondPpo'
  },
  {
    slug: 'beyond-human-feedback',
    title: 'Beyond Human Feedback: RLAIF, Process Rewards, RLVR',
    category: 'Aligning LMs',
    excerpt: 'Human labels are expensive and slow. RLAIF replaces the labeller with a model; process reward models densify the signal; RLVR removes the learned reward entirely in verifiable domains.',
    component: 'BeyondHumanFeedback'
  },
  {
    slug: 'positional-encodings',
    title: 'Positional Encodings: A Tour',
    category: 'Positions in Transformers',
    excerpt: 'Self-attention is permutation-invariant. Five additive position-encoding schemes — absolute, Shaw, T5, Swin, ALiBi — and where each one injects position.',
    component: 'PositionalEncodings'
  },
  {
    slug: 'rope',
    title: 'Rotary Positional Embeddings (RoPE)',
    category: 'Positions in Transformers',
    excerpt: 'RoPE encodes token position by rotating each query and key vector — the dot product naturally depends on relative offset, with zero new parameters and a clean path to long context.',
    component: 'RotaryPositionalEmbeddings'
  },
  {
    slug: 'training-cost',
    title: 'Training Cost of a Transformer',
    category: 'Transformer Costs',
    excerpt: "Training Llama-2-7B took ~1.7M GPU-hours. To know why, count what one transformer layer actually does — forward and backward — and what has to be held in memory between them.",
    component: 'TrainingCost'
  },
  {
    slug: 'training-cost-optimization',
    title: 'Optimizing Training Cost',
    category: 'Transformer Costs',
    excerpt: "Gradient accumulation, gradient checkpointing, mixed precision, ZeRO/FSDP — four optimizations that each target a different memory bucket in the training stack.",
    component: 'TrainingCostOptimization'
  },
  {
    slug: 'inference-cost',
    title: 'Inference Cost of a Transformer',
    category: 'Transformer Costs',
    excerpt: "Why the first token takes 500 ms and the rest stream at 20 ms each — the prefill/decode split, the KV cache, and the arithmetic-intensity argument for why decode is memory-bound.",
    component: 'InferenceCost'
  },
  {
    slug: 'inference-cost-optimization',
    title: 'Optimizing Inference Cost',
    category: 'Transformer Costs',
    excerpt: "Decode is bandwidth-bound, so every optimization shrinks the KV cache or packs the GPU better: GQA / MQA, PagedAttention, and continuous batching.",
    component: 'InferenceCostOptimization'
  },
  {
    slug: 'nlp-history',
    title: 'From TF-IDF to Attention',
    category: 'Transformers',
    excerpt: "How NLP got from counting words to attention — tokens, TF-IDF, embeddings, and the RNN/CNN limits that attention was built to escape. Part 1 of a 3-part series on Transformers.",
    component: 'NlpHistory'
  },
  {
    slug: 'self-attention',
    title: 'Inside Self-Attention: Q, K, V, and Softmax',
    category: 'Transformers',
    excerpt: "How a single self-attention layer works — three projections (Q, K, V), a scaled dot-product, a softmax, and a weighted sum. Every number on the page is live; hover any cell to see the math.",
    component: 'SelfAttention'
  },
  {
    slug: 'from-attention-to-transformer',
    title: 'From Attention to Transformer',
    category: 'Transformers',
    excerpt: "A single attention operation isn't yet a transformer. Add multiple heads, an MLP per token, a causal mask, and positional information — and you have the block that powers every modern language model.",
    component: 'FromAttentionToTransformer'
  },
  {
    slug: 'mini-max',
    title: 'Minimax Search',
    category: 'Algorithm',
    excerpt: 'Simple decision-making algorithm for two-player games — walk every move sequence, assume the opponent plays optimally, pick the move that minimizes your worst case.',
    component: 'MinimaxSearch'
  },
  {
    slug: 'alpha-beta',
    title: 'Alpha-Beta Pruning',
    category: 'Algorithm',
    excerpt: 'An enhancement to minimax that skips branches it can prove can\'t change the answer — same chosen move, far fewer nodes evaluated. The reason real chess engines can look further ahead.',
    component: 'AlphaBetaPruning'
  },
  {
    slug: 'monte-carlo',
    title: 'Monte Carlo Tree Search',
    category: 'Algorithm',
    excerpt: 'Simulation-based search: random rollouts replace exhaustive evaluation. Selection (UCB1), expansion, simulation, backpropagation — the four steps that beat humans at Go.',
    component: 'MonteCarloTreeSearch'
  }
];
