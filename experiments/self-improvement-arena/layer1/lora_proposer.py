"""Layer 1, arms 2 & 3: greedy / risk-seeking LoRA fine-tuning.

Where arm 1 (``llm_evolution.py``) improves by feeding the best results back into
the *prompt* (in-context, no weight updates), these two arms improve by updating
the model's *weights* — a LoRA adapter — from the batch rewards. They are the LLM
analogue of Layer 0's ``rl_greedy`` / ``rl_risk``: same ask/tell seam, same
verifier, same budget, and — crucially — the SAME per-sample weighting formulas
(imported from ``sia.objectives``). The ONLY difference between the greedy and
risk arms is how rewards become weights, exactly as in Layer 0:

    greedy   : w_i = R_i - mean(R)                         (objectives.greedy_weights)
    risk     : w_i = (R_i - q) on the top-eps, else 0      (objectives.quantile_weights)
               or  w_i propto e^{beta R_i}, centered       (objectives.entropic_weights)

The training step is a reward-weighted policy-gradient / weighted-SFT update:

    maximize  sum_i  w_i * sum_t log pi(completion_token_t^i | prompt)
    <=>  minimize  loss = sum_i  w_i * sum_t CE(logits_t^i, completion_token_t^i)

over the LoRA parameters only (the base model is frozen). Gradient steps do NOT
count against the verifier-call budget; only generations do (1 verifier call per
candidate, same accounting as the evolution arm).

MLX / Apple-Silicon note
------------------------
Every ``mlx`` / ``mlx_lm`` import is LAZY (inside methods), so importing this
module never fails on a non-Mac box and Layer 0 stays runnable everywhere. The
MLX-touching work is isolated in three methods — ``_setup_model`` (attach LoRA),
``_sample_batch`` (generate), ``_lora_step`` (the weighted-PG update) — so the
ask/tell/weighting bookkeeping can be unit-tested with a fake model (no MLX, no
GPU); see ``tests/test_layer1.py``.

HONESTY: this file was written and the non-MLX parts were unit-tested on a Linux
box with NO Apple Silicon, so the actual MLX LoRA training step has NOT been
executed here. The MLX call sites follow mlx-lm's own LoRA tuner/trainer patterns
(``linear_to_lora_layers`` + ``nn.value_and_grad`` + ``optimizer.update``), but
the exact ``mlx_lm`` API can drift across versions. The on-device smoke test and
expected behavior are documented in ``layer1/README.md``; treat the MLX path as
"needs M4 verification" until that smoke run passes.
"""
from __future__ import annotations

import numpy as np

from sia.expression import Node, leaf, parse_expression
from sia.objectives import (cvar_weights, entropic_weights, greedy_weights,
                            quantile_weights)
from sia.proposers.base import Proposer
from sia.verifier import Result

# A candidate that always scores 0 (used when the model's output can't be parsed).
# The wasted generation still costs one verifier call (fair budgeting) and the
# generated tokens still get a (low) weight in the gradient step — exactly like an
# invalid sample in Layer 0, which scores reward 0 and is mean-baselined down.
INVALID = Node("/", [leaf("0.0"), leaf("0.0")])


def load_model(model_id: str):
    """Load an mlx-lm model + tokenizer once (lazy import). Same as the evolution
    arm; re-exported here so callers can load without importing evolution."""
    from mlx_lm import load
    return load(model_id)


class LoRAProposer(Proposer):
    """Greedy or risk-seeking LoRA proposer. ``arm`` selects the objective:

    - ``"greedy"``  -> greedy_weights (expected reward; the collapse-prone arm)
    - ``"risk"``    -> quantile_weights (DSR) or entropic_weights, per ``mode``

    Same ask/tell contract as Layer 0, so ``run_layer1.run_one`` drives it
    unchanged. Construct with an already-loaded ``model`` + ``tokenizer`` (from
    ``load_model``); the constructor attaches a fresh LoRA adapter to that model.

    IMPORTANT: attaching LoRA *mutates* the model in place, so each run must get
    its own freshly-loaded model — do not share one model across LoRA runs.
    """

    def __init__(self, task, rng, model, tokenizer, arm: str = "greedy",
                 batch_size: int = 8, temperature: float = 1.0, max_tokens: int = 48,
                 n_data_shown: int = 12, const_placeholder: bool = False,
                 # LoRA / optimizer
                 lora_layers: int = 8, lora_rank: int = 8, lora_scale: float = 2.0,
                 lora_dropout: float = 0.0, lr: float = 1e-4,
                 # trust region: KL penalty to the frozen BASE model (prevents the
                 # policy from drifting off-distribution into token-spam). This is the
                 # ingredient real LLM-RL (GRPO/RS-GRPO/PPO) uses and pure weighted-SFT
                 # lacks. NOTE: distinct from target_kl below (that is the entropic
                 # beta-rule's target, unrelated). GRPO uses 0.04, but our smaller
                 # setup (tiny batch, no advantage normalization) needs more KL to
                 # stay stable -- 0.04 re-broke it in smoke tests, 0.1 held.
                 kl_coef: float = 0.1,
                 # risk-arm objective knobs (mirror Layer 0's rl_risk)
                 mode: str = "quantile", epsilon: float = 0.25, beta: float = 2.0,
                 beta_rule: str = "fixed", target_ess: float = 0.3,
                 target_kl: float = 1.0, **hp):
        super().__init__(task, rng, **hp)
        if arm not in ("greedy", "risk"):
            raise ValueError(f"arm must be 'greedy' or 'risk', got {arm!r}")
        if mode not in ("quantile", "entropic", "cvar"):
            raise ValueError(f"mode must be quantile/entropic/cvar, got {mode!r}")
        if beta_rule not in ("fixed", "ess", "kl"):
            raise ValueError(f"beta_rule must be fixed/ess/kl, got {beta_rule!r}")
        self.model = model
        self.tok = tokenizer
        self.arm = arm
        self.batch_size = batch_size
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.n_data_shown = n_data_shown
        self.const_placeholder = const_placeholder  # DSR-style C placeholder (C:=1)
        # LoRA / optim config
        self.lora_layers = lora_layers
        self.lora_rank = lora_rank
        self.lora_scale = lora_scale
        self.lora_dropout = lora_dropout
        self.lr = lr
        self.kl_coef = kl_coef
        # risk objective config
        self.mode = mode
        self.epsilon = epsilon
        self.beta = beta
        self.beta_rule = beta_rule
        self.target_ess = target_ess
        self.target_kl = target_kl
        # state carried between ask() and tell()
        self._pending_prompt = None          # token ids of the prompt (list[int])
        self._pending_completions = None     # list[list[int]] generated tokens
        self._last_valid_frac = float("nan")
        self._last_beta = float("nan")
        self._last_loss = float("nan")
        self._last_pg = float("nan")
        self._last_kl = float("nan")
        self._last_kl_term = float("nan")
        self._last_prompt = ""               # decoded prompt text (for the app)
        self._last_responses: list[str] = []  # raw model outputs (for the app)
        self._lora_modules = []              # LoRALinear layers (for toggling the adapter)
        self.opt = None
        self._setup_model()  # attach LoRA + build optimizer (MLX)

    # --- prompt construction (data only, NO archive inspiration) --------------
    # Mirrors llm_evolution._prompt minus the program-database block: the LoRA
    # arms learn from rewards via weight updates, not from in-context exemplars.
    def _data_block(self) -> str:
        x, y = self.task.x_train, self.task.y_train
        idx = np.linspace(0, len(x) - 1, min(self.n_data_shown, len(x))).astype(int)
        return "\n".join(f"  x = {x[i]:+.3f}   y = {y[i]:+.3f}" for i in idx)

    def _prompt_tokens(self):
        if self.const_placeholder:
            vocab = ("Allowed: the variable x, operators + - * /, the functions sin and "
                     "cos, and the constant placeholder C. Write C in place of every "
                     "numeric constant or coefficient -- for example write C*x + C "
                     "instead of 2.5*x + 1.3. Each C's value is chosen automatically.")
        else:
            vocab = ("Allowed: the variable x, operators + - * /, the functions sin and "
                     "cos, and numeric constants.")
        rules = ("You are doing symbolic regression. Find a formula y = f(x) that fits "
                 f"the data.\n{vocab}\nThe data may be nonlinear or "
                 "periodic — consider terms like x*x, x*x*x, or sin/cos, not just "
                 "straight lines.\nReply with ONLY the formula for f(x) on a single "
                 "line — no words, no 'y =', no code fences.")
        msg = f"{rules}\n\nData points:\n{self._data_block()}\n\nNew formula for f(x):"
        self._last_prompt = msg  # surfaced to the app for inspection
        # apply_chat_template(tokenize=True default) -> token ids, same as evolution
        return self.tok.apply_chat_template(
            [{"role": "user", "content": msg}], add_generation_prompt=True)

    # --- the ask / tell seam -------------------------------------------------
    def ask(self) -> list[Node]:
        prompt = self._prompt_tokens()
        samples = self._sample_batch(prompt)  # list[(text, token_ids)]
        cands, completions, n_valid = [], [], 0
        for text, toks in samples:
            if self.const_placeholder:           # C placeholder + BFGS constant-fitting
                from layer1.constfit import parse_and_fit
                node = parse_and_fit(text, self.task.x_train, self.task.y_train)
            else:
                node = parse_expression(text)
            if node is None:
                cands.append(INVALID)
            else:
                cands.append(node)
                n_valid += 1
            completions.append(list(toks))
        self._last_valid_frac = n_valid / max(self.batch_size, 1)
        self._last_responses = [text for text, _ in samples]  # raw outputs for the app
        # stash for tell(): the gradient step needs the exact prompt + completions
        self._pending_prompt = prompt
        self._pending_completions = completions
        return cands

    def _weights(self, results: list[Result]) -> np.ndarray:
        """Rewards -> per-sample weights, via the SHARED objectives (the SAME
        formulas as Layer 0). The only thing that differs between the two LoRA
        arms is this function's branch."""
        R = np.array([r.reward for r in results], dtype=float)
        if self.arm == "greedy":
            return greedy_weights(R)
        if self.mode == "quantile":
            return quantile_weights(R, self.epsilon)
        if self.mode == "cvar":  # risk-averse lower-tail mirror of quantile
            return cvar_weights(R, self.epsilon)
        w, self._last_beta = entropic_weights(R, self.beta_rule, self.beta,
                                              self.target_ess, self.target_kl)
        return w

    def tell(self, candidates: list[Node], results: list[Result]) -> None:
        if self._pending_completions is None:
            return  # tell() without a preceding ask(); nothing to learn from
        weights = self._weights(results)
        # One LoRA gradient step (does NOT cost a verifier call). All completions
        # are included with their weights — invalids carry reward 0, so greedy
        # pushes them down and quantile ignores them, exactly as in Layer 0.
        self._lora_step(self._pending_prompt, self._pending_completions, weights)
        self._pending_prompt = self._pending_completions = None

    def diagnostics(self) -> dict:
        return {"policy_entropy": float("nan"),      # no token-entropy probe for the LLM
                "valid_fraction": self._last_valid_frac,
                "beta": self._last_beta,
                "loss": self._last_loss,
                "pg_loss": self._last_pg,            # PG term (signed)
                "kl": self._last_kl,                 # raw KL (k3)
                "kl_term": self._last_kl_term}       # kl_coef * KL (its loss contribution)

    def last_io(self) -> dict:
        """The prompt sent and the raw responses from the last ask() -- for the app
        to display what the LLM actually saw and produced."""
        return {"prompt": self._last_prompt, "responses": list(self._last_responses)}

    # --- MLX-touching methods (lazy imports; the only non-portable code) ------
    def _setup_model(self) -> None:
        """Freeze the base model and attach a trainable LoRA adapter, then build
        the optimizer. Follows mlx-lm's own tuner pattern.

        NOTE (version sensitivity): ``linear_to_lora_layers`` takes a config dict
        with keys ``rank`` / ``scale`` / ``dropout`` on recent mlx-lm; older
        builds nest these under ``lora_parameters`` or use ``alpha``. If the call
        below raises on your installed version, this is the one line to adapt — the
        rest of the arm is API-stable. See layer1/README.md for the pinned version.
        """
        import mlx.optimizers as optim
        from mlx_lm.tuner.lora import LoRALinear
        from mlx_lm.tuner.utils import linear_to_lora_layers
        from mlx_lm.sample_utils import make_sampler

        self.model.freeze()  # base weights stay fixed; only LoRA params train
        lora_config = {"rank": self.lora_rank, "scale": self.lora_scale,
                       "dropout": self.lora_dropout}
        linear_to_lora_layers(self.model, self.lora_layers, lora_config)
        self.model.train()
        self.opt = optim.Adam(learning_rate=self.lr)
        self._sampler = make_sampler(temp=self.temperature)
        # Handles to the LoRA layers so we can momentarily zero the adapter
        # (scale=0 -> pure base model) to get reference logprobs for the KL penalty.
        self._lora_modules = [m for _, m in self.model.named_modules()
                              if isinstance(m, LoRALinear)]

    def _sample_batch(self, prompt_tokens) -> list[tuple[str, list[int]]]:
        """Sample ``batch_size`` completions from the current LoRA-adapted policy.
        Returns each as ``(decoded_text, generated_token_ids)``; the text is parsed
        into a Node, the token ids drive the gradient step in ``_lora_step``."""
        from mlx_lm import stream_generate

        out: list[tuple[str, list[int]]] = []
        for _ in range(self.batch_size):
            text, toks = "", []
            for resp in stream_generate(self.model, self.tok, prompt=prompt_tokens,
                                        max_tokens=self.max_tokens,
                                        sampler=self._sampler):
                text += resp.text
                toks.append(int(resp.token))
            out.append((text, toks))
        return out

    def _ref_logprobs(self, inputs, targets):
        """Per-token log pi of ``targets`` under the BASE model (LoRA disabled).
        Computed with the adapter momentarily zeroed (scale=0) so no second model
        copy is needed; returned as a constant (no gradient)."""
        import mlx.core as mx
        import mlx.nn as nn
        saved = [m.scale for m in self._lora_modules]
        for m in self._lora_modules:
            m.scale = 0.0  # adapter off -> pure base model
        try:
            logits = self.model(inputs)
            ref_logp = -nn.losses.cross_entropy(logits, targets)  # (B, T-1)
            mx.eval(ref_logp)
        finally:
            for m, s in zip(self._lora_modules, saved):
                m.scale = s
        return mx.stop_gradient(ref_logp)

    def _lora_step(self, prompt_tokens, completions, weights) -> None:
        """One reward-weighted policy-gradient step on the LoRA params, with a
        trust-region KL penalty to the frozen base model.

        loss = sum_i w_i * sum_t CE(logits_t, token_t)            (PG term)
             + kl_coef * sum_{i,t} [ exp(d) - d - 1 ],  d = logpi_ref - logpi_theta
                                                          (k3 KL(theta||ref) >= 0)

        The PG term pushes probability toward high-weight completions; the KL term
        keeps the policy near the base distribution so it cannot reward-hack into
        off-distribution token-spam (the failure mode of pure weighted-SFT). This is
        the GRPO/RS-GRPO/PPO trust region, applied to the LoRA params (base frozen).
        """
        import mlx.core as mx
        import mlx.nn as nn

        prompt = list(prompt_tokens)
        P = len(prompt)
        seqs = [prompt + list(c) for c in completions]
        seqs = [s for s in seqs if len(s) > P]  # drop empty completions (nothing to train)
        if not seqs:
            return
        # keep weights aligned with the kept sequences
        keep = [i for i, c in enumerate(completions) if len(c) > 0]
        w = np.asarray(weights, dtype=np.float32)[keep]
        B = len(seqs)
        T = max(len(s) for s in seqs)
        pad_id = getattr(self.tok, "eos_token_id", 0) or 0

        batch = np.full((B, T), pad_id, dtype=np.int32)
        mask = np.zeros((B, T), dtype=np.float32)  # 1 at completion-token positions
        for i, s in enumerate(seqs):
            batch[i, :len(s)] = s
            mask[i, P:len(s)] = 1.0  # completion tokens occupy [P, len(s))
        batch = mx.array(batch)
        # logits at position t predict token t+1, so targets/mask shift left by one
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        tgt_mask = mx.array(mask[:, 1:])
        wv = mx.array(w)
        # reference (base-model) per-token logprobs -- a constant for the KL term
        ref_logp = self._ref_logprobs(inputs, targets) if self.kl_coef > 0 else None

        def loss_fn(model):
            logits = model(inputs)                          # (B, T-1, V)
            ce = nn.losses.cross_entropy(logits, targets)  # (B, T-1), -log pi per token
            per_sample = (ce * tgt_mask).sum(axis=1)        # sum over completion tokens
            pg = (wv * per_sample).sum()                    # reward-weighted PG loss
            if ref_logp is None:
                kl = mx.array(0.0)
            else:
                d = ref_logp - (-ce)                        # logpi_ref - logpi_theta
                kl_tok = mx.exp(d) - d - 1.0                 # k3 estimator, >= 0
                kl = (kl_tok * tgt_mask).sum()
            return pg + self.kl_coef * kl, (pg, kl)         # aux: the two components

        (loss, (pg, kl)), grads = nn.value_and_grad(self.model, loss_fn)(self.model)
        self.opt.update(self.model, grads)
        mx.eval(self.model.parameters(), self.opt.state)
        self._last_loss = float(loss)
        self._last_pg = float(pg)                # PG term (signed)
        self._last_kl = float(kl)                # raw KL (k3)
        self._last_kl_term = float(self.kl_coef * kl)  # KL's contribution to the loss
