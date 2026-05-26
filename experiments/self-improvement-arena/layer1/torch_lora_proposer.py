"""PyTorch/Hugging-Face LLM proposer with GRPO (runs on NVIDIA GPUs / RunPod).

This is the portable counterpart to the MLX ``lora_proposer`` (Apple-only). It keeps
the EXACT same seam as Layer 0 and the MLX arm: a batch of LLM samples is parsed into
the shared ``Node`` representation, scored by the shared ``Verifier``, and rewards are
turned into per-sample advantages by the shared ``sia.objectives`` (greedy / quantile /
entropic) -- only that mapping differs between arms.

The update is GRPO as used by ThetaEvolve / TTT-Discover:
  - group-relative advantage (per arm, via objectives),
  - clipped surrogate with ASYMMETRIC clipping (clip_low 0.2, clip_high 0.28),
  - TRUNCATED importance sampling (clamp the ratio) for stability,
  - multi-epoch reuse of each batch,
  - NO KL / NO entropy regularization (matching ThetaEvolve's stated setup).

Designed to run remotely on a GPU pod; nothing here is Apple-specific.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

# Reuse the pure-numpy seam (works on any platform).
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "src"))
sys.path.insert(0, os.path.dirname(_HERE))  # for `import layer1.constfit`
from sia.expression import Node, leaf, parse_expression, to_infix  # noqa: E402
from sia.objectives import (cvar_weights, entropic_weights, greedy_weights,  # noqa: E402
                            quantile_weights)
from sia.verifier import Result  # noqa: E402

INVALID = leaf("1.0")  # placeholder scored ~0 by the verifier (an unparseable sample)


class TorchLoRAProposer:
    def __init__(self, task, model_id="Qwen/Qwen2.5-0.5B-Instruct", arm="risk",
                 mode="quantile", epsilon=0.25, beta=2.0, beta_rule="fixed",
                 target_ess=0.3, target_kl=1.0, batch_size=16, micro_batch=8, lr=1e-6,
                 weight_decay=0.1, lora_rank=16, lora_alpha=32, lora_dropout=0.0,
                 max_new_tokens=48, temperature=1.0, top_p=0.95,
                 const_placeholder=True, n_data_shown=12,
                 ppo_epochs=2, clip_low=0.2, clip_high=0.28, trunc_is=2.0,
                 std_normalize=False, reasoning=False, thinking_budget=0,
                 answer_budget=64, seed=0, dtype="bfloat16"):
        if arm not in ("greedy", "risk", "best_of_n", "evolution"):
            raise ValueError("arm must be greedy, risk, best_of_n, or evolution")
        self.archive: list[tuple[float, str]] = []  # (reward, expr) for the evolution arm
        self.archive_k = 6                           # top-K exemplars fed back into the prompt
        if mode not in ("quantile", "entropic", "cvar"):
            raise ValueError("mode must be quantile/entropic/cvar")
        self.task = task
        self.arm, self.mode = arm, mode
        self.epsilon, self.beta, self.beta_rule = epsilon, beta, beta_rule
        self.target_ess, self.target_kl = target_ess, target_kl
        self.batch_size = batch_size
        self.micro = micro_batch   # forward/backward chunk size (bounds peak GPU memory)
        self.max_new_tokens, self.temperature, self.top_p = max_new_tokens, temperature, top_p
        self.const_placeholder, self.n_data_shown = const_placeholder, n_data_shown
        self.ppo_epochs = ppo_epochs
        self.clip_low, self.clip_high, self.trunc_is = clip_low, clip_high, trunc_is
        self.std_normalize = std_normalize
        self.reasoning = reasoning
        # Two-stage reasoning (Qwen3 "soft budget"): if thinking_budget > 0, the
        # generator first produces up to `thinking_budget` tokens of CoT; any row
        # that hasn't emitted `</think>` by then has it forcibly spliced in, and
        # the model then continues for up to `answer_budget` tokens to write the
        # final formula. The injected </think> tokens are MASKED from comp_mask
        # so the gradient ignores them (they aren't a sample from pi_theta).
        self.thinking_budget = thinking_budget
        self.answer_budget = answer_budget
        if reasoning and thinking_budget == 0 and max_new_tokens < 256:
            print(f"[torch] WARN: reasoning=True but max_new_tokens={max_new_tokens} -- "
                  f"thinking traces typically need 1024+ tokens; the model will be cut off mid-CoT",
                  flush=True)
        self._last_beta = float("nan")
        self._last_loss = float("nan")
        self._last_valid_frac = float("nan")
        self._last_responses: list[str] = []
        torch.manual_seed(seed)

        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        td = {"bfloat16": torch.bfloat16, "float16": torch.float16,
              "float32": torch.float32}[dtype if self.device == "cuda" else "float32"]
        self.tok = AutoTokenizer.from_pretrained(model_id)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        base = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=td)
        # Gradient checkpointing: re-run forward inside backward instead of keeping
        # activations resident. Essential at long max_new_tokens (e.g. reasoning mode at
        # 2048 tokens) where the per-layer attention activations + 152k-wide logits
        # OOM an 80GB A100 otherwise. PEFT idiom: `enable_input_require_grads` is
        # needed because the base is frozen; without it checkpointing has no inputs
        # with requires_grad and silently no-ops.
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        base.enable_input_require_grads()
        base.config.use_cache = False  # incompatible with grad checkpointing during training
        lcfg = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                          target_modules="all-linear", task_type="CAUSAL_LM")
        self.model = get_peft_model(base, lcfg).to(self.device)
        self.opt = torch.optim.AdamW(
            (p for p in self.model.parameters() if p.requires_grad),
            lr=lr, weight_decay=weight_decay)
        print(f"[torch] device={self.device} dtype={td} model={model_id} "
              f"trainable params={sum(p.numel() for p in self.model.parameters() if p.requires_grad)}",
              flush=True)

    # --- prompt (identical text to the MLX arm, for comparability) ------------
    def _data_block(self) -> str:
        x, y = self.task.x_train, self.task.y_train
        idx = np.linspace(0, len(x) - 1, min(self.n_data_shown, len(x))).astype(int)
        return "\n".join(f"  x = {x[i]:+.3f}   y = {y[i]:+.3f}" for i in idx)

    def _prompt_text(self) -> str:
        if self.const_placeholder:
            vocab = ("Allowed: the variable x, operators + - * /, the functions sin and "
                     "cos, and the constant placeholder C. Write C in place of every "
                     "numeric constant or coefficient -- for example write C*x + C "
                     "instead of 2.5*x + 1.3. Each C's value is chosen automatically.")
        else:
            vocab = ("Allowed: the variable x, operators + - * /, the functions sin and "
                     "cos, and numeric constants.")
        archive_block = ""
        if self.arm == "evolution" and self.archive:  # AlphaEvolve-style: best-so-far in prompt
            ex = "\n".join(f"  f(x) = {expr}   (score {r:.3f})" for r, expr in self.archive)
            archive_block = ("\n\nBest formulas found so far -- propose a DIFFERENT formula "
                             f"that fits better than these:\n{ex}")
        if self.reasoning:
            # Thinking-mode prompt with an EXPLICIT budget + commitment contract.
            # Small reasoning models (Qwen3-1.7B) ruminate the whole budget without
            # committing if left unconstrained -- 0/5 of our initial smoke samples
            # ever emitted `</think>`. We add a budget hint here and, in _generate,
            # splice a TTT-Discover-style "out of thinking tokens, sending my final
            # message" sentinel right before forcibly closing the think block.
            budget_hint = (f"You have ~{self.thinking_budget} tokens to think; "
                           "use them efficiently. " if self.thinking_budget > 0 else "")
            instructions = (
                f"{budget_hint}Think step by step inside the thinking block: scan the "
                "(x, y) values for sign, symmetry, growth and periodicity; list 2-3 "
                "plausible skeletons (polynomial, polynomial + trig, rational); pick one "
                "and COMMIT. Do not enumerate forever.\nWhen you are done, write "
                "`</think>` on its own and then output ONLY the formula for f(x) on the "
                "very next line -- no words, no 'y =', no code fences, just the expression.")
        else:
            instructions = ("Reply with ONLY the formula for f(x) on a single line -- no words, "
                            "no 'y =', no code fences.")
        return ("You are doing symbolic regression. Find a formula y = f(x) that fits "
                f"the data.\n{vocab}\nThe data may be nonlinear or periodic -- consider "
                f"terms like x*x, x*x*x, or sin/cos, not just straight lines.\n{instructions}\n\n"
                f"Data points:\n{self._data_block()}{archive_block}\n\nNew formula for f(x):")

    def _prompt_ids(self) -> torch.Tensor:
        # Qwen3 (and other dual-mode reasoning models) honour `enable_thinking=...` in
        # apply_chat_template; older tokenizers (Qwen2.5, Llama) ignore the kwarg or
        # raise. Pass it only when reasoning is on, and fall back if rejected.
        msgs = [{"role": "user", "content": self._prompt_text()}]
        try:
            ids = self.tok.apply_chat_template(
                msgs, add_generation_prompt=True, return_tensors="pt",
                enable_thinking=self.reasoning)
        except TypeError:
            ids = self.tok.apply_chat_template(
                msgs, add_generation_prompt=True, return_tensors="pt")
        return ids.to(self.device)

    @staticmethod
    def _extract_formula(text: str) -> str:
        """Pull the final-formula line out of an LLM reply.

        Non-reasoning mode: the model is told to reply with one line, so the first
        non-empty line is the formula.  Reasoning mode: the model may emit
        `<think>...</think>\nformula` (Qwen3-style) or otherwise dump a long
        explanation; the formula is the LAST non-empty line after the thinking
        block. This handles both cases robustly.
        """
        if "</think>" in text:
            text = text.rsplit("</think>", 1)[1]
        lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
        if not lines:
            return ""
        # Last line tends to be the answer for reasoning models; first-line is
        # equivalent when the model obeys the single-line instruction.
        return lines[-1]

    # --- generation ----------------------------------------------------------
    # Forced wrap-up sentence (TTT-Discover style): in the model's own voice so the
    # transition stays in-distribution. Spliced inside the thinking block when the
    # budget is hit and the model hasn't closed </think> on its own.
    FORCED_END = ("\n\nOkay, I am out of thinking tokens. I need to send my final "
                  "message now.\n</think>\n")

    @torch.no_grad()
    def _generate_two_stage(self, inp, plen):
        """Two-stage budgeted reasoning.

        Stage 1: generate up to `thinking_budget` tokens. For any row that hasn't
        emitted `</think>` by then, splice the FORCED_END sentence in (in the
        model's own voice -- in-distribution wrap-up). Stage 2: generate up to
        `answer_budget` more tokens, which is where the final formula appears.

        Returns (seqs, forced_mask) where forced_mask marks the spliced-in tokens
        so the GRPO loss can ignore them (they aren't samples from pi_theta).
        """
        # Stage 1: free thinking.
        stage1 = self.model.generate(
            inp, attention_mask=torch.ones_like(inp),
            do_sample=True, temperature=self.temperature, top_p=self.top_p,
            max_new_tokens=self.thinking_budget,
            pad_token_id=self.tok.pad_token_id)

        # Per-row: did the model close </think> on its own?
        think_close_ids = self.tok.encode("</think>", add_special_tokens=False)
        forced_ids = self.tok.encode(self.FORCED_END, add_special_tokens=False)
        forced_ids_t = torch.tensor(forced_ids, device=self.device, dtype=stage1.dtype)

        # Build per-row stage-1 sequences with optional FORCED_END splice.
        rows = []
        forced_lens = []
        for r in range(stage1.shape[0]):
            tail = stage1[r, plen:]
            # strip padding for the check
            text_tail = self.tok.decode(tail, skip_special_tokens=False)
            if "</think>" in text_tail:
                # Model already closed thinking -- leave its tokens alone.
                rows.append(stage1[r])
                forced_lens.append(0)
            else:
                # Trim trailing pad/eos before splicing.
                end = tail.shape[0]
                while end > 0 and tail[end - 1].item() in (self.tok.pad_token_id,
                                                            self.tok.eos_token_id):
                    end -= 1
                row = torch.cat([stage1[r, :plen + end], forced_ids_t], dim=0)
                rows.append(row)
                forced_lens.append(len(forced_ids))

        # Right-pad to the longest stage-1+force row, then generate stage 2.
        max_len = max(r.shape[0] for r in rows)
        pid = self.tok.pad_token_id
        padded = torch.full((len(rows), max_len), pid, device=self.device,
                            dtype=stage1.dtype)
        attn1 = torch.zeros_like(padded)
        for i, r in enumerate(rows):
            padded[i, :r.shape[0]] = r
            attn1[i, :r.shape[0]] = 1

        seqs = self.model.generate(
            padded, attention_mask=attn1,
            do_sample=True, temperature=self.temperature, top_p=self.top_p,
            max_new_tokens=self.answer_budget,
            pad_token_id=pid)

        # Mark the spliced FORCED_END tokens for the loss mask. Their position in
        # each row is [rows[i].shape[0] - forced_lens[i] : rows[i].shape[0]].
        forced_mask = torch.zeros_like(seqs, dtype=torch.float32)
        for i, r in enumerate(rows):
            fl = forced_lens[i]
            if fl > 0:
                forced_mask[i, r.shape[0] - fl : r.shape[0]] = 1.0
        return seqs, forced_mask

    @torch.no_grad()
    def _generate(self, prompt_ids):
        plen = prompt_ids.shape[1]
        inp = prompt_ids.repeat(self.batch_size, 1)
        # Generation needs the KV cache; training-time forwards disable it (see __init__
        # for the grad-checkpointing wiring). Toggle around generate() so the rest of
        # the proposer can leave use_cache=False as the steady state.
        self.model.config.use_cache = True
        try:
            if self.reasoning and self.thinking_budget > 0:
                seqs, forced_mask = self._generate_two_stage(inp, plen)
            else:
                seqs = self.model.generate(
                    inp, attention_mask=torch.ones_like(inp),
                    do_sample=True, temperature=self.temperature, top_p=self.top_p,
                    max_new_tokens=self.max_new_tokens,
                    pad_token_id=self.tok.pad_token_id)  # (B, plen+gen)
                forced_mask = torch.zeros_like(seqs, dtype=torch.float32)
        finally:
            self.model.config.use_cache = False
        comp_mask = torch.zeros_like(seqs, dtype=torch.float32)
        comp_mask[:, plen:] = (seqs[:, plen:] != self.tok.pad_token_id).float()
        # Tokens we forcibly injected aren't samples from pi_theta -- mask them out so
        # they don't contribute to the policy gradient.
        comp_mask = comp_mask * (1.0 - forced_mask)
        texts = self.tok.batch_decode(seqs[:, plen:], skip_special_tokens=True)
        # pi_old per-token, computed in no-grad micro-chunks so peak memory stays bounded
        old_logp = torch.cat([self._token_logprobs(seqs[s:s + self.micro])
                              for s in range(0, seqs.shape[0], self.micro)], dim=0)
        return seqs, comp_mask[:, 1:], old_logp, texts, plen

    def _token_logprobs(self, seqs, grad=False):
        """Per-token log pi(token_t | <t) aligned to seqs[:,1:]. (B, T-1).

        Memory-efficient: log p(tgt) = logit_tgt - logsumexp(logits). Avoids
        materializing a full (B, T, vocab) log_softmax tensor (vocab ~152k -> ~5GB
        per copy; the naive version OOMs a 16GB GPU at batch 32). Keeps logits in the
        model dtype (bf16); only the (B, T) results are upcast to fp32.
        """
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            logits = self.model(seqs).logits[:, :-1, :]              # (B, T-1, V)
            tgt = seqs[:, 1:].unsqueeze(-1)                          # (B, T-1, 1)
            tgt_logit = logits.gather(-1, tgt).squeeze(-1)          # (B, T-1)
            lse = torch.logsumexp(logits, dim=-1)                   # (B, T-1), reduction
            return (tgt_logit - lse).float()

    # --- ask / tell ----------------------------------------------------------
    def ask(self) -> list[Node]:
        prompt_ids = self._prompt_ids()
        seqs, comp_mask, old_logp, texts, _ = self._generate(prompt_ids)
        self._pending = (seqs, comp_mask, old_logp)
        cands, n_valid = [], 0
        for text in texts:
            text = self._extract_formula(text)
            if self.const_placeholder:
                from layer1.constfit import parse_and_fit
                node = parse_and_fit(text, self.task.x_train, self.task.y_train)
            else:
                node = parse_expression(text)
            if node is None:
                cands.append(INVALID)
            else:
                cands.append(node); n_valid += 1
        self._last_valid_frac = n_valid / max(self.batch_size, 1)
        self._last_responses = texts
        return cands

    def _weights(self, results: list[Result]) -> np.ndarray:
        R = np.array([r.reward for r in results], dtype=float)
        if self.arm == "greedy":
            w = greedy_weights(R)
        elif self.mode == "quantile":
            w = quantile_weights(R, self.epsilon)
        elif self.mode == "cvar":
            w = cvar_weights(R, self.epsilon)
        else:
            w, self._last_beta = entropic_weights(R, self.beta_rule, self.beta,
                                                  self.target_ess, self.target_kl)
        if self.std_normalize:                # standard-GRPO style normalization
            w = w / (w.std() + 1e-8)
        return w

    def tell(self, candidates, results: list[Result]) -> None:
        if getattr(self, "_pending", None) is None:
            return
        if self.arm in ("best_of_n", "evolution"):  # no weight update for these arms
            if self.arm == "evolution":              # update the in-context archive instead
                for cand, r in zip(candidates, results):
                    if cand is not INVALID:
                        self.archive.append((r.reward, to_infix(cand)))
                # dedupe by expr, keep the top-K by reward
                best = {}
                for r, e in self.archive:
                    if e not in best or r > best[e]:
                        best[e] = r
                self.archive = sorted(((r, e) for e, r in best.items()), reverse=True)[:self.archive_k]
            self._pending = None
            return
        seqs, comp_mask, old_logp = self._pending
        adv = torch.tensor(self._weights(results), dtype=torch.float32, device=self.device)
        cl, ch, tc = 1 - self.clip_low, 1 + self.clip_high, self.trunc_is
        B = seqs.shape[0]
        total_mask = comp_mask.sum().clamp(min=1)   # normalize over the FULL batch's tokens
        for _ in range(self.ppo_epochs):
            self.opt.zero_grad()
            ep_loss = 0.0
            # Micro-batch with gradient accumulation: each chunk's graph is freed by its
            # own backward(), so peak memory ~ self.micro, not the full batch (attention
            # activations are O(T^2)*batch*layers and OOM the GPU at batch 32).
            for s in range(0, B, self.micro):
                sl = slice(s, s + self.micro)
                logp = self._token_logprobs(seqs[sl], grad=True)        # (m, T-1)
                ratio = torch.clamp(torch.exp(logp - old_logp[sl]), max=tc)  # truncated IS
                a = adv[sl].unsqueeze(1)
                surr = torch.minimum(ratio * a, torch.clamp(ratio, cl, ch) * a)  # GRPO surrogate
                loss = -(surr * comp_mask[sl]).sum() / total_mask
                loss.backward()
                ep_loss += float(loss.detach())
            torch.nn.utils.clip_grad_norm_(
                (p for p in self.model.parameters() if p.requires_grad), 1.0)
            self.opt.step()
            self._last_loss = ep_loss
        self._pending = None

    def diagnostics(self) -> dict:
        return {"policy_entropy": float("nan"), "valid_fraction": self._last_valid_frac,
                "beta": self._last_beta, "loss": self._last_loss}

    def last_io(self) -> dict:
        return {"prompt": self._prompt_text(), "responses": list(self._last_responses)}
