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
from sia.expression import Node, leaf, parse_expression  # noqa: E402
from sia.objectives import (cvar_weights, entropic_weights, greedy_weights,  # noqa: E402
                            quantile_weights)
from sia.verifier import Result  # noqa: E402

INVALID = leaf("1.0")  # placeholder scored ~0 by the verifier (an unparseable sample)


class TorchLoRAProposer:
    def __init__(self, task, model_id="Qwen/Qwen2.5-0.5B-Instruct", arm="risk",
                 mode="quantile", epsilon=0.25, beta=2.0, beta_rule="fixed",
                 target_ess=0.3, target_kl=1.0, batch_size=16, lr=1e-6,
                 weight_decay=0.1, lora_rank=16, lora_alpha=32, lora_dropout=0.0,
                 max_new_tokens=48, temperature=1.0, top_p=0.95,
                 const_placeholder=True, n_data_shown=12,
                 ppo_epochs=2, clip_low=0.2, clip_high=0.28, trunc_is=2.0,
                 std_normalize=False, seed=0, dtype="bfloat16"):
        if arm not in ("greedy", "risk"):
            raise ValueError("arm must be greedy or risk")
        if mode not in ("quantile", "entropic", "cvar"):
            raise ValueError("mode must be quantile/entropic/cvar")
        self.task = task
        self.arm, self.mode = arm, mode
        self.epsilon, self.beta, self.beta_rule = epsilon, beta, beta_rule
        self.target_ess, self.target_kl = target_ess, target_kl
        self.batch_size = batch_size
        self.max_new_tokens, self.temperature, self.top_p = max_new_tokens, temperature, top_p
        self.const_placeholder, self.n_data_shown = const_placeholder, n_data_shown
        self.ppo_epochs = ppo_epochs
        self.clip_low, self.clip_high, self.trunc_is = clip_low, clip_high, trunc_is
        self.std_normalize = std_normalize
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
        return ("You are doing symbolic regression. Find a formula y = f(x) that fits "
                f"the data.\n{vocab}\nThe data may be nonlinear or periodic -- consider "
                "terms like x*x, x*x*x, or sin/cos, not just straight lines.\nReply with "
                "ONLY the formula for f(x) on a single line -- no words, no 'y =', no code "
                f"fences.\n\nData points:\n{self._data_block()}\n\nNew formula for f(x):")

    def _prompt_ids(self) -> torch.Tensor:
        ids = self.tok.apply_chat_template(
            [{"role": "user", "content": self._prompt_text()}],
            add_generation_prompt=True, return_tensors="pt")
        return ids.to(self.device)

    # --- generation ----------------------------------------------------------
    @torch.no_grad()
    def _generate(self, prompt_ids):
        plen = prompt_ids.shape[1]
        inp = prompt_ids.repeat(self.batch_size, 1)
        seqs = self.model.generate(
            inp, attention_mask=torch.ones_like(inp),
            do_sample=True, temperature=self.temperature, top_p=self.top_p,
            max_new_tokens=self.max_new_tokens, pad_token_id=self.tok.pad_token_id)  # (B, plen+gen)
        comp_mask = torch.zeros_like(seqs, dtype=torch.float32)
        comp_mask[:, plen:] = (seqs[:, plen:] != self.tok.pad_token_id).float()
        texts = self.tok.batch_decode(seqs[:, plen:], skip_special_tokens=True)
        old_logp = self._token_logprobs(seqs)  # pi_old, per-token (B, T-1)
        return seqs, comp_mask[:, 1:], old_logp, texts, plen

    def _token_logprobs(self, seqs, grad=False):
        """Per-token log pi(token_t | <t) aligned to seqs[:,1:]. (B, T-1)."""
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            logits = self.model(seqs).logits[:, :-1, :]      # predict tokens 1..T-1
            logp = F.log_softmax(logits.float(), dim=-1)
            return logp.gather(-1, seqs[:, 1:].unsqueeze(-1)).squeeze(-1)

    # --- ask / tell ----------------------------------------------------------
    def ask(self) -> list[Node]:
        prompt_ids = self._prompt_ids()
        seqs, comp_mask, old_logp, texts, _ = self._generate(prompt_ids)
        self._pending = (seqs, comp_mask, old_logp)
        cands, n_valid = [], 0
        for text in texts:
            text = text.strip().splitlines()[0] if text.strip() else ""
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
        seqs, comp_mask, old_logp = self._pending
        adv = torch.tensor(self._weights(results), dtype=torch.float32, device=self.device)
        adv_tok = adv.unsqueeze(1)            # (B,1) broadcast over completion tokens
        cl, ch, tc = 1 - self.clip_low, 1 + self.clip_high, self.trunc_is
        for _ in range(self.ppo_epochs):
            logp = self._token_logprobs(seqs, grad=True)        # (B, T-1)
            ratio = torch.exp(logp - old_logp)
            ratio = torch.clamp(ratio, max=tc)                  # truncated importance sampling
            unclipped = ratio * adv_tok
            clipped = torch.clamp(ratio, cl, ch) * adv_tok
            surr = torch.minimum(unclipped, clipped)            # GRPO clipped surrogate
            loss = -(surr * comp_mask).sum() / comp_mask.sum().clamp(min=1)
            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                (p for p in self.model.parameters() if p.requires_grad), 1.0)
            self.opt.step()
            self._last_loss = float(loss.detach())
        self._pending = None

    def diagnostics(self) -> dict:
        return {"policy_entropy": float("nan"), "valid_fraction": self._last_valid_frac,
                "beta": self._last_beta, "loss": self._last_loss}

    def last_io(self) -> dict:
        return {"prompt": self._prompt_text(), "responses": list(self._last_responses)}
