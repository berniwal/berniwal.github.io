"""Interactive visualizer for the Self-Improvement Arena.

Step the search/RL methods batch-by-batch on the SAME task and watch the dynamics
the offline figures only summarize: best-so-far reward, the greedy collapse (batch
diversity / policy entropy crashing to ~0), and whether the current best actually
fits the data. Layer 0 runs instantly (numpy); the Layer 1 (LLM/LoRA) panel lights
up only on Apple Silicon and otherwise explains why it is disabled.

    pip install -e ".[app]"          # then:
    streamlit run streamlit_app.py

Layout: no sidebar. Only the target/seed are global; each tab owns its own controls
(so the Layer-0 knobs don't clutter the Layer-1 view) and is a two-column view --
plots + parameters on the left, expression/reward detail (and, for Layer 1, the
prompt + raw responses) on the right. All stateful logic lives in app_engine.py.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st

import app_engine as eng
from sia.expression import to_infix
from sia.plotting import STYLE

st.set_page_config(page_title="Self-Improvement Arena", layout="wide")


def _line_fig(states, attr, ylabel, ylim=None):
    """Overlay one per-step metric across the selected methods, using the shared
    STYLE colors so the live view matches the offline figures."""
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    for key, s in states.items():
        ys = getattr(s, attr)
        if ys:
            ax.plot(s.calls, ys, color=STYLE[key][0], label=STYLE[key][1], lw=2)
    ax.set_xlabel("verifier calls")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    if ylim:
        ax.set_ylim(*ylim)
    if any(getattr(s, attr) for s in states.values()):
        ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


def _fit_fig(task, expr, color="#d95f02"):
    xs, y_target, y_pred = eng.fit_overlay(task, expr)
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    ax.scatter(task.x_train, task.y_train, s=18, color="#333333",
               label="training data", zorder=3)
    ax.plot(xs, y_target, color="#1b9e77", lw=2, label="target f(x)")
    if y_pred is not None:
        ax.plot(xs, y_pred, color=color, lw=2, ls="--",
                label=f"best: {to_infix(expr)}")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# --- top bar: only the shared task controls (no sidebar) ---------------------
st.title("Self-Improvement Arena")
tc = st.columns([1, 1, 1, 3])
target = tc[0].selectbox(
    "Target  f(x)",
    ["easy", "medium", "harder"] + [f"nguyen-{i}" for i in range(1, 9)], index=1,
    help="easy/medium/harder use the base grammar; nguyen-* use DSR's Koza grammar "
         "{+,-,*,/,sin,cos,exp,log,x} (no constants).")
seed = int(tc[1].number_input("Seed", min_value=0, max_value=9999, value=0, step=1))
reward_mode = tc[2].selectbox(
    "Reward", ["mse", "nrmse"], index=0,
    help="mse: 1/(1+MSE). nrmse: 1/(1+RMSE/std(y)) — scale-invariant (DSR-style); "
         "predict-the-mean = 0.5 on every target, gentler gradient.")
task = eng.make_task(target, seed=seed)
tc[3].caption(f"f(x) = {eng.target_label(task)}  ·  reward = **{reward_mode}**  ·  "
              f"grammar = {len(task.grammar.tokens)} tokens")

tab_l0, tab_l1 = st.tabs(["Layer 0  (search / RL)", "Layer 1  (LLM / MLX)"])

# =====================================================================
# Layer 0 tab: its own controls + plots (left), scoreboard + batch (right)
# =====================================================================
with tab_l0:
    ctrl = st.columns([1, 1, 1, 2])
    batch_size = int(ctrl[0].select_slider("Batch size", options=[100, 200, 500, 1000],
                                           value=1000,
                                           help="DSR uses 1000; lower it for snappier "
                                                "stepping."))
    constraints = ctrl[1].checkbox(
        "Constraints", value=True,
        help="DSR's 4 a-priori constraints (no nested trig / all-constant operands / "
             "inverse-of-unary; min+max length). Off = the collapse regime.")
    hierarchical = ctrl[2].checkbox(
        "Hierarchical entropy", value=False,
        help="Discount the entropy bonus by gamma^t so early/structural tokens weigh "
             "most (Landajuela 2021): sets gamma=0.85, ent_coef=0.02.")
    methods = ctrl[3].multiselect(
        "Methods", options=list(eng.LAYER0_PRESETS),
        default=list(eng.LAYER0_PRESETS), format_func=lambda k: STYLE[k][1])

    sig = (target, seed, batch_size, tuple(methods), reward_mode, constraints, hierarchical)
    if st.session_state.get("sig") != sig:
        st.session_state.states = {
            m: eng.build_layer0_state(m, task, seed, batch_size, reward_mode,
                                      constraints, hierarchical)
            for m in methods}
        st.session_state.sig = sig
    states = st.session_state.states

    sc = st.columns([1, 1, 1, 1, 1, 3])
    n_step = 0
    if sc[0].button("Step +1", use_container_width=True):
        n_step = 1
    if sc[1].button("+10", use_container_width=True):
        n_step = 10
    if sc[2].button("+100", use_container_width=True):
        n_step = 100
    if sc[3].button("+1000", use_container_width=True):
        n_step = 1000
    if sc[4].button("Reset", use_container_width=True):
        st.session_state.states = {
            m: eng.build_layer0_state(m, task, seed, batch_size, reward_mode,
                                      constraints, hierarchical)
            for m in methods}
        states = st.session_state.states
    if n_step:
        for s in states.values():
            eng.step(s, n_step)
    cur_calls = max((s.verifier.calls for s in states.values()), default=0)
    sc[5].caption(f"verifier calls / method: **{cur_calls:,}**  ·  same task, same "
                  "verifier, same budget — only the proposer differs.")

    if not states:
        st.info("Select at least one method above.")
    else:
        # --- Scoreboard: full width across the top ---
        st.markdown("**Scoreboard**")
        ranked = sorted(states.items(), key=lambda kv: kv[1].best, reverse=True)
        st.table([{"method": STYLE[m][1],
                   "best reward": f"{s.best:.4f}",
                   "success (batch)": f"{s.success_frac[-1]:.2f}" if s.success_frac else "-",
                   "diversity": f"{s.diversity[-1]:.2f}" if s.diversity else "-",
                   "solved@calls": f"{s.evals_to_solve:,}" if s.success else "—"}
                  for m, s in ranked])

        # --- 2x2 grid of dynamics plots ---
        ga = st.columns(2)
        f1 = _line_fig(states, "best_reward", "best-so-far reward")
        ga[0].pyplot(f1); plt.close(f1)
        fs = _line_fig(states, "success_frac", "batch success rate (held-out)",
                       ylim=(-0.03, 1.03))
        ga[1].pyplot(fs); plt.close(fs)
        ga[1].caption("Fraction of each batch that recovers the target on held-out data.")
        gb = st.columns(2)
        f2 = _line_fig(states, "diversity", "batch diversity (unique fraction)",
                       ylim=(-0.03, 1.03))
        gb[0].pyplot(f2); plt.close(f2)
        gb[0].caption("Diversity → 0 means the policy mode-collapsed onto one expression.")
        f3 = _line_fig(states, "entropy", "policy entropy (nats)")
        gb[1].pyplot(f3); plt.close(f3)
        gb[1].caption("Policy entropy (RL arms) — same collapse story as diversity.")

        # --- Inspect one method: fit curve + latest batch, same height ---
        focus = st.selectbox("Inspect method", options=list(states),
                             format_func=lambda k: STYLE[k][1])
        ins = st.columns([3, 2])
        with ins[0]:
            ffit = _fit_fig(task, states[focus].best_expr, color=STYLE[focus][0])
            st.pyplot(ffit); plt.close(ffit)
        with ins[1]:
            st.markdown(f"**Latest batch — {STYLE[focus][1]}** "
                        "(one row dominating = collapse):")
            rows = eng.batch_summary(states[focus], k=8)
            if rows:
                st.table([{"expression": e, "count": n, "reward": f"{r:.4f}"}
                          for e, n, r in rows])
            else:
                st.caption("Step the search to populate a batch.")

# =====================================================================
# Layer 1 tab: shared controls, then (A) live step and (B) full run + replay
# =====================================================================
with tab_l1:
    st.markdown("#### Layer 1 — LLM proposer (evolution / greedy LoRA / risk LoRA)")
    if not eng.mlx_available():
        st.warning(
            "Layer 1 needs Apple Silicon + `mlx-lm`, not available here "
            f"(platform: **{eng.platform_label()}**). Everything else works "
            "regardless. On a Mac: `pip install -e \".[layer1]\"` and reload.")
        st.caption("Layer 1 swaps the proposer for an LLM behind the SAME ask/tell "
                   "seam; the LoRA arms reuse Layer 0's exact reward weighting "
                   "(sia/objectives.py). See layer1/README.md.")
    else:
        cc = st.columns([1, 3])
        arm = cc[0].selectbox("Arm", list(eng.LAYER1_ARMS), index=1)
        model_id = cc[1].text_input("Model", "mlx-community/Qwen2.5-3B-Instruct-4bit")
        p1, p2, p3, p4 = st.columns(4)
        l1_bs = int(p1.slider("Batch size", 2, 16, 8))
        l1_tok = int(p2.slider("Max tokens", 16, 96, 48))
        l1_temp = float(p3.slider("Temperature", 0.0, 1.5, 1.0, 0.1,
                                  help="Higher = more diverse samples. "
                                       "0 = greedy/deterministic (identical samples)."))
        l1_kl = float(p4.slider("KL coef", 0.0, 0.3, 0.1, 0.02,
                                help="Trust-region strength toward the base model "
                                     "(LoRA arms). Higher = more stable but snaps back "
                                     "off solutions; lower = commits more but risks "
                                     "degeneration. GRPO uses 0.04."))

        # ---------- A. Full run + step-through (the main investigation tool) ----------
        st.markdown("##### Full run, then step through it")
        st.caption("Compute a whole run at once (like one run from the experiment), "
                   "then scrub through every batch: best-so-far, this batch's "
                   "proposals, and the exact prompt + responses.")
        fr1, fr2 = st.columns([1, 2])
        full_budget = int(fr1.slider("Budget (generations)", 32, 512, 256, 32))
        n_batches = full_budget // l1_bs
        fr2.caption(f"= {n_batches} batches of {l1_bs}  ·  arm **{arm}**, temp {l1_temp}. "
                    "Runs all at once (minutes); steps below are instant replay.")
        if st.button(f"▶ Run full run — {full_budget} generations (takes minutes)",
                     use_container_width=True):
            from layer1.lora_proposer import load_model
            t1 = eng.make_task(target, seed=seed)
            with st.spinner(f"Loading {model_id} ..."):
                model, tok = eng.mlx_call(load_model, model_id)  # fresh model (LoRA mutates)
                state = eng.mlx_call(
                    eng.build_layer1_state, arm, t1, seed, model, tok,
                    batch_size=l1_bs, max_tokens=l1_tok, temperature=l1_temp,
                    reward_mode=reward_mode, kl_coef=l1_kl)
            # Drive the loop here (one mlx_call per batch) so the progress bar updates.
            pbar = st.progress(0.0, text=f"batch 0/{n_batches}")
            frames = []
            for i in range(n_batches):
                fr = eng.mlx_call(eng.run_batch, state)
                frames.append(fr)
                pbar.progress((i + 1) / n_batches,
                              text=f"batch {i+1}/{n_batches}  ·  best {fr.best:.3f}  ·  "
                                   f"batch-mean {fr.batch_mean:.3f}  ·  valid {fr.valid_frac:.2f}")
            pbar.empty()
            st.session_state.l1_replay = eng.RunReplay(arm, t1.name, full_budget,
                                                       l1_bs, frames)
            st.session_state.l1_replay_task = t1

        if "l1_replay" in st.session_state:
            rep = st.session_state.l1_replay
            rt = st.session_state.l1_replay_task
            nf = len(rep.frames)
            bi = st.slider("Step through batches", 1, nf, nf,
                           help="Replay any batch of the finished run.") - 1
            fr = rep.frames[bi]
            rl, rr = st.columns([3, 2])
            with rl:
                st.caption(f"**{rep.arm}** on **{rep.target}** — batch {fr.batch}/{nf} "
                           f"({fr.calls} generations)")
                fig, ax = plt.subplots(figsize=(6.2, 3.0))
                xs = [f.calls for f in rep.frames]
                lh = []
                lh.append(ax.plot(xs, [f.best for f in rep.frames], color="#7570b3",
                                  lw=2, label="best-so-far")[0])
                lh.append(ax.plot(xs, [f.batch_mean for f in rep.frames], color="#999999",
                                  lw=1.2, ls=":", label="batch mean")[0])
                ax.axvline(fr.calls, color="#d95f02", lw=1.2)
                ax.set_xlabel("verifier calls"); ax.set_ylabel("reward"); ax.grid(alpha=0.3)
                ax2 = ax.twinx()  # batch diversity on the right axis -> see collapse vs reward
                lh.append(ax2.plot(xs, [f.diversity for f in rep.frames], color="#e7298a",
                                   lw=1.2, label="batch diversity")[0])
                ax2.set_ylabel("batch diversity", color="#e7298a"); ax2.set_ylim(-0.03, 1.03)
                ax2.tick_params(axis="y", colors="#e7298a")
                ax.legend(handles=lh, fontsize=7, loc="best")
                fig.tight_layout()
                st.pyplot(fig); plt.close(fig)

                # loss components (LoRA arms only) -- shows the PG-vanishes / KL-dominates story
                pgs = [abs(f.pg_loss) for f in rep.frames]
                if any(p == p for p in pgs):  # not all NaN -> a LoRA arm
                    eps = 1e-4  # floor so log scale can show "PG -> 0"
                    figl, axl = plt.subplots(figsize=(6.2, 2.6))
                    axl.plot(xs, [max(abs(f.pg_loss), eps) for f in rep.frames],
                             color="#1b9e77", lw=1.5, label="|PG loss|")
                    axl.plot(xs, [max(f.kl_term, eps) for f in rep.frames],
                             color="#d95f02", lw=1.5, label="KL term (kl_coef·KL)")
                    axl.axvline(fr.calls, color="#d95f02", lw=1.0, alpha=0.5)
                    axl.set_yscale("log")
                    axl.set_xlabel("verifier calls"); axl.set_ylabel("loss component (log)")
                    axl.grid(alpha=0.3, which="both"); axl.legend(fontsize=7)
                    figl.tight_layout()
                    st.pyplot(figl); plt.close(figl)
                    st.caption("When **|PG| → 0** (batch rewards equalize) but **KL stays "
                               "large**, the KL term dominates and drags the policy back "
                               "toward the base model — the snap-back off a converged solution.")

                ffit = _fit_fig(rt, fr.best_expr)
                st.pyplot(ffit); plt.close(ffit)
                g1, g2, g3 = st.columns(3)
                g1.metric("best so far", f"{fr.best:.4f}")
                g2.metric("batch mean", f"{fr.batch_mean:.4f}")
                g3.metric("valid frac", f"{fr.valid_frac:.2f}")
            with rr:
                st.markdown(f"**Batch {fr.batch} proposals** "
                            f"({len(fr.rows)} unique of {rep.batch_size})")
                st.table([{"expression": e, "count": n, "reward": f"{r:.4f}"}
                          for e, n, r in fr.rows])
                with st.expander("Prompt & raw responses (this batch)"):
                    st.markdown("**Prompt:**")
                    st.code(fr.prompt or "(none)", language="text")
                    st.markdown("**Raw responses:**")
                    for i, r in enumerate(fr.responses):
                        st.code(f"[{i}] {r}", language="text")

        # ---------- B. Live single-step (quick poking) ----------
        st.divider()
        with st.expander("Live single-step (poke one batch at a time)"):
            a, b = st.columns(2)
            if a.button("Load model & (re)start", use_container_width=True):
                with st.spinner(f"Loading {model_id} ..."):
                    from layer1.lora_proposer import load_model
                    model, tok = eng.mlx_call(load_model, model_id)
                t1 = eng.make_task(target, seed=seed)
                st.session_state.l1 = eng.mlx_call(
                    eng.build_layer1_state, arm, t1, seed, model, tok,
                    batch_size=l1_bs, max_tokens=l1_tok, temperature=l1_temp,
                    reward_mode=reward_mode, kl_coef=l1_kl)
                st.session_state.l1_task = t1
            if b.button("Step (1 batch)", use_container_width=True,
                        disabled="l1" not in st.session_state):
                with st.spinner("Generating batch + LoRA gradient step ..."):
                    eng.mlx_call(eng.step, st.session_state.l1, 1)
            if "l1" in st.session_state:
                s = st.session_state.l1
                m1, m2, m3 = st.columns(3)
                m1.metric("best reward", f"{s.best:.4f}")
                m2.metric("valid fraction",
                          f"{s.valid_frac[-1]:.2f}" if s.valid_frac else "-")
                m3.metric("generations", f"{s.verifier.calls}")
                rows = eng.batch_summary(s, k=l1_bs)
                if rows:
                    st.table([{"expression": e, "count": n, "reward": f"{r:.4f}"}
                              for e, n, r in rows])
