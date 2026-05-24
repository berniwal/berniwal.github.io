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
tc = st.columns([1, 1, 3])
target = tc[0].selectbox("Target  f(x)", ["easy", "medium", "harder"], index=1)
seed = int(tc[1].number_input("Seed", min_value=0, max_value=9999, value=0, step=1))
task = eng.make_task(target, seed=seed)
tc[2].caption(f"Target formula:  f(x) = {to_infix(task.target_expr)}")

tab_l0, tab_l1 = st.tabs(["Layer 0  (search / RL)", "Layer 1  (LLM / MLX)"])

# =====================================================================
# Layer 0 tab: its own controls + plots (left), scoreboard + batch (right)
# =====================================================================
with tab_l0:
    ctrl = st.columns([1, 3])
    batch_size = int(ctrl[0].select_slider("Batch size", options=[20, 50, 100, 200],
                                           value=200))
    methods = ctrl[1].multiselect(
        "Methods", options=list(eng.LAYER0_PRESETS),
        default=list(eng.LAYER0_PRESETS), format_func=lambda k: STYLE[k][1])

    sig = (target, seed, batch_size, tuple(methods))
    if st.session_state.get("sig") != sig:
        st.session_state.states = {
            m: eng.build_layer0_state(m, task, seed, batch_size) for m in methods}
        st.session_state.sig = sig
    states = st.session_state.states

    sc = st.columns([1, 1, 1, 1, 3])
    n_step = 0
    if sc[0].button("Step +1", use_container_width=True):
        n_step = 1
    if sc[1].button("+10", use_container_width=True):
        n_step = 10
    if sc[2].button("+100", use_container_width=True):
        n_step = 100
    if sc[3].button("Reset", use_container_width=True):
        st.session_state.states = {
            m: eng.build_layer0_state(m, task, seed, batch_size) for m in methods}
        states = st.session_state.states
    if n_step:
        for s in states.values():
            eng.step(s, n_step)
    cur_calls = max((s.verifier.calls for s in states.values()), default=0)
    sc[4].caption(f"verifier calls / method: **{cur_calls:,}**  ·  same task, same "
                  "verifier, same budget — only the proposer differs.")

    if not states:
        st.info("Select at least one method above.")
    else:
        left, right = st.columns([3, 2])
        with left:
            f1 = _line_fig(states, "best_reward", "best-so-far reward")
            st.pyplot(f1); plt.close(f1)
            f2 = _line_fig(states, "diversity", "batch diversity (unique fraction)",
                           ylim=(-0.03, 1.03))
            st.pyplot(f2); plt.close(f2)
            st.caption("Diversity → 0 means the policy mode-collapsed onto one expression.")
            focus = st.selectbox("Inspect method", options=list(states),
                                 format_func=lambda k: STYLE[k][1])
            ffit = _fit_fig(task, states[focus].best_expr, color=STYLE[focus][0])
            st.pyplot(ffit); plt.close(ffit)
        with right:
            st.markdown("**Scoreboard**")
            st.table([{"method": STYLE[m][1],
                       "best reward": f"{s.best:.4f}",
                       "diversity": f"{s.diversity[-1]:.2f}" if s.diversity else "-",
                       "solved@calls": f"{s.evals_to_solve:,}" if s.success else "—"}
                      for m, s in states.items()])
            st.caption("Policy entropy (RL arms) — same collapse story as diversity:")
            f3 = _line_fig(states, "entropy", "policy entropy (nats)")
            st.pyplot(f3); plt.close(f3)
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
        p1, p2, p3 = st.columns(3)
        l1_bs = int(p1.slider("Batch size", 2, 16, 8))
        l1_tok = int(p2.slider("Max tokens", 16, 96, 48))
        l1_temp = float(p3.slider("Temperature", 0.0, 1.5, 1.0, 0.1,
                                  help="Higher = more diverse samples. "
                                       "0 = greedy/deterministic (identical samples)."))

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
                    batch_size=l1_bs, max_tokens=l1_tok, temperature=l1_temp)
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
                ax.plot(xs, [f.best for f in rep.frames], color="#7570b3", lw=2,
                        label="best-so-far")
                ax.plot(xs, [f.batch_mean for f in rep.frames], color="#999999", lw=1.2,
                        ls=":", label="batch mean (learning signal)")
                ax.axvline(fr.calls, color="#d95f02", lw=1.2)
                ax.set_xlabel("verifier calls"); ax.set_ylabel("reward")
                ax.grid(alpha=0.3); ax.legend(fontsize=7)
                fig.tight_layout()
                st.pyplot(fig); plt.close(fig)
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
                    batch_size=l1_bs, max_tokens=l1_tok, temperature=l1_temp)
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
