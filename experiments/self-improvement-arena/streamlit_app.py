"""Interactive visualizer for the Self-Improvement Arena.

Step the search/RL methods batch-by-batch on the SAME task and watch the dynamics
the offline figures only summarize: best-so-far reward, the greedy collapse (batch
diversity / policy entropy crashing to ~0), and whether the current best actually
fits the data. Layer 0 runs instantly (numpy); the Layer 1 (LLM/LoRA) panel lights
up only on Apple Silicon and otherwise explains why it is disabled.

    pip install -e ".[app]"          # then:
    streamlit run streamlit_app.py

All the stateful logic lives in app_engine.py (no Streamlit there, so it is
unit-tested separately); this file is just the view.
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
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
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


# --- sidebar: controls -------------------------------------------------------
st.sidebar.title("Self-Improvement Arena")
st.sidebar.caption("Step the methods batch-by-batch and watch evolution vs. RL "
                   "dynamics emerge.")

target = st.sidebar.selectbox("Target  f(x)", ["easy", "medium", "harder"], index=1)
seed = int(st.sidebar.number_input("Seed", min_value=0, max_value=9999, value=0, step=1))
batch_size = int(st.sidebar.select_slider(
    "Batch size (verifier calls per step)", options=[20, 50, 100, 200], value=200))
methods = st.sidebar.multiselect(
    "Methods", options=list(eng.LAYER0_PRESETS), default=list(eng.LAYER0_PRESETS),
    format_func=lambda k: STYLE[k][1])

# (Re)build state when the config changes -- a different task/seed/batch/method set
# is a different experiment, so it resets cleanly and reproducibly.
sig = (target, seed, batch_size, tuple(methods))
if st.session_state.get("sig") != sig:
    task = eng.make_task(target, seed=seed)
    st.session_state.task = task
    st.session_state.states = {
        m: eng.build_layer0_state(m, task, seed, batch_size) for m in methods}
    st.session_state.sig = sig

task = st.session_state.task
states = st.session_state.states

st.sidebar.markdown("**Step the search**")
b1, b2, b3 = st.sidebar.columns(3)
n_step = 0
if b1.button("Step +1", use_container_width=True):
    n_step = 1
if b2.button("+10", use_container_width=True):
    n_step = 10
if b3.button("+100", use_container_width=True):
    n_step = 100
if st.sidebar.button("Reset", use_container_width=True):
    st.session_state.states = {
        m: eng.build_layer0_state(m, task, seed, batch_size) for m in methods}
    states = st.session_state.states
if n_step:
    for s in states.values():
        eng.step(s, n_step)

cur_steps = max((s.steps for s in states.values()), default=0)
cur_calls = max((s.verifier.calls for s in states.values()), default=0)
st.sidebar.metric("Steps taken", cur_steps)
st.sidebar.metric("Verifier calls (per method)", f"{cur_calls:,}")

# --- main: header + per-method scoreboard ------------------------------------
st.subheader(f"Target: {target}   —   f(x) = {to_infix(task.target_expr)}")
st.caption("Same task, same verifier, same budget — only the proposer differs. "
           "Watch greedy's diversity collapse while evolution / risk-seeking stay broad.")

if states:
    cols = st.columns(len(states))
    for col, (m, s) in zip(cols, states.items()):
        col.markdown(f"**{STYLE[m][1]}**")
        col.metric("best reward", f"{s.best:.4f}")
        col.metric("batch diversity", f"{s.diversity[-1]:.2f}" if s.diversity else "-")
        col.caption(f"solved at {s.evals_to_solve:,} calls" if s.success else "not solved")
else:
    st.info("Select at least one method in the sidebar.")

tab_prog, tab_fit, tab_l1 = st.tabs(
    ["Progress", "Fit & batch detail", "Layer 1 (LLM / MLX)"])

with tab_prog:
    if states:
        c1, c2 = st.columns(2)
        f1 = _line_fig(states, "best_reward", "best-so-far reward")
        c1.pyplot(f1); plt.close(f1)
        f2 = _line_fig(states, "diversity", "batch diversity (unique fraction)",
                       ylim=(-0.03, 1.03))
        c2.pyplot(f2); plt.close(f2)
        st.caption("Diversity near 0 = the policy has mode-collapsed onto a single "
                   "expression. Below: policy entropy (RL arms only).")
        f3 = _line_fig(states, "entropy", "policy entropy (nats)")
        st.pyplot(f3); plt.close(f3)
    else:
        st.write("Nothing to plot yet.")

with tab_fit:
    if states:
        focus = st.selectbox("Focus method", options=list(states),
                             format_func=lambda k: STYLE[k][1])
        s = states[focus]
        xs, y_target, y_pred = eng.fit_overlay(task, s.best_expr)
        fig, ax = plt.subplots(figsize=(7, 3.6))
        ax.scatter(task.x_train, task.y_train, s=18, color="#333333",
                   label="training data", zorder=3)
        ax.plot(xs, y_target, color="#1b9e77", lw=2, label="target f(x)")
        if y_pred is not None:
            ax.plot(xs, y_pred, color=STYLE[focus][0], lw=2, ls="--",
                    label=f"best so far: {to_infix(s.best_expr)}")
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
        fig.tight_layout()
        st.pyplot(fig); plt.close(fig)

        st.markdown("**Latest batch — most frequent expressions** "
                    "(one row dominating the count = collapse):")
        rows = eng.batch_summary(s, k=8)
        if rows:
            st.table([{"expression": e, "count": n, "reward": f"{r:.4f}"}
                      for e, n, r in rows])
        else:
            st.caption("Step the search to populate a batch.")
    else:
        st.write("Select a method to inspect its fit.")

with tab_l1:
    st.markdown("### Layer 1 — LLM proposer (evolution / greedy LoRA / risk LoRA)")
    if not eng.mlx_available():
        st.warning(
            "Layer 1 needs Apple Silicon + `mlx-lm`, which is not available here "
            f"(detected platform: **{eng.platform_label()}**). Everything else in "
            "this app works regardless. On a Mac, install it with "
            "`pip install -e \".[layer1]\"` and reload.")
        st.caption("Layer 1 swaps the proposer for an LLM behind the SAME ask/tell "
                   "seam; the LoRA arms reuse Layer 0's exact reward weighting "
                   "(sia/objectives.py). See layer1/README.md.")
    else:
        st.caption("One LLM generation per candidate, so steps here take seconds, "
                   "not milliseconds — use a small batch and a few steps to build "
                   "intuition, not a full sweep.")
        arm = st.selectbox("Arm", list(eng.LAYER1_ARMS), index=1)
        model_id = st.text_input("Model", "mlx-community/Qwen2.5-3B-Instruct-4bit")
        c1, c2 = st.columns(2)
        l1_bs = int(c1.slider("Batch size", 2, 16, 8))
        l1_tok = int(c2.slider("Max tokens", 16, 96, 48))

        a, b = st.columns(2)
        if a.button("Load model & (re)start run", use_container_width=True):
            with st.spinner(f"Loading {model_id} ... (first load downloads weights)"):
                from layer1.lora_proposer import load_model
                model, tok = load_model(model_id)
            t1 = eng.make_task(target, seed=seed)
            st.session_state.l1 = eng.build_layer1_state(
                arm, t1, seed, model, tok, batch_size=l1_bs, max_tokens=l1_tok)
            st.session_state.l1_task = t1
        if b.button("Step (1 batch)", use_container_width=True, disabled="l1" not in st.session_state):
            with st.spinner("Generating batch + LoRA gradient step ..."):
                eng.step(st.session_state.l1, 1)

        if "l1" in st.session_state:
            s = st.session_state.l1
            t1 = st.session_state.l1_task
            m1, m2, m3 = st.columns(3)
            m1.metric("best reward", f"{s.best:.4f}")
            m2.metric("valid fraction", f"{s.valid_frac[-1]:.2f}" if s.valid_frac else "-")
            m3.metric("generations", f"{s.verifier.calls}")
            xs, y_target, y_pred = eng.fit_overlay(t1, s.best_expr)
            fig, ax = plt.subplots(figsize=(7, 3.4))
            ax.scatter(t1.x_train, t1.y_train, s=18, color="#333333",
                       label="training data", zorder=3)
            ax.plot(xs, y_target, color="#1b9e77", lw=2, label="target f(x)")
            if y_pred is not None:
                ax.plot(xs, y_pred, color="#d95f02", lw=2, ls="--",
                        label=f"best: {to_infix(s.best_expr)}")
            ax.set_xlabel("x"); ax.set_ylabel("y"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
            fig.tight_layout()
            st.pyplot(fig); plt.close(fig)
            rows = eng.batch_summary(s, k=8)
            if rows:
                st.table([{"expression": e, "count": n, "reward": f"{r:.4f}"}
                          for e, n, r in rows])
