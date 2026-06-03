"""
Main Streamlit entry point — Flatland 5 Agents RL Dashboard.

Launch: streamlit run streamlit_app/app.py
"""

import os
import sys
import subprocess
import time

import streamlit as st
import pandas as pd
import plotly.express as px

# Ensure project root on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils.metrics_utils import load_metrics, get_best_algorithm, compare_all_algorithms
from streamlit_app.components.reward_chart import plot_rewards

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Flatland 5 Agents — RL Dashboard",
    page_icon="🚂",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state defaults ────────────────────────────────────────────────────
if "training_proc" not in st.session_state:
    st.session_state.training_proc = None
if "selected_algo" not in st.session_state:
    st.session_state.selected_algo = "PPO"
if "num_agents" not in st.session_state:
    st.session_state.num_agents = 5
if "map_frame" not in st.session_state:
    st.session_state.map_frame = None

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🚂 Flatland RL")
    st.markdown("---")

    mode = st.radio("Mode", ["Comparer tous", "PPO", "APEX", "MARWIL"], index=0)
    st.session_state.selected_algo = mode

    st.session_state.num_agents = st.slider("Nombre d'agents", 1, 5, 5)

    st.markdown("---")
    st.subheader("Entraînement")

    algo_for_train = st.selectbox("Algorithme à entraîner", ["PPO", "APEX", "MARWIL"])
    smoke = st.checkbox("Smoke test (rapide)", value=True)

    col_train, col_stop = st.columns(2)
    with col_train:
        if st.button("🚀 Entraîner", use_container_width=True):
            cmd = [
                sys.executable,
                os.path.join(PROJECT_ROOT, "train.py"),
                "--algorithm", algo_for_train.lower(),
                "--num-cpus", "4",
            ]
            if smoke:
                cmd.append("--smoke-test")
            st.session_state.training_proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, cwd=PROJECT_ROOT,
            )
            st.success(f"Entraînement {algo_for_train} lancé !")

    with col_stop:
        if st.button("⏹ Arrêter", use_container_width=True):
            if st.session_state.training_proc:
                st.session_state.training_proc.terminate()
                st.session_state.training_proc = None
                st.warning("Entraînement arrêté.")

    st.markdown("---")
    if st.button("▶️ Évaluer", use_container_width=True):
        eval_cmd = [
            sys.executable,
            os.path.join(PROJECT_ROOT, "evaluate.py"),
            "--algorithm", algo_for_train.lower(),
            "--num-episodes", "3",
        ]
        subprocess.Popen(eval_cmd, cwd=PROJECT_ROOT)
        st.info("Évaluation lancée en arrière-plan.")

# ── Main header ───────────────────────────────────────────────────────────────
st.title("🚂 Flatland 5 Agents — RL Dashboard")
st.markdown("Entraînement distribué avec **PPO**, **APEX** et **MARWIL** · Carte ferroviaire fixe (seed=42)")

# ── Training progress bar ─────────────────────────────────────────────────────
if st.session_state.training_proc and st.session_state.training_proc.poll() is None:
    st.info("⏳ Entraînement en cours…")
    progress_bar = st.progress(0)
    log_area = st.empty()

    df_live = load_metrics(algo_for_train.lower())
    if not df_live.empty:
        max_expected = 100
        pct = min(len(df_live) / max_expected, 1.0)
        progress_bar.progress(pct)
        log_area.text(f"Épisodes enregistrés : {len(df_live)}")
elif st.session_state.training_proc and st.session_state.training_proc.poll() is not None:
    st.success("✅ Entraînement terminé.")
    st.session_state.training_proc = None

# ── 3-column live dashboard ───────────────────────────────────────────────────
col1, col2, col3 = st.columns([2, 1, 2])

with col1:
    st.subheader("🗺 Carte Flatland")
    map_placeholder = st.empty()
    try:
        from envs.flatland_env import FlatlandMultiAgentEnv
        if "live_env" not in st.session_state:
            st.session_state.live_env = FlatlandMultiAgentEnv()
            st.session_state.live_env.reset()
        frame_bytes = st.session_state.live_env.get_render_frame()
        map_placeholder.image(frame_bytes, caption="Carte fixe (seed 42)", use_column_width=True)
    except Exception as exc:
        map_placeholder.warning(f"Carte indisponible : {exc}")

with col2:
    st.subheader("📊 Métriques Live")
    target = mode if mode != "Comparer tous" else algo_for_train
    df_m = load_metrics(target.lower())
    if not df_m.empty:
        last = df_m.iloc[-1]
        st.metric("Reward (dernier ep.)", f"{last['reward']:.2f}")
        arrival_rate = last.get("arrival_rate", last.get("arrivals", 0) / 5.0)
        st.metric("Taux d'arrivée", f"{arrival_rate*100:.1f}%")
        st.metric("Steps", int(last["steps"]))
        st.metric("Deadlocks", int(last["deadlocks"]))
        st.metric("Épisodes total", len(df_m))
    else:
        st.info("Lance un entraînement pour voir les métriques ici.")

with col3:
    st.subheader("📈 Reward par épisode")
    target_algo = mode if mode != "Comparer tous" else algo_for_train
    df_chart = load_metrics(target_algo.lower())
    if not df_chart.empty:
        fig = plot_rewards(df_chart, target_algo)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Aucune donnée d'entraînement disponible.")

# ── Best algorithm banner ─────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🏆 Meilleur Algorithme")

comp_df = compare_all_algorithms()
best_algo = get_best_algorithm()

if comp_df["score"].max() > 0:
    best_row = comp_df[comp_df["algorithm"] == best_algo].iloc[0]
    st.success(
        f"**{best_algo}** — Score : `{best_row['score']:.4f}` | "
        f"Reward moyen : `{best_row['mean_reward']}` | "
        f"Taux d'arrivée : `{best_row['arrival_rate']*100:.1f}%`"
    )

    st.dataframe(
        comp_df.style.highlight_max(subset=["score"], color="#D5F5E3"),
        use_container_width=True,
    )
else:
    st.info("Entraîne au moins un algorithme pour obtenir le classement.")

# ── Navigation hint ───────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "**Pages disponibles (sidebar gauche) :**  \n"
    "📋 `01_train` · 🎯 `02_evaluate` · 📊 `03_comparison` · 🗺 `04_live_map`"
)
