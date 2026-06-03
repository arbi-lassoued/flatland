"""
Page 02 — Évaluation d'un algorithme entraîné.
"""

import os
import sys
import subprocess
import glob

import streamlit as st
import pandas as pd
import plotly.express as px

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from utils.metrics_utils import load_metrics

st.set_page_config(page_title="Évaluation — Flatland RL", page_icon="🎯", layout="wide")

# ── Session state ─────────────────────────────────────────────────────────────
if "eval_proc_02" not in st.session_state:
    st.session_state.eval_proc_02 = None
if "eval_algo_02" not in st.session_state:
    st.session_state.eval_algo_02 = "PPO"

st.title("🎯 Évaluation — Flatland 5 Agents")

# ── Controls ──────────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 2])

with col1:
    algo = st.selectbox("Algorithme", ["PPO", "APEX", "MARWIL"])
    st.session_state.eval_algo_02 = algo
    num_episodes = st.slider("Nombre d'épisodes", 1, 20, 5)

    # Find available checkpoints
    ckpt_pattern = os.path.join(PROJECT_ROOT, "results", algo.lower(), "**", "checkpoint_*")
    checkpoints = sorted(glob.glob(ckpt_pattern, recursive=True))
    ckpt_options = ["auto"] + checkpoints
    selected_ckpt = st.selectbox("Checkpoint", ckpt_options)

with col2:
    st.markdown("### Instructions")
    st.markdown(
        "1. Sélectionnez un algorithme entraîné\n"
        "2. Choisissez un checkpoint (ou `auto` pour le dernier)\n"
        "3. Définissez le nombre d'épisodes à évaluer\n"
        "4. Cliquez **Évaluer**"
    )

st.markdown("---")

# ── Evaluate button ───────────────────────────────────────────────────────────
col_btn, col_stop, col_stat = st.columns([1, 1, 3])

with col_btn:
    if st.button("▶️ Évaluer", use_container_width=True):
        cmd = [
            sys.executable,
            os.path.join(PROJECT_ROOT, "evaluate.py"),
            "--algorithm", algo.lower(),
            "--checkpoint", selected_ckpt,
            "--num-episodes", str(num_episodes),
        ]
        st.session_state.eval_proc_02 = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=PROJECT_ROOT,
        )
        st.success(f"Évaluation {algo} lancée.")

with col_stop:
    if st.button("⏹ Arrêter", use_container_width=True):
        if st.session_state.eval_proc_02:
            st.session_state.eval_proc_02.terminate()
            st.session_state.eval_proc_02 = None

with col_stat:
    proc = st.session_state.eval_proc_02
    if proc is None:
        st.info("Aucune évaluation active.")
    elif proc.poll() is None:
        st.success(f"⏳ Évaluation en cours (PID {proc.pid})")
    else:
        rc = proc.returncode
        st.success("✅ Évaluation terminée.") if rc == 0 else st.error(f"❌ Code retour {rc}")

# ── Live map animation placeholder ───────────────────────────────────────────
st.subheader("🗺 Aperçu carte")
map_placeholder = st.empty()
try:
    from envs.flatland_env import FlatlandMultiAgentEnv
    if "eval_env_02" not in st.session_state:
        st.session_state.eval_env_02 = FlatlandMultiAgentEnv()
        st.session_state.eval_env_02.reset()
    frame = st.session_state.eval_env_02.get_render_frame()
    map_placeholder.image(frame, caption="État de la carte (statique sans checkpoint actif)", use_column_width=True)
except Exception as exc:
    map_placeholder.warning(f"Carte indisponible : {exc}")

# ── Eval results ──────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📊 Résultats d'évaluation")

eval_csv = os.path.join(PROJECT_ROOT, "results", algo.lower(), "eval_results.csv")
if os.path.exists(eval_csv):
    df_eval = pd.read_csv(eval_csv)
    st.dataframe(df_eval, use_container_width=True)

    # Box plot of rewards
    if "reward" in df_eval.columns:
        fig_box = px.box(df_eval, y="reward", title=f"Distribution des Rewards — {algo}",
                         template="plotly_white", height=300)
        st.plotly_chart(fig_box, use_container_width=True)

    # Download button
    csv_bytes = df_eval.to_csv(index=False).encode("utf-8")
    st.download_button(
        "💾 Exporter CSV",
        data=csv_bytes,
        file_name=f"eval_{algo.lower()}.csv",
        mime="text/csv",
    )
else:
    df_m = load_metrics(algo.lower())
    if not df_m.empty:
        st.info("Affichage des métriques d'entraînement (pas encore d'évaluation dédiée).")
        st.dataframe(df_m.tail(20), use_container_width=True)
        fig_box = px.box(df_m, y="reward", title=f"Distribution des Rewards (entraînement) — {algo}",
                         template="plotly_white", height=300)
        st.plotly_chart(fig_box, use_container_width=True)
    else:
        st.info("Lance une évaluation pour voir les résultats ici.")

# ── GIF display ───────────────────────────────────────────────────────────────
gif_path = os.path.join(PROJECT_ROOT, "results", algo.lower(), "episode_eval.gif")
if os.path.exists(gif_path):
    st.subheader("🎬 GIF de l'épisode évalué")
    with open(gif_path, "rb") as f:
        st.image(f.read(), caption="Episode GIF", use_column_width=False, width=400)
