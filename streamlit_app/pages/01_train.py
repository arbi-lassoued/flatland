"""
Page 01 — Entraînement des algorithmes RL.
"""

import os
import sys
import subprocess
import time
import yaml

import streamlit as st
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from utils.metrics_utils import load_metrics
from streamlit_app.components.reward_chart import plot_rewards

st.set_page_config(page_title="Entraînement — Flatland RL", page_icon="🚀", layout="wide")

# ── Session state ─────────────────────────────────────────────────────────────
if "train_proc_01" not in st.session_state:
    st.session_state.train_proc_01 = None
if "train_algo_01" not in st.session_state:
    st.session_state.train_algo_01 = "PPO"

st.title("🚀 Entraînement — Flatland 5 Agents")

# ── Algorithm selection ───────────────────────────────────────────────────────
col_sel, col_opt = st.columns([1, 2])

with col_sel:
    algo = st.selectbox("Algorithme", ["PPO", "APEX", "MARWIL"])
    st.session_state.train_algo_01 = algo
    num_cpus = st.slider("CPUs", 1, 8, 4)
    smoke_test = st.checkbox("🔥 Smoke test (5 000 steps)", value=False)

# ── Show config ───────────────────────────────────────────────────────────────
cfg_file = os.path.join(PROJECT_ROOT, "configs", f"{algo.lower()}_config.yaml")
with open(cfg_file) as f:
    cfg_content = yaml.safe_load(f)

with st.expander(f"⚙️ Configuration {algo}", expanded=False):
    st.code(yaml.dump(cfg_content, default_flow_style=False), language="yaml")

st.markdown("---")

# ── Launch controls ───────────────────────────────────────────────────────────
col_launch, col_stop, col_status = st.columns([1, 1, 3])

with col_launch:
    if st.button("🚀 Lancer l'entraînement", use_container_width=True):
        if st.session_state.train_proc_01 and st.session_state.train_proc_01.poll() is None:
            st.warning("Un entraînement est déjà en cours.")
        else:
            cmd = [
                sys.executable,
                os.path.join(PROJECT_ROOT, "train.py"),
                "--algorithm", algo.lower(),
                "--num-cpus", str(num_cpus),
            ]
            if smoke_test:
                cmd.append("--smoke-test")
            st.session_state.train_proc_01 = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=PROJECT_ROOT,
            )
            st.session_state.train_algo_01 = algo
            st.success(f"✅ Entraînement {algo} lancé (PID {st.session_state.train_proc_01.pid})")

with col_stop:
    if st.button("⏹ Arrêter", use_container_width=True):
        if st.session_state.train_proc_01:
            st.session_state.train_proc_01.terminate()
            st.session_state.train_proc_01 = None
            st.warning("⏹ Entraînement arrêté.")

with col_status:
    proc = st.session_state.train_proc_01
    if proc is None:
        st.info("Aucun entraînement actif.")
    elif proc.poll() is None:
        st.success(f"⏳ En cours — PID {proc.pid}")
    else:
        rc = proc.returncode
        if rc == 0:
            st.success("✅ Entraînement terminé avec succès.")
        else:
            st.error(f"❌ Entraînement terminé avec code {rc}.")

st.markdown("---")

# ── Live log output ───────────────────────────────────────────────────────────
st.subheader("📋 Logs en temps réel")
log_box = st.empty()

proc = st.session_state.train_proc_01
if proc and proc.poll() is None:
    log_lines = []
    try:
        for _ in range(30):
            line = proc.stdout.readline()
            if line:
                log_lines.append(line.rstrip())
    except Exception:
        pass
    log_box.text_area("Logs", value="\n".join(log_lines[-50:]), height=200)
else:
    log_box.info("Lance un entraînement pour voir les logs ici.")

# ── Live reward chart ─────────────────────────────────────────────────────────
st.subheader("📈 Reward en temps réel")
display_algo = st.session_state.train_algo_01

df_live = load_metrics(display_algo.lower())
if not df_live.empty:
    fig = plot_rewards(df_live, display_algo)
    st.plotly_chart(fig, use_container_width=True)

    last = df_live.iloc[-1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Épisodes", len(df_live))
    c2.metric("Reward moyen", f"{df_live['reward'].mean():.2f}")
    c3.metric("Taux arrivée", f"{df_live['arrival_rate'].mean()*100:.1f}%")
    c4.metric("Steps moyen", f"{df_live['steps'].mean():.0f}")
else:
    st.info("Les métriques apparaîtront ici après quelques épisodes d'entraînement.")

# ── Checkpoint info ───────────────────────────────────────────────────────────
import glob
checkpoint_dir = os.path.join(PROJECT_ROOT, "results", display_algo.lower())
checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, "**", "checkpoint_*"), recursive=True))
if checkpoints:
    st.subheader("💾 Checkpoints disponibles")
    st.code("\n".join(checkpoints[-5:]), language="bash")
