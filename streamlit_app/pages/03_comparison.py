"""
Page 03 — Comparaison des algorithmes PPO, APEX, MARWIL.
"""

import os
import sys

import streamlit as st
import pandas as pd
import plotly.express as px

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from utils.metrics_utils import (
    load_metrics,
    compare_all_algorithms,
    get_best_algorithm,
    compute_algorithm_score,
)
from streamlit_app.components.algo_comparison import (
    create_comparison_table,
    create_radar_chart,
    create_reward_bar,
    create_reward_evolution,
    create_arrival_vs_reward_scatter,
    create_stacked_reward_breakdown,
)

st.set_page_config(page_title="Comparaison — Flatland RL", page_icon="📊", layout="wide")
st.title("📊 Comparaison des Algorithmes")

# ── Load data ─────────────────────────────────────────────────────────────────
ppo_df = load_metrics("ppo")
apex_df = load_metrics("apex")
marwil_df = load_metrics("marwil")

any_data = not (ppo_df.empty and apex_df.empty and marwil_df.empty)

if not any_data:
    st.warning("Aucune donnée d'entraînement disponible. Lance au moins un algorithme depuis la page **01_train**.")
    st.stop()

# ── Comparative table ─────────────────────────────────────────────────────────
st.subheader("📋 Tableau comparatif")
table_df = create_comparison_table(ppo_df, apex_df, marwil_df)
st.dataframe(table_df, use_container_width=True, hide_index=True)

# ── Best algorithm banner ─────────────────────────────────────────────────────
best = get_best_algorithm()
comp = compare_all_algorithms()
if comp["score"].max() > 0:
    best_row = comp[comp["algorithm"] == best].iloc[0]
    score_txt = (
        f"**Score** : `{best_row['score']:.4f}` · "
        f"**Reward moyen** : `{best_row['mean_reward']}` · "
        f"**Taux arrivée** : `{float(best_row['arrival_rate'])*100:.1f}%` · "
        f"**Steps moyen** : `{best_row['mean_steps']}`"
    )
    st.success(f"🏆 Meilleur algorithme : **{best}**\n\n{score_txt}")
    with st.expander("Pourquoi ?"):
        st.markdown(
            "Le score est calculé selon la formule :\n\n"
            "> `score = (reward_norm × 0.4) + (arrival_rate × 0.4) + (rapidité × 0.2)`\n\n"
            f"**{best}** présente le meilleur compromis entre récompense totale, "
            "taux d'arrivée des agents, et rapidité à terminer les épisodes."
        )

st.markdown("---")

# ── Charts row 1 ─────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Reward Moyen")
    st.plotly_chart(create_reward_bar(ppo_df, apex_df, marwil_df), use_container_width=True)

with col2:
    st.subheader("🎯 Arrivée vs Reward")
    st.plotly_chart(create_arrival_vs_reward_scatter(ppo_df, apex_df, marwil_df), use_container_width=True)

# ── Reward evolution ──────────────────────────────────────────────────────────
st.subheader("📈 Évolution du Reward par Algorithme")
st.plotly_chart(create_reward_evolution(ppo_df, apex_df, marwil_df), use_container_width=True)

# ── Radar chart ───────────────────────────────────────────────────────────────
st.subheader("🕸 Radar Multi-Dimensions")
scores_dict = {"PPO": ppo_df, "APEX": apex_df, "MARWIL": marwil_df}
st.plotly_chart(create_radar_chart(scores_dict), use_container_width=True)

# ── Reward breakdown stacked bar ─────────────────────────────────────────────
st.subheader("🧩 Décomposition des Récompenses")

def _mean_breakdown(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"arrived": 0, "step_penalty": 0, "deadlock": 0, "collision": 0, "cooperative": 0, "invalid_action": 0}
    # Approximate breakdown from global stats
    arrived_r = df["arrivals"].mean() * 10.0
    step_r = df["steps"].mean() * 0.01
    deadlock_r = df["deadlocks"].mean() * 5.0
    return {
        "arrived": arrived_r,
        "step_penalty": step_r,
        "deadlock": deadlock_r,
        "collision": 0.0,
        "cooperative": 0.0,
        "invalid_action": 0.0,
    }

breakdown_by_algo = {
    "PPO": _mean_breakdown(ppo_df),
    "APEX": _mean_breakdown(apex_df),
    "MARWIL": _mean_breakdown(marwil_df),
}
st.plotly_chart(create_stacked_reward_breakdown(breakdown_by_algo), use_container_width=True)
