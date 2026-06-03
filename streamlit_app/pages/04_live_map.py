"""
Page 04 — Carte Flatland en direct avec contrôle manuel ou IA.
"""

import os
import sys
import time
import glob

import streamlit as st
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from envs.flatland_env import FlatlandMultiAgentEnv
from streamlit_app.components.map_renderer import render_flatland_streamlit
from utils.render_utils import get_agent_positions

st.set_page_config(page_title="Carte Live — Flatland RL", page_icon="🗺", layout="wide")
st.title("🗺 Carte Flatland en Direct")

# ── Session state ─────────────────────────────────────────────────────────────
if "live_env_04" not in st.session_state:
    st.session_state.live_env_04 = None
if "live_obs_04" not in st.session_state:
    st.session_state.live_obs_04 = None
if "live_playing_04" not in st.session_state:
    st.session_state.live_playing_04 = False
if "live_step_04" not in st.session_state:
    st.session_state.live_step_04 = 0
if "live_rewards_04" not in st.session_state:
    st.session_state.live_rewards_04 = {f"agent_{i}": 0.0 for i in range(5)}
if "live_mode_04" not in st.session_state:
    st.session_state.live_mode_04 = "Manuel"


def _init_env():
    env = FlatlandMultiAgentEnv()
    obs = env.reset()
    st.session_state.live_env_04 = env
    st.session_state.live_obs_04 = obs
    st.session_state.live_step_04 = 0
    st.session_state.live_rewards_04 = {f"agent_{i}": 0.0 for i in range(5)}
    st.session_state.live_playing_04 = False


if st.session_state.live_env_04 is None:
    _init_env()

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("⚙️ Contrôles")
    mode = st.radio("Mode", ["Manuel", "Agent IA"], index=0)
    st.session_state.live_mode_04 = mode
    speed = st.slider("Vitesse (steps/s)", 1, 10, 3)

    st.markdown("---")
    col_play, col_pause, col_reset = st.columns(3)
    with col_play:
        if st.button("▶️"):
            st.session_state.live_playing_04 = True
    with col_pause:
        if st.button("⏸"):
            st.session_state.live_playing_04 = False
    with col_reset:
        if st.button("⏹"):
            _init_env()

    if mode == "Agent IA":
        st.markdown("---")
        algo_ia = st.selectbox("Algorithme IA", ["PPO", "APEX", "MARWIL"])
        ckpt_pattern = os.path.join(PROJECT_ROOT, "results", algo_ia.lower(), "**", "checkpoint_*")
        ckpts = sorted(glob.glob(ckpt_pattern, recursive=True))
        ckpt_sel = st.selectbox("Checkpoint", ["(aucun)"] + ckpts)
    else:
        ckpt_sel = "(aucun)"

    st.markdown("---")
    st.metric("Step actuel", st.session_state.live_step_04)

# ── Main display ──────────────────────────────────────────────────────────────
col_map, col_panel = st.columns([3, 1])

env = st.session_state.live_env_04

with col_map:
    map_slot = st.empty()
    positions = get_agent_positions(env.rail_env)
    img_bytes = render_flatland_streamlit(env.rail_env, positions, width=640, height=640)
    map_slot.image(img_bytes, caption=f"Step {st.session_state.live_step_04}", use_column_width=True)

with col_panel:
    st.subheader("🚂 Agents")
    agent_colors = ["🔴", "🔵", "🟢", "🟡", "🟣"]
    for i in range(5):
        agent_id = f"agent_{i}"
        pos_info = positions.get(agent_id)
        cum_r = st.session_state.live_rewards_04.get(agent_id, 0.0)
        if pos_info:
            row, col, direction = pos_info
            direction_names = {0: "↑Nord", 1: "→Est", 2: "↓Sud", 3: "←Ouest"}
            st.markdown(
                f"{agent_colors[i]} **Agent {i}**  \n"
                f"Pos: ({row},{col}) {direction_names.get(direction,'?')}  \n"
                f"Reward: `{cum_r:.2f}`"
            )
        else:
            st.markdown(f"{agent_colors[i]} **Agent {i}** — hors carte  \nReward: `{cum_r:.2f}`")
        st.markdown("---")

# ── Manual action buttons ─────────────────────────────────────────────────────
ACTION_NAMES = ["DO_NOTHING", "LEFT", "FORWARD", "RIGHT", "STOP"]

if mode == "Manuel":
    st.subheader("🕹 Contrôle Manuel")
    obs_dict = st.session_state.live_obs_04 or {}
    action_dict_manual = {}

    col_agents = st.columns(5)
    for i, col in enumerate(col_agents):
        with col:
            agent_id = f"agent_{i}"
            st.markdown(f"**Agent {i}**")
            selected_action = st.selectbox(
                f"Action {i}", ACTION_NAMES,
                index=2,
                key=f"action_agent_{i}",
                label_visibility="collapsed",
            )
            action_dict_manual[agent_id] = ACTION_NAMES.index(selected_action)

    if st.button("➡️ Exécuter step"):
        obs, rewards, dones, infos = env.step(action_dict_manual)
        st.session_state.live_obs_04 = obs
        for agent_id, r in rewards.items():
            st.session_state.live_rewards_04[agent_id] = (
                st.session_state.live_rewards_04.get(agent_id, 0.0) + r
            )
        st.session_state.live_step_04 += 1

        if dones.get("__all__", False):
            st.success("🏁 Tous les agents ont terminé !")

        # Refresh map
        positions = get_agent_positions(env.rail_env)
        img_bytes = render_flatland_streamlit(env.rail_env, positions, width=640, height=640)
        map_slot.image(img_bytes, caption=f"Step {st.session_state.live_step_04}", use_column_width=True)
        st.rerun()

# ── Auto play (IA mode) ───────────────────────────────────────────────────────
elif mode == "Agent IA" and st.session_state.live_playing_04:
    obs_dict = st.session_state.live_obs_04 or {}

    # Default to random actions if no checkpoint
    action_dict_ia = {agent_id: env.action_space.sample() for agent_id in obs_dict}

    if ckpt_sel and ckpt_sel != "(aucun)":
        try:
            if "ia_agent_04" not in st.session_state:
                import ray
                from ray.rllib.models import ModelCatalog
                from ray import tune
                from models.custom_model import FlatlandModel

                ray.init(ignore_reinit_error=True, num_cpus=1)
                tune.register_env("flatland_5agents", lambda cfg: FlatlandMultiAgentEnv(cfg))
                ModelCatalog.register_custom_model("flatland_model", FlatlandModel)

                from ray.rllib.agents.ppo import PPOTrainer
                ia_trainer = PPOTrainer(config={
                    "env": "flatland_5agents",
                    "framework": "torch",
                    "num_workers": 0,
                    "model": {"custom_model": "flatland_model"},
                    "multiagent": {
                        "policies": {"shared_policy"},
                        "policy_mapping_fn": lambda agent_id, *args, **kwargs: "shared_policy",
                    },
                })
                ia_trainer.restore(ckpt_sel)
                st.session_state.ia_agent_04 = ia_trainer

            ia_agent = st.session_state.ia_agent_04
            for agent_id, o in obs_dict.items():
                action_dict_ia[agent_id] = ia_agent.compute_single_action(o, policy_id="shared_policy")
        except Exception as exc:
            st.warning(f"IA non disponible ({exc}), actions aléatoires utilisées.")

    obs, rewards, dones, infos = env.step(action_dict_ia)
    st.session_state.live_obs_04 = obs
    for agent_id, r in rewards.items():
        st.session_state.live_rewards_04[agent_id] = (
            st.session_state.live_rewards_04.get(agent_id, 0.0) + r
        )
    st.session_state.live_step_04 += 1

    if dones.get("__all__", False):
        st.success("🏁 Épisode terminé — Réinitialisation…")
        time.sleep(1)
        _init_env()

    positions = get_agent_positions(env.rail_env)
    img_bytes = render_flatland_streamlit(env.rail_env, positions, width=640, height=640)
    map_slot.image(img_bytes, caption=f"Step {st.session_state.live_step_04}", use_column_width=True)

    time.sleep(1.0 / speed)
    st.rerun()
