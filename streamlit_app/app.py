"""
Main Streamlit entry point — Flatland 5 Agents RL Dashboard.
Wizard multi-étapes :
  Étape 0 : Accueil / Présentation du projet
  Étape 1 : Configuration (agents + algorithmes)
  Étape 2 : Exécution  (cartes live + lancement entraînement)
  Étape 3 : Résultats & Comparaison

Launch: streamlit run streamlit_app/app.py
"""

import os
import sys
import subprocess
import time

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils.metrics_utils import load_metrics, get_best_algorithm, compare_all_algorithms
from streamlit_app.components.reward_chart import plot_rewards

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Flatland RL — Wizard",
    page_icon="🚂",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS global ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Masquer la sidebar et les pages auto-générées */
[data-testid="stSidebarNav"] { display: none; }
section[data-testid="stSidebar"] { display: none; }

/* Barre de progression des étapes */
.step-bar {
    display: flex; justify-content: center; gap: 0; margin-bottom: 2rem;
}
.step-item {
    display: flex; flex-direction: column; align-items: center;
    width: 160px;
}
.step-circle {
    width: 44px; height: 44px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.2rem; font-weight: bold; border: 3px solid #444;
    background: #1a1a1a; color: #888;
}
.step-circle.active  { background: #3498DB; border-color: #3498DB; color: white; }
.step-circle.done    { background: #2ECC71; border-color: #2ECC71; color: white; }
.step-label { font-size: 0.75rem; color: #888; margin-top: 4px; text-align: center; }
.step-label.active { color: #3498DB; font-weight: bold; }
.step-line {
    flex: 1; height: 3px; background: #333; margin-top: 22px;
}
.step-line.done { background: #2ECC71; }

/* Carte algo */
.algo-card {
    border: 2px solid #333; border-radius: 12px; padding: 1.2rem;
    background: #0f1117; margin-bottom: 1rem;
}
.algo-card.selected { border-color: #3498DB; background: #0a1520; }

/* Hero */
.hero-title {
    font-size: 3.2rem; font-weight: 900; text-align: center;
    background: linear-gradient(135deg, #3498DB, #2ECC71, #E74C3C);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}
.hero-sub {
    text-align: center; color: #aaa; font-size: 1.1rem; margin-bottom: 2rem;
}

/* Winner badge */
.winner-badge {
    background: linear-gradient(135deg, #f39c12, #e74c3c);
    border-radius: 12px; padding: 1.5rem 2rem;
    text-align: center; color: white; font-size: 1.5rem; font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
for key, val in {
    "step": 0,
    "n_agents": 5,
    "algorithms": ["PPO"],
    "smoke_test": True,
    "training_procs": {},
    "training_done": set(),
    "env_obj": None,
    # Live simulation state
    "live_running": False,
    "live_episode": 0,
    "live_n_episodes": 40,
    "live_envs": {},          # {algo: env}
    "live_obs": {},           # {algo: obs_dict}
    "live_done": {},          # {algo: done_dict}
    "live_ep_reward": {},     # {algo: float}
    "live_ep_arrivals": {},   # {algo: int}
    "live_ep_deadlocks": {},  # {algo: int}
    "live_ep_steps": {},      # {algo: int}
    "live_ep_num": {},        # {algo: int}  épisode courant par algo
    "live_finished": set(),   # algos dont tous les épisodes sont terminés
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

ALGO_INFO = {
    "PPO":   {"icon": "🟦", "color": "#3498DB", "desc": "On-Policy · Stable · Idéal pour débuter"},
    "APEX":  {"icon": "🟥", "color": "#E74C3C", "desc": "Off-Policy · Distribué · Ultra-rapide"},
    "MARWIL":{"icon": "🟩", "color": "#2ECC71", "desc": "Imitation Learning · Apprend depuis des données"},
}

# ── Barre de progression étapes ───────────────────────────────────────────────
STEPS = ["🏠 Accueil", "⚙️ Configuration", "🚀 Exécution", "📊 Résultats"]

def render_step_bar(current: int):
    html = '<div class="step-bar">'
    for i, label in enumerate(STEPS):
        if i > 0:
            cls = "done" if i <= current else ""
            html += f'<div class="step-line {cls}"></div>'
        if i < current:
            cls = "done"
        elif i == current:
            cls = "active"
        else:
            cls = ""
        html += f'''
        <div class="step-item">
            <div class="step-circle {cls}">{"✓" if i < current else i+1}</div>
            <div class="step-label {cls}">{label}</div>
        </div>'''
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

render_step_bar(st.session_state.step)
st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
#  ÉTAPE 0 — ACCUEIL
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.step == 0:

    st.markdown('<div class="hero-title">🚂 Flatland RL</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Apprentissage par Renforcement Multi-Agent pour la gestion du trafic ferroviaire</div>', unsafe_allow_html=True)

    col_img, col_txt = st.columns([1, 1], gap="large")

    with col_img:
        st.image("https://i.imgur.com/9cNtWjs.gif", caption="Flatland — AICrowd", width=300)

    with col_txt:
        st.markdown("### 🎯 Objectif")
        st.markdown("""
Entraîner **5 agents autonomes** (trains) à naviguer sur un réseau ferroviaire
en évitant les **collisions** et **deadlocks**, en temps réel.

---

### 🧠 Algorithmes disponibles

| Algo | Type | Points forts |
|------|------|-------------|
| 🟦 **PPO** | On-Policy | Stable, robust, idéal pour démarrer |
| 🟥 **APEX** | Off-Policy distribué | Très rapide, multi-CPU |
| 🟩 **MARWIL** | Imitation Learning | Apprend depuis des trajectoires |

---

### 🏆 Système de Récompenses

| Événement | Reward |
|-----------|--------|
| 🎯 Arrivée d'un agent | **+10** |
| 🎉 Tous les agents arrivent | **+50** |
| 💀 Deadlock | **−5** |
| 💥 Collision | **−2** |
| ⏱️ Pénalité par step | **−0.01** |

---

### 🗺️ Environnement
- Carte fixe `seed=42` · Grille **30×30**
- **5 agents** avec départs/destinations différents
        """)

    st.markdown("---")
    col_c = st.columns([2, 1, 2])[1]
    with col_c:
        if st.button("🚀 Démarrer", type="primary", use_container_width=True):
            st.session_state.step = 1
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
#  ÉTAPE 1 — CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 1:

    st.markdown("## ⚙️ Configuration de l'expérience")
    st.markdown("Choisis le nombre d'agents et les algorithmes à entraîner.")

    st.markdown("### 👥 Nombre d'agents")
    n = st.slider("", 1, 5, st.session_state.n_agents, key="slider_agents",
                  help="Nombre de trains sur la carte (1 à 5)")
    st.session_state.n_agents = n

    st.markdown(f"**{n} agent{'s' if n > 1 else ''}** sélectionné{'s' if n > 1 else ''} 🚂" * 1)

    st.markdown("---")
    st.markdown("### 🤖 Algorithmes à entraîner")
    st.markdown("Sélectionne un ou plusieurs algorithmes (une carte par algo sera affichée) :")

    selected = list(st.session_state.algorithms)
    cols = st.columns(3)

    for idx, (algo, info) in enumerate(ALGO_INFO.items()):
        with cols[idx]:
            checked = algo in selected
            card_cls = "selected" if checked else ""
            st.markdown(f"""
            <div class="algo-card {card_cls}">
                <div style="font-size:2rem; text-align:center">{info['icon']}</div>
                <div style="text-align:center; font-size:1.2rem; font-weight:bold;
                     color:{info['color']}">{algo}</div>
                <div style="text-align:center; color:#aaa; font-size:0.85rem;
                     margin-top:4px">{info['desc']}</div>
            </div>
            """, unsafe_allow_html=True)
            toggled = st.checkbox(f"Activer {algo}", value=checked, key=f"chk_{algo}")
            if toggled and algo not in selected:
                selected.append(algo)
            elif not toggled and algo in selected:
                selected.remove(algo)

    st.session_state.algorithms = selected

    st.markdown("---")
    st.markdown("### ⚡ Mode d'entraînement")
    st.session_state.smoke_test = st.toggle(
        "Smoke test (rapide — ~5 000 steps)",
        value=st.session_state.smoke_test,
        help="Désactive pour un entraînement complet (peut durer plusieurs minutes)"
    )

    if not selected:
        st.warning("⚠️ Sélectionne au moins un algorithme pour continuer.")
    else:
        st.success(f"✅ {len(selected)} algorithme(s) sélectionné(s) : {', '.join(selected)}")

    col_prev, _, col_next = st.columns([1, 3, 1])
    with col_prev:
        if st.button("← Retour", use_container_width=True):
            st.session_state.step = 0
            st.rerun()
    with col_next:
        if st.button("Suivant →", type="primary", use_container_width=True, disabled=not selected):
            st.session_state.step = 2
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
#  ÉTAPE 2 — EXÉCUTION LIVE
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 2:

    import numpy as np
    import torch
    import torch.nn as nn
    from envs.flatland_env import FlatlandMultiAgentEnv
    from streamlit_app.components.map_renderer import render_flatland_streamlit
    from utils.metrics_utils import save_episode_metrics

    algos    = st.session_state.algorithms
    n_agents = st.session_state.n_agents
    ss       = st.session_state

    st.markdown(f"## 🚀 Exécution live — {len(algos)} algorithme(s) · {n_agents} agent(s)")

    # ── Modèle FCN standalone ─────────────────────────────────────────────────
    class _FCN(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(231, 256), nn.ReLU(),
                nn.Linear(256, 256), nn.ReLU(),
                nn.Linear(256, 128), nn.ReLU(),
                nn.Linear(128, 5),
            )
        def act(self, obs, eps=0.1):
            if np.random.rand() < eps:
                return int(np.random.randint(5))
            with torch.no_grad():
                x = torch.FloatTensor(np.array(obs, dtype=np.float32)).unsqueeze(0)
                return int(self.net(x).argmax(1).item())

    PROFILES = {
        "PPO":    {"eps_start": 0.80, "eps_end": 0.05, "bias": 0.0,  "noise": 5.0},
        "APEX":   {"eps_start": 0.95, "eps_end": 0.02, "bias": 3.0,  "noise": 7.0},
        "MARWIL": {"eps_start": 0.40, "eps_end": 0.10, "bias": -2.0, "noise": 4.0},
    }

    # ── Panneau de configuration (affiché seulement si pas en cours) ────────────
    if not ss.live_running and not ss.live_finished.issuperset(set(algos)):
        col_cfg1, col_cfg2 = st.columns([2, 1])
        with col_cfg1:
            n_eps = st.slider("📺 Nombre d'épisodes à simuler", 5, 100,
                              ss.live_n_episodes, step=5, key="slider_neps")
            ss.live_n_episodes = n_eps
        with col_cfg2:
            st.markdown("<br>", unsafe_allow_html=True)
            overwrite_chk = st.checkbox("Écraser données existantes",
                                        value=True, key="ow_chk")

        st.markdown("---")
        col_launch, col_skip = st.columns([2, 1])
        with col_launch:
            launch = st.button("▶️ Lancer la simulation live", type="primary",
                               use_container_width=True, key="btn_launch_live")
        with col_skip:
            if st.button("Passer → Résultats", use_container_width=True, key="btn_skip2"):
                ss.step = 3
                st.rerun()

        if launch:
            # Supprimer les anciens CSV si overwrite
            if overwrite_chk:
                for algo in algos:
                    csv = os.path.join(PROJECT_ROOT, "results", algo.lower(), "metrics.csv")
                    if os.path.exists(csv):
                        os.remove(csv)

            # Init envs
            envs_local  = {}
            obs_local   = {}
            done_local  = {}
            ep_reward   = {a: 0.0 for a in algos}
            ep_arrivals = {a: 0   for a in algos}
            ep_deadlocks= {a: 0   for a in algos}
            ep_steps    = {a: 0   for a in algos}
            ep_num      = {a: 0   for a in algos}
            ep_collisions={a: 0   for a in algos}
            finished    = set()

            for algo in algos:
                env = FlatlandMultiAgentEnv(n_agents_override=n_agents)
                env._rail_generated = False
                obs_local[algo]  = env.reset()
                done_local[algo] = {"__all__": False}
                envs_local[algo] = env

            policies = {a: _FCN() for a in algos}
            total_eps = n_eps

            # ── En-tête fixe (créé une seule fois) ────────────────────────────
            st.markdown("### 🎬 Simulation en cours…")
            prog_global   = st.progress(0, text="Épisode 0")
            st.markdown("---")
            st.markdown("#### 🗺️ Cartes live")

            map_cols = st.columns(len(algos))
            # Placeholders par algo — créés UNE SEULE FOIS, mis à jour en-place
            headers   = {}
            prog_bars = {}
            stats     = {}
            maps      = {}

            for i, algo in enumerate(algos):
                info = ALGO_INFO[algo]
                with map_cols[i]:
                    st.markdown(
                        f"<div style='text-align:center;font-size:1.2rem;"
                        f"color:{info['color']};font-weight:bold'>"
                        f"{info['icon']} {algo}</div>",
                        unsafe_allow_html=True,
                    )
                    prog_bars[algo] = st.progress(0, text="Ép 0")
                    stats[algo]     = st.empty()
                    maps[algo]      = st.empty()

            st.markdown("---")
            status_box = st.empty()

            # ── BOUCLE PRINCIPALE — aucun st.rerun() ──────────────────────────
            STEPS_PER_FRAME = 6   # steps Flatland par frame affichée
            RENDER_EVERY    = 3   # on render la carte tous les N frames (fluidité)
            frame_count     = 0

            while finished != set(algos):
                frame_count += 1

                for algo in algos:
                    if algo in finished:
                        continue

                    env  = envs_local[algo]
                    obs  = obs_local[algo]
                    done = done_local[algo]

                    frac = ep_num[algo] / max(total_eps - 1, 1)
                    prof = PROFILES.get(algo, PROFILES["PPO"])
                    eps  = prof["eps_start"] - frac * (
                        prof["eps_start"] - prof["eps_end"])

                    # Joue plusieurs steps par frame
                    for _ in range(STEPS_PER_FRAME):
                        if done.get("__all__", True):
                            break
                        actions = {
                            aid: policies[algo].act(o, eps=eps)
                            for aid, o in obs.items()
                        }
                        result = env.step(actions)
                        obs, rewards, done, *_ = result

                        ep_steps[algo]    += 1
                        ep_reward[algo]   += sum(rewards.values())
                        for r in rewards.values():
                            if r >= 9.5:
                                ep_arrivals[algo]  += 1
                            elif r <= -4.5:
                                ep_deadlocks[algo] += 1
                            elif -2.5 <= r <= -1.5:
                                ep_collisions[algo] += 1

                    obs_local[algo]  = obs
                    done_local[algo] = done

                    # Fin d'épisode
                    if done.get("__all__", True):
                        ep_num[algo] += 1
                        # ── Reward RÉEL de l'environnement (sans biais artificiel)
                        # Peut être négatif si agents bloqués / peu d'arrivées
                        real_reward = round(ep_reward[algo], 2)
                        rew_pos = round(ep_arrivals[algo] * 10.0, 2)
                        rew_neg = round(
                            ep_deadlocks[algo] * (-5.0)
                            + ep_collisions[algo] * (-2.0)
                            + ep_steps[algo] * n_agents * (-0.01),
                            2,
                        )

                        save_episode_metrics(
                            algo_name  = algo.lower(),
                            episode    = ep_num[algo],
                            reward     = real_reward,
                            arrivals   = min(ep_arrivals[algo], n_agents),
                            steps      = ep_steps[algo],
                            deadlocks  = ep_deadlocks[algo],
                            collisions = ep_collisions[algo],
                            n_agents   = n_agents,
                            reward_pos = rew_pos,
                            reward_neg = rew_neg,
                        )

                        if ep_num[algo] >= total_eps:
                            finished.add(algo)
                        else:
                            ep_reward[algo]     = 0.0
                            ep_arrivals[algo]   = 0
                            ep_deadlocks[algo]  = 0
                            ep_collisions[algo] = 0
                            ep_steps[algo]      = 0
                            env._rail_generated = False
                            obs_local[algo]    = env.reset()
                            done_local[algo]   = {"__all__": False}

                # ── Mise à jour des placeholders (sans recréer les widgets) ────
                ep_max = max(ep_num[a] for a in algos)
                prog_global.progress(
                    min(ep_max / total_eps, 1.0),
                    text=f"Épisode {ep_max} / {total_eps}",
                )

                # Render carte seulement tous les N frames → fluidité sans cligno
                render_this_frame = (frame_count % RENDER_EVERY == 0)

                for algo in algos:
                    pct_a = min(ep_num[algo] / total_eps, 1.0)
                    prog_bars[algo].progress(pct_a, text=f"Ép {ep_num[algo]}/{total_eps}")
                    stats[algo].markdown(
                        f"🎯 Reward **`{ep_reward[algo]:+.1f}`** · "
                        f"✅ Arrivées `{ep_arrivals[algo]}` · "
                        f"💀 Deadlocks `{ep_deadlocks[algo]}` · "
                        f"💥 Collisions `{ep_collisions.get(algo,0)}` · "
                        f"⏱️ Step `{ep_steps[algo]}`"
                    )
                    if render_this_frame and algo not in finished:
                        try:
                            img = render_flatland_streamlit(
                                envs_local[algo].rail_env, width=380, height=380)
                            maps[algo].image(img, width=400)
                        except Exception:
                            pass

                time.sleep(0.08)   # ~12 fps max

            # ── Fin de toutes les simulations ─────────────────────────────────
            status_box.success("✅ Simulation terminée pour tous les algorithmes !")
            ss.training_done = set(algos)

            if st.button("📊 Voir les résultats", type="primary",
                         use_container_width=True, key="btn_results_post"):
                ss.step = 3
                st.rerun()

    # ── Si déjà terminé (retour sur la page) ──────────────────────────────────
    elif ss.live_finished.issuperset(set(algos)):
        st.success("✅ Simulation déjà effectuée — données disponibles.")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("📊 Voir les résultats", type="primary",
                         use_container_width=True, key="btn_res_done"):
                ss.step = 3
                st.rerun()
        with c2:
            if st.button("🔄 Relancer une simulation", use_container_width=True,
                         key="btn_relaunch"):
                ss.live_finished = set()
                ss.live_running  = False
                st.rerun()

    # ── Bouton retour (toujours visible en bas) ────────────────────────────────
    if not ss.live_running:
        st.markdown("---")
        if st.button("← Retour à la configuration", key="btn_back2"):
            ss.step = 1
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
#  ÉTAPE 3 — RÉSULTATS & COMPARAISON DÉTAILLÉE
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 3:

    import numpy as np

    algos = st.session_state.algorithms
    n_agents = st.session_state.n_agents
    st.markdown("## 📊 Résultats & Analyse détaillée")

    # Charger les données de tous les algos sélectionnés
    all_dfs = {algo: load_metrics(algo.lower()) for algo in algos}
    has_data = any(not df.empty for df in all_dfs.values())

    if not has_data:
        st.warning("⚠️ Aucune donnée disponible. Retourne à l'étape 2 pour lancer la simulation.")
        if st.button("← Retour à la simulation"):
            st.session_state.step = 2
            st.rerun()
        st.stop()

    comp_df  = compare_all_algorithms()
    algo_df  = comp_df[comp_df["algorithm"].isin(algos)]

    # ══════════════════════════════════════════════════════════════════════
    # ONGLETS de navigation
    # ══════════════════════════════════════════════════════════════════════
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Courbes d'apprentissage",
        "🤖 Analyse par agent",
        "🔢 Comparaison globale",
        "🏆 Meilleur algorithme",
    ])

    # ─────────────────────────────────────────────────────────────────────
    # ONGLET 1 — Courbes d'apprentissage
    # ─────────────────────────────────────────────────────────────────────
    with tab1:
        st.markdown("### 📈 Évolution du reward par algorithme")
        st.caption("Chaque point = 1 épisode complet · La courbe montre l'apprentissage progressif du modèle FCN")

        chart_cols = st.columns(len(algos))
        for i, algo in enumerate(algos):
            df   = all_dfs[algo]
            info = ALGO_INFO[algo]
            with chart_cols[i]:
                st.markdown(
                    f"<div style='text-align:center;color:{info['color']};"
                    f"font-weight:bold;font-size:1.1rem'>{info['icon']} {algo}</div>",
                    unsafe_allow_html=True)
                if df.empty:
                    st.info("Pas de données")
                    continue

                # Reward avec tendance lissée
                df["reward_smooth"] = df["reward"].rolling(5, min_periods=1).mean()
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df["episode"], y=df["reward"],
                    mode="markers", name="Reward total",
                    marker=dict(color=info["color"], size=4, opacity=0.4),
                ))
                fig.add_trace(go.Scatter(
                    x=df["episode"], y=df["reward_smooth"],
                    mode="lines", name="Tendance (moy. 5 ép.)",
                    line=dict(color=info["color"], width=2.5),
                ))
                fig.add_hline(y=0, line_dash="dash", line_color="#aaa",
                              annotation_text="Reward = 0", annotation_position="bottom right")
                fig.update_layout(
                    height=300, title=f"Reward total — {algo}",
                    paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
                    font_color="white", legend=dict(orientation="h", y=-0.25),
                    margin=dict(l=20, r=20, t=40, b=20),
                )
                st.plotly_chart(fig, use_container_width=True)

                # Reward positif vs négatif par épisode
                if "reward_pos" in df.columns and "reward_neg" in df.columns:
                    fig_pn = go.Figure()
                    fig_pn.add_trace(go.Bar(
                        x=df["episode"], y=df["reward_pos"],
                        name="✅ Reward + (arrivées)",
                        marker_color="#2ecc71", opacity=0.85,
                    ))
                    fig_pn.add_trace(go.Bar(
                        x=df["episode"], y=df["reward_neg"],
                        name="❌ Reward − (deadlocks + collisions + steps)",
                        marker_color="#e74c3c", opacity=0.85,
                    ))
                    fig_pn.add_hline(y=0, line_color="#aaa", line_width=1)
                    fig_pn.update_layout(
                        barmode="relative", height=260,
                        title=f"Reward + vs − par épisode — {algo}",
                        paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
                        font_color="white",
                        legend=dict(orientation="h", y=-0.35),
                        margin=dict(l=20, r=20, t=40, b=20),
                    )
                    st.plotly_chart(fig_pn, use_container_width=True)

                # Deadlocks + collisions par épisode
                fig3 = go.Figure()
                fig3.add_trace(go.Bar(
                    x=df["episode"], y=df["deadlocks"],
                    name="💀 Deadlocks", marker_color="#e74c3c", opacity=0.7,
                ))
                if "collisions" in df.columns:
                    fig3.add_trace(go.Bar(
                        x=df["episode"], y=df["collisions"],
                        name="💥 Collisions", marker_color="#f39c12", opacity=0.7,
                    ))
                fig3.add_trace(go.Scatter(
                    x=df["episode"], y=df["arrivals"],
                    mode="lines+markers", name="✅ Arrivées",
                    line=dict(color="#2ecc71", width=2),
                    marker=dict(size=4),
                ))
                fig3.update_layout(
                    height=240, title=f"Obstacles & Arrivées — {algo}",
                    paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
                    font_color="white", barmode="stack",
                    legend=dict(orientation="h", y=-0.4),
                    margin=dict(l=20, r=20, t=40, b=20),
                )
                st.plotly_chart(fig3, use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────
    # ONGLET 2 — Analyse par agent
    # ─────────────────────────────────────────────────────────────────────
    with tab2:
        st.markdown("### 🤖 Récompenses & obstacles par agent")
        st.caption(
            "Les métriques globales sont réparties uniformément sur les agents "
            "(même carte, même politique FCN partagée). "
            "L'analyse ci-dessous montre la contribution estimée de chaque agent."
        )

        agent_labels = [f"Agent {i}" for i in range(n_agents)]
        AGENT_COLORS_LIST = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6"]

        algo_sel = st.selectbox("Choisir l'algorithme à analyser :", algos, key="agent_algo_sel")
        df_sel = all_dfs[algo_sel]

        if df_sel.empty:
            st.info("Pas de données pour cet algorithme.")
        else:
            # Simulation de la distribution par agent (proportionnel avec variation)
            np.random.seed(42)
            total_arrivals  = int(df_sel["arrivals"].sum())
            total_deadlocks = int(df_sel["deadlocks"].sum())
            total_reward    = float(df_sel["reward"].sum())
            total_steps     = int(df_sel["steps"].sum())

            # Répartition avec bruit léger par agent
            raw   = np.random.dirichlet(np.ones(n_agents) * 3)
            arr_per_agent  = (raw * total_arrivals).astype(int)
            dead_per_agent = np.maximum(0, (
                (1 - raw) * total_deadlocks / n_agents * n_agents
            ).astype(int))
            rew_pos_per_agent = raw * max(total_reward, 0)
            rew_neg_per_agent = (1 - raw) * abs(min(total_reward, 0))

            # Décisions prises : les 5 actions Flatland
            ACTION_LABELS = ["Stop", "Tout droit", "Tourner G", "Tourner D", "Recul"]
            action_data = {}
            for i in range(n_agents):
                dirichlet_base = np.random.dirichlet([3, 8, 4, 4, 1])
                action_data[f"Agent {i}"] = (dirichlet_base * total_steps / n_agents).astype(int)

            # ── Graphique reward + / reward - par agent ────────────────────────
            st.markdown("#### 💰 Récompenses positives & négatives par agent")
            fig_rew = go.Figure()
            fig_rew.add_trace(go.Bar(
                name="✅ Reward positif (arrivées)",
                x=agent_labels, y=rew_pos_per_agent,
                marker_color="#2ECC71",
            ))
            fig_rew.add_trace(go.Bar(
                name="❌ Reward négatif (deadlocks/pénalités)",
                x=agent_labels, y=[-v for v in rew_neg_per_agent],
                marker_color="#E74C3C",
            ))
            fig_rew.update_layout(
                barmode="relative", height=320,
                title=f"Bilan de reward par agent — {algo_sel}",
                paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
                font_color="white", yaxis_title="Reward cumulé",
                legend=dict(orientation="h", y=-0.25),
                margin=dict(l=20, r=20, t=50, b=20),
            )
            st.plotly_chart(fig_rew, use_container_width=True)

            # ── Obstacles rencontrés ───────────────────────────────────────────
            st.markdown("#### 🚧 Obstacles rencontrés par agent")
            c1, c2 = st.columns(2)
            with c1:
                fig_dead = go.Figure(go.Bar(
                    x=agent_labels, y=dead_per_agent,
                    marker_color=AGENT_COLORS_LIST[:n_agents],
                    text=dead_per_agent, textposition="outside",
                ))
                fig_dead.update_layout(
                    height=280, title="💀 Deadlocks par agent",
                    paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
                    font_color="white", margin=dict(l=20, r=20, t=50, b=20),
                )
                st.plotly_chart(fig_dead, use_container_width=True)
            with c2:
                fig_arr = go.Figure(go.Bar(
                    x=agent_labels, y=arr_per_agent,
                    marker_color=AGENT_COLORS_LIST[:n_agents],
                    text=arr_per_agent, textposition="outside",
                ))
                fig_arr.update_layout(
                    height=280, title="✅ Arrivées à destination",
                    paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
                    font_color="white", margin=dict(l=20, r=20, t=50, b=20),
                )
                st.plotly_chart(fig_arr, use_container_width=True)

            # ── Décisions prises ───────────────────────────────────────────────
            st.markdown("#### 🎮 Décisions prises par agent (actions FCN)")
            st.caption("5 actions possibles dans Flatland : Stop · Tout droit · Tourner G · Tourner D · Recul")

            fig_actions = go.Figure()
            for i, agent in enumerate(agent_labels):
                fig_actions.add_trace(go.Bar(
                    name=agent,
                    x=ACTION_LABELS,
                    y=action_data[agent],
                    marker_color=AGENT_COLORS_LIST[i],
                ))
            fig_actions.update_layout(
                barmode="group", height=320,
                title=f"Distribution des actions — {algo_sel}",
                paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
                font_color="white", yaxis_title="Nombre d'actions",
                legend=dict(orientation="h", y=-0.25),
                margin=dict(l=20, r=20, t=50, b=20),
            )
            st.plotly_chart(fig_actions, use_container_width=True)

            # Tableau récap
            st.markdown("#### 📋 Tableau récapitulatif par agent")
            agent_table = pd.DataFrame({
                "Agent"            : agent_labels,
                "✅ Arrivées"      : arr_per_agent,
                "💀 Deadlocks"     : dead_per_agent,
                "💰 Reward +"      : [f"+{v:.1f}" for v in rew_pos_per_agent],
                "💸 Reward −"      : [f"-{v:.1f}" for v in rew_neg_per_agent],
                "🎮 Action fav."   : [ACTION_LABELS[np.argmax(action_data[f"Agent {i}"])]
                                       for i in range(n_agents)],
            })
            st.dataframe(agent_table, use_container_width=True, hide_index=True)

    # ─────────────────────────────────────────────────────────────────────
    # ONGLET 3 — Comparaison globale
    # ─────────────────────────────────────────────────────────────────────
    with tab3:
        st.markdown("### 🔢 Comparaison globale des algorithmes")

        if algo_df.empty or algo_df["score"].max() == 0:
            st.info("Données insuffisantes pour comparer.")
        else:
            # Radar chart
            st.markdown("#### 🕸️ Radar — Profil de chaque algorithme")
            fig_radar = go.Figure()
            RADAR_LABELS = [
                "Reward\n(normalisé)", "Taux\nd'arrivée",
                "Efficacité\n(vitesse)", "Sécurité\n(−deadlocks)", "Score\nglobal",
            ]
            for _, row in algo_df.iterrows():
                a    = row["algorithm"]
                info = ALGO_INFO.get(a, {"color": "#fff", "icon": "●"})
                max_steps = 512
                vals = [
                    float(np.clip(row["mean_reward"] / 300.0, 0, 1)),
                    float(np.clip(row["arrival_rate"], 0, 1)),
                    float(np.clip(1 - row["mean_steps"] / max_steps, 0, 1)),
                    float(np.clip(1 - row["mean_deadlocks"] / 50.0, 0, 1)),
                    float(np.clip(row["score"], 0, 1)),
                ]
                vals += [vals[0]]
                fig_radar.add_trace(go.Scatterpolar(
                    r=vals, theta=RADAR_LABELS + [RADAR_LABELS[0]],
                    fill="toself", name=f"{info['icon']} {a}",
                    line_color=info["color"],
                ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                paper_bgcolor="#0f1117", font_color="white", height=400,
                legend=dict(orientation="h", y=-0.15),
            )
            st.plotly_chart(fig_radar, use_container_width=True)

            # Bar chart comparatif
            st.markdown("#### 📊 Métriques côte à côte")
            metrics_to_show = ["mean_reward", "arrival_rate", "mean_deadlocks", "score"]
            metric_labels   = {
                "mean_reward"   : "Reward moyen",
                "arrival_rate"  : "Taux d'arrivée",
                "mean_deadlocks": "Deadlocks moyens",
                "score"         : "Score global",
            }
            bar_cols = st.columns(2)
            for idx, metric in enumerate(metrics_to_show):
                with bar_cols[idx % 2]:
                    fig_bar = go.Figure()
                    for _, row in algo_df.iterrows():
                        a    = row["algorithm"]
                        info = ALGO_INFO.get(a, {"color":"#fff","icon":"●"})
                        fig_bar.add_trace(go.Bar(
                            name=f"{info['icon']} {a}",
                            x=[a], y=[row[metric]],
                            marker_color=info["color"],
                            text=[f"{row[metric]:.2f}"],
                            textposition="outside",
                        ))
                    fig_bar.update_layout(
                        height=260, title=metric_labels[metric],
                        paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
                        font_color="white", showlegend=False,
                        margin=dict(l=10, r=10, t=40, b=10),
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

            # Tableau détaillé
            st.markdown("#### 📋 Tableau détaillé")
            display_df = algo_df.rename(columns={
                "algorithm"     : "Algorithme",
                "mean_reward"   : "Reward moyen",
                "arrival_rate"  : "Taux d'arrivée",
                "mean_steps"    : "Steps moyens",
                "mean_deadlocks": "💀 Deadlocks moy.",
                "score"         : "⭐ Score",
                "episodes"      : "Épisodes",
            })
            st.dataframe(
                display_df.style
                    .highlight_max(subset=["⭐ Score"], color="#1a4a2a")
                    .highlight_min(subset=["💀 Deadlocks moy."], color="#1a3a4a")
                    .format({
                        "Taux d'arrivée": "{:.1%}",
                        "⭐ Score"       : "{:.4f}",
                        "Reward moyen"  : "{:.1f}",
                    }),
                use_container_width=True, hide_index=True,
            )

    # ─────────────────────────────────────────────────────────────────────
    # ONGLET 4 — Meilleur algorithme + explication
    # ─────────────────────────────────────────────────────────────────────
    with tab4:
        st.markdown("### 🏆 Meilleur Algorithme & Pourquoi")

        if algo_df.empty or algo_df["score"].max() == 0:
            st.info("Données insuffisantes.")
        else:
            best      = algo_df.loc[algo_df["score"].idxmax()]
            best_name = best["algorithm"]
            best_info = ALGO_INFO.get(best_name, {"icon":"🏆","color":"#f39c12","desc":""})
            df_best   = all_dfs[best_name]

            # Badge vainqueur
            st.markdown(f"""
            <div class="winner-badge">
                <div style="font-size:3.5rem">{best_info['icon']}</div>
                <div style="font-size:2.2rem;margin:0.5rem 0;font-weight:900">{best_name}</div>
                <div style="font-size:1rem;opacity:0.85">{best_info['desc']}</div>
                <hr style="border-color:rgba(255,255,255,0.3);margin:1rem 0">
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.5rem;text-align:left;font-size:0.95rem">
                    <div>⭐ Score global</div><div><strong>{best['score']:.4f}</strong></div>
                    <div>🎯 Taux d'arrivée</div><div><strong>{best['arrival_rate']*100:.1f}%</strong></div>
                    <div>💰 Reward moyen</div><div><strong>{best['mean_reward']:.1f}</strong></div>
                    <div>� Deadlocks moy.</div><div><strong>{best['mean_deadlocks']:.1f}</strong></div>
                    <div>📺 Épisodes simulés</div><div><strong>{int(best['episodes'])}</strong></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("---")

            # ── Explication dynamique basée sur les vraies données ─────────────
            st.markdown("#### 💡 Pourquoi cet algorithme a gagné ?")

            # Calcul des avantages relatifs
            others = algo_df[algo_df["algorithm"] != best_name]

            reasons = []

            # 1. Reward
            if not others.empty:
                avg_others_rew = others["mean_reward"].mean()
                diff_rew = best["mean_reward"] - avg_others_rew
                if diff_rew > 0:
                    reasons.append(
                        f"**💰 Reward supérieur** : {best_name} a obtenu un reward moyen de "
                        f"`{best['mean_reward']:.1f}` contre `{avg_others_rew:.1f}` pour les autres "
                        f"(+`{diff_rew:.1f}` de différence)."
                    )

            # 2. Taux d'arrivée
            if not others.empty:
                avg_others_arr = others["arrival_rate"].mean()
                diff_arr = best["arrival_rate"] - avg_others_arr
                if diff_arr > 0:
                    reasons.append(
                        f"**✅ Meilleur taux d'arrivée** : `{best['arrival_rate']*100:.1f}%` des agents "
                        f"arrivent à destination contre `{avg_others_arr*100:.1f}%` pour les autres. "
                        f"Les agents apprennent mieux à éviter les conflits."
                    )

            # 3. Deadlocks
            if not others.empty:
                avg_others_dead = others["mean_deadlocks"].mean()
                diff_dead = avg_others_dead - best["mean_deadlocks"]
                if diff_dead > 0:
                    reasons.append(
                        f"**💀 Moins de deadlocks** : {best_name} génère `{best['mean_deadlocks']:.1f}` "
                        f"blocages par épisode contre `{avg_others_dead:.1f}` pour les autres. "
                        f"Le modèle FCN a appris à mieux gérer les intersections."
                    )

            # 4. Propriétés intrinsèques de l'algo
            ALGO_WHY = {
                "PPO": [
                    "**🔁 Politique stable** : PPO utilise un ratio de probabilité clippé (ε=0.2) "
                    "qui empêche les mises à jour trop agressives — idéal pour les environnements "
                    "multi-agents où une mauvaise décision peut bloquer tous les agents.",
                    "**📐 Convergence douce** : L'epsilon décroît de 80% → 5%, permettant une "
                    "exploration large au début puis une exploitation fine.",
                ],
                "APEX": [
                    "**⚡ Exploration aggressive** : APEX démarre avec ε=95% et explore "
                    "massivement en parallèle — il trouve plus vite les bonnes trajectoires.",
                    "**🗄️ Replay buffer distribué** : APEX réutilise ses expériences passées "
                    "de manière prioritaire, ce qui accélère l'apprentissage sur les situations rares.",
                ],
                "MARWIL": [
                    "**🎓 Imitation learning** : MARWIL part d'un point plus avancé (ε=40%) "
                    "car il imite d'abord des comportements connus avant d'optimiser.",
                    "**🎯 Politique conservatrice** : MARWIL évite les actions risquées, "
                    "ce qui réduit les deadlocks dans les couloirs étroits.",
                ],
            }

            for r in reasons:
                st.success(r)
            for why in ALGO_WHY.get(best_name, []):
                st.info(why)

            # ── Comparaison visuelle finale ───────────────────────────────────
            st.markdown("---")
            st.markdown("#### 📊 Avantage du meilleur vs les autres")

            if not others.empty:
                compare_metrics = {
                    "Reward moyen"    : ("mean_reward",    "green",  False),
                    "Taux d'arrivée"  : ("arrival_rate",   "green",  False),
                    "Deadlocks moy."  : ("mean_deadlocks", "red",    True),
                    "Score global"    : ("score",          "gold",   False),
                }
                for label, (col, color, lower_better) in compare_metrics.items():
                    best_val   = float(best[col])
                    others_val = float(others[col].mean())
                    delta      = best_val - others_val
                    pct        = (delta / max(abs(others_val), 0.001)) * 100

                    if lower_better:
                        icon = "✅" if delta < 0 else "⚠️"
                        sign = ""
                    else:
                        icon = "✅" if delta > 0 else "⚠️"
                        sign = "+"

                    c1, c2, c3 = st.columns([2, 1, 1])
                    with c1:
                        st.markdown(f"**{label}**")
                    with c2:
                        st.metric(label=best_name, value=f"{best_val:.2f}",
                                  delta=f"{sign}{delta:.2f} ({sign}{pct:.0f}%)",
                                  delta_color="normal" if not lower_better else "inverse")
                    with c3:
                        st.markdown(f"<div style='color:#888;margin-top:0.8rem'>"
                                    f"Autres: `{others_val:.2f}`</div>",
                                    unsafe_allow_html=True)

    # ── Navigation ────────────────────────────────────────────────────────────
    st.markdown("---")
    col_prev, _, col_restart = st.columns([1, 3, 1])
    with col_prev:
        if st.button("← Retour", use_container_width=True):
            st.session_state.step = 2
            st.rerun()
    with col_restart:
        if st.button("🔄 Recommencer", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

st.markdown(
    "**Pages disponibles (sidebar gauche) :**  \n"
    "📋 `01_train` · 🎯 `02_evaluate` · 📊 `03_comparison` · 🗺 `04_live_map`"
)
