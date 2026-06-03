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
        st.image("https://i.imgur.com/9cNtWjs.gif", caption="Flatland — AICrowd", use_column_width=True)

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
#  ÉTAPE 2 — EXÉCUTION
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 2:

    algos  = st.session_state.algorithms
    n_agents = st.session_state.n_agents
    smoke  = st.session_state.smoke_test

    st.markdown(f"## 🚀 Exécution — {len(algos)} algorithme(s) · {n_agents} agent(s)")

    # ── Cartes live ───────────────────────────────────────────────────────────
    st.markdown("### 🗺️ Cartes Flatland par algorithme")
    map_cols = st.columns(len(algos))

    # Initialise l'env si besoin
    if st.session_state.env_obj is None:
        with st.spinner("Initialisation de l'environnement…"):
            try:
                from envs.flatland_env import FlatlandMultiAgentEnv
                from streamlit_app.components.map_renderer import render_flatland_streamlit
                env = FlatlandMultiAgentEnv(n_agents_override=n_agents)
                env.reset()
                st.session_state.env_obj = env
            except Exception as e:
                st.error(f"Erreur environnement : {e}")

    env = st.session_state.env_obj
    for i, algo in enumerate(algos):
        info = ALGO_INFO[algo]
        with map_cols[i]:
            st.markdown(f"<div style='text-align:center; font-size:1.3rem; "
                        f"color:{info['color']}; font-weight:bold'>"
                        f"{info['icon']} {algo}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='text-align:center; color:#aaa; "
                        f"font-size:0.8rem'>{info['desc']}</div>", unsafe_allow_html=True)
            st.markdown("**🧠 Modèle :**")
            st.code("Input(231) → FC(256) → FC(256) → FC(256)\n"
                    "→ Policy(5 actions) + Value(1)", language="text")
            if env is not None:
                try:
                    from streamlit_app.components.map_renderer import render_flatland_streamlit
                    img = render_flatland_streamlit(env.rail_env, width=400, height=400)
                    st.image(img, caption=f"Carte — {n_agents} agents — seed=42",
                             use_column_width=True)
                except Exception as ex:
                    st.warning(f"Rendu : {ex}")
            else:
                st.info("Carte indisponible")

    st.markdown("---")

    # ── Lancement entraînement ────────────────────────────────────────────────
    st.markdown("### ▶️ Générer les données & Lancer l'entraînement")

    # ── Option 1 : Générer données via FCN (rollouts réels) ──────────────────
    with st.expander("🧠 **Option 1 — Générer des données avec le modèle FCN** (recommandé)", expanded=True):
        st.markdown("""
**Principe :** Le réseau FCN (**Input 231 → FC256 → FC256 → FC128 → Policy 5 actions**)
joue des épisodes réels sur l'environnement Flatland.
Les métriques (reward, taux d'arrivée, deadlocks) sont sauvegardées et visualisées à l'étape suivante.
        """)

        n_eps = st.slider("Nombre d'épisodes par algorithme", 10, 100, 40, step=10,
                          key="n_eps_gen")
        overwrite_data = st.checkbox("Écraser les données existantes", value=False, key="overwrite_chk")

        if st.button("🚀 Générer avec le FCN", type="primary", use_container_width=True,
                     key="btn_generate"):
            from utils.generate_data import generate_for_algo

            for algo in algos:
                st.markdown(f"**{ALGO_INFO[algo]['icon']} {algo}** — génération en cours…")
                prog_bar = st.progress(0, text=f"{algo} — démarrage…")

                try:
                    generate_for_algo(
                        algo=algo.lower(),
                        n_episodes=n_eps,
                        n_agents=n_agents,
                        overwrite=overwrite_data,
                        streamlit_progress=lambda v, text="": prog_bar.progress(v, text=text),
                    )
                    prog_bar.progress(1.0, text=f"✅ {algo} terminé")
                    st.success(f"✅ {algo} — {n_eps} épisodes générés")
                except Exception as e:
                    st.error(f"❌ {algo} — Erreur : {e}")

            st.session_state.training_done = set(algos)
            st.balloons()

    st.markdown("---")

    # ── Option 2 : Entraînement RLlib complet ────────────────────────────────
    with st.expander("⚙️ Option 2 — Entraînement complet via RLlib (PPO/APEX/MARWIL)", expanded=False):
        active_procs = st.session_state.training_procs
        running = [a for a, p in active_procs.items() if p.poll() is None]
        done_algos = st.session_state.training_done

        if running:
            st.info(f"⏳ En cours : {', '.join(running)}")
            progress_cols = st.columns(len(running))
            for i, algo in enumerate(running):
                with progress_cols[i]:
                    df_live = load_metrics(algo.lower())
                    ep = len(df_live)
                    total = 50 if smoke else 500
                    pct = min(ep / total, 1.0)
                    st.markdown(f"**{algo}** — épisode {ep}/{total}")
                    st.progress(pct)
            time.sleep(2)
            st.rerun()

        newly_done = [a for a, p in active_procs.items()
                      if p.poll() is not None and a not in done_algos]
        for a in newly_done:
            done_algos.add(a)
        st.session_state.training_done = done_algos

        if st.button("⚡ Lancer l'entraînement RLlib", use_container_width=True,
                     disabled=bool(running), key="btn_rllib"):
            for algo in algos:
                cmd = [sys.executable, os.path.join(PROJECT_ROOT, "train.py"),
                       "--algorithm", algo.lower(), "--num-cpus", "4"]
                if smoke:
                    cmd.append("--smoke-test")
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT,
                                        text=True, cwd=PROJECT_ROOT)
                active_procs[algo] = proc
            st.session_state.training_procs = active_procs
            st.success(f"Entraînement RLlib lancé : {', '.join(algos)}")
            st.rerun()

    done_algos = st.session_state.training_done
    all_done = bool(done_algos) and all(a in done_algos for a in algos)

    if all_done:
        st.success(f"✅ Données disponibles pour : {', '.join(done_algos)}")
        col_c = st.columns([2, 1, 2])[1]
        with col_c:
            if st.button("📊 Voir les résultats", type="primary", use_container_width=True):
                st.session_state.step = 3
                st.rerun()

    col_prev, col_skip_btn = st.columns([1, 1])
    with col_prev:
        if st.button("← Retour", use_container_width=True, key="btn_back_step2"):
            st.session_state.step = 1
            st.rerun()
    with col_skip_btn:
        if st.button("Passer → Résultats (données existantes)", use_container_width=True,
                     key="btn_skip_step2"):
            st.session_state.step = 3
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
#  ÉTAPE 3 — RÉSULTATS & COMPARAISON
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 3:

    algos = st.session_state.algorithms
    st.markdown("## 📊 Résultats & Comparaison")

    # ── Courbes de récompense ─────────────────────────────────────────────────
    st.markdown("### 📈 Courbes d'apprentissage")
    chart_cols = st.columns(len(algos))
    has_data = False

    for i, algo in enumerate(algos):
        df = load_metrics(algo.lower())
        info = ALGO_INFO[algo]
        with chart_cols[i]:
            st.markdown(f"<div style='text-align:center; color:{info['color']};"
                        f"font-weight:bold; font-size:1.1rem'>{info['icon']} {algo}</div>",
                        unsafe_allow_html=True)
            if not df.empty:
                has_data = True
                fig = px.line(df, x="episode", y="reward",
                              title=f"Reward — {algo}",
                              color_discrete_sequence=[info['color']])
                fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20),
                                  paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
                                  font_color="white")
                st.plotly_chart(fig, use_container_width=True)

                # Taux d'arrivée
                fig2 = px.line(df, x="episode", y="arrival_rate",
                               title=f"Taux d'arrivée — {algo}",
                               color_discrete_sequence=[info['color']])
                fig2.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20),
                                   paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
                                   font_color="white", yaxis_range=[0, 1])
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info(f"Aucune donnée pour {algo}.\nLance l'entraînement à l'étape précédente.")

    st.markdown("---")

    # ── Tableau de comparaison ────────────────────────────────────────────────
    st.markdown("### 🔢 Tableau comparatif")
    comp_df = compare_all_algorithms()
    algo_df = comp_df[comp_df["algorithm"].isin(algos)]

    if not algo_df.empty and algo_df["score"].max() > 0:
        # Radar chart
        st.markdown("#### 🕸️ Radar — Profil des algorithmes")
        cats = ["mean_reward", "arrival_rate", "score"]
        fig_radar = go.Figure()
        for _, row in algo_df.iterrows():
            algo = row["algorithm"]
            info = ALGO_INFO.get(algo, {"color": "#fff", "icon": "●"})
            vals = [
                min(float(row["mean_reward"]) / 100, 1.0),
                float(row["arrival_rate"]),
                float(row["score"]),
            ]
            vals += [vals[0]]
            labels = ["Reward\n(normalisé)", "Taux\nd'arrivée", "Score\nglobal", "Reward\n(normalisé)"]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals, theta=labels, fill="toself",
                name=f"{info['icon']} {algo}",
                line_color=info["color"],
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
            font_color="white", height=350,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        st.markdown("#### 📋 Tableau détaillé")
        display_df = algo_df.rename(columns={
            "algorithm": "Algorithme",
            "mean_reward": "Reward moyen",
            "arrival_rate": "Taux d'arrivée",
            "mean_steps": "Steps moyens",
            "mean_deadlocks": "Deadlocks",
            "score": "⭐ Score",
            "episodes": "Épisodes",
        })
        st.dataframe(
            display_df.style.highlight_max(subset=["⭐ Score"], color="#1a4a2a")
                            .highlight_min(subset=["Deadlocks"], color="#1a3a4a")
                            .format({"Taux d'arrivée": "{:.1%}", "⭐ Score": "{:.4f}"}),
            use_container_width=True, hide_index=True,
        )

        st.markdown("---")

        # ── Vainqueur ──────────────────────────────────────────────────────────
        st.markdown("### 🏆 Meilleur Algorithme")
        best = algo_df.loc[algo_df["score"].idxmax()]
        best_name = best["algorithm"]
        best_info = ALGO_INFO.get(best_name, {"icon": "🏆", "color": "#f39c12", "desc": ""})

        st.markdown(f"""
        <div class="winner-badge">
            <div style="font-size:3rem">{best_info['icon']}</div>
            <div style="font-size:2rem; margin: 0.5rem 0">{best_name}</div>
            <div style="font-size:1rem; opacity:0.85">{best_info['desc']}</div>
            <hr style="border-color:rgba(255,255,255,0.3)">
            <div>⭐ Score global : <strong>{best['score']:.4f}</strong></div>
            <div>🎯 Taux d'arrivée : <strong>{best['arrival_rate']*100:.1f}%</strong></div>
            <div>💰 Reward moyen : <strong>{best['mean_reward']}</strong></div>
            <div>🚀 Épisodes entraînés : <strong>{int(best['episodes'])}</strong></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### 💡 Conclusion")
        conclusions = {
            "PPO":    "PPO est **stable et fiable**. Idéal pour des environnements complexes avec peu de ressources.",
            "APEX":   "APEX est **le plus rapide** grâce au parallélisme. Parfait pour explorer rapidement les politiques.",
            "MARWIL": "MARWIL excelle quand on a des **données expertes**. Très efficace en apprentissage par imitation.",
        }
        st.info(f"**{best_name}** a obtenu le meilleur score global. {conclusions.get(best_name, '')}")

    else:
        st.warning("Aucune donnée disponible. Retourne à l'étape 2 pour lancer l'entraînement.")

    st.markdown("---")
    col_prev, _, col_restart = st.columns([1, 3, 1])
    with col_prev:
        if st.button("← Retour", use_container_width=True):
            st.session_state.step = 2
            st.rerun()
    with col_restart:
        if st.button("🔄 Recommencer", use_container_width=True):
            for key in ["step", "algorithms", "training_procs", "training_done", "env_obj"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

st.markdown(
    "**Pages disponibles (sidebar gauche) :**  \n"
    "📋 `01_train` · 🎯 `02_evaluate` · 📊 `03_comparison` · 🗺 `04_live_map`"
)
