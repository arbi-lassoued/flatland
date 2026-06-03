import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from utils.metrics_utils import compute_algorithm_score, load_metrics


ALGO_COLORS = {"PPO": "#3498DB", "APEX": "#E74C3C", "MARWIL": "#2ECC71"}


def create_comparison_table(ppo_df: pd.DataFrame, apex_df: pd.DataFrame, marwil_df: pd.DataFrame) -> pd.DataFrame:
    """Build a formatted comparison DataFrame for st.dataframe()."""
    rows = []
    for name, df in [("PPO", ppo_df), ("APEX", apex_df), ("MARWIL", marwil_df)]:
        if df.empty:
            rows.append({
                "Algorithme": name,
                "Reward Moyen": "—",
                "Taux Arrivée": "—",
                "Steps Moyen": "—",
                "Deadlocks Moyen": "—",
                "Score Global": "—",
                "Épisodes": 0,
            })
            continue
        score = compute_algorithm_score(df)
        arrival = df["arrival_rate"].mean() if "arrival_rate" in df.columns else (df["arrivals"].mean() / 5.0)
        rows.append({
            "Algorithme": name,
            "Reward Moyen": f"{df['reward'].mean():.2f}",
            "Taux Arrivée": f"{arrival*100:.1f}%",
            "Steps Moyen": f"{df['steps'].mean():.0f}",
            "Deadlocks Moyen": f"{df['deadlocks'].mean():.2f}",
            "Score Global": f"{score:.4f}",
            "Épisodes": len(df),
        })
    return pd.DataFrame(rows)


def create_radar_chart(scores_dict: dict):
    """
    Create a radar chart comparing algorithms on 5 dimensions.
    scores_dict: {"PPO": df, "APEX": df, "MARWIL": df}
    """
    dimensions = ["Reward", "Arrivées", "Rapidité", "Stabilité", "Efficacité"]
    fig = go.Figure()

    for algo, df in scores_dict.items():
        color = ALGO_COLORS.get(algo, "#95A5A6")
        if df.empty:
            values = [0.0] * 5
        else:
            mean_reward = min(df["reward"].mean() / 100.0, 1.0)
            arrival = df["arrival_rate"].mean() if "arrival_rate" in df.columns else df["arrivals"].mean() / 5.0
            rapidity = max(0.0, 1.0 - df["steps"].mean() / 512.0)
            stability = max(0.0, 1.0 - df["reward"].std() / (abs(df["reward"].mean()) + 1e-6))
            deadlock_rate = df["deadlocks"].mean() / (df["steps"].mean() + 1e-6)
            efficacy = max(0.0, 1.0 - deadlock_rate * 10)
            values = [mean_reward, arrival, rapidity, stability, efficacy]

        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=dimensions + [dimensions[0]],
            fill="toself",
            name=algo,
            line_color=color,
            fillcolor=color,
            opacity=0.25,
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], tickformat=".0%"),
        ),
        showlegend=True,
        title="Comparaison Multi-Dimensions des Algorithmes",
        template="plotly_white",
        height=420,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def create_reward_bar(ppo_df: pd.DataFrame, apex_df: pd.DataFrame, marwil_df: pd.DataFrame):
    """Bar chart comparing mean reward across algorithms."""
    data = []
    for name, df in [("PPO", ppo_df), ("APEX", apex_df), ("MARWIL", marwil_df)]:
        mean_r = df["reward"].mean() if not df.empty else 0.0
        data.append({"Algorithme": name, "Reward Moyen": mean_r})

    df_bar = pd.DataFrame(data)
    fig = px.bar(
        df_bar, x="Algorithme", y="Reward Moyen",
        color="Algorithme",
        color_discrete_map=ALGO_COLORS,
        title="Reward Moyen par Algorithme",
        template="plotly_white",
        height=320,
        text_auto=".2f",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(margin=dict(l=30, r=20, t=50, b=30))
    return fig


def create_reward_evolution(ppo_df: pd.DataFrame, apex_df: pd.DataFrame, marwil_df: pd.DataFrame):
    """Line chart showing reward evolution per episode for all 3 algos."""
    fig = go.Figure()
    for name, df in [("PPO", ppo_df), ("APEX", apex_df), ("MARWIL", marwil_df)]:
        color = ALGO_COLORS.get(name, "#95A5A6")
        if df.empty:
            continue
        d = df.copy().reset_index(drop=True)
        d["episode"] = range(1, len(d) + 1)
        d["rolling"] = d["reward"].rolling(10, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=d["episode"], y=d["rolling"],
            mode="lines", name=name,
            line=dict(color=color, width=2),
        ))

    fig.update_layout(
        title="Évolution du Reward (moyenne mobile 10 ep.) par Algorithme",
        xaxis_title="Épisode",
        yaxis_title="Récompense",
        template="plotly_white",
        height=350,
        legend=dict(orientation="h", y=1.05),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def create_arrival_vs_reward_scatter(ppo_df: pd.DataFrame, apex_df: pd.DataFrame, marwil_df: pd.DataFrame):
    """Scatter chart: taux d'arrivée vs reward moyen."""
    data = []
    for name, df in [("PPO", ppo_df), ("APEX", apex_df), ("MARWIL", marwil_df)]:
        if df.empty:
            data.append({"Algorithme": name, "Reward Moyen": 0, "Taux Arrivée": 0})
        else:
            arrival = df["arrival_rate"].mean() if "arrival_rate" in df.columns else df["arrivals"].mean() / 5.0
            data.append({"Algorithme": name, "Reward Moyen": df["reward"].mean(), "Taux Arrivée": arrival})

    df_sc = pd.DataFrame(data)
    fig = px.scatter(
        df_sc, x="Taux Arrivée", y="Reward Moyen",
        color="Algorithme", text="Algorithme",
        color_discrete_map=ALGO_COLORS,
        size=[40] * len(df_sc),
        title="Taux d'Arrivée vs Reward Moyen",
        template="plotly_white",
        height=320,
    )
    fig.update_traces(textposition="top center")
    fig.update_layout(margin=dict(l=30, r=20, t=50, b=30))
    return fig


def create_stacked_reward_breakdown(breakdown_by_algo: dict):
    """
    Stacked bar chart of reward components per algorithm.
    breakdown_by_algo: {"PPO": {"arrived": x, "deadlock": x, ...}, ...}
    """
    components = ["arrived", "step_penalty", "deadlock", "collision", "cooperative", "invalid_action"]
    comp_colors = {
        "arrived": "#2ECC71",
        "step_penalty": "#95A5A6",
        "deadlock": "#E74C3C",
        "collision": "#E67E22",
        "cooperative": "#3498DB",
        "invalid_action": "#9B59B6",
    }

    algos = list(breakdown_by_algo.keys())
    fig = go.Figure()

    for comp in components:
        values = [abs(breakdown_by_algo.get(algo, {}).get(comp, 0)) for algo in algos]
        fig.add_trace(go.Bar(
            name=comp.replace("_", " ").title(),
            x=algos,
            y=values,
            marker_color=comp_colors.get(comp, "#BDC3C7"),
        ))

    fig.update_layout(
        barmode="stack",
        title="Décomposition des Récompenses par Algorithme",
        xaxis_title="Algorithme",
        yaxis_title="Valeur absolue des composantes",
        template="plotly_white",
        height=350,
        legend=dict(orientation="h", y=1.1),
        margin=dict(l=30, r=20, t=70, b=30),
    )
    return fig
