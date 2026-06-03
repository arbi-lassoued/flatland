import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_rewards(metrics_df: pd.DataFrame, algo_name: str):
    """
    Return a Plotly figure showing episode reward with a rolling-mean overlay.
    Compatible with st.plotly_chart().
    """
    if metrics_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="Aucune donnée disponible", showarrow=False,
                           xref="paper", yref="paper", x=0.5, y=0.5, font=dict(size=16))
        fig.update_layout(title=f"Rewards — {algo_name}")
        return fig

    df = metrics_df.copy().reset_index(drop=True)
    df["episode"] = range(1, len(df) + 1)
    df["rolling_mean"] = df["reward"].rolling(window=10, min_periods=1).mean()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["episode"],
        y=df["reward"],
        mode="lines",
        name="Reward par épisode",
        line=dict(color="#3498DB", width=1.2),
        opacity=0.6,
    ))

    fig.add_trace(go.Scatter(
        x=df["episode"],
        y=df["rolling_mean"],
        mode="lines",
        name="Moyenne mobile (10 ep.)",
        line=dict(color="#E74C3C", width=2, dash="dot"),
    ))

    fig.update_layout(
        title=f"Évolution des Récompenses — {algo_name}",
        xaxis_title="Épisode",
        yaxis_title="Récompense totale",
        legend=dict(orientation="h", y=1.05),
        template="plotly_white",
        height=350,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def plot_reward_breakdown(breakdown_dict: dict):
    """
    Return a Plotly pie chart of reward component proportions.
    breakdown_dict: {"arrived": float, "deadlock": float, ...}
    """
    labels = list(breakdown_dict.keys())
    values = [abs(v) for v in breakdown_dict.values()]

    color_map = {
        "arrived": "#2ECC71",
        "step_penalty": "#95A5A6",
        "deadlock": "#E74C3C",
        "collision": "#E67E22",
        "cooperative": "#3498DB",
        "invalid_action": "#9B59B6",
    }
    colors = [color_map.get(lbl, "#BDC3C7") for lbl in labels]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        marker=dict(colors=colors),
        hole=0.3,
        textinfo="label+percent",
        hovertemplate="%{label}: %{value:.2f}<extra></extra>",
    ))
    fig.update_layout(
        title="Décomposition des Récompenses",
        template="plotly_white",
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=False,
    )
    return fig
