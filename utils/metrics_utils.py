import os
import pandas as pd
import numpy as np


RESULTS_BASE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
ALGO_NAMES = ["ppo", "apex", "marwil"]


def save_episode_metrics(
    algo_name: str,
    episode: int,
    reward: float,
    arrivals: int,
    steps: int,
    deadlocks: int,
) -> None:
    """Append one episode's metrics to results/{algo}/metrics.csv."""
    algo_dir = os.path.join(RESULTS_BASE, algo_name.lower())
    os.makedirs(algo_dir, exist_ok=True)
    csv_path = os.path.join(algo_dir, "metrics.csv")

    row = {
        "episode": episode,
        "reward": reward,
        "arrivals": arrivals,
        "steps": steps,
        "deadlocks": deadlocks,
        "arrival_rate": arrivals / 5.0,
    }
    df_new = pd.DataFrame([row])

    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new

    df_combined.to_csv(csv_path, index=False)


def load_metrics(algo_name: str) -> pd.DataFrame:
    """Load metrics CSV for a given algorithm. Returns empty DataFrame if not found."""
    csv_path = os.path.join(RESULTS_BASE, algo_name.lower(), "metrics.csv")
    if not os.path.exists(csv_path):
        return pd.DataFrame(columns=["episode", "reward", "arrivals", "steps", "deadlocks", "arrival_rate"])
    return pd.read_csv(csv_path)


def compute_algorithm_score(df: pd.DataFrame) -> float:
    """Compute a global score: (mean_reward * 0.4) + (arrival_rate * 0.4) + (1/mean_steps * 0.2)."""
    if df.empty:
        return 0.0
    mean_reward = df["reward"].mean()
    arrival_rate = df["arrival_rate"].mean() if "arrival_rate" in df.columns else (df["arrivals"].mean() / 5.0)
    mean_steps = df["steps"].mean()

    # Normalize reward to [0, 1] assuming max possible ~100
    norm_reward = np.clip(mean_reward / 100.0, 0.0, 1.0)
    # arrival_rate already in [0, 1]
    norm_arrival = np.clip(arrival_rate, 0.0, 1.0)
    # Rapidité: fewer steps = better; normalize assuming max 512 steps
    norm_speed = np.clip(1.0 - (mean_steps / 512.0), 0.0, 1.0)

    score = (norm_reward * 0.4) + (norm_arrival * 0.4) + (norm_speed * 0.2)
    return float(score)


def compare_all_algorithms() -> pd.DataFrame:
    """Load metrics for all algorithms and return a comparative DataFrame with scores."""
    rows = []
    for algo in ALGO_NAMES:
        df = load_metrics(algo)
        if df.empty:
            rows.append({
                "algorithm": algo.upper(),
                "mean_reward": 0.0,
                "arrival_rate": 0.0,
                "mean_steps": 0.0,
                "mean_deadlocks": 0.0,
                "score": 0.0,
                "episodes": 0,
            })
            continue
        score = compute_algorithm_score(df)
        arrival_rate = df["arrival_rate"].mean() if "arrival_rate" in df.columns else (df["arrivals"].mean() / 5.0)
        rows.append({
            "algorithm": algo.upper(),
            "mean_reward": round(df["reward"].mean(), 3),
            "arrival_rate": round(float(arrival_rate), 3),
            "mean_steps": round(df["steps"].mean(), 1),
            "mean_deadlocks": round(df["deadlocks"].mean(), 2),
            "score": round(score, 4),
            "episodes": len(df),
        })
    return pd.DataFrame(rows)


def get_best_algorithm() -> str:
    """Return the name of the algorithm with the highest global score."""
    comparison = compare_all_algorithms()
    if comparison.empty or comparison["score"].max() == 0:
        return "Aucun algorithme entraîné"
    best_idx = comparison["score"].idxmax()
    return comparison.loc[best_idx, "algorithm"]
