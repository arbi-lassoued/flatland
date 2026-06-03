#!/usr/bin/env python3
"""
Evaluation script for Flatland 5-agents RL.

Usage:
    python evaluate.py --algorithm ppo --checkpoint results/ppo/.../checkpoint_000050
    python evaluate.py --algorithm ppo --checkpoint auto --num-episodes 5
"""

import argparse
import os
import sys
import glob

import ray
from ray.rllib.models import ModelCatalog
from ray import tune

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from envs.flatland_env import FlatlandMultiAgentEnv
from models.custom_model import FlatlandModel
from utils.metrics_utils import save_episode_metrics
from utils.render_utils import render_episode_gif

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")


def find_latest_checkpoint(algo: str) -> str:
    pattern = os.path.join(RESULTS_DIR, algo, "**", "checkpoint_*", "checkpoint-*")
    checkpoints = sorted(glob.glob(pattern, recursive=True))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint found in results/{algo}/")
    return os.path.dirname(checkpoints[-1])


def restore_agent(algo: str, checkpoint_path: str):
    algo_name_map = {"ppo": "PPO", "apex": "APEX", "marwil": "MARWIL"}
    rllib_algo = algo_name_map[algo.lower()]

    tune.register_env("flatland_5agents", lambda cfg: FlatlandMultiAgentEnv(cfg))
    ModelCatalog.register_custom_model("flatland_model", FlatlandModel)

    from ray.rllib.agents.ppo import PPOTrainer
    from ray.rllib.agents.dqn import ApexTrainer
    from ray.rllib.agents.marwil import MARWILTrainer

    trainer_cls = {"PPO": PPOTrainer, "APEX": ApexTrainer, "MARWIL": MARWILTrainer}[rllib_algo]

    config = {
        "env": "flatland_5agents",
        "framework": "torch",
        "num_workers": 0,
        "multiagent": {
            "policies": {"shared_policy"},
            "policy_mapping_fn": lambda agent_id, *args, **kwargs: "shared_policy",
        },
        "model": {"custom_model": "flatland_model"},
    }

    agent = trainer_cls(config=config)
    agent.restore(checkpoint_path)
    return agent


def run_evaluation(algo: str, checkpoint_path: str, num_episodes: int) -> list:
    ray.init(ignore_reinit_error=True, num_cpus=2)

    agent = restore_agent(algo, checkpoint_path)
    env = FlatlandMultiAgentEnv()

    all_stats = []

    print(f"\nEvaluating {algo.upper()} for {num_episodes} episodes…\n")

    for ep in range(num_episodes):
        obs = env.reset()
        episode_reward = 0.0
        done = {"__all__": False}
        steps = 0

        while not done["__all__"]:
            action_dict = {}
            for agent_id, o in obs.items():
                action = agent.compute_single_action(o, policy_id="shared_policy")
                action_dict[agent_id] = action

            obs, rewards, done, infos = env.step(action_dict)
            episode_reward += sum(rewards.values())
            steps += 1

        stats = env.get_episode_stats()
        stats["reward"] = episode_reward
        all_stats.append(stats)

        save_episode_metrics(
            algo_name=algo,
            episode=ep,
            reward=episode_reward,
            arrivals=stats["arrivals"],
            steps=stats["steps"],
            deadlocks=stats["deadlocks"],
        )

        print(
            f"  Episode {ep+1:3d} | reward={episode_reward:8.2f} | "
            f"arrivals={stats['arrivals']}/5 | steps={stats['steps']} | "
            f"deadlocks={stats['deadlocks']}"
        )

    # Summary
    import pandas as pd
    df = pd.DataFrame(all_stats)
    print(f"\n{'='*60}")
    print(f"  Summary ({algo.upper()}, {num_episodes} episodes)")
    print(f"{'='*60}")
    print(df.describe().to_string())

    eval_dir = os.path.join(RESULTS_DIR, algo.lower())
    os.makedirs(eval_dir, exist_ok=True)
    csv_path = os.path.join(eval_dir, "eval_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved → {csv_path}")

    # GIF
    try:
        gif_path = os.path.join(eval_dir, "episode_eval.gif")
        render_episode_gif(env, agent, gif_path)
        print(f"GIF saved    → {gif_path}")
    except Exception as exc:
        print(f"GIF skipped  → {exc}")

    ray.shutdown()
    return all_stats


def main():
    parser = argparse.ArgumentParser(description="Evaluate Flatland 5-agents RL")
    parser.add_argument("--algorithm", "-a", default="ppo", choices=["ppo", "apex", "marwil"])
    parser.add_argument("--checkpoint", "-c", default="auto",
                        help="Path to checkpoint dir or 'auto' to find latest")
    parser.add_argument("--num-episodes", "-n", type=int, default=10)
    args = parser.parse_args()

    checkpoint_path = (
        find_latest_checkpoint(args.algorithm)
        if args.checkpoint == "auto"
        else args.checkpoint
    )
    print(f"Using checkpoint: {checkpoint_path}")
    run_evaluation(args.algorithm, checkpoint_path, args.num_episodes)


if __name__ == "__main__":
    main()
