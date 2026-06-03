#!/usr/bin/env python3
"""
Main training script for Flatland 5-agents RL.

Usage:
    python train.py --algorithm ppo --num-cpus 4
    python train.py --algorithm apex --num-cpus 4
    python train.py --algorithm marwil --num-cpus 4
    python train.py --algorithm ppo --smoke-test
"""

import argparse
import os
import sys
import yaml

import ray
from ray import tune
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import MultiAgentEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils.typing import PolicyID

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from envs.flatland_env import FlatlandMultiAgentEnv
from models.custom_model import FlatlandModel
from utils.metrics_utils import save_episode_metrics

CONFIGS_DIR = os.path.join(PROJECT_ROOT, "configs")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")


# ── Callbacks ─────────────────────────────────────────────────────────────────

class FlatlandCallbacks(DefaultCallbacks):
    """Log per-episode metrics to CSV after each episode ends."""

    def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
        algo_name = episode.last_info_for().get("_algo_name", "unknown")
        info = episode.last_info_for(agent_id=list(episode.agent_rewards.keys())[0])
        arrivals = info.get("episode_arrivals", 0)
        deadlocks = info.get("episode_deadlocks", 0)
        steps = info.get("episode_steps", 1)
        total_reward = sum(episode.agent_rewards.values())

        ep_num = episode.episode_id
        save_episode_metrics(
            algo_name=algo_name,
            episode=ep_num,
            reward=total_reward,
            arrivals=arrivals,
            steps=steps,
            deadlocks=deadlocks,
        )


# ── Config building ───────────────────────────────────────────────────────────

def load_yaml_config(algo: str) -> dict:
    cfg_file = os.path.join(CONFIGS_DIR, f"{algo.lower()}_config.yaml")
    with open(cfg_file) as f:
        return yaml.safe_load(f)


def build_rllib_config(algo: str, algo_cfg: dict, smoke_test: bool = False) -> dict:
    env_config = {}

    shared_policy = PolicySpec(
        observation_space=None,
        action_space=None,
        config={
            "model": {
                "custom_model": "flatland_model",
                "custom_model_config": {},
            }
        },
    )

    def policy_mapping_fn(agent_id, episode, worker=None, **kwargs):
        return "shared_policy"

    base = {
        "env": "flatland_5agents",
        "env_config": env_config,
        "framework": algo_cfg.get("framework", "torch"),
        "num_workers": algo_cfg.get("num_workers", 2),
        "gamma": algo_cfg.get("gamma", 0.99),
        "lr": algo_cfg.get("lr", 3e-4),
        "rollout_fragment_length": algo_cfg.get("rollout_fragment_length", 200),
        "multiagent": {
            "policies": {"shared_policy": shared_policy},
            "policy_mapping_fn": policy_mapping_fn,
        },
        "callbacks": FlatlandCallbacks,
    }

    if smoke_test:
        base["num_workers"] = 1
        base["train_batch_size"] = 200
        base["rollout_fragment_length"] = 50

    algo_upper = algo.upper()

    if algo_upper == "PPO":
        base.update({
            "train_batch_size": algo_cfg.get("train_batch_size", 4000) if not smoke_test else 200,
            "sgd_minibatch_size": algo_cfg.get("sgd_minibatch_size", 256) if not smoke_test else 64,
            "num_sgd_iter": algo_cfg.get("num_sgd_iter", 10),
            "clip_param": algo_cfg.get("clip_param", 0.2),
            "entropy_coeff": algo_cfg.get("entropy_coeff", 0.01),
            "vf_loss_coeff": algo_cfg.get("vf_loss_coeff", 0.5),
            "lambda": algo_cfg.get("lambda", 0.95),
        })

    elif algo_upper == "APEX":
        base.update({
            "train_batch_size": algo_cfg.get("train_batch_size", 512) if not smoke_test else 200,
            "buffer_size": algo_cfg.get("buffer_size", 50000) if not smoke_test else 1000,
            "learning_starts": algo_cfg.get("learning_starts", 1000) if not smoke_test else 100,
            "target_network_update_freq": algo_cfg.get("target_network_update_freq", 500),
        })

    elif algo_upper == "MARWIL":
        input_path = algo_cfg.get("input", "./results/ppo/demos")
        if not os.path.isabs(input_path):
            input_path = os.path.join(PROJECT_ROOT, input_path)
        base.update({
            "train_batch_size": algo_cfg.get("train_batch_size", 2000) if not smoke_test else 200,
            "beta": algo_cfg.get("beta", 1.0),
            "input": input_path,
        })

    return base


def build_stop_criteria(algo_cfg: dict, smoke_test: bool) -> dict:
    if smoke_test:
        return {"timesteps_total": 5000}
    return algo_cfg.get("stop", {"timesteps_total": 2_000_000})


# ── Entrypoint ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train Flatland 5-agents RL")
    parser.add_argument("--algorithm", "-a", default="ppo", choices=["ppo", "apex", "marwil"],
                        help="RL algorithm to use")
    parser.add_argument("--config", "-c", default=None, help="Path to custom YAML config")
    parser.add_argument("--num-cpus", type=int, default=4, help="Number of CPUs for Ray")
    parser.add_argument("--smoke-test", action="store_true", help="Quick test with minimal steps")
    args = parser.parse_args()

    algo = args.algorithm.lower()
    algo_cfg = load_yaml_config(algo) if args.config is None else yaml.safe_load(open(args.config))

    ray.init(num_cpus=args.num_cpus, ignore_reinit_error=True)

    tune.register_env("flatland_5agents", lambda cfg: FlatlandMultiAgentEnv(cfg))
    ModelCatalog.register_custom_model("flatland_model", FlatlandModel)

    rllib_cfg = build_rllib_config(algo, algo_cfg, smoke_test=args.smoke_test)
    stop = build_stop_criteria(algo_cfg, args.smoke_test)

    local_dir = os.path.join(RESULTS_DIR, algo)
    os.makedirs(local_dir, exist_ok=True)

    algo_name_map = {"ppo": "PPO", "apex": "APEX", "marwil": "MARWIL"}
    rllib_algo_name = algo_name_map[algo]

    print(f"\n{'='*60}")
    print(f"  Training {rllib_algo_name} — {'SMOKE TEST' if args.smoke_test else 'FULL RUN'}")
    print(f"  Results → {local_dir}")
    print(f"{'='*60}\n")

    results = tune.run(
        rllib_algo_name,
        config=rllib_cfg,
        stop=stop,
        local_dir=local_dir,
        checkpoint_freq=50 if not args.smoke_test else 1,
        checkpoint_at_end=True,
        verbose=1,
    )

    best_trial = results.get_best_trial("episode_reward_mean", "max", "last")
    if best_trial:
        print(f"\nBest checkpoint: {best_trial.checkpoint.value}")
    ray.shutdown()


if __name__ == "__main__":
    main()
