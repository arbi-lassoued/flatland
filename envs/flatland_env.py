import os
import yaml
import numpy as np
import gymnasium as gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
try:
    from flatland.envs.line_generators import sparse_line_generator as sparse_schedule_generator
except ImportError:
    from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv

from utils.observation_utils import normalize_observation

# ── Reward constants ──────────────────────────────────────────────────────────
REWARD_AGENT_ARRIVED = 10.0
REWARD_ALL_AGENTS_ARRIVED = 50.0
REWARD_STEP_PENALTY = -0.01
REWARD_DEADLOCK = -5.0
REWARD_COLLISION = -2.0
REWARD_INVALID_ACTION = -0.5
REWARD_COOPERATIVE_BONUS = 2.0

OBS_SIZE = 231
N_ACTIONS = 5
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "map_config.yaml")


class FlatlandMultiAgentEnv(MultiAgentEnv):
    """
    Multi-agent Flatland environment compatible with RLlib.

    The rail map is generated once (seed=42) and never changes between episodes.
    Schedules (agent start/goal positions) are regenerated each reset.
    """

    def __init__(self, config: dict = None, n_agents_override: int = None):
        super().__init__()
        config = config or {}
        self._load_map_config(config)

        # Permet de surcharger le nombre d'agents depuis l'interface
        if n_agents_override is not None:
            self.n_agents = int(n_agents_override)

        self._rail_generated = False
        self._prev_positions: dict = {}
        self._deadlock_counters: dict = {}

        # Spaces (per-agent mappings required by newer RLlib env runners)
        # Use ordered list of agent ids so RLlib sees consistent keys
        self._agent_ids = [f"agent_{i}" for i in range(self.n_agents)]

        single_obs_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32
        )
        single_act_space = gym.spaces.Discrete(N_ACTIONS)

        # RLlib expects observation_space/action_space to be iterable/mapping for multi-agent
        # Provide dict mapping agent_id -> space
        self.observation_space = {agent_id: single_obs_space for agent_id in self._agent_ids}
        self.action_space = {agent_id: single_act_space for agent_id in self._agent_ids}

        # Build the Flatland RailEnv
        self._build_rail_env()

        # Episode tracking
        self._episode_arrivals = 0
        self._episode_deadlocks = 0
        self._episode_collisions = 0
        self._episode_steps = 0
        self._cumulative_rewards: dict = {f"agent_{i}": 0.0 for i in range(self.n_agents)}

    # ── Construction ─────────────────────────────────────────────────────────

    def _load_map_config(self, override: dict):
        with open(CONFIG_PATH) as f:
            file_cfg = yaml.safe_load(f).get("map", {})
        cfg = {**file_cfg, **override}

        self.width = int(cfg.get("width", 30))
        self.height = int(cfg.get("height", 30))
        self.seed = int(cfg.get("seed", 42))
        self.n_agents = int(cfg.get("number_of_agents", 5))
        self.max_num_cities = int(cfg.get("max_num_cities", 5))
        self.grid_mode = bool(cfg.get("grid_mode", False))
        self.max_rails_between_cities = int(cfg.get("max_rails_between_cities", 2))
        self.max_rails_in_city = int(cfg.get("max_rails_in_city", 3))
        # Episode length limit (used to set truncated flag)
        self.max_steps = int(cfg.get("max_steps", 512))

    def _build_rail_env(self):
        tree_obs = TreeObsForRailEnv(
            max_depth=2,
            predictor=ShortestPathPredictorForRailEnv(max_depth=20),
        )
        rail_gen = sparse_rail_generator(
            max_num_cities=self.max_num_cities,
            grid_mode=self.grid_mode,
            max_rails_between_cities=self.max_rails_between_cities,
            max_rail_pairs_in_city=self.max_rails_in_city,
            seed=self.seed,
        )
        line_gen = sparse_schedule_generator()

        self.rail_env = RailEnv(
            width=self.width,
            height=self.height,
            rail_generator=rail_gen,
            line_generator=line_gen,
            number_of_agents=self.n_agents,
            obs_builder_object=tree_obs,
        )

    # ── MultiAgentEnv interface ───────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        """Gymnasium-style reset.

        Accepts keyword-only seed and options. Returns a tuple (obs_dict, info_dict).
        """
        if not self._rail_generated:
            raw_obs, _ = self.rail_env.reset(
                regenerate_rail=True,
                regenerate_schedule=True,
                random_seed=self.seed if seed is None else seed,
            )
            self._rail_generated = True
        else:
            raw_obs, _ = self.rail_env.reset(
                regenerate_rail=False,
                regenerate_schedule=True,
                random_seed=seed,
            )

        # Episode bookkeeping
        self._episode_arrivals = 0
        self._episode_deadlocks = 0
        self._episode_collisions = 0
        self._episode_steps = 0
        self._cumulative_rewards = {f"agent_{i}": 0.0 for i in range(self.n_agents)}
        self._prev_positions = {}
        self._deadlock_counters = {f"agent_{i}": 0 for i in range(self.n_agents)}
        self._arrived = {f"agent_{i}": False for i in range(self.n_agents)}
        # Ensure the all-agents-arrived bonus is only given once per episode
        self._all_arrived_bonus_given = False

        obs = self._build_obs_dict(raw_obs)
        info = {}
        return obs, info

    def step(self, action_dict: dict):
        """Step following gymnasium-style returns.

        Returns: obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict
        """
        # increment step counter
        self._episode_steps += 1

        # Map RLlib agent IDs → Flatland integer actions
        flatland_actions = {}
        for i in range(self.n_agents):
            agent_id = f"agent_{i}"
            flatland_actions[i] = int(action_dict.get(agent_id, 0))

        raw_obs, raw_rewards, dones, infos = self.rail_env.step(flatland_actions)

        obs_dict = self._build_obs_dict(raw_obs)
        reward_dict = {}
        terminated_dict = {}
        truncated_dict = {}
        info_dict = {}

        all_positions = {}
        for i, agent in enumerate(self.rail_env.agents):
            if agent.position is not None:
                pos_key = agent.position
                all_positions.setdefault(pos_key, []).append(i)

        for i in range(self.n_agents):
            agent_id = f"agent_{i}"
            agent = self.rail_env.agents[i]
            breakdown = {
                "arrived": 0.0,
                "deadlock": 0.0,
                "collision": 0.0,
                "step_penalty": REWARD_STEP_PENALTY,
                "cooperative": 0.0,
                "invalid_action": 0.0,
            }

            reward = REWARD_STEP_PENALTY

            # Arrival
            if dones[i] and not self._arrived[agent_id]:
                self._arrived[agent_id] = True
                reward += REWARD_AGENT_ARRIVED
                breakdown["arrived"] = REWARD_AGENT_ARRIVED
                self._episode_arrivals += 1

            # Deadlock detection: agent not done but stuck for several steps
            curr_pos = agent.position
            prev_pos = self._prev_positions.get(agent_id)
            if not dones[i] and curr_pos is not None and curr_pos == prev_pos:
                self._deadlock_counters[agent_id] = self._deadlock_counters.get(agent_id, 0) + 1
                if self._deadlock_counters[agent_id] >= 5:
                    reward += REWARD_DEADLOCK
                    breakdown["deadlock"] = REWARD_DEADLOCK
                    self._episode_deadlocks += 1
                    self._deadlock_counters[agent_id] = 0
            else:
                self._deadlock_counters[agent_id] = 0

            # Collision detection
            if curr_pos is not None and len(all_positions.get(curr_pos, [])) > 1:
                reward += REWARD_COLLISION
                breakdown["collision"] = REWARD_COLLISION
                self._episode_collisions += 1

            self._prev_positions[agent_id] = curr_pos
            self._cumulative_rewards[agent_id] += reward
            reward_dict[agent_id] = reward

            # terminated: natural done (arrived or other end)
            terminated = bool(dones[i])
            terminated_dict[agent_id] = terminated

            # truncated: time-limit exceeded
            truncated = bool(self._episode_steps >= self.max_steps)
            truncated_dict[agent_id] = truncated

            info_dict[agent_id] = {
                "reward_breakdown": breakdown,
                "cumulative_reward": self._cumulative_rewards[agent_id],
                "episode_arrivals": self._episode_arrivals,
                "episode_deadlocks": self._episode_deadlocks,
                "episode_collisions": self._episode_collisions,
                "episode_steps": self._episode_steps,
            }

        # All-agents-arrived bonus: give only once per episode
        if all(self._arrived.values()) and not getattr(self, "_all_arrived_bonus_given", False):
            for agent_id in self._arrived:
                reward_dict[agent_id] += REWARD_ALL_AGENTS_ARRIVED
                info_dict[agent_id]["reward_breakdown"]["arrived"] += REWARD_ALL_AGENTS_ARRIVED
            self._all_arrived_bonus_given = True

        # __all__ flags
        terminated_all = all(terminated_dict.get(f"agent_{i}", False) for i in range(self.n_agents))
        truncated_all = all(truncated_dict.get(f"agent_{i}", False) for i in range(self.n_agents))

        terminated_dict["__all__"] = terminated_all
        truncated_dict["__all__"] = truncated_all

        return obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_obs_dict(self, raw_obs: dict) -> dict:
        obs_dict = {}
        for i in range(self.n_agents):
            agent_id = f"agent_{i}"
            agent_obs = raw_obs.get(i, None)
            obs_dict[agent_id] = normalize_observation(agent_obs, tree_depth=2, num_features_per_node=11)
        return obs_dict

    def get_render_frame(self) -> np.ndarray:
        """Return an RGB numpy array of the current map state."""
        from utils.render_utils import get_map_frame
        return get_map_frame(self.rail_env)

    def get_episode_stats(self) -> dict:
        return {
            "arrivals": self._episode_arrivals,
            "deadlocks": self._episode_deadlocks,
            "collisions": self._episode_collisions,
            "steps": self._episode_steps,
        }
