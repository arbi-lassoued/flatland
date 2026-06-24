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
    Multi-agent Flatland environment compatible with RLlib (gymnasium API).

    The rail map is generated with a fixed seed (42) and stays identical
    between episodes; schedules (agent start/goal) are regenerated each reset.
    """

    def __init__(self, config: dict = None, n_agents_override: int = None):
        super().__init__()
        config = config or {}
        self._load_map_config(config)

        if n_agents_override is not None:
            self.n_agents = int(n_agents_override)

        self._rail_generated = False
        self._prev_positions: dict = {}
        self._deadlock_counters: dict = {}

        # Spaces (gymnasium)
        _obs = gym.spaces.Box(low=-1.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32)
        _act = gym.spaces.Discrete(N_ACTIONS)
        self.observation_space = gym.spaces.Dict({f"agent_{i}": _obs for i in range(self.n_agents)})
        self.action_space = gym.spaces.Dict({f"agent_{i}": _act for i in range(self.n_agents)})

        # Agent identifiers (RLlib new API stack requires agents / possible_agents)
        self._agent_ids = {f"agent_{i}" for i in range(self.n_agents)}
        self.possible_agents = [f"agent_{i}" for i in range(self.n_agents)]
        self.agents = list(self.possible_agents)

        self._build_rail_env()

        # Episode tracking
        self._episode_arrivals = 0
        self._episode_deadlocks = 0
        self._episode_collisions = 0
        self._episode_steps = 0
        self._cumulative_rewards = {f"agent_{i}": 0.0 for i in range(self.n_agents)}

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
        self.max_steps = int(cfg.get("max_step", 512))

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

    # ── MultiAgentEnv interface (gymnasium) ───────────────────────────────────

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.seed = int(seed)

        # Régénération complète à chaque épisode (carte figée car seed fixe)
        raw_obs, _ = self.rail_env.reset(
            regenerate_rail=True,
            regenerate_schedule=True,
            random_seed=self.seed,
        )
        self._rail_generated = True

        self._episode_arrivals = 0
        self._episode_deadlocks = 0
        self._episode_collisions = 0
        self._episode_steps = 0
        self._cumulative_rewards = {f"agent_{i}": 0.0 for i in range(self.n_agents)}
        self._prev_positions = {}
        self._deadlock_counters = {f"agent_{i}": 0 for i in range(self.n_agents)}
        self._arrived = {f"agent_{i}": False for i in range(self.n_agents)}
        self._done_emitted = set()

        self.agents = list(self.possible_agents)

        obs_dict = self._build_obs_dict(raw_obs)
        info_dict = {agent_id: {} for agent_id in obs_dict}
        return obs_dict, info_dict

    def step(self, action_dict: dict):
        self._episode_steps += 1
        if not hasattr(self, "_done_emitted"):
            self._done_emitted = set()

        # Actions Flatland (pour tous les agents internes, meme deja arrives)
        flatland_actions = {}
        for i in range(self.n_agents):
            aid = f"agent_{i}"
            flatland_actions[i] = int(action_dict.get(aid, 0))

        raw_obs, raw_rewards, dones, infos = self.rail_env.step(flatland_actions)

        # Positions pour la detection de collision
        all_positions = {}
        for i, agent in enumerate(self.rail_env.agents):
            if agent.position is not None:
                all_positions.setdefault(agent.position, []).append(i)

        active_obs, active_rew, term, truncd, active_info = {}, {}, {}, {}, {}
        trunc = self._episode_steps >= self.max_steps

        # On itere SEULEMENT sur les agents pas encore emis comme done
        for i in range(self.n_agents):
            aid = f"agent_{i}"
            if aid in self._done_emitted:
                continue

            agent = self.rail_env.agents[i]
            reward = REWARD_STEP_PENALTY
            is_done_flat = bool(dones[i])

            # Arrivee (premiere fois seulement)
            if is_done_flat and not self._arrived[aid]:
                self._arrived[aid] = True
                reward += REWARD_AGENT_ARRIVED
                self._episode_arrivals += 1

            # Deadlock
            curr_pos = agent.position
            prev_pos = self._prev_positions.get(aid)
            if not is_done_flat and curr_pos is not None and curr_pos == prev_pos:
                self._deadlock_counters[aid] = self._deadlock_counters.get(aid, 0) + 1
                if self._deadlock_counters[aid] >= 5:
                    reward += REWARD_DEADLOCK
                    self._episode_deadlocks += 1
                    self._deadlock_counters[aid] = 0
            else:
                self._deadlock_counters[aid] = 0

            # Collision
            if curr_pos is not None and len(all_positions.get(curr_pos, [])) > 1:
                reward += REWARD_COLLISION
                self._episode_collisions += 1

            self._prev_positions[aid] = curr_pos
            self._cumulative_rewards[aid] += reward

            # Construction de l'obs uniquement pour les agents encore actifs
            normed = normalize_observation(raw_obs.get(i, None), tree_depth=2, num_features_per_node=11)
            active_obs[aid] = np.asarray(normed, dtype=np.float32)
            active_rew[aid] = reward
            term[aid] = is_done_flat
            truncd[aid] = bool(trunc)
            active_info[aid] = {}

            if is_done_flat or trunc:
                self._done_emitted.add(aid)

        # Bonus collectif (une seule fois)
        if all(self._arrived.values()) and not getattr(self, "_all_bonus_given", False):
            self._all_bonus_given = True
            for aid in list(active_rew.keys()):
                active_rew[aid] += REWARD_ALL_AGENTS_ARRIVED

        # Fin globale : tous emis OU dones["__all__"] OU truncated global
        all_emitted = (len(self._done_emitted) == self.n_agents)
        term["__all__"] = bool(all_emitted or dones.get("__all__", False))
        truncd["__all__"] = bool(trunc)

        return active_obs, active_rew, term, truncd, active_info

    def _build_obs_dict(self, raw_obs: dict) -> dict:
        obs_dict = {}
        for i in range(self.n_agents):
            agent_id = f"agent_{i}"
            agent_obs = raw_obs.get(i, None)
            normed = normalize_observation(agent_obs, tree_depth=2, num_features_per_node=11)
            obs_dict[agent_id] = np.asarray(normed, dtype=np.float32)
        return obs_dict

    def get_render_frame(self) -> np.ndarray:
        from utils.render_utils import get_map_frame
        return get_map_frame(self.rail_env)

    def get_episode_stats(self) -> dict:
        return {
            "arrivals": self._episode_arrivals,
            "deadlocks": self._episode_deadlocks,
            "collisions": self._episode_collisions,
            "steps": self._episode_steps,
        }