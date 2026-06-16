# Flatland 5 Agents — RL Dashboard

Multi-agent reinforcement learning for railway traffic management.  
Trains 5 agents on a **fixed map** (seed=42, never changes) using **PPO**, **APEX**, and **MARWIL** via RLlib, then compares them in an interactive Streamlit interface.

**Status**: ✅ Fully compatible with Python 3.13 | **Dashboard**: Live on http://localhost:8508

## Requirements

- **Python 3.13+**
- **PyTorch 2.12.0**
- **Ray[rllib] 2.55.1**
- **Streamlit 1.58.0+**
- **Flatland 3.0.15**

## Installation

```bash
cd flatland_5agents
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Launch the Streamlit dashboard

```bash
source .venv/bin/activate
streamlit run streamlit_app/app.py
```

The dashboard will be available at **http://localhost:8508** with full support for:
- **Training**: PPO, APEX, MARWIL algorithms with real-time monitoring
- **Evaluation**: Load checkpoints and test agent performance
- **Comparison**: Side-by-side algorithm performance metrics
- **Live Visualization**: Interactive map rendering with agent trajectories

## Command-line training

```bash
# Full training
python train.py --algorithm ppo  --num-cpus 4
python train.py --algorithm apex --num-cpus 4
python train.py --algorithm marwil --num-cpus 4

# Quick smoke test (5 000 steps)
python train.py --algorithm ppo --smoke-test
```

## Evaluation

```bash
python evaluate.py --algorithm ppo --checkpoint auto --num-episodes 10
```

## TensorBoard

```bash
tensorboard --logdir ./results
```

## Reward system

| Event | Reward |
|---|---|
| Agent arrives at destination | +10.0 |
| All 5 agents arrive | +50.0 bonus |
| Step penalty (speed incentive) | −0.01 |
| Deadlock detected (stuck ≥5 steps) | −5.0 |
| Collision (two agents on same cell) | −2.0 |
| Invalid action attempted | −0.5 |
| Cooperative bonus | +2.0 |

## Algorithm scoring

```
score = (normalized_reward × 0.4) + (arrival_rate × 0.4) + (speed × 0.2)
```

## Project structure

```
flatland_5agents/
├── envs/flatland_env.py        ← FlatlandMultiAgentEnv (RLlib MultiAgentEnv)
├── models/custom_model.py      ← FlatlandModel (231→256→256→128, policy+value heads)
├── utils/
│   ├── observation_utils.py    ← TreeObs → float32[231] normalizer
│   ├── metrics_utils.py        ← CSV logging + algorithm scoring
│   └── render_utils.py         ← matplotlib renderer + GIF exporter
├── configs/                    ← YAML configs for map + each algorithm
├── streamlit_app/
│   ├── app.py                  ← Main dashboard
│   └── pages/
│       ├── 01_train.py         ← Training page
│       ├── 02_evaluate.py      ← Evaluation page
│       ├── 03_comparison.py    ← Algorithm comparison
│       └── 04_live_map.py      ← Interactive live map
├── train.py                    ← CLI training entry point
└── evaluate.py                 ← CLI evaluation entry point
```

## Python 3.13 Compatibility

This project has been fully updated to support **Python 3.13** with the following modernizations:

### Version Updates
- NumPy 2.4.6 (with compatibility patches for array indexing)
- Ray[rllib] 2.55.1 (replaces unavailable 2.9.0)
- PyTorch 2.12.0 (replaces 2.0.0)
- Gym 0.14.0 (compatible with flatland-rl)
- importlib-resources <2,>=1.0.1 (for resource loading in Python 3.13)

### Compatibility Patches Applied
1. **importlib_resources**: Fixed `typing.io` removal in Python 3.13
2. **Flatland rail_generators**: NumPy 2.x array indexing compatibility
3. **Flatland transition_map**: NumPy 2.x scalar conversion fixes
4. **Gym**: distutils module fallback for Python 3.12+
5. **Streamlit**: Removed deprecated `use_column_width` parameter

All patches are automatically applied during installation.

## Troubleshooting

**Dashboard won't start?**
- Ensure you're using Python 3.13: `python3 --version`
- Activate the virtual environment: `source .venv/bin/activate`
- Check port availability: `lsof -i :8508`

**CUDA errors?**
- The project supports both CPU and GPU. For GPU:
  ```bash
  pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cu118
  ```
