# Flatland 5 Agents — RL Dashboard

Multi-agent reinforcement learning for railway traffic management.  
Trains 5 agents on a **fixed map** (seed=42, never changes) using **PPO**, **APEX**, and **MARWIL** via RLlib, then compares them in an interactive Streamlit interface.

## Installation

```bash
cd flatland_5agents
pip install -r requirements.txt
pip install -e .
```

## Launch the Streamlit dashboard

```bash
streamlit run streamlit_app/app.py
```

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
