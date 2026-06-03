import io
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

AGENT_COLORS = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6"]
DIRECTION_ARROWS = {0: "↑", 1: "→", 2: "↓", 3: "←"}


def get_agent_positions(rail_env) -> dict:
    """Return dict mapping 'agent_i' → (row, col, direction) for each active agent."""
    positions = {}
    for i, agent in enumerate(rail_env.agents):
        if agent.position is not None:
            positions[f"agent_{i}"] = (agent.position[0], agent.position[1], agent.direction)
        else:
            positions[f"agent_{i}"] = None
    return positions


def get_map_frame(rail_env) -> np.ndarray:
    """Render the Flatland environment to an RGB numpy array using matplotlib."""
    height = rail_env.height
    width = rail_env.width

    fig, ax = plt.subplots(figsize=(8, 8), dpi=80)
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(-0.5, height - 0.5)
    ax.set_aspect("equal")
    ax.set_facecolor("#F8F9FA")
    ax.invert_yaxis()
    ax.set_title("Flatland Railway Map", fontsize=14, fontweight="bold", pad=10)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    # Draw grid
    for r in range(height):
        for c in range(width):
            cell = rail_env.rail.get_full_transitions(r, c)
            if cell != 0:
                ax.add_patch(mpatches.Rectangle(
                    (c - 0.4, r - 0.4), 0.8, 0.8,
                    linewidth=0, facecolor="#7F8C8D", alpha=0.6, zorder=1
                ))

    # Draw agent targets
    for i, agent in enumerate(rail_env.agents):
        color = AGENT_COLORS[i % len(AGENT_COLORS)]
        if agent.target is not None:
            tr, tc = agent.target
            ax.plot(tc, tr, marker="*", markersize=14, color=color,
                    markeredgecolor="black", markeredgewidth=0.5, zorder=3)
            ax.text(tc + 0.3, tr - 0.3, f"T{i}", fontsize=6, color=color, fontweight="bold", zorder=4)

    # Draw agents
    for i, agent in enumerate(rail_env.agents):
        color = AGENT_COLORS[i % len(AGENT_COLORS)]
        if agent.position is not None:
            ar, ac = agent.position
            circle = plt.Circle((ac, ar), 0.35, color=color, zorder=5, ec="black", linewidth=0.8)
            ax.add_patch(circle)
            arrow = DIRECTION_ARROWS.get(agent.direction, "?")
            ax.text(ac, ar, f"{i}{arrow}", ha="center", va="center",
                    fontsize=7, color="white", fontweight="bold", zorder=6)

    # Legend
    legend_patches = [
        mpatches.Patch(color=AGENT_COLORS[i], label=f"Agent {i}") for i in range(len(rail_env.agents))
    ]
    legend_patches.append(mpatches.Patch(color="#7F8C8D", label="Rail"))
    ax.legend(handles=legend_patches, loc="upper right", fontsize=7, framealpha=0.8)

    ax.grid(True, alpha=0.15, linewidth=0.3)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    arr = np.array(img)
    plt.close(fig)
    return arr


def render_episode_gif(env, agent, path: str, max_steps: int = 200) -> None:
    """Run one episode and save a GIF animation."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    frames = []

    obs, _ = env.reset() if hasattr(env, "_is_new_api") else (env.reset(), {})
    if isinstance(obs, tuple):
        obs = obs[0]

    for _ in range(max_steps):
        frame_arr = get_map_frame(env.rail_env if hasattr(env, "rail_env") else env)
        frames.append(Image.fromarray(frame_arr))

        if agent is not None:
            action_dict = {}
            for agent_id, o in obs.items():
                action = agent.compute_single_action(o, policy_id="shared_policy")
                action_dict[agent_id] = action
        else:
            action_dict = {agent_id: 2 for agent_id in obs}

        obs, _, dones, _ = env.step(action_dict)
        if dones.get("__all__", False):
            break

    if frames:
        frames[0].save(
            path,
            save_all=True,
            append_images=frames[1:],
            loop=0,
            duration=100,
        )
