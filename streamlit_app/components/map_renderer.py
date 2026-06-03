import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

AGENT_COLORS = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6"]
DIRECTION_ARROWS = {0: "↑", 1: "→", 2: "↓", 3: "←"}


def render_flatland_streamlit(rail_env, agent_positions: dict = None, width: int = 600, height: int = 600) -> bytes:
    """
    Render the Flatland rail environment as a PNG image (bytes).
    Returns raw bytes suitable for st.image().
    """
    dpi = 80
    fig_w = width / dpi
    fig_h = height / dpi

    env_h = rail_env.height
    env_w = rail_env.width

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.set_xlim(-0.5, env_w - 0.5)
    ax.set_ylim(-0.5, env_h - 0.5)
    ax.set_aspect("equal")
    ax.set_facecolor("#ECF0F1")
    ax.invert_yaxis()

    ax.set_title("Flatland Railway Map (Carte Fixe — Seed 42)", fontsize=10, fontweight="bold", pad=6)
    ax.tick_params(labelsize=6)

    # Draw rail cells
    for r in range(env_h):
        for c in range(env_w):
            cell = rail_env.rail.get_full_transitions(r, c)
            if cell != 0:
                ax.add_patch(mpatches.FancyBboxPatch(
                    (c - 0.4, r - 0.4), 0.8, 0.8,
                    boxstyle="round,pad=0.05",
                    linewidth=0.3,
                    edgecolor="#5D6D7E",
                    facecolor="#BDC3C7",
                    zorder=1,
                ))

    # Draw agent targets (stars)
    for i, agent in enumerate(rail_env.agents):
        color = AGENT_COLORS[i % len(AGENT_COLORS)]
        if agent.target is not None:
            tr, tc = agent.target
            ax.plot(tc, tr, marker="*", markersize=12, color=color,
                    markeredgecolor="black", markeredgewidth=0.5, zorder=3, alpha=0.85)
            ax.text(tc + 0.35, tr - 0.35, f"T{i}", fontsize=5.5,
                    color=color, fontweight="bold", zorder=4)

    # Draw agents
    for i, agent in enumerate(rail_env.agents):
        color = AGENT_COLORS[i % len(AGENT_COLORS)]
        pos = None
        direction = 0

        if agent_positions and f"agent_{i}" in agent_positions:
            info = agent_positions[f"agent_{i}"]
            if info is not None:
                pos = (info[0], info[1])
                direction = info[2]
        elif agent.position is not None:
            pos = agent.position
            direction = agent.direction

        if pos is not None:
            ar, ac = pos
            circle = plt.Circle((ac, ar), 0.32, color=color, zorder=5,
                                 ec="black", linewidth=0.8, alpha=0.9)
            ax.add_patch(circle)
            arrow = DIRECTION_ARROWS.get(direction, "?")
            ax.text(ac, ar, f"{i}{arrow}", ha="center", va="center",
                    fontsize=6.5, color="white", fontweight="bold", zorder=6)

    # Legend
    legend_patches = [
        mpatches.Patch(color=AGENT_COLORS[i], label=f"Agent {i}") for i in range(len(rail_env.agents))
    ]
    legend_patches.append(mpatches.Patch(color="#BDC3C7", label="Rail"))
    ax.legend(handles=legend_patches, loc="upper right", fontsize=6,
              framealpha=0.85, ncol=2)

    ax.grid(True, alpha=0.1, linewidth=0.3)
    plt.tight_layout(pad=0.5)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)
    buf.seek(0)
    img_bytes = buf.read()
    plt.close(fig)
    return img_bytes
