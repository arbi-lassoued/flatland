"""
map_renderer.py — Rendu officiel Flatland via RenderTool (PILSVG backend).
Produit exactement la même carte visuelle que flatland.aicrowd.com.
"""
import io
import numpy as np
from PIL import Image
from flatland.utils.rendertools import RenderTool, AgentRenderVariant

# Cache du renderer pour éviter de le recréer à chaque frame
_renderer_cache: dict = {}

# ── Palette de couleurs (gardée pour compatibilité) ──────────────────────────
AGENT_COLORS = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6"]
DIRECTION_ARROWS = {0: "↑", 1: "→", 2: "↓", 3: "←"}


def render_flatland_streamlit(
    rail_env,
    agent_positions: dict = None,
    width: int = 800,
    height: int = 800,
) -> bytes:
    """
    Render the Flatland rail environment using the official RenderTool (PILSVG).
    Returns raw PNG bytes suitable for st.image().
    Produit exactement la même carte que flatland.aicrowd.com.
    """
    global _renderer_cache

    env_id = id(rail_env)

    # Crée ou réutilise le renderer pour cet environnement
    if env_id not in _renderer_cache:
        _renderer_cache.clear()
        renderer = RenderTool(
            rail_env,
            gl="PILSVG",
            agent_render_variant=AgentRenderVariant.AGENT_SHOWS_OPTIONS_AND_BOX,
            show_debug=False,
            clear_debug_text=True,
            screen_width=width,
            screen_height=height,
        )
        _renderer_cache[env_id] = renderer
    else:
        renderer = _renderer_cache[env_id]
        renderer.reset()

    # Render
    renderer.render_env(
        show=False,
        show_agents=True,
        show_inactive_agents=True,
        show_predictions=False,
        selected_agent=None,
    )

    # Récupère l'image numpy RGBA (H, W, 4)
    img_array = renderer.get_image()

    # Convertit en PNG bytes
    pil_img = Image.fromarray(img_array)
    pil_img = pil_img.resize((width, height), Image.LANCZOS)

    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


# ─────────────────────────────────────────────────────────────────────────────
# Tout ce qui suit est MORT — remplacé par RenderTool officiel ci-dessus
# ─────────────────────────────────────────────────────────────────────────────
BG_COLOR        = "#0a0a0a"
GRASS_COLOR     = "#1a2a1a"
RAIL_BED_COLOR  = "#2c2c2c"
RAIL_COLOR      = "#d0d0d0"
RAIL_EDGE_COLOR = "#888888"
DIRECTION_DX_DY = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}
