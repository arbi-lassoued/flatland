import numpy as np


def normalize_observation(obs, tree_depth: int = 2, num_features_per_node: int = 11) -> np.ndarray:
    """Flatten and normalize a TreeObs observation into a fixed-size float32 vector of shape (231,)."""
    flat = _flatten_tree(obs, tree_depth, num_features_per_node)
    # Pad or truncate to exactly 231 features
    target_size = _compute_tree_obs_size(tree_depth, num_features_per_node)
    if len(flat) < target_size:
        flat.extend([-1.0] * (target_size - len(flat)))
    else:
        flat = flat[:target_size]
    arr = np.array(flat, dtype=np.float32)
    arr = np.where(np.isinf(arr) & (arr > 0), 1.0, arr)
    arr = np.where(np.isinf(arr) & (arr < 0), -1.0, arr)
    arr = np.where(np.isnan(arr), -1.0, arr)
    arr = np.clip(arr, -1.0, 1.0)
    return arr


def _flatten_tree(node, remaining_depth: int, num_features: int) -> list:
    """Recursively flatten a TreeObs node into a flat list of floats."""
    if node is None or isinstance(node, float):
        # Represent missing/empty node as -1 for all features + all children
        size = _compute_tree_obs_size(remaining_depth, num_features)
        return [-1.0] * size

    # Extract node features
    features = [
        _safe_val(node.dist_own_target_encountered),
        _safe_val(node.dist_other_target_encountered),
        _safe_val(node.dist_other_agent_encountered),
        _safe_val(node.dist_potential_conflict),
        _safe_val(node.dist_unusable_switch),
        _safe_val(node.dist_to_next_branch),
        _safe_val(node.dist_min_to_target),
        _safe_val(node.num_agents_same_direction),
        _safe_val(node.num_agents_opposite_direction),
        _safe_val(node.num_agents_malfunctioning),
        _safe_val(node.speed_min_fractional),
    ]

    if remaining_depth == 0:
        return features

    # Recurse into children
    children_flat = []
    childs = node.childs if hasattr(node, "childs") else {}
    for direction in ["L", "F", "R", "B"]:
        child = childs.get(direction, None)
        children_flat.extend(_flatten_tree(child, remaining_depth - 1, num_features))

    return features + children_flat


def _safe_val(val) -> float:
    """Convert a tree observation feature value to a safe float."""
    if val is None:
        return -1.0
    if isinstance(val, float) and (np.isinf(val) or np.isnan(val)):
        return 1.0 if val > 0 else -1.0
    try:
        f = float(val)
        return np.clip(f, -1.0, 1.0)
    except (TypeError, ValueError):
        return -1.0


def _compute_tree_obs_size(depth: int, num_features: int) -> int:
    """Compute total number of features in a tree observation of given depth."""
    total = 0
    for d in range(depth + 1):
        total += (4 ** d) * num_features
    return total
