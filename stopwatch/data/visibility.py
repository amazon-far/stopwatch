"""STOP visibility from locally rendered sparse counts, without torch imports."""


def stop_goal_visible(sparse, goal_class_ids, visibility_thresholds):
    """Apply the recipe's accepted-class area thresholds to the final frame."""
    lo, hi = sparse["offsets"][-2:]
    pixels = dict(zip(sparse["ids"][lo:hi], sparse["pixels"][lo:hi], strict=True))
    return any(pixels.get(i + 1, 0) > 0 and
               float(pixels[i + 1]) / (640 * 360) >= visibility_thresholds[i]
               for i in goal_class_ids)
