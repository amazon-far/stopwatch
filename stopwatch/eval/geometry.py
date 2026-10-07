"""Accepted-instance destination sets and scoring distance geometry."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from habitat_sim import Simulator


@lru_cache(maxsize=128)
def load_scene_instances(root: Path, split: str, scene_id: str) -> list[dict[str, Any]]:
    """Read the publisher-format instance records; validation splits map to val."""
    if split != "train" and split != "val" and not split.startswith("val_"):
        raise ValueError(f"Unknown instance-viewpoint split {split!r}")
    scene = _scene_name(scene_id)
    path = root / ("train" if split == "train" else "val") / f"{scene}.json"
    return json.loads(path.read_text())["instances"]


def _scene_name(scene: str) -> str:
    return Path(scene).name.removesuffix(".scene_instance.json").removesuffix(".glb").removesuffix(".basis")


@lru_cache(maxsize=8)
def _load_goal_sets(root: Path) -> dict[str, dict[str, dict[str, frozenset[int]]]]:
    """Load frozen scoring memberships for the released benchmarks."""
    path = root.expanduser() / "vocab/objectnav_goal_sets.json"
    payload = json.loads(path.read_text())
    return {"scoring_class_ids": {
        benchmark: {category: frozenset(ids) for category, ids in categories.items()}
        for benchmark, categories in payload["scoring_class_ids"].items()
    }}


def destination_targets(instances: Sequence[dict[str, Any]], goal_class_ids: Sequence[int]) -> NDArray[np.float32]:
    """Return all accepted-instance viewpoint positions [V,3] for scoring."""
    # The caller passes the complete accepted set. Frozen scoring membership
    # must not be expanded a second time through the prompt vocabulary.
    accepted = frozenset(goal_class_ids)
    return np.asarray([
        vp["agent_state"]["position"] for instance in instances
        if instance["class_id"] in accepted for vp in instance["view_points"]
    ], dtype=np.float32).reshape(-1, 3)


def geodesic_distance(sim: Simulator, position: Sequence[float], targets: NDArray[np.float32]) -> float:
    """Use a fresh MultiGoalShortestPath; return infinity for no finite path."""
    import habitat_sim

    if len(targets) == 0:
        return math.inf
    path = habitat_sim.MultiGoalShortestPath()
    path.requested_start = np.asarray(position, dtype=np.float32)
    path.requested_ends = np.asarray(targets, dtype=np.float32)
    if not sim.pathfinder.find_path(path):
        return math.inf
    value = float(path.geodesic_distance)
    return value if math.isfinite(value) else math.inf
