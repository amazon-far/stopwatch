"""Score our criterion over the released benchmark populations."""

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import TypedDict

import numpy as np

from stopwatch.config import EvalConfig
from stopwatch.eval.dispatch import _read_records
from stopwatch.eval.env import _habitat_config, _install_navmesh, _load_dataset
from stopwatch.eval.geometry import (
    _load_goal_sets,
    _scene_name,
    destination_targets,
    geodesic_distance,
    load_scene_instances,
)


class CriterionMetrics(TypedDict):
    successes: int
    episodes: int
    success_rate: float


def score_records(config: EvalConfig, records_root: Path, *, limit: int | None = None) -> dict[str, CriterionMetrics]:
    """Deduplicate records and score the selected episodes on the protocol navmesh."""
    import habitat

    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    records = _read_records(records_root)
    dataset = _load_dataset(config)
    all_ids = {(_scene_name(ep.scene_id), str(ep.episode_id)) for ep in dataset.episodes}
    if not records.keys() <= all_ids:
        raise ValueError("Output contains completed episodes outside this benchmark")
    dataset.episodes = dataset.episodes[:limit]
    episodes = {(_scene_name(ep.scene_id), str(ep.episode_id)): ep for ep in dataset.episodes}
    records = {identity: row for identity, row in records.items() if identity in episodes}
    if records.keys() != episodes.keys():
        raise ValueError(f"Incomplete frozen population, missing={len(episodes.keys() - records.keys())}, "
                         f"extra={len(records.keys() - episodes.keys())}")
    if any(row["split"] != config.dataset.split for row in records.values()):
        raise ValueError("Record split differs from the recipe")
    benchmark = config.dataset.benchmark
    definitions = _load_goal_sets(config.paths.root)["scoring_class_ids"][benchmark]
    grouped = defaultdict(list)
    for identity, episode in episodes.items():
        if episode.object_category not in definitions:
            raise ValueError(f"Missing frozen scoring membership for {episode.object_category}")
        grouped[episode.scene_id].append(identity)
    # Rebuild the protocol navmesh independently of rollout measurements.
    # Keep the RGB sensor so the pinned simulator owns a renderer. Habitat-Lab
    # closes it with destroy=False on scene changes, which dereferences the
    # renderer even when a sensorless configuration did not create one.
    cfg = _habitat_config(config, 0)
    env = habitat.Env(config=cfg, dataset=dataset)
    try:
        _install_navmesh(env.sim, config)
        for scene, identities in grouped.items():
            env.current_episode = episodes[identities[0]]
            env.reset()
            instances = load_scene_instances(config.paths.resolve(config.dataset.instance_viewpoints),
                                             config.dataset.split, scene)
            targets = {}
            for identity in identities:
                row, episode = records[identity], episodes[identity]
                category = episode.object_category
                if category not in targets:
                    targets[category] = destination_targets(instances, definitions[category])
                ours = math.inf
                if row["ended_by_stop"]:
                    position = row["final_position"]
                    if np.asarray(position).shape != (3,) or not np.isfinite(position).all():
                        raise ValueError(f"Invalid terminal position for {identity}")
                    ours = geodesic_distance(env.sim, position, targets[category])
                row.update(ours_distance_m=ours if math.isfinite(ours) else None,
                           ours_success=ours < 0.25,
                           viewpoint_count=len(targets[category]))
    finally:
        env.close()

    def metrics(population, field):
        successes = sum(bool(records[identity][field]) for identity in population)
        return {"successes": successes, "episodes": len(population), "success_rate": successes / len(population)}

    result = {"ours": metrics(records, "ours_success")}

    def finite(value):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {key: finite(item) for key, item in value.items()}
        if isinstance(value, list):
            return [finite(item) for item in value]
        return value

    output = records_root / "records.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(finite([records[key] for key in sorted(records)]), ensure_ascii=False,
                                     allow_nan=False) + "\n")
    temporary.replace(output)
    return result
