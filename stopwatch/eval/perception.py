"""Turn perception JSON into policy observations."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor, device

from stopwatch.config import EvalConfig
from stopwatch.data.vocab import Vocabulary
from stopwatch.eval.env import Observation, batch_observations
from stopwatch.model.rollout import Rollout

_DISTANCES = {"<0.5m": 0.25, "0.5-1m": 0.75, "1-2m": 1.5, "2-4m": 3.0, ">4m": 5.0}


def _parse(text: str) -> tuple[list[str], str | None]:
    completion = str(text).strip()
    prefix, separator, _ = completion.partition(',"action":"')
    if separator:
        completion = prefix + "}"
    try:
        payload = json.loads(completion)
    except (TypeError, ValueError):
        return [], None
    objects = payload.get("objects") if isinstance(payload, dict) else None
    if not isinstance(objects, list) or any(not isinstance(name, str) for name in objects):
        return [], None
    target = payload.get("target")
    if target != "not seen" and target not in _DISTANCES:
        return [], None
    return [name.strip() for name in objects if name.strip()], target if target in _DISTANCES else None


class PerceptionProvider:
    def __init__(self, rollout: Rollout, vocabulary: Vocabulary, config: EvalConfig, *, target: device) -> None:
        if config.perception.distance_bin_edges != [0.5, 1.0, 2.0, 4.0]:
            raise ValueError("Perception requires the five distance bins")
        self.rollout, self.vocabulary, self.config, self.target = rollout, vocabulary, config, target
        self._category_to_id = {name: i + 1 for i, name in enumerate(vocabulary.categories)}

    @torch.inference_mode()
    def provide_observations(self, observations: Sequence[Observation], previous_actions: Tensor, masks: Tensor, *, active_indices: Sequence[int]) -> list[Observation]:
        """Return policy observations.

        Keep all B slots, including inactive slots. Insert semantic_bow_ids
        int64 [16] and goal_distance float32 [1] into each observation.
        """
        if len(observations) != self.config.num_environments:
            raise ValueError("Perception inputs must have one row per slot")
        batch = batch_observations(observations, self.target)
        self.rollout.step(batch, previous_actions, masks, active_indices=active_indices)
        texts = self.rollout.predictions
        if len(texts) != len(observations):
            raise RuntimeError("Perception returned an incorrect prediction count")
        updated = [dict(observation) for observation in observations]
        for index in active_indices:
            raw, distance = _parse(texts[index])
            names = [name for name in raw if name in self._category_to_id][:16]
            ids, metres = perception_policy_values(names, distance, self._category_to_id)
            updated[index]["semantic_bow_ids"] = np.asarray(ids, dtype=np.int64)
            updated[index]["goal_distance"] = np.asarray([metres], dtype=np.float32)
        for index in range(len(updated)):
            if "semantic_bow_ids" not in updated[index]:
                updated[index]["semantic_bow_ids"] = np.zeros(16, dtype=np.int64)
                updated[index]["goal_distance"] = np.asarray([math.nan], dtype=np.float32)
        return updated


def perception_policy_values(objects: Sequence[str], distance_bin: str | None, category_to_id: dict[str, int]) -> tuple[list[int], float]:
    """Return 16 one-based IDs with zero padding and distance or NaN."""
    ids = [category_to_id[name] for name in objects if name in category_to_id][:16]
    return ids + [0] * (16 - len(ids)), _DISTANCES.get(distance_bin, math.nan)


class StuckRecovery:
    def __init__(self) -> None:
        self.blocked_forwards = 0
        self.turns_remaining = 0

    def action(self, proposed: int) -> int:
        """Preserve STOP; otherwise consume a pending forced left turn."""
        return 2 if self.turns_remaining and proposed != 0 else proposed

    def observe(self, action: int, before: Sequence[float], after: Sequence[float], *, forced: bool) -> None:
        """After three FORWARD moves under 0.01 m, request six left turns."""
        if forced:
            self.turns_remaining -= 1
            self.blocked_forwards = 0
            return
        if action == 0:
            self.turns_remaining = 0
        displacement = math.hypot(after[0] - before[0], after[2] - before[2])
        if action == 1 and displacement < 0.01:
            self.blocked_forwards += 1
            if self.blocked_forwards == 3:
                self.turns_remaining = 6
        else:
            self.blocked_forwards = 0
