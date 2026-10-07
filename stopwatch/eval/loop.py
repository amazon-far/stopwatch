"""Episode lifecycle and durable per-episode evaluation records."""

from __future__ import annotations

import time
from pathlib import Path
from typing import NotRequired, TypedDict

import numpy as np
import torch

from stopwatch.config import EvalConfig
from stopwatch.eval.dispatch import SceneStealingDispatcher, append_record
from stopwatch.eval.env import VectorEnv, batch_observations
from stopwatch.eval.perception import PerceptionProvider, StuckRecovery
from stopwatch.model.rollout import Rollout


class EpisodeRecord(TypedDict):
    """Terminal state consumed by scoring and restart skipping."""

    episode_id: str
    scene: str
    split: str
    ended_by_stop: bool
    final_position: list[float]
    complete: NotRequired[bool]
    completed_at_ns: NotRequired[int]


@torch.inference_mode()
def run_rollouts(config: EvalConfig, envs: VectorEnv, perception: PerceptionProvider, policy: Rollout, dispatcher: SceneStealingDispatcher, *, records_path: Path) -> None:
    """Run greedy four-action navigation, writing one durable row per terminal episode."""
    count, store = config.num_environments, dispatcher.store
    target = perception.target
    pending = sum(map(len, dispatcher.scene_episodes))
    initial_complete = config.dataset.expected_population - pending
    previous_actions = torch.zeros((count, 1), dtype=torch.long, device=target)
    masks = torch.zeros((count, 1), dtype=torch.float32, device=target)
    observations = [{"rgb": np.zeros((360, 640, 3), dtype=np.uint8),
                     "objectgoal": np.zeros(1, dtype=np.int64), "goal_name": "chair",
                     "goal_class_ids": frozenset()} for _ in range(count)]
    slots = [None] * count
    scenes = [None] * count
    reported = -1

    def publish(slot, action):
        state = slots[slot]
        episode = envs.episodes[state["index"]]
        row: EpisodeRecord = {
            "episode_id": str(episode.episode_id), "scene": episode.scene_id,
            "split": config.dataset.split, "ended_by_stop": action == 0,
            "final_position": state["position"],
        }
        append_record(records_path, row)
        store.add("completed", 1)
        slots[slot] = None
        masks[slot] = 0

    while int(store.add("completed", 0)) < pending:
        if store.check(["fatal_error"]):
            raise RuntimeError(store.get("fatal_error").decode())
        for slot in range(count):
            if slots[slot] is not None:
                continue
            scene, index = dispatcher.next_work(scenes[slot])
            if index is None:
                continue
            scenes[slot] = scene
            slots[slot] = {"index": index, "steps": 0, "recovery": StuckRecovery()}
            masks[slot], previous_actions[slot] = 0, 0
            observations[slot] = envs.reset_at(slot, index)
            slots[slot]["position"], _ = envs.agent_pose_at(slot)
        active = [i for i, state in enumerate(slots) if state is not None]
        if not active:
            time.sleep(0.5)
            continue
        policy_observations = perception.provide_observations(
            observations, previous_actions, masks,
            active_indices=active)
        logits = policy.step(batch_observations(policy_observations, target), previous_actions, masks,
                             active_indices=active)
        selected = logits.argmax(dim=-1).tolist()
        for slot in active:
            state = slots[slot]
            recovery = state["recovery"]
            proposed = selected[slot]
            action = recovery.action(proposed)
            forced = recovery.turns_remaining > 0 and proposed != 0
            state["next"] = action, forced
            envs.async_step_at(slot, action)
        for slot in active:
            state = slots[slot]
            action, forced = state["next"]
            observation, done = envs.wait_step_at(slot)
            position, _ = envs.agent_pose_at(slot)
            state["recovery"].observe(action, state["position"], position, forced=forced)
            state["position"] = position
            state["steps"] += 1
            observations[slot] = observation
            previous_actions[slot], masks[slot] = action, 1
            if done or action == 0 or state["steps"] >= config.max_episode_steps:
                publish(slot, action)
        completed = initial_complete + int(store.add("completed", 0))
        if completed // config.progress_interval != reported:
            reported = completed // config.progress_interval
            print(f"ObjectNav eval progress: {completed}/{config.dataset.expected_population}", flush=True)
