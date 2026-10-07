"""Scene stealing and durable episode skipping."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Sequence
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from torch.distributed import TCPStore

from stopwatch.eval.geometry import _scene_name

if TYPE_CHECKING:
    from stopwatch.eval.loop import EpisodeRecord


EpisodeIdentity = tuple[str, str]  # normalized scene, string episode ID


def create_store(*, host_name: str, port: int, world_size: int, is_master: bool, timeout: timedelta) -> TCPStore:
    """Create a TCPStore with wait_for_workers=False and no rollout collectives."""
    return TCPStore(host_name=host_name, port=port, world_size=world_size,
                    is_master=is_master, timeout=timeout, wait_for_workers=False)


def _read_records(records_root: Path) -> dict[EpisodeIdentity, dict]:
    records, newest = {}, {}
    for path in sorted(records_root.glob("rank*.jsonl")):
        with path.open() as handle:
            for index, line in enumerate(handle):
                if not line.endswith("\n"):
                    continue
                row = json.loads(line)
                if row.get("complete") is not True:
                    continue
                identity = (_scene_name(row["scene"]), str(row["episode_id"]))
                order = (row["completed_at_ns"], path.name, index)
                if identity not in newest or order > newest[identity]:
                    records[identity], newest[identity] = row, order
    return records


def completed_episodes(records_root: Path) -> set[EpisodeIdentity]:
    """Read durable terminal records, deduplicating by scene and episode ID."""
    return set(_read_records(records_root))


def append_record(path: Path, record: EpisodeRecord) -> None:
    """Append and flush a terminal JSONL row before publishing completion."""
    path.parent.mkdir(parents=True, exist_ok=True)
    row = dict(record, complete=True, completed_at_ns=time.time_ns())
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


class SceneStealingDispatcher:
    def __init__(self, store: TCPStore, scene_episodes: Sequence[Sequence[int]]) -> None:
        self.store = store
        self.scene_episodes = [list(indices) for indices in scene_episodes]

    def _claim_episode(self, scene: int) -> int | None:
        cursor = int(self.store.add(f"scene_{scene}", 1)) - 1
        indices = self.scene_episodes[scene]
        return indices[cursor] if cursor < len(indices) else None

    def next_work(self, current_scene: int | None) -> tuple[int | None, int | None]:
        """Return scene index and episode index, or (None,None) when exhausted."""
        if current_scene is not None:
            episode = self._claim_episode(current_scene)
            if episode is not None:
                return current_scene, episode
        while True:
            scene = int(self.store.add("next_scene", 1)) - 1
            if scene >= len(self.scene_episodes):
                break
            episode = self._claim_episode(scene)
            if episode is not None:
                return scene, episode
        for scene in range(len(self.scene_episodes)):
            episode = self._claim_episode(scene)
            if episode is not None:
                return scene, episode
        return None, None
