"""The 540-class semantic space, our vocabulary and within-class name sampling."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import TYPE_CHECKING

from stopwatch.config import Source

if TYPE_CHECKING:
    from stopwatch.model.qwen import PromptNames


SEMANTIC_CLASSES = 540
VOCAB_FILES = (
    "semantic_540.json", "goal_vocabulary.json",
    "within_class_members.json", "class_visibility_thresholds.json",
)
NAME_SAMPLING_ALGORITHM = "semantic-540-within-class-members-python-choice-v1"


class Vocabulary:
    categories: tuple[str, ...]
    goal_to_class_ids: tuple[int, ...]
    member_pools: tuple[tuple[str, ...], ...]
    member_pool_sha256: str
    visibility_thresholds: tuple[float, ...]

    def __init__(self, root: Path) -> None:
        """Read root/vocab files, retaining category and member-pool order."""
        root = root.expanduser()
        self.root = root
        payloads = {name: json.loads((root / "vocab" / name).read_text()) for name in VOCAB_FILES}
        semantic = payloads["semantic_540.json"]
        self.categories = tuple(semantic["categories"])
        self.goal_to_class_ids = tuple(semantic["goal_to_class_ids"])
        members = payloads["within_class_members.json"]["classes"]
        self.member_pools = tuple(tuple(row["label"] for row in members[name]["members"])
                                  for name in self.categories)
        encoded = json.dumps(list(zip(self.categories, self.member_pools)),
                             ensure_ascii=False, separators=(",", ":"))
        self.member_pool_sha256 = hashlib.sha256(encoded.encode()).hexdigest()
        table = payloads["class_visibility_thresholds.json"]
        self.visibility_thresholds = tuple(float(table["classes"][str(i)]["threshold"])
                                           for i in range(SEMANTIC_CLASSES))
        self._accepted = {}
        indices = {name: index for index, name in enumerate(self.categories)}
        for group in payloads["goal_vocabulary.json"]["groups"]:
            ids = tuple(sorted(indices[name] for name in group))
            self._accepted.update((i, ids) for i in ids)

    def accepted_class_ids(self, class_id: int) -> tuple[int, ...]:
        """Expand a zero-based semantic class through our vocabulary."""
        if not 0 <= class_id < SEMANTIC_CLASSES:
            raise ValueError("semantic class ID outside 0..539")
        return self._accepted.get(class_id, (class_id,))

    def goal_class_ids(self, source: Source, goal_index: int) -> tuple[int, ...]:
        """Map a source-local goal index to accepted zero-based semantic classes."""
        if source == "ovon":
            if not 0 <= goal_index < len(self.goal_to_class_ids):
                raise ValueError("OVON goal index outside 0..378")
            class_id = self.goal_to_class_ids[goal_index]
        elif source in ("mp3d", "hssd"):
            class_id = goal_index
        else:
            raise ValueError(f"unknown source {source!r}")
        return self.accepted_class_ids(class_id)


def sample_within_class_names(goal: int, vocabulary: Vocabulary, *, probability: float, seed: int, source: Source, key: str, occurrence: int) -> PromptNames:
    """Sample immutable name tables with an occurrence-specific RNG seed."""
    identity = [NAME_SAMPLING_ALGORITHM, vocabulary.member_pool_sha256, int(seed), source, key, int(occurrence)]
    rng = random.Random(json.dumps(identity, ensure_ascii=False, separators=(",", ":")))
    if rng.random() >= probability:
        return {"semantic_names": None, "goal_name": None}
    names = tuple(rng.choice(pool) for pool in vocabulary.member_pools)
    class_id = vocabulary.goal_to_class_ids[goal] if source == "ovon" else goal
    return {"semantic_names": names, "goal_name": names[class_id]}
