"""Read local episode parquet and sample training windows.

No semantic masks are decoded. Sparse class counts produce the same supervision.
Training data and rendered semantic counts are documented in docs/training.md.
"""

from __future__ import annotations

import io
import random
from collections import OrderedDict
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, RandomSampler, Sampler

from stopwatch.config import SourceConfig, TrainConfig
from stopwatch.data.visibility import stop_goal_visible

if TYPE_CHECKING:
    from stopwatch.data.vocab import Vocabulary
    from stopwatch.model.qwen import Prompt, PromptBuilder


class EpisodeRow(TypedDict):
    """One parquet row, with flattened pre-action poses."""

    key: str
    scene_id: str
    scene_dataset_config: str
    object_category: str
    category_index: int  # int32, OVON goal index or MP3D/HSSD semantic class
    goal_category_ids: list[int]  # zero-based semantic classes, never OVON goal IDs
    n_frames: int
    actions: list[int]
    poses: list[float]
    geo_dists: list[float]


class EpisodeWindow(TypedDict):
    """CPU window. T frames precede their corresponding expert actions."""

    frames: Tensor  # uint8 [T, 360, 640, 3]
    actions: Tensor  # int64 [T], Habitat action IDs 0..3
    T: int
    prompt: Prompt


SampleIndex = tuple[int, int, int]  # row index, target-frame seed, occurrence ordinal


class EpisodeDataset(Dataset[EpisodeWindow]):
    def __init__(self, config: TrainConfig, source: SourceConfig, vocabulary: Vocabulary, prompt_builder: PromptBuilder, *, rank: int = 0, world_size: int = 1) -> None:
        """Build the STOP-visible index in original parquet row order."""
        from pyarrow import parquet

        if not 0 <= rank < world_size:
            raise ValueError("dataset rank must belong to a positive world size")
        self.config, self.source = config, source
        self.vocabulary, self.prompt_builder = vocabulary, prompt_builder
        root = config.paths.resolve(source.episodes)
        self.parts = tuple(sorted(root.glob("part-*.parquet")))
        if not self.parts:
            raise ValueError(f"no episode parts in {root}")
        locations = []
        for part_index, part in enumerate(self.parts):
            file = parquet.ParquetFile(part)
            for group in range(file.num_row_groups):
                table = file.read_row_group(group, columns=["key", "category_index", "goal_category_ids", "n_frames"])
                for offset, row in enumerate(table.to_pylist()):
                    members = vocabulary.goal_class_ids(source.name, row["category_index"])
                    if tuple(row["goal_category_ids"]) != members:
                        raise ValueError("STOP members disagree with the complete accepted goal set")
                    path = config.paths.resolve("semantic") / f"{row['key']}.npz"
                    with np.load(path, allow_pickle=False) as counts:
                        if len(counts["offsets"]) != row["n_frames"] + 1:
                            raise ValueError(f"Semantic frame count differs for {row['key']}")
                        if stop_goal_visible(counts, members, vocabulary.visibility_thresholds):
                            locations.append((part_index, group, offset))
        if world_size > len(self.parts):
            locations = locations[rank::world_size]
        else:
            locations = [location for location in locations if location[0] % world_size == rank]
        self.locations = np.asarray(locations, dtype=np.int64).reshape(-1, 3)
        if not len(self.locations):
            raise ValueError("source contains no STOP-visible episodes on this rank")
        self._groups = OrderedDict()

    def __len__(self) -> int:
        return len(self.locations)

    def __getitem__(self, index: SampleIndex) -> EpisodeWindow:
        """Sample a window and tokenize it in the worker."""
        row_index, target_seed, occurrence = index
        return self._window(self._row(row_index), target_seed, occurrence)

    def _row(self, index: int) -> EpisodeRow:
        from pyarrow import parquet

        part, group, offset = map(int, self.locations[index])
        identity = (part, group)
        table = self._groups.pop(identity, None)
        if table is None:
            table = parquet.ParquetFile(self.parts[part]).read_row_group(group)
        self._groups[identity] = table
        while len(self._groups) > 2:
            self._groups.popitem(last=False)
        return table.slice(offset, 1).to_pylist()[0]

    def _window(self, row: EpisodeRow, target_seed: int, occurrence: int) -> EpisodeWindow:
        from decord import VideoReader, bridge, cpu

        from stopwatch.data.vocab import sample_within_class_names

        path = self.config.paths.resolve(self.config.paths.rgb) / f"{row['key']}.mp4"
        payload = path.read_bytes()
        bridge.set_bridge("native")
        try:
            video = VideoReader(io.BytesIO(payload), ctx=cpu(0))
            total = len(video)
        except Exception as error:
            raise RuntimeError(f"Cannot decode episode {path}") from error
        if total != row["n_frames"] or total <= 0:
            raise ValueError(f"RGB frame count {total} differs from the required positive count {row['n_frames']} in {path}")
        perception = self.config.model.role == "perception"
        if perception:
            target = random.Random(target_seed).randint(0, total - 1)
            start = max(0, target - self.config.model.frame_window + 1)
            length = target - start + 1
        else:
            length = min(total, self.config.model.frame_window)
            last_start = total - length
            start = 0
            if last_start > 0:
                start = (last_start if random.random() < self.config.data.terminal_window_probability
                         else random.randint(0, last_start - 1))
        try:
            decoded = video.get_batch(list(range(start, start + length))).asnumpy()
        except Exception as error:
            raise RuntimeError(f"Cannot decode episode {path}") from error
        if decoded.shape != (length, 360, 640, 3) or decoded.dtype != np.uint8:
            raise ValueError(f"RGB geometry differs from the released camera in {path}")
        actions = torch.tensor(row["actions"][start:start + length], dtype=torch.int64)
        goal = row["category_index"]
        members = row["goal_category_ids"]
        frame_indices = [start + length - 1] if perception else list(range(start, start + length))
        semantic_path = self.config.paths.resolve("semantic") / f"{row['key']}.npz"
        with np.load(semantic_path, allow_pickle=False) as counts:
            semantic, visible = semantic_targets(counts, frame_indices, members, self.vocabulary)
        history = torch.zeros(length, dtype=torch.bool) if perception else visible
        if perception:
            history[-1] = visible[0]
        distances = torch.tensor(row["geo_dists"][start:start + length], dtype=torch.float32)
        include = history & torch.isfinite(distances)
        distances = torch.nan_to_num(distances)
        result = EpisodeWindow(frames=torch.from_numpy(decoded), actions=actions, T=length)
        names = sample_within_class_names(
            goal, self.vocabulary, probability=self.config.data.within_class_names_probability,
            seed=self.config.data.seed, source=self.source.name, key=row["key"], occurrence=occurrence,
        )
        result["prompt"] = self.prompt_builder.build_prompt(
            goal, actions.tolist(), semantic[0] if perception else semantic,
            goal_visible=history.tolist(),
            goal_distances=[float(d) if bool(mask) else None for d, mask in zip(distances, include)],
            **names,
        )
        return result


class SeededRandomSampler(Sampler[SampleIndex]):
    def __init__(self, dataset: EpisodeDataset, *, sample_seed: int, generator_seed: int, num_samples: int) -> None:
        """Use replacement sampling and attach sample_seed + ordinal and ordinal."""
        self.dataset, self.sample_seed = dataset, sample_seed
        self.generator_seed, self.num_samples = generator_seed, num_samples

    def __iter__(self) -> Iterator[SampleIndex]:
        sampler = RandomSampler(self.dataset, replacement=True, num_samples=self.num_samples,
                                generator=torch.Generator().manual_seed(self.generator_seed))
        for ordinal, index in enumerate(sampler):
            yield index, self.sample_seed + ordinal, ordinal

    def __len__(self) -> int:
        return self.num_samples


def read_episodes(parts_root: Path) -> Iterator[EpisodeRow]:
    """Read sorted part files in original row order."""
    from pyarrow import parquet

    parts = sorted(parts_root.glob("part-*.parquet"))
    if not parts:
        raise ValueError(f"no episode parts in {parts_root}")
    for part in parts:
        file = parquet.ParquetFile(part)
        for batch in file.iter_batches(batch_size=256):
            yield from batch.to_pylist()


def semantic_targets(sparse, frame_indices: list[int], goal_class_ids: list[int], vocabulary: Vocabulary) -> tuple[Tensor, Tensor]:
    """Return int16 [F,16] ranked one-based IDs and bool [F] goal membership."""
    ranked = np.zeros((len(frame_indices), 16), dtype=np.int16)
    visible = np.zeros(len(frame_indices), dtype=np.bool_)
    minimum = np.ceil(np.asarray(vocabulary.visibility_thresholds, dtype=np.float64) * (640 * 360))
    for index, frame in enumerate(frame_indices):
        lo, hi = sparse["offsets"][frame:frame + 2]
        counts = np.zeros(541, dtype=np.int64)
        counts[sparse["ids"][lo:hi]] = sparse["pixels"][lo:hi]
        counts[1:][counts[1:] < minimum] = 0
        candidates = [i + 1 for i in goal_class_ids if counts[i + 1] > 0]
        goal_id = max(candidates, key=lambda i: counts[i]) if candidates else None
        visible[index] = goal_id is not None
        present = np.flatnonzero(counts)
        order = present[np.lexsort((present, -counts[present]))][:16]
        if goal_id is not None and goal_id not in order:
            order = np.concatenate((order[:-1], [goal_id]))
        ranked[index, :len(order)] = order
    return torch.from_numpy(ranked), torch.from_numpy(visible)
