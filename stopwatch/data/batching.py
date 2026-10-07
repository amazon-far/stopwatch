# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2026 Amazon.com, Inc. or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modification note. StopWatch retains unequal-size window partitions and
# reference loss coefficients, with local parquet streams and CPU transport.

"""Source mixing, reference-bin construction and Karmarkar-Karp balancing."""

from __future__ import annotations

import heapq
import math
import random
from bisect import bisect_right
from collections.abc import Iterator
from copy import deepcopy
from functools import partial
from itertools import accumulate, islice
from typing import TYPE_CHECKING, NotRequired, TypedDict

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, get_worker_info

from stopwatch.config import TrainConfig
from stopwatch.data.dataset import (
    EpisodeDataset,
    EpisodeWindow,
    SeededRandomSampler,
)

if TYPE_CHECKING:
    from stopwatch.data.vocab import Vocabulary
    from stopwatch.model.qwen import PromptBuilder, QwenNTP


class PackedBatch(TypedDict):
    """Padded prompt rows plus concatenated frames, not Hugging Face labels."""

    frames: Tensor  # uint8 [N,360,640,3]
    seq_lens: Tensor  # int64 [E], sum N
    actions: Tensor  # int64 [N]
    prompt_input_ids: Tensor  # int64 [E,L], right padded with tokenizer pad ID
    prompt_attention_mask: Tensor  # int64 [E,L], 1 for real tokens
    prompt_action_mask: Tensor  # bool [E,L], predecessor sites
    prompt_semantic_mask: Tensor  # bool [E,L], predecessor sites
    prompt_semantic_targets: Tensor  # int64 [S], row-major mask order
    source_name: NotRequired[str]
    action_loss_coefficients: NotRequired[Tensor]  # policy float32 [N]


class WorkerState(TypedDict):
    worker: int
    python: tuple
    numpy: tuple
    torch: Tensor


class StreamState(TypedDict):
    """Consumed loader positions, including all reference streams."""

    consumed: list[int]
    workers: list[dict[int, WorkerState]]
    num_workers: list[int]


class WindowDescriptor(TypedDict):
    id: int
    frames: int
    token_lengths: list[int]


class ComputeCoefficients(TypedDict):
    vision_per_frame: float
    decoder_per_token: float
    causal_per_token_squared: float


def pack_episodes(episodes: list[EpisodeWindow], target_frames: int, *, pad_token_id: int, current_step_only: bool) -> PackedBatch:
    """Take the ordered whole-window prefix that fits; pad prompts on CPU."""
    chosen, frames = [], 0
    for episode in episodes:
        if frames + episode["T"] > target_frames:
            break
        chosen.append(episode)
        frames += episode["T"]
    if not chosen:
        raise ValueError("reference bin has no fitting episode")
    result = {key: torch.cat([ep[key] for ep in chosen]) for key in _FRAME_FIELDS if key in chosen[0]}
    result["seq_lens"] = torch.tensor([ep["T"] for ep in chosen], dtype=torch.int64)
    prompts = [ep["prompt"] for ep in chosen]
    width = max(len(prompt[0]) for prompt in prompts)
    result["prompt_input_ids"] = torch.full((len(chosen), width), pad_token_id, dtype=torch.int64)
    result["prompt_attention_mask"] = torch.zeros((len(chosen), width), dtype=torch.int64)
    for name in ("action", "semantic"):
        result[f"prompt_{name}_mask"] = torch.zeros((len(chosen), width), dtype=torch.bool)
    semantic_targets = []
    for index, (episode, prompt) in enumerate(zip(chosen, prompts)):
        ids, action, semantic, targets = prompt
        if len(action) != (1 if current_step_only else episode["T"]):
            raise ValueError("action sites disagree with window length or model role")
        if len(semantic) != len(targets):
            raise ValueError("prompt sites and target counts disagree")
        for name, sites in (("action", action), ("semantic", semantic)):
            if len(set(sites)) != len(sites) or any(not 0 <= site < len(ids) for site in sites):
                raise ValueError(f"invalid {name} predecessor sites")
            result[f"prompt_{name}_mask"][index, torch.tensor(sites, dtype=torch.int64)] = True
        if list(semantic) != sorted(semantic):
            raise ValueError("prompt targets must follow row-major mask order")
        result["prompt_input_ids"][index, :len(ids)] = torch.tensor(ids, dtype=torch.int64)
        result["prompt_attention_mask"][index, :len(ids)] = 1
        semantic_targets.extend(targets)
    result["prompt_semantic_targets"] = torch.tensor(semantic_targets, dtype=torch.int64)
    return result


def karmarkar_karp(costs: list[float], partitions: int) -> list[list[int]]:
    """Retain verl's heap ordering and deterministic opposite-order merges."""
    class Set:
        def __init__(self):
            self.total, self.items = 0, []

        def add(self, index, value):
            self.items.append((index, value))
            self.total += value

        def merge(self, other):
            self.items.extend(other.items)
            self.total += other.total

        def __lt__(self, other):
            return (self.total, len(self.items), self.items) < (other.total, len(other.items), other.items)

    class State:
        def __init__(self, index, value):
            self.sets = [Set() for _ in range(partitions)]
            self.sets[0].add(index, value)
            self.sets.sort(reverse=True)

        def merge(self, other):
            for left, right in zip(self.sets, reversed(other.sets), strict=True):
                left.merge(right)
            self.sets.sort(reverse=True)

        def __lt__(self, other):
            spread = self.sets[0].total - self.sets[-1].total
            other_spread = other.sets[0].total - other.sets[-1].total
            return spread > other_spread if spread != other_spread else self.sets[0] > other.sets[0]

    if partitions < 1 or len(costs) < partitions or any(value <= 0 for value in costs):
        raise ValueError("partitioning requires positive costs and at least one window per group")
    heap = []
    for value, index in sorted((value, index) for index, value in enumerate(costs)):
        heapq.heappush(heap, State(index, value))
    while len(heap) > 1:
        left, right = heapq.heappop(heap), heapq.heappop(heap)
        left.merge(right)
        heapq.heappush(heap, left)
    result = [sorted(index for index, _ in part.items) for part in heap[0].sets]
    if any(not group for group in result) or sorted(i for group in result for i in group) != list(range(len(costs))):
        raise RuntimeError("window partition lost or duplicated an item")
    return result


def compute_coefficients(model: QwenNTP, sequence_parallel_size: int) -> ComputeCoefficients:
    """Derive vision, linear decoder and causal costs from Qwen dimensions."""
    text = model.qwen.config.text_config
    h, intermediate, depth = text.hidden_size, text.intermediate_size, text.num_hidden_layers
    head = getattr(text, "head_dim", None) or h // text.num_attention_heads
    query, kv = text.num_attention_heads * head, text.num_key_value_heads * head
    attention = 6 * depth * query / sequence_parallel_size
    linear = 3 * depth * (4 * h * query + 4 * h * kv + 6 * h * intermediate) / sequence_parallel_size
    vision = model.qwen.config.vision_config
    width, mid, layers = vision.hidden_size, vision.intermediate_size, vision.depth
    patches = (model.config.image_height // vision.patch_size) * (model.config.image_width // vision.patch_size)
    merge = vision.spatial_merge_size ** 2
    merger_width = width * merge
    mergers = 1 + len(vision.deepstack_visual_indexes)
    stem = 2 * patches * vision.in_channels * vision.temporal_patch_size * vision.patch_size ** 2 * width
    tower = layers * ((8 * width ** 2 + 4 * width * mid) * patches + 4 * width * patches ** 2)
    merger = 3 * mergers * (patches // merge) * (2 * merger_width ** 2 + 2 * merger_width * h)
    return {"vision_per_frame": (stem + tower + merger) / sequence_parallel_size,
            "decoder_per_token": linear + attention, "causal_per_token_squared": attention}


def split_bin(batch: PackedBatch, *, bin_count: int, policy: bool) -> list[PackedBatch]:
    """Slice windows without rebuilding prompts; attach unweighted loss coefficients."""
    lengths = batch["seq_lens"].tolist()
    offsets = [0, *batch["seq_lens"].cumsum(0).tolist()]
    targets = {"prompt_semantic_targets": "prompt_semantic_mask"}
    target_offsets = {key: [0, *batch[mask].sum(1).cumsum(0).tolist()] for key, mask in targets.items()}
    frame_count = batch["actions"].new_tensor(batch["actions"].numel(), dtype=torch.float32)
    result = []
    for index, length in enumerate(lengths):
        window = {}
        lo, hi = offsets[index:index + 2]
        tokens = int(batch["prompt_attention_mask"][index].sum())
        for key, value in batch.items():
            if key in _FRAME_FIELDS:
                window[key] = value[lo:hi]
            elif key == "seq_lens":
                window[key] = value[index:index + 1]
            elif key in targets:
                start, end = target_offsets[key][index:index + 2]
                window[key] = value[start:end]
            elif key.startswith("prompt_") and isinstance(value, Tensor) and value.ndim == 2:
                window[key] = value[index:index + 1, :tokens]
            elif isinstance(value, list) and len(value) == len(lengths):
                window[key] = value[index:index + 1]
            elif not isinstance(value, (Tensor, list)):
                window[key] = value
            else:
                raise ValueError(f"no window slicing rule for {key}")
        if policy:
            window["action_loss_coefficients"] = torch.ones(length, dtype=torch.float32) / frame_count / bin_count
        result.append(window)
    return result


def plan_windows(windows: list[WindowDescriptor], original: list[list[int]], coefficients: ComputeCoefficients, frame_cap: int) -> list[list[int]]:
    """Return the original ownership if proposed max cost or frame cap regresses."""
    by_id = {item["id"]: item for item in windows}
    costs = {item["id"]: _window_cost(item, coefficients) for item in windows}
    candidate = [[windows[index]["id"] for index in group]
                 for group in karmarkar_karp([costs[item["id"]] for item in windows], len(original))]
    maximum_cost = lambda groups: max(sum(costs[index] for index in group) for group in groups)
    if (maximum_cost(candidate) > maximum_cost(original)
            or max(sum(by_id[index]["frames"] for index in group) for group in candidate) > frame_cap):
        return original
    return candidate


class ReferenceBinLoader:
    def __init__(self, config: TrainConfig, vocabulary: Vocabulary, prompt_builder: PromptBuilder, model: QwenNTP, *, rank: int, start_step: int = 0) -> None:
        """Build virtual-rank streams and transfer balanced windows to their owners."""
        self.config, self.rank, self.start_step = config, rank, start_step
        self.batch_size, self.dataset = None, self
        self.policy = config.model.role == "policy"
        self.sp = config.parallel.sequence_parallel_size
        self.world = config.parallel.world_size
        self.pad_token_id = prompt_builder.pad_token_id
        self.coefficients = compute_coefficients(model, self.sp)
        self._source_sampler = _SourceSampler([source.weight for source in config.data.sources], config.data.seed)
        self._consumed = self._source_sampler.counts_before(start_step)
        self._workers = [{} for _ in config.data.sources]
        self._loaders, self._samplers = [], []
        self._started = False
        for index, source in enumerate(config.data.sources):
            builder = prompt_builder if index == 0 else model.make_prompt_builder(source.name)
            dataset = EpisodeDataset(config, source, vocabulary, builder,
                                     rank=rank, world_size=config.parallel.reference_bins)
            sampler = SeededRandomSampler(dataset, sample_seed=config.data.seed + rank,
                                          generator_seed=config.data.seed + rank, num_samples=2 ** 48)
            collator = _CheckpointCollator(config, builder.pad_token_id)
            loader = DataLoader(
                dataset, batch_size=config.data.candidate_episodes, sampler=sampler,
                num_workers=config.data.num_workers, prefetch_factor=config.data.prefetch_factor,
                persistent_workers=True, pin_memory=True, collate_fn=collator,
                worker_init_fn=partial(_initialize_worker, collator=collator),
                generator=torch.Generator().manual_seed(config.data.seed + index * 1000003),
            )
            self._loaders.append(loader)
            self._samplers.append(sampler)

    def __iter__(self) -> Iterator[PackedBatch]:
        """Yield the complete SP-group batch once per optimizer update."""
        import torch.distributed as dist

        if self._started:
            raise RuntimeError("reference loader iterator may only be started once")
        self._started = True
        if dist.get_rank() != self.rank or dist.get_world_size() != self.world:
            raise ValueError("distributed layout disagrees with reference streams")
        group = dist.new_group(backend="gloo")
        iterators = {}
        for step in range(self.start_step, self.config.schedule.total_updates):
            source_index = self._source_sampler.source_at(step)
            if source_index not in iterators:
                loader = self._loaders[source_index]
                loader.batch_sampler.sampler = _OffsetSampler(
                    self._samplers[source_index],
                    self._consumed[source_index] * self.config.data.candidate_episodes,
                )
                iterators[source_index] = iter(loader)
            batch = next(iterators[source_index])
            worker = batch.pop("_restart_worker_state")
            self._workers[source_index][worker["worker"]] = worker
            self._consumed[source_index] += 1
            source = self.config.data.sources[source_index]
            batch["source_name"] = source.name
            windows = {
                self.rank * 1024 + index: window for index, window in enumerate(split_bin(
                    batch, bin_count=self.config.parallel.reference_bins, policy=self.policy,
                ))
            }
            descriptions = _describe(windows, self.rank)
            gathered = [None] * self.world
            dist.all_gather_object(gathered, descriptions, group=group)
            items = sorted((item for rows in gathered for item in rows), key=lambda item: item["id"])
            bins = []
            for identity in range(self.world):
                members = [item for item in items if item["owner"] == identity]
                if not members or any(item["plain"]["source_name"] != source.name for item in members):
                    raise RuntimeError("reference stream identities or source schedule changed")
                bins.append({"id": identity, "owner": identity,
                             "frames": sum(item["frames"] for item in members),
                             "token_lengths": [length for item in members for length in item["token_lengths"]]})
            original_bins = _original_groups(bins, self.sp)
            original_windows = [[item["id"] for identity in owners for item in items if item["owner"] == identity]
                                for owners in original_bins]
            groups = plan_windows(items, original_windows, self.coefficients, self.config.parallel.group_frame_cap)
            decoded = _exchange(windows, items, groups, self.rank, self.sp, group)
            result = _merge([decoded[identity] for identity in groups[self.rank // self.sp]], self.pad_token_id)
            if (result["frames"].shape[0] > self.config.parallel.group_frame_cap
                    or int(result["seq_lens"].max()) > self.config.model.frame_window):
                raise ValueError("balanced batch exceeds frame or history cap")
            for key, value in result.items():
                if isinstance(value, Tensor) and not value.is_pinned():
                    result[key] = value.pin_memory()
            yield result

    def __len__(self) -> int:
        return self.config.schedule.total_updates - self.start_step

    def state_dict(self) -> list[StreamState]:
        """Return consumed states in this rank's ordered local reference streams."""
        return [deepcopy({"consumed": self._consumed, "workers": self._workers,
                          "num_workers": [loader.num_workers for loader in self._loaders]})]

    def load_state_dict(self, state: list[StreamState]) -> None:
        """Restore consumed positions before starting the next update."""
        if self._started:
            raise RuntimeError("restore reference state before starting its iterator")
        if len(state) != 1:
            raise ValueError("checkpoint reference stream ownership differs")
        stream = state[0]
        restored_step = sum(stream["consumed"])
        if not 0 <= restored_step <= self.config.schedule.total_updates:
            raise ValueError("checkpoint position is outside the training schedule")
        if self.start_step not in (0, restored_step):
            raise ValueError("checkpoint differs from the requested starting update")
        if stream["consumed"] != self._source_sampler.counts_before(restored_step):
            raise ValueError("checkpoint consumed positions differ from the source schedule")
        if stream["num_workers"] != [loader.num_workers for loader in self._loaders]:
            raise ValueError("checkpoint worker layout differs")
        self.start_step = restored_step
        self._consumed = list(stream["consumed"])
        self._workers = deepcopy(stream["workers"])
        for loader, consumed, workers in zip(self._loaders, self._consumed, self._workers):
            loader.collate_fn.resume = {"rotation": consumed % loader.num_workers, "workers": deepcopy(workers)}


_FRAME_FIELDS = {"frames", "actions"}


def _window_cost(item, coefficients):
    length = sum(item["token_lengths"])
    return (coefficients["vision_per_frame"] * item["frames"]
            + coefficients["decoder_per_token"] * length
            + coefficients["causal_per_token_squared"] * length ** 2)


class _SourceSampler:
    def __init__(self, weights, seed):
        scaled = [value / max(weights) for value in weights]
        total = math.fsum(scaled)
        self.cumulative = list(accumulate(value / total for value in scaled))
        self.seed = seed

    def source_at(self, step):
        draw = random.Random(f"{self.seed}:{step}").random()
        return min(bisect_right(self.cumulative, draw), len(self.cumulative) - 1)

    def counts_before(self, step):
        counts = [0] * len(self.cumulative)
        for previous in range(step):
            counts[self.source_at(previous)] += 1
        return counts


class _OffsetSampler:
    def __init__(self, sampler, offset):
        self.sampler, self.offset = sampler, offset

    def __iter__(self):
        return islice(iter(self.sampler), self.offset, None)

    def __len__(self):
        return len(self.sampler) - self.offset


class _CheckpointCollator:
    def __init__(self, config, pad_token_id):
        self.config, self.pad_token_id = config, pad_token_id
        self.resume = None

    def __call__(self, samples):
        batch = pack_episodes(samples, self.config.parallel.reference_bin_frames,
                              pad_token_id=self.pad_token_id,
                              current_step_only=self.config.model.role == "perception")
        info = get_worker_info()
        identity = (info.id + self.resume["rotation"]) % info.num_workers if self.resume else info.id
        batch["_restart_worker_state"] = {"worker": identity, "python": random.getstate(),
                                         "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
        return batch


def _initialize_worker(worker_id, *, collator):
    info = get_worker_info()
    resume = collator.resume
    identity = (worker_id + resume["rotation"]) % info.num_workers if resume else worker_id
    random.seed(int(info.seed) % (2 ** 32))
    np.random.seed(int(info.seed) % (2 ** 32))
    torch.manual_seed(int(info.seed))
    if resume and identity in resume["workers"]:
        state = resume["workers"][identity]
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])


def _original_groups(bins, sp):
    groups = [[index] for index in range(len(bins))]
    if sp == 2:
        # Retain the reference perception-bin pairing fit before window balancing.
        def cost(group):
            lengths = [length for index in group for length in bins[index]["token_lengths"]]
            return (-31.474364931961812
                    + 3.182496273110452 * sum(bins[index]["frames"] for index in group)
                    + 0.01595939170955446 * sum(lengths)
                    + 15.008669965417708 * len(lengths)
                    + 9.732922426202912e-07 * sum(value ** 2 for value in lengths)
                    + 0.0023374289527025313 * max(lengths))
        order = sorted(groups, key=lambda group: (cost(group), min(group)))
        groups = [order[index] + order[-1 - index] for index in range(len(order) // 2)]
    assigned = {}

    def assign(index, visited):
        owners = [bins[item]["owner"] // sp for item in groups[index]]
        candidates = sorted(set(owners), key=lambda owner: (-owners.count(owner), owner))
        for owner in candidates:
            if owner in visited:
                continue
            visited.add(owner)
            if owner not in assigned or assign(assigned[owner], visited):
                assigned[owner] = index
                return True
        return False

    for index in range(len(groups)):
        if not assign(index, set()):
            raise RuntimeError("reference groups cannot retain a locally owned bin")
    return [groups[assigned[owner]] for owner in range(len(groups))]


def _describe(windows, rank):
    descriptions = []
    for identity, window in windows.items():
        tensors, plain = {}, {}
        for key, value in window.items():
            if isinstance(value, Tensor):
                window[key] = value.contiguous()
                tensors[key] = {"shape": list(value.shape), "dtype": str(value.dtype).removeprefix("torch.")}
            else:
                plain[key] = value
        descriptions.append({"id": identity, "owner": rank, "tensors": tensors, "plain": plain,
                             "frames": window["frames"].shape[0],
                             "token_lengths": window["prompt_attention_mask"].sum(1).tolist()})
    return descriptions


def _exchange(local, items, groups, rank, sp, group):
    import torch.distributed as dist

    by_id = {item["id"]: item for item in items}
    destinations = {identity: list(range(index * sp, (index + 1) * sp))
                    for index, owners in enumerate(groups) for identity in owners}
    mine = groups[rank // sp]
    decoded = {identity: local[identity] for identity in mine if identity in local}
    for identity in mine:
        if identity not in decoded:
            item = by_id[identity]
            decoded[identity] = dict(item["plain"])
            for key, spec in item["tensors"].items():
                decoded[identity][key] = torch.empty(spec["shape"], dtype=getattr(torch, spec["dtype"]), pin_memory=True)
    operations, buffers = [], []
    for item in items:
        identity, owner = item["id"], item["owner"]
        for peer in destinations[identity]:
            if peer == owner or rank not in (owner, peer):
                continue
            for slot, key in enumerate(sorted(item["tensors"])):
                value = local[identity][key] if rank == owner else decoded[identity][key]
                if not value.numel():
                    continue
                buffer = value.reshape(-1).view(torch.uint8)
                buffers.append(buffer)
                operation = dist.isend if rank == owner else dist.irecv
                other = peer if rank == owner else owner
                operations.append(dist.P2POp(operation, buffer, other, group=group, tag=identity * 1024 + slot))
    for request in dist.batch_isend_irecv(operations) if operations else []:
        request.wait()
    return decoded


def _merge(windows, pad_token_id):
    from torch.nn import functional

    result = {}
    if any(set(window) != set(windows[0]) for window in windows):
        raise ValueError("reference windows have different fields")
    for key in windows[0]:
        values = [window[key] for window in windows]
        if isinstance(values[0], Tensor):
            if key.startswith("prompt_") and values[0].ndim == 2:
                width = max(value.shape[1] for value in values)
                values = [functional.pad(value, (0, width - value.shape[1]),
                                         value=pad_token_id if key == "prompt_input_ids" else 0) for value in values]
            result[key] = torch.cat(values)
        elif isinstance(values[0], list):
            result[key] = [item for value in values for item in value]
        elif all(value == values[0] for value in values):
            result[key] = values[0]
        else:
            raise ValueError(f"reference windows disagree on {key}")
    return result
