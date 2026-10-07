"""Filesystem asynchronous checkpoints and stock bf16 safetensors export.

Distributed checkpointing writes to a directory visible to every rank, with atomic completion
publication and no transport or retention service.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.staging import DefaultStager, StagingOptions
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.fsdp import FSDPModule
from torch.optim import Optimizer
from transformers import AutoModelForImageTextToText

from stopwatch.config import TrainConfig
from stopwatch.data.batching import ReferenceBinLoader
from stopwatch.data.vocab import VOCAB_FILES, Vocabulary
from stopwatch.model.qwen import QwenNTP
from stopwatch.train.fsdp import full_model_state, restore_rng, rng_state


def _digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _layout(config: TrainConfig, vocabulary: Vocabulary) -> dict:
    recipe = asdict(config)
    for field in ("paths", "checkpoint", "log_interval"):
        recipe.pop(field)
    # JSON normalization makes paths comparable after serialization.
    recipe = json.loads(json.dumps(recipe, default=str))
    return {
        "world": dist.get_world_size(),
        "parameter_sharding": "full_dp_sp_world",
        "torch_version": str(torch.__version__),
        "recipe": recipe,
        "vocabulary": {name: _digest(config.paths.resolve(Path("vocab") / name))
                       for name in VOCAB_FILES},
        "member_pool_sha256": vocabulary.member_pool_sha256,
        "sources": {source.name: _digest(config.paths.resolve(source.episodes) / "manifest.json")
                    for source in config.data.sources},
    }


def _read_metadata(directory: Path) -> dict:
    record = json.loads((directory / "_COMPLETE").read_text())
    if record["format"] != 1:
        raise ValueError("Unsupported restart format")
    metadata = torch.load(directory / "training_metadata.pt", map_location="cpu", weights_only=False)
    if record["update"] != metadata["update"]:
        raise ValueError("Restart completion record differs from metadata")
    return metadata


class AsyncCheckpoint:
    def __init__(self, config: TrainConfig, model: QwenNTP, optimizer: Optimizer, loader: ReferenceBinLoader, vocabulary: Vocabulary) -> None:
        """Bind async DCP to local/shared storage and the exact recipe layout."""
        self.config, self.model, self.optimizer, self.loader = config, model, optimizer, loader
        self._layout = _layout(config, vocabulary)
        self._root = config.paths.resolve(config.paths.output) / "restart"
        self._group = dist.new_group(backend="gloo", timeout=timedelta(hours=2))
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="checkpoint-write")
        self._pending = None
        self._completed = 0
        self._last_saved = None
        self._restored_rng = None

    def restore(self) -> int:
        """Restore model, optimizer, per-rank RNG and consumed worker states.

        Return completed updates, or zero without a complete restart. A null
        resume_from selects the latest complete local restart automatically.
        Reject layout, model revision or vocabulary digest changes.
        """
        selected = [None]
        if dist.get_rank() == 0:
            explicit = self.config.checkpoint.resume_from
            if explicit is not None:
                selected[0] = str(self.config.paths.resolve(explicit))
            else:
                candidates = []
                for marker in self._root.glob("step_*/_COMPLETE"):
                    record = json.loads(marker.read_text())
                    candidates.append((int(record["update"]), marker.parent))
                if candidates:
                    selected[0] = str(max(candidates)[1])
        dist.broadcast_object_list(selected, src=0, group=self._group)
        if selected[0] is None:
            return 0
        directory = Path(selected[0])
        metadata = _read_metadata(directory)
        if metadata["layout"] != self._layout:
            fields = sorted(key for key in self._layout if self._layout[key] != metadata["layout"].get(key))
            raise ValueError(f"Restart layout differs in {fields}")
        model, optimizer = get_state_dict(self.model, self.optimizer)
        state = {"model": model, "optimizer": optimizer}
        dcp.load(state, checkpoint_id=directory, process_group=self._group)
        set_state_dict(self.model, self.optimizer,
                       model_state_dict=state["model"], optim_state_dict=state["optimizer"])
        rank_state = torch.load(directory / f"rank{dist.get_rank()}.pt",
                                map_location="cpu", weights_only=False)
        self.loader.load_state_dict(rank_state["stream"])
        self._restored_rng = rank_state["rng"]
        restore_rng(self._restored_rng)
        self._completed = int(metadata["update"])
        self._last_saved = self._completed
        return self._completed

    def save(self, update: int) -> None:
        """Stage DCP and consumed loader state; publish completion after writing."""
        if update == self._last_saved:
            return
        if not 0 <= update <= self.config.schedule.total_updates:
            raise ValueError("Restart update is outside the training schedule")
        self.wait()
        identity = [f"step_{update:06d}_{uuid.uuid4().hex}" if dist.get_rank() == 0 else None]
        dist.broadcast_object_list(identity, src=0, group=self._group)
        directory = self._root / identity[0]
        directory.mkdir(parents=True, exist_ok=True)
        for module in self.model.modules():
            if isinstance(module, FSDPModule):
                module.reshard()
        rank_state = deepcopy({"rng": rng_state(), "stream": self.loader.state_dict()})
        metadata = {"update": update, "layout": self._layout, "config": asdict(self.config)}
        model, optimizer = get_state_dict(self.model, self.optimizer)
        # Stage before another optimizer step can mutate parameters or moments.
        # Only the background Gloo group participates in checkpoint collectives.
        stager = DefaultStager(StagingOptions(
            use_pinned_memory=True, use_shared_memory=False,
            use_async_staging=False, use_non_blocking_copy=True,
        ))
        try:
            future = dcp.async_save({"model": model, "optimizer": optimizer},
                                    checkpoint_id=directory, process_group=self._group,
                                    async_stager=stager)
        finally:
            stager.close()
        self._pending = self._executor.submit(self._publish, future, directory, metadata, rank_state)
        self._last_saved = update

    def _publish(self, future, directory: Path, metadata: dict, rank_state: dict) -> None:
        future.result()
        torch.save(rank_state, directory / f"rank{dist.get_rank()}.pt")
        if dist.get_rank() == 0:
            torch.save(metadata, directory / "training_metadata.pt")
        dist.barrier(group=self._group)
        if dist.get_rank() == 0:
            temporary = directory / "_COMPLETE.tmp"
            temporary.write_text(json.dumps({"format": 1, "update": metadata["update"]}))
            temporary.replace(directory / "_COMPLETE")
        dist.barrier(group=self._group)

    def wait(self) -> None:
        """Finish pending writes and propagate writer failures."""
        if self._pending is not None:
            self._pending.result()
            self._pending = None


def export_weights(model: QwenNTP, destination: Path) -> None:
    """Collectively gather, then rank zero writes stock bf16 safetensors/processor."""
    state = full_model_state(model)
    # File I/O must not hold other ranks inside an NCCL watchdog interval.
    group = dist.new_group(backend="gloo", timeout=timedelta(hours=2))
    try:
        if dist.get_rank() == 0:
            _save_stock(model, state, destination)
        dist.barrier(group=group)
    finally:
        dist.destroy_process_group(group)


def _save_stock(model: QwenNTP, state: dict, destination: Path) -> None:
    if any(not name.startswith("qwen.") for name in state):
        raise ValueError("Training weights contain non-Qwen state")
    weights = {name.removeprefix("qwen."): value.to(torch.bfloat16)
               for name, value in state.items()}
    saved_rng = rng_state()
    try:
        config = deepcopy(model.qwen.config)
        config.dtype = torch.bfloat16
        # Build only metadata before assigning the gathered CPU tensors. This
        # leaves the live FSDP model and its fp32 optimizer masters untouched.
        with torch.device("meta"):
            stock = AutoModelForImageTextToText.from_config(config, dtype=torch.bfloat16)
        stock.load_state_dict(weights, strict=True, assign=True)
        stock.tie_weights()
        stock.save_pretrained(destination, safe_serialization=True)
        model.processor.save_pretrained(destination)
    finally:
        restore_rng(saved_rng)
