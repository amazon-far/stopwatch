"""Strict, self-contained YAML recipes and dotted command-line overrides.

Paths other than Paths.root are relative to the user data root unless absolute.
Scene dataset configuration files resolve under Paths.scenes. Loading a recipe
does not import the model or simulator.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from dataclasses import dataclass, fields, is_dataclass
from itertools import pairwise
from pathlib import Path
from types import UnionType
from typing import Literal, get_args, get_origin, get_type_hints

import yaml

Role = Literal["perception", "policy"]
Source = Literal["ovon", "mp3d", "hssd"]
Benchmark = Literal["ovon_val_unseen", "mp3dov", "hssdov"]


@dataclass(frozen=True)
class Paths:
    root: Path
    scenes: Path
    output: Path

    def resolve(self, path: str | Path) -> Path:
        """Resolve a data, scene, RGB, checkpoint or output path against root."""
        path = Path(path).expanduser()
        return path if path.is_absolute() else self.root.expanduser() / path


@dataclass(frozen=True)
class TrainPaths(Paths):
    rgb: Path


@dataclass(frozen=True)
class ModelConfig:
    role: Role
    image_width: int
    image_height: int
    frame_window: int
    distance_bin_edges: list[float]


@dataclass(frozen=True)
class TrainModelConfig(ModelConfig):
    model_id: str
    revision: str


@dataclass(frozen=True)
class SourceConfig:
    name: Source
    episodes: Path
    weight: float


@dataclass(frozen=True)
class DataConfig:
    sources: list[SourceConfig]
    seed: int
    num_workers: int
    prefetch_factor: int
    candidate_episodes: int
    within_class_names_probability: float


@dataclass(frozen=True)
class PolicyDataConfig(DataConfig):
    terminal_window_probability: float


@dataclass(frozen=True)
class ParallelConfig:
    world_size: int
    sequence_parallel_size: int
    reference_bins: int
    reference_bin_frames: int
    group_frame_cap: int


@dataclass(frozen=True)
class OptimizerConfig:
    encoder_lr: float
    trunk_lr: float
    eps: float
    weight_decay: float
    max_grad_norm: float


@dataclass(frozen=True)
class ScheduleConfig:
    total_updates: int
    warmup_updates: int
    min_lr_ratio: float


@dataclass(frozen=True)
class CheckpointConfig:
    restart_every_updates: int
    export_every_updates: int
    resume_from: Path | None


@dataclass(frozen=True)
class TrainConfig:
    kind: Literal["train"]
    paths: TrainPaths
    model: TrainModelConfig
    data: DataConfig | PolicyDataConfig
    parallel: ParallelConfig
    optimizer: OptimizerConfig
    schedule: ScheduleConfig
    checkpoint: CheckpointConfig
    log_interval: int


@dataclass(frozen=True)
class EvalCheckpoints:
    perception: Path
    policy: Path


@dataclass(frozen=True)
class EvalDatasetConfig:
    benchmark: Benchmark
    split: Literal["val", "val_unseen"]
    episodes: Path
    expected_population: int
    scene_dataset_config: Path | None
    instance_viewpoints: Path
    navmesh_max_climb: float
    navmesh_include_static_objects: bool


@dataclass(frozen=True)
class EvalConfig:
    kind: Literal["eval"]
    paths: Paths
    checkpoints: EvalCheckpoints
    perception: ModelConfig
    policy: ModelConfig
    dataset: EvalDatasetConfig
    seed: int
    num_environments: int
    max_episode_steps: int
    perception_max_new_tokens: int
    progress_interval: int


Config = TrainConfig | EvalConfig


class _StrictLoader(yaml.SafeLoader):
    """Reject duplicate YAML keys rather than silently replacing a value."""


def _mapping(loader: _StrictLoader, node: yaml.MappingNode) -> dict:
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node)
        if not isinstance(key, str):
            raise TypeError("YAML mapping keys must be strings")
        if key in result:
            raise ValueError(f"Duplicate YAML key {key!r}")
        result[key] = loader.construct_object(value_node)
    return result


_StrictLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def _decode(annotation: object, value: object, location: str) -> object:
    origin, args = get_origin(annotation), get_args(annotation)
    if origin is UnionType:
        for choice in args:
            try:
                return _decode(choice, value, location)
            except (TypeError, ValueError):
                pass
        raise ValueError(f"{location} must match {annotation}")
    if origin is Literal:
        if not any(type(value) is type(item) and value == item for item in args):
            raise ValueError(f"{location} must be one of {args}")
        return value
    if is_dataclass(annotation):
        if not isinstance(value, dict):
            raise ValueError(f"{location} must be a mapping")
        names = {field.name for field in fields(annotation)}
        unknown, missing = value.keys() - names, names - value.keys()
        if unknown or missing:
            raise ValueError(f"{location} unknown keys {sorted(unknown)}, missing keys {sorted(missing)}")
        hints = get_type_hints(annotation)
        return annotation(**{name: _decode(hints[name], item, f"{location}.{name}")
                             for name, item in value.items()})
    if origin is list:
        if not isinstance(value, list):
            raise ValueError(f"{location} must be a list")
        return [_decode(args[0], item, f"{location}[{index}]") for index, item in enumerate(value)]
    if annotation is Path:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{location} must be a nonempty path string")
        return Path(value)
    if annotation is float:
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"{location} must be a finite number")
        return float(value)
    if type(value) is not annotation:
        raise ValueError(f"{location} must be {annotation}")
    return value


def _validate(config: Config) -> None:
    models = [config.model] if isinstance(config, TrainConfig) else [config.perception, config.policy]
    for model in models:
        expected = (640, 352, 128) if model.role == "perception" else (320, 160, 500)
        if (model.image_width, model.image_height, model.frame_window) != expected:
            raise ValueError(f"{model.role} requires image width, height and history {expected}")
        edges = model.distance_bin_edges
        if not edges or edges[0] <= 0 or any(left >= right for left, right in pairwise(edges)):
            raise ValueError("Distance bin edges must be positive and strictly increasing")
    if isinstance(config, TrainConfig):
        if not config.model.model_id or not config.model.revision:
            raise ValueError("Model identity and revision must be nonempty")
        p = config.parallel
        expected = (64, 2, 64, 128, 256) if config.model.role == "perception" else (16, 1, 16, 512, 512)
        if (p.world_size, p.sequence_parallel_size, p.reference_bins, p.reference_bin_frames, p.group_frame_cap) != expected:
            raise ValueError(
                f"{config.model.role} training requires {expected[0]} physical ranks and "
                f"sequence_parallel_size={expected[1]} to preserve reference-bin update membership; "
                f"expected (world_size, sequence_parallel_size, reference_bins, reference_bin_frames, "
                f"group_frame_cap)={expected}"
            )
        if [source.name for source in config.data.sources] != ["ovon", "mp3d", "hssd"]:
            raise ValueError("Training sources must be ordered ovon, mp3d, hssd")
        if any(source.weight <= 0 for source in config.data.sources) or not math.isclose(sum(source.weight for source in config.data.sources), 1.0):
            raise ValueError("Source weights must be positive and sum to one")
        if isinstance(config.data, PolicyDataConfig) != (config.model.role == "policy"):
            raise ValueError("Only policy training requires terminal_window_probability")
        probabilities = [config.data.within_class_names_probability]
        if isinstance(config.data, PolicyDataConfig):
            probabilities.append(config.data.terminal_window_probability)
        for probability in probabilities:
            if not 0 <= probability <= 1:
                raise ValueError("Sampling probabilities must be in [0, 1]")
        if config.data.num_workers != 16 or config.data.prefetch_factor != 4:
            raise ValueError("Release data loading requires 16 workers and prefetch factor 4")
        if config.data.candidate_episodes <= 0 or config.log_interval <= 0:
            raise ValueError("Candidate count and log interval must be positive")
        s, o, c = config.schedule, config.optimizer, config.checkpoint
        if not 0 <= s.warmup_updates < s.total_updates or not 0 <= s.min_lr_ratio <= 1:
            raise ValueError("Invalid warmup or cosine schedule")
        if min(o.encoder_lr, o.trunk_lr, o.eps, o.max_grad_norm) <= 0 or o.weight_decay < 0:
            raise ValueError("Invalid optimizer values")
        if min(c.restart_every_updates, c.export_every_updates) <= 0:
            raise ValueError("Checkpoint intervals must be positive")
    else:
        if config.perception.role != "perception" or config.policy.role != "policy":
            raise ValueError("Evaluation requires perception and policy models in their named fields")
        if config.num_environments != 2 or config.max_episode_steps != 500 or config.perception_max_new_tokens != 192:
            raise ValueError("Evaluation requires two environments, 500 steps and 192 generated tokens")
        if min(config.dataset.expected_population, config.progress_interval) <= 0:
            raise ValueError("Episode population and progress interval must be positive")
        if config.dataset.navmesh_max_climb not in (0.1, 0.2):
            raise ValueError("Unsupported protocol navmesh climb")


def load_config(path: str | Path, overrides: Sequence[str] = ()) -> Config:
    """Load a complete recipe, applying repeated --set dotted.key=YAML-value.

    Unknown keys, duplicates, missing fields and wrong value types raise
    ValueError. Overrides may replace existing scalar fields or entire lists;
    list indexing and implicit config inheritance are not supported.
    """
    with Path(path).open(encoding="utf-8") as handle:
        document = yaml.load(handle, Loader=_StrictLoader)
    if not isinstance(document, dict):
        raise TypeError("Recipe must be a YAML mapping")
    for override in overrides:
        key, separator, raw = override.partition("=")
        if not separator or not key or any(not part for part in key.split(".")):
            raise ValueError(f"Expected dotted.key=value, got {override!r}")
        parent = document
        parts = key.split(".")
        for part in parts[:-1]:
            if not isinstance(parent, dict) or part not in parent:
                raise ValueError(f"Unknown override {key!r}")
            parent = parent[part]
        if not isinstance(parent, dict) or parts[-1] not in parent:
            raise ValueError(f"Unknown override {key!r}")
        parent[parts[-1]] = yaml.load(raw, Loader=_StrictLoader)
    kind = document.get("kind")
    if kind not in ("train", "eval"):
        raise ValueError("Recipe kind must be train or eval")
    config = _decode(TrainConfig if kind == "train" else EvalConfig, document, "config")
    _validate(config)
    return config


def parse_config(argv: Sequence[str] | None = None) -> Config:
    """Parse the common --config and repeatable --set command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args(argv)
    return load_config(args.config, args.overrides)
