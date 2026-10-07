"""Stretch ObjectNav environments and observation-to-tensor batching."""

from __future__ import annotations

import json
import multiprocessing as mp
import traceback
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, NotRequired, TypedDict

import numpy as np
import torch
from numpy.typing import NDArray
from torch import device

from stopwatch.config import EvalConfig
from stopwatch.data.semantics import prepare_scene
from stopwatch.data.vocab import Vocabulary
from stopwatch.eval.geometry import _scene_name
from stopwatch.model.rollout import ObservationBatch

if TYPE_CHECKING:
    from habitat.tasks.nav.object_nav_task import ObjectGoalNavEpisode


class Observation(TypedDict):
    rgb: NDArray[np.uint8]  # [360,640,3]
    objectgoal: NDArray[np.int64]  # [1]
    goal_name: str
    goal_class_ids: frozenset[int]
    semantic_bow_ids: NotRequired[NDArray[np.int64]]  # [16]
    goal_distance: NotRequired[NDArray[np.float32]]  # [1], NaN if absent


def _dataset_config(config: EvalConfig):
    from habitat.config.default_structured_configs import DatasetConfig
    from omegaconf import OmegaConf

    return OmegaConf.structured(DatasetConfig(
        type="ObjectNav-v1", split=config.dataset.split,
        data_path=str(config.paths.resolve(config.dataset.episodes)),
        scenes_dir=str(config.paths.resolve(config.paths.scenes)), content_scenes=["*"],
    ))


def _load_dataset(config: EvalConfig):
    from habitat.datasets.object_nav.object_nav_dataset import ObjectNavDatasetV1

    class ReleasedDataset(ObjectNavDatasetV1):
        def from_json(self, json_str, scenes_dir=None):
            start = len(self.episodes)
            identities = [str(ep["episode_id"]) for ep in json.loads(json_str)["episodes"]]
            super().from_json(json_str, scenes_dir)
            # Preserve dataset episode IDs for resume and scoring.
            for episode, identity in zip(self.episodes[start:], identities, strict=True):
                episode.episode_id = identity

    dataset = ReleasedDataset(config=_dataset_config(config))
    scene_config = config.dataset.scene_dataset_config
    for episode in dataset.episodes:
        episode.scene_dataset_config = (str(config.paths.resolve(config.paths.scenes) / scene_config)
                                        if scene_config is not None else "default")
    identities = {(_scene_name(ep.scene_id), str(ep.episode_id)) for ep in dataset.episodes}
    if len(identities) != len(dataset.episodes) or len(identities) != config.dataset.expected_population:
        raise ValueError("Episode inventory differs from the frozen benchmark population")
    return dataset


def load_episodes(config: EvalConfig) -> list[ObjectGoalNavEpisode]:
    """Load all episodes and atomically prepare their scenes before environment creation."""
    episodes = list(_load_dataset(config).episodes)
    scene_config = config.dataset.scene_dataset_config
    if scene_config is not None:
        scenes_root = config.paths.resolve(config.paths.scenes)
        for scene_id in sorted({episode.scene_id for episode in episodes}):
            row = {"scene_id": str(Path(scene_id).relative_to(scenes_root)),
                   "scene_dataset_config": str(scene_config)}
            prepare_scene(row, scenes_root, config.paths.resolve("vocab"))
    return episodes


def batch_observations(observations: Sequence[Observation], target: device) -> ObservationBatch:
    """Stack model observation arrays without dtype conversion."""
    batch: ObservationBatch = {
        "rgb": torch.as_tensor(np.stack([obs["rgb"] for obs in observations]), device=target),
        "goal_name": [obs["goal_name"] for obs in observations],
        "goal_class_ids": [obs["goal_class_ids"] for obs in observations],
    }
    for name in ("semantic_bow_ids", "goal_distance"):
        if any(name in obs for obs in observations):
            if not all(name in obs for obs in observations):
                raise ValueError(f"Observation field {name} is absent from some slots")
            batch[name] = torch.as_tensor(np.stack([obs[name] for obs in observations]), device=target)
    return batch


def _habitat_config(config: EvalConfig, gpu_id: int):
    from habitat.config.default_structured_configs import (
        AgentConfig,
        EnvironmentConfig,
        HabitatConfig,
        HabitatSimRGBSensorConfig,
        HabitatSimV0Config,
        IteratorOptionsConfig,
        MoveForwardActionConfig,
        ObjectGoalSensorConfig,
        SimulatorConfig,
        StopActionConfig,
        TaskConfig,
        TurnLeftActionConfig,
        TurnRightActionConfig,
    )
    from omegaconf import OmegaConf

    task = TaskConfig(type="ObjectNav-v1", end_on_success=True, goal_sensor_uuid="objectgoal")
    task.actions = {
        "stop": StopActionConfig(), "move_forward": MoveForwardActionConfig(),
        "turn_left": TurnLeftActionConfig(), "turn_right": TurnRightActionConfig(),
    }
    task.lab_sensors = {"objectgoal_sensor": ObjectGoalSensorConfig()}
    task.measurements = {}
    scene_config = config.dataset.scene_dataset_config
    simulator = SimulatorConfig(
        seed=config.seed, forward_step_size=0.25, turn_angle=30,
        default_agent_navmesh=False,
        navmesh_include_static_objects=config.dataset.navmesh_include_static_objects,
        scene_dataset=(str(config.paths.resolve(config.paths.scenes) / scene_config)
                       if scene_config is not None else "default"),
        habitat_sim_v0=HabitatSimV0Config(gpu_device_id=gpu_id, allow_sliding=True),
        agents={"main_agent": AgentConfig(height=1.41, radius=0.17, sim_sensors={
            "rgb_sensor": HabitatSimRGBSensorConfig(width=640, height=360, hfov=120,
                                                   position=[0.0, 1.31, 0.0]),
        })}, agents_order=["main_agent"],
    )
    habitat = HabitatConfig(seed=config.seed, simulator=simulator, task=task,
                            environment=EnvironmentConfig(max_episode_steps=500,
                                iterator_options=IteratorOptionsConfig(shuffle=False, group_by_scene=False,
                                    cycle=False, num_episode_sample=-1, max_scene_repeat_episodes=-1,
                                    max_scene_repeat_steps=-1)),
                            dataset=_dataset_config(config))
    return OmegaConf.create({"habitat": OmegaConf.to_container(OmegaConf.structured(habitat))})


def _install_navmesh(sim, config: EvalConfig) -> None:
    import habitat_sim

    settings = habitat_sim.NavMeshSettings()
    values = {
        "cell_size": 0.05, "cell_height": 0.05, "agent_height": 1.41, "agent_radius": 0.17,
        "agent_max_climb": config.dataset.navmesh_max_climb, "agent_max_slope": 45.0,
        "region_min_size": 20.0, "region_merge_size": 20.0, "edge_max_len": 12.0,
        "edge_max_error": 1.3, "verts_per_poly": 6.0, "detail_sample_dist": 6.0,
        "detail_sample_max_error": 1.0, "filter_low_hanging_obstacles": True,
        "filter_ledge_spans": True, "filter_walkable_low_height_spans": True,
        "include_static_objects": config.dataset.navmesh_include_static_objects,
    }
    for key, value in values.items():
        setattr(settings, key, value)
    original = sim.reconfigure
    active_scene = None

    def rebuild(sim_config):
        nonlocal active_scene
        if sim_config.scene != active_scene:
            if not sim.recompute_navmesh(sim.pathfinder, settings):
                raise RuntimeError(f"Protocol navmesh build failed for {sim_config.scene}")
            active_scene = sim_config.scene

    def reconfigure(sim_config, *args, **kwargs):
        result = original(sim_config, *args, **kwargs)
        rebuild(sim_config)
        return result

    rebuild(sim.habitat_config)
    sim.reconfigure = reconfigure


def _resolve_goal(config, episode, sensor_id, vocabulary):
    benchmark = config.dataset.benchmark
    if benchmark in {"mp3dov", "hssdov"}:
        class_id = sensor_id
        if vocabulary.categories[class_id] != episode.object_category:
            raise ValueError("Open-vocabulary episode goal ID/name mismatch")
        name = vocabulary.categories[class_id]
    else:
        class_id = vocabulary.goal_to_class_ids[sensor_id]
        name = vocabulary.categories[class_id]
    accepted = frozenset(vocabulary.accepted_class_ids(class_id))
    return name, accepted


def _worker(connection, config, episodes, gpu_id):
    import habitat

    env = None
    try:
        dataset = _load_dataset(config)
        dataset.episodes = list(episodes)
        env = habitat.Env(config=_habitat_config(config, gpu_id), dataset=dataset)
        env.seed(config.seed)
        _install_navmesh(env.sim, config)
        vocabulary = Vocabulary(config.paths.root)

        def observation(raw):
            episode = env.current_episode
            name, accepted = _resolve_goal(
                config, episode, int(raw["objectgoal"][0]), vocabulary)
            goal_id = int(raw["objectgoal"][0])
            return {
                "rgb": raw["rgb"], "objectgoal": np.asarray([goal_id], dtype=np.int64),
                "goal_name": name, "goal_class_ids": accepted,
            }

        connection.send((True, None))
        while True:
            command, argument = connection.recv()
            if command == "close":
                break
            if command == "reset":
                env.current_episode = episodes[argument]
                raw = env.reset()
                result = observation(raw)
            elif command == "step":
                raw = env.step(argument)
                result = observation(raw), env.episode_over
            elif command == "pose":
                state = env.sim.get_agent(0).get_state()
                rotation = state.sensor_states["rgb"].rotation
                result = state.position.tolist(), [rotation.w, rotation.x, rotation.y, rotation.z]
            else:
                raise ValueError(f"Unknown environment command {command}")
            connection.send((True, result))
    except EOFError:
        pass
    except Exception:  # noqa: BLE001 -- transmit the traceback to the parent process
        connection.send((False, traceback.format_exc()))
    finally:
        if env is not None:
            env.close()
        connection.close()


class VectorEnv:
    def __init__(self, config: EvalConfig, episodes: Sequence[ObjectGoalNavEpisode], *, gpu_id: int) -> None:
        """Create two subprocess environments with protocol navmesh settings."""
        self.config, self.episodes, self.gpu_id = config, list(episodes), gpu_id
        self._context = mp.get_context("spawn")
        self._connections, self._processes = [None] * config.num_environments, [None] * config.num_environments
        self._pending = [False] * config.num_environments
        try:
            for slot in range(config.num_environments):
                self._start(slot)
        except BaseException:
            self.close()
            raise

    def _start(self, slot):
        connection, child = self._context.Pipe()
        slot_config = replace(self.config, seed=self.config.seed + slot)
        process = self._context.Process(target=_worker, args=(child, slot_config, self.episodes, self.gpu_id), daemon=True)
        process.start()
        child.close()
        self._connections[slot], self._processes[slot] = connection, process
        self._receive(slot)

    def _receive(self, slot):
        ok, result = self._connections[slot].recv()
        if not ok:
            # Python/configuration errors are fatal, rather than scored as crashes.
            raise RuntimeError(result)
        return result

    def _call(self, slot, command, argument=None):
        if self._pending[slot]:
            raise RuntimeError("A simulator step is still pending")
        self._connections[slot].send((command, argument))
        return self._receive(slot)

    def reset_at(self, slot: int, episode_index: int) -> Observation:
        """Assign an episode and return its initial pre-action observation."""
        return self._call(slot, "reset", episode_index)

    def async_step_at(self, slot: int, action: int) -> None:
        if self._pending[slot]:
            raise RuntimeError("A simulator step is still pending")
        self._connections[slot].send(("step", action))
        self._pending[slot] = True

    def wait_step_at(self, slot: int) -> tuple[Observation, bool]:
        """Return observation and done without automatic reset."""
        if not self._pending[slot]:
            raise RuntimeError("No simulator step is pending")
        self._pending[slot] = False
        return self._receive(slot)

    def agent_pose_at(self, slot: int) -> tuple[list[float], list[float]]:
        """Return world position [3] and camera quaternion [qw,qx,qy,qz]."""
        return self._call(slot, "pose")

    def _stop(self, slot):
        connection, process = self._connections[slot], self._processes[slot]
        if connection is not None:
            connection.close()
        if process is not None:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join()
        self._connections[slot] = self._processes[slot] = None

    def close(self) -> None:
        for slot in range(self.config.num_environments):
            self._stop(slot)
