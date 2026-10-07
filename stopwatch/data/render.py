"""Render pre-action RGB and sparse semantic counts without retaining masks."""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from pathlib import Path

from numpy import uint8
from numpy.typing import NDArray

from stopwatch.config import Source, TrainConfig
from stopwatch.data.dataset import EpisodeRow


def _make_simulator(row: EpisodeRow, scenes_root: Path, gpu_id: int):
    import habitat_sim

    sim_config = habitat_sim.SimulatorConfiguration()
    sim_config.scene_id = str(scenes_root / row["scene_id"])
    sim_config.gpu_device_id = gpu_id
    sim_config.enable_physics = False
    sim_config.load_semantic_mesh = True
    sim_config.use_semantic_textures = False
    sim_config.allow_sliding = True
    if row["scene_dataset_config"]:
        sim_config.scene_dataset_config_file = str(scenes_root / row["scene_dataset_config"])
    sensor = habitat_sim.CameraSensorSpec()
    sensor.uuid = "rgb"
    sensor.sensor_type = habitat_sim.SensorType.COLOR
    sensor.resolution = [360, 640]
    sensor.position = [0.0, 1.31, 0.0]
    sensor.hfov = 120
    semantic = habitat_sim.CameraSensorSpec()
    semantic.uuid = "semantic"
    semantic.sensor_type = habitat_sim.SensorType.SEMANTIC
    semantic.resolution = sensor.resolution
    semantic.position = sensor.position
    semantic.hfov = sensor.hfov
    agent_config = habitat_sim.agent.AgentConfiguration()
    agent_config.height, agent_config.radius = 1.41, 0.17
    agent_config.sensor_specifications = [sensor, semantic]
    return habitat_sim.Simulator(habitat_sim.Configuration(sim_config, [agent_config]))


def _teleport(sim, pose) -> None:
    import habitat_sim
    import quaternion

    sim.agents[0].set_state(habitat_sim.AgentState(
        position=pose[:3], rotation=quaternion.quaternion(*map(float, pose[3:])),
    ), reset_sensors=True)


def render_episode(row: EpisodeRow, scenes_root: Path, *, gpu_id: int,
                   source: Source, vocab_root: Path, semantic_output: Path) -> Iterator[NDArray[uint8]]:
    """Yield RGB and save counts from the same pre-action sensor observations."""
    import numpy as np

    from stopwatch.data.semantics import instance_lut, prepare_scene

    prepare_scene(row, scenes_root, vocab_root)
    sim = _make_simulator(row, scenes_root, gpu_id)
    try:
        lut = instance_lut(sim, source, vocab_root)
        offsets, ids, pixels = [0], [], []
        for pose in np.asarray(row["poses"], dtype=np.float32).reshape(-1, 7):
            _teleport(sim, pose)
            observations = sim.get_sensor_observations()
            raw = np.asarray(observations["semantic"]).squeeze()
            valid = (raw >= 0) & (raw < len(lut))
            labels = np.where(valid, lut[np.clip(raw, 0, len(lut) - 1)], 0)
            counts = np.bincount(labels.ravel(), minlength=541)
            present = np.flatnonzero(counts[1:]) + 1
            ids.extend(present.tolist())
            pixels.extend(counts[present].tolist())
            offsets.append(len(ids))
            yield observations["rgb"][..., :3].copy()
        if len(offsets) != row["n_frames"] + 1:
            raise ValueError(f"Semantic frame count differs for {row['key']}")
        semantic_output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=semantic_output.parent, suffix=".npz", delete=False) as handle:
            temporary = Path(handle.name)
        try:
            np.savez(temporary, offsets=np.asarray(offsets, dtype=np.int32),
                     ids=np.asarray(ids, dtype=np.uint16), pixels=np.asarray(pixels, dtype=np.uint32))
            temporary.replace(semantic_output)
        finally:
            temporary.unlink(missing_ok=True)
    finally:
        sim.close()


def encode_rgb(frames: Iterator[NDArray[uint8]], output: Path, *, n_frames: int) -> None:
    """Write H.264 High yuv420p, CRF 18, fast, no B-frames, GOP 16, 10 fps."""
    import imageio.v2 as imageio

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".mp4", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        count = 0
        with imageio.get_writer(
            temporary, fps=10, codec="libx264", pixelformat="yuv420p", macro_block_size=1,
            output_params=["-profile:v", "high", "-crf", "18", "-preset", "fast",
                           "-bf", "0", "-g", "16", "-x264-params",
                           "rc-lookahead=0:scenecut=0:mbtree=0:keyint=16:min-keyint=16"],
        ) as writer:
            for frame in frames:
                if frame.shape != (360, 640, 3) or frame.dtype != uint8:
                    raise ValueError("renderer must produce uint8 [360,640,3] RGB")
                writer.append_data(frame)
                count += 1
        if count != n_frames or count == 0:
            raise ValueError(f"rendered {count} frames, expected {n_frames}")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def render_dataset(config: TrainConfig, source: Source, *, gpu_id: int) -> None:
    """Render RGB and counts, resuming episodes whose two outputs are complete."""
    import numpy as np
    from decord import DECORDError, VideoReader, cpu

    from stopwatch.data.dataset import read_episodes

    source_config = next(item for item in config.data.sources if item.name == source)
    for row in read_episodes(config.paths.resolve(source_config.episodes)):
        output = config.paths.resolve(config.paths.rgb) / f"{row['key']}.mp4"
        semantic = config.paths.resolve("semantic") / f"{row['key']}.npz"
        if output.is_file() and semantic.is_file():
            try:
                video = VideoReader(str(output), ctx=cpu(0), num_threads=1)
                with np.load(semantic, allow_pickle=False) as counts:
                    if len(video) == row["n_frames"] and len(counts["offsets"]) == row["n_frames"] + 1:
                        continue
            except (DECORDError, RuntimeError, ValueError):
                pass
        encode_rgb(render_episode(row, config.paths.resolve(config.paths.scenes), gpu_id=gpu_id,
                                  source=source, vocab_root=config.paths.resolve("vocab"), semantic_output=semantic),
                   output, n_frames=row["n_frames"])
