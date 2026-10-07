"""Source sensor IDs mapped to the shared one-based semantic classes."""

import json
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np


def _write(path: Path, body: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(body)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


@lru_cache(maxsize=1)
def _hssd_templates(root: Path) -> dict[str, Path]:
    templates = {}
    for path in root.rglob("*.object_config.json"):
        name = path.name.removesuffix(".object_config.json")
        if name in templates:
            raise ValueError(f"Duplicate HSSD template {name}")
        templates[name] = path.resolve()
    return templates


def prepare_scene(row, scenes_root: Path, vocab_root: Path) -> None:
    """Resolve exact HSSD template handles and install the MP3D orientation config."""
    config = scenes_root / row["scene_dataset_config"]
    if row["scene_id"].startswith("mp3d/"):
        _write(config, (vocab_root / "mp3d.scene_dataset_config.json").read_bytes())
    elif row["scene_id"].startswith("hssd-hab/"):
        root = (scenes_root / "hssd-hab").resolve()
        scene = scenes_root / row["scene_id"]
        payload = json.loads((root / "scenes" / scene.name).read_text())
        templates = _hssd_templates(root)
        for placement in payload["object_instances"]:
            if placement["motion_type"] != "STATIC":
                raise ValueError("HSSD rendering requires static placements")
            placement["template_name"] = str(templates[placement["template_name"]])
        stage = payload["stage_instance"]
        stage["template_name"] = str(root / (stage["template_name"] + ".stage_config.json"))
        _write(scene, json.dumps(payload).encode())
        payload = json.loads((root / "hssd-hab.scene_dataset_config.json").read_text())
        payload["scene_instances"]["paths"] = {".json": ["derived_scenes"]}
        _write(config, json.dumps(payload).encode())


def instance_lut(sim, source: str, vocab_root: Path) -> np.ndarray:
    """Map instance pixels; zero and unknown instances remain background."""
    mapping = json.loads((vocab_root / f"{source}_semantic_lut.json").read_text())
    if source == "hssd":
        manager = sim.get_rigid_object_manager()
        objects = sorted((manager.get_object_by_handle(h) for h in manager.get_object_handles()),
                         key=lambda obj: obj.object_id)
        lut = np.zeros(len(objects) + 1, dtype=np.uint16)
        for sid, obj in enumerate(objects, 1):
            template = Path(obj.creation_attributes.handle).name.removesuffix(".object_config.json")
            lut[sid] = mapping[template]
            obj.semantic_id = sid
            if not obj.visual_scene_nodes or any(node.semantic_id != sid for node in obj.visual_scene_nodes):
                raise ValueError(f"HSSD instance assignment failed for {template}")
        return lut
    objects = [obj for obj in sim.semantic_scene.objects if obj is not None and obj.category is not None]
    lut = np.zeros(max([0, *(int(obj.semantic_id) for obj in objects)]) + 1, dtype=np.uint16)
    for obj in objects:
        sid = int(obj.semantic_id)
        if sid <= 0:
            continue
        if source == "mp3d":
            raw = int(obj.category.index("raw"))
            if 0 < raw < len(mapping):
                lut[sid] = mapping[raw]
        else:
            lut[sid] = mapping.get(obj.category.name().strip().lower(), 0)
    return lut
