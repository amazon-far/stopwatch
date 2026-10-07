# Training StopWatch

Install the environment using [Getting Started](../README.md#1-environment).
Run the commands below from the repository root with `.venv` activated.

## Download trajectories

```bash
hf download Amazon-FAR/stopwatch-data --repo-type dataset \
  --include 'train/*' --include 'vocab/*' --include README.md --local-dir data
```

The download contains trajectories, not videos or per-frame semantic labels.

## Scene assets

Use the publisher assets and directory layout in
[Run inference](../README.md#2-run-inference), including the training scenes.
Rendering needs the following semantic assets as well.

| Source | Required assets |
|---|---|
| OVON | HM3D v0.2 Habitat meshes, HM3DSem v0.2 annotations and semantic configs. Download `hm3d-train-habitat-v0.2.tar`, `hm3d-train-semantic-annots-v0.2.tar` and `hm3d-train-semantic-configs-v0.2.tar` from [the publisher](https://github.com/matterport/habitat-matterport-3dresearch#-downloading-hm3d-v02). |
| MP3D | The Matterport3D Habitat package, including `<scene>.glb`, `<scene>.house`, `<scene>_semantic.ply` and `<scene>.navmesh`. |
| HSSD | The static HSSD-Hab scenes with their stage and object configs, render and collision meshes, and publisher scene dataset config. |

Extract the HM3D training packages under `data/scenes/hm3d/train/`, preserving
the publisher layout, `.basis.glb`, `.semantic.glb`, `.semantic.txt` and
`hm3d_annotated_train_basis.scene_dataset_config.json`.

## Render episodes

Render every downloaded episode before training. The same rendered files serve
both models.

```bash
python scripts/render.py --config configs/train_perception_4b.yaml --source ovon --gpu-id 0
python scripts/render.py --config configs/train_perception_4b.yaml --source mp3d --gpu-id 0
python scripts/render.py --config configs/train_perception_4b.yaml --source hssd --gpu-id 0
```

Rendered files go to `data/rgb/<key>.mp4` and `data/semantic/<key>.npz`. Rendering
all 3.1M training episodes takes about 10 TB of disk.

Restart the same command to skip episodes with complete RGB and count files.
The loader drops episodes whose goal is invisible at STOP under the per-class
visibility thresholds.

## Train

Training requires 64 ranks for perception and 16 for policy. The official recipes
used B200 GPUs. The examples below use eight GPUs per node. Run the corresponding
command on every node with a unique zero-based `NODE_RANK` and the same
`MASTER_ADDR`, the address of node zero. Data and outputs must be accessible at
the same paths on every rank.

```bash
torchrun --nnodes=8 --nproc_per_node=8 --node_rank="$NODE_RANK" --master_addr="$MASTER_ADDR" --master_port=29500 \
  -m stopwatch.train --config configs/train_perception_4b.yaml
torchrun --nnodes=2 --nproc_per_node=8 --node_rank="$NODE_RANK" --master_addr="$MASTER_ADDR" --master_port=29500 \
  -m stopwatch.train --config configs/train_policy_2b.yaml
```

Keep the total rank count when changing node layouts.

Use `--set dotted.key=value` to override recipe fields. Paths resolve under
`paths.root` (default `data`), except scene configs under `paths.scenes`.
Absolute paths are accepted. Evaluation checkpoints also accept Hugging Face model IDs.

## Checkpoints and resume

| Model | Output directory | Restart interval | Weight export interval |
|---|---|---:|---:|
| perception | `data/outputs/train_perception_4b/` | 600 updates | 5,000 updates |
| policy | `data/outputs/train_policy_2b/` | 1,000 updates | 5,000 updates |

Restart the same training command to resume the latest complete checkpoint in
`<output>/restart/step_<update>_<id>/`. To select a particular checkpoint,
append `--set checkpoint.resume_from=/absolute/path/to/restart/step_<update>_<id>`.

## Weight export

Evaluation weights are saved to `<output>/step_<update>/` every 5,000 updates and
`<output>/final/` at completion. Use `restart/` checkpoints to resume training.

For example, evaluate the final trained models on OVON with the downloaded
evaluation data and validation scenes.

```bash
hf download Amazon-FAR/stopwatch-data --repo-type dataset \
  --include 'eval/*' --include 'vocab/*' --include 'protocol/*' --include README.md --local-dir data
torchrun --standalone --nproc_per_node=1 -m stopwatch.evaluate \
  --config configs/eval_ovon_val_unseen.yaml \
  --set checkpoints.perception=outputs/train_perception_4b/final \
  --set checkpoints.policy=outputs/train_policy_2b/final \
  --set paths.output=outputs/eval_trained_ovon
```
