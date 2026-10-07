"""Entrypoint for torchrun -m stopwatch.evaluate."""

import argparse
import hashlib
import json
import os
import random
import time
from collections import defaultdict
from collections.abc import Sequence
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

from stopwatch.config import EvalConfig, load_config


def evaluate(config: EvalConfig, *, limit: int | None = None) -> None:
    """Load both bf16 models, resume unfinished episodes and write final scores."""
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    # Set process controls before constructing models or initializing CUDA.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    import numpy as np
    import torch

    from stopwatch.data.vocab import Vocabulary
    from stopwatch.eval.dispatch import (
        SceneStealingDispatcher,
        completed_episodes,
        create_store,
    )
    from stopwatch.eval.env import VectorEnv, load_episodes
    from stopwatch.eval.geometry import _load_goal_sets, _scene_name
    from stopwatch.eval.loop import run_rollouts
    from stopwatch.eval.perception import PerceptionProvider
    from stopwatch.eval.score import score_records
    from stopwatch.model.qwen import QwenNTP
    from stopwatch.model.rollout import Rollout

    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_world = int(os.environ["LOCAL_WORLD_SIZE"])
    if not 0 <= local_rank < local_world <= torch.cuda.device_count():
        raise ValueError("Evaluation requires one visible GPU per local rank")
    torch.cuda.set_device(local_rank)
    target = torch.device("cuda", local_rank)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    store = create_store(host_name=os.environ["MASTER_ADDR"], port=int(os.environ["MASTER_PORT"]) + 100,
                         world_size=world, is_master=rank == 0, timeout=timedelta(hours=24))
    output = config.paths.resolve(config.paths.output)
    output.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        store.set("invocation", uuid4().hex)
        # Freeze one resume snapshot for every rank before this invocation writes.
        store.set("completed_snapshot", json.dumps(sorted(completed_episodes(output))))
    invocation = store.get("invocation").decode()
    complete = {tuple(row) for row in json.loads(store.get("completed_snapshot"))}
    episodes = load_episodes(config)
    membership = _load_goal_sets(config.paths.root)["scoring_class_ids"][config.dataset.benchmark]
    if any(episode.object_category not in membership for episode in episodes):
        raise ValueError("Missing scoring membership for an episode goal")
    all_ids = {(_scene_name(ep.scene_id), str(ep.episode_id)) for ep in episodes}
    if not complete <= all_ids:
        raise ValueError("Output contains completed episodes outside this benchmark")
    episodes = episodes[:limit]
    partial = len(episodes) < len(all_ids)
    if rank == 0 and partial:
        print(f"Partial run, first {len(episodes)} of {len(all_ids)} episodes in dataset order", flush=True)
    grouped = defaultdict(list)
    for index, episode in enumerate(episodes):
        if (_scene_name(episode.scene_id), str(episode.episode_id)) not in complete:
            grouped[episode.scene_id].append(index)
    blocks = list(grouped.values())
    digest = hashlib.sha256(json.dumps(blocks, separators=(",", ":")).encode()).hexdigest()
    if rank == 0:
        store.set("dispatch_digest", digest)
    elif store.get("dispatch_digest").decode() != digest:
        raise ValueError("Ranks disagree on the frozen episode dispatch order")
    envs = None
    try:
        if blocks:
            vocabulary = Vocabulary(config.paths.root)
            perception_model = QwenNTP(config.perception, vocabulary,
                                       checkpoint=config.checkpoints.perception, data_root=config.paths.root).to(target).eval()
            policy_model = QwenNTP(config.policy, vocabulary,
                                   checkpoint=config.checkpoints.policy, data_root=config.paths.root).to(target).eval()
            perception_rollout = Rollout(perception_model, num_environments=config.num_environments,
                                          max_new_tokens=config.perception_max_new_tokens)
            policy = Rollout(policy_model, num_environments=config.num_environments)
            provider = PerceptionProvider(perception_rollout, vocabulary, config, target=target)
            dispatcher = SceneStealingDispatcher(store, blocks)
            envs = VectorEnv(config, episodes, gpu_id=local_rank)
            run_rollouts(config, envs, provider, policy, dispatcher,
                         records_path=output / f"rank{rank}-{invocation}.jsonl")
            envs.close()
            envs = None
            del provider, perception_rollout, policy, perception_model, policy_model
            torch.cuda.empty_cache()
        store.set(f"finished_{rank}", "1")
        if rank == 0:
            while not store.check([f"finished_{i}" for i in range(world)]):
                if store.check(["fatal_error"]):
                    raise RuntimeError(store.get("fatal_error").decode())
                time.sleep(0.5)
            metrics = {**score_records(config, output, limit=limit), "partial_run": partial}
            temporary = output / "metrics.json.tmp"
            temporary.write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
            temporary.replace(output / "metrics.json")
            print(f"Eval {config.dataset.split}: {json.dumps(metrics)}", flush=True)
            store.set("scoring_finished", "1")
        else:
            while not store.check(["scoring_finished"]):
                if store.check(["fatal_error"]):
                    raise RuntimeError(store.get("fatal_error").decode())
                time.sleep(0.5)
        # Keep rank zero's store alive until peers acknowledge their final read.
        store.set(f"ack_{rank}", "1")
        if rank == 0:
            store.wait([f"ack_{i}" for i in range(world)])
    except BaseException as error:
        store.set("fatal_error", f"Evaluation rank {rank} failed with {type(error).__name__}: {error}")
        raise
    finally:
        if envs is not None:
            envs.close()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--limit", type=int, help="Evaluate only the first N episodes in dataset order")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    config = load_config(args.config, args.overrides)
    if not isinstance(config, EvalConfig):
        raise TypeError("Evaluation requires an evaluation recipe")
    evaluate(config, limit=args.limit)


if __name__ == "__main__":
    main()
