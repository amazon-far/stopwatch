"""TRL SFTTrainer integration and the two unweighted release objectives.

The objectives use the inherited Trainer update ordering.
"""

from __future__ import annotations

import logging
import math
import os
import random
from datetime import timedelta
from typing import Unpack

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import Dataset
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from transformers import TrainerCallback
from trl import SFTConfig, SFTTrainer

from stopwatch.config import TrainConfig
from stopwatch.data.batching import PackedBatch, ReferenceBinLoader
from stopwatch.data.vocab import Vocabulary
from stopwatch.model.parallel import (
    clip_grad_norm,
    compile_token_regions,
    install_parallel,
)
from stopwatch.model.qwen import PackedPrompt, QwenNTP
from stopwatch.train.checkpoint import AsyncCheckpoint, export_weights
from stopwatch.train.fsdp import restore_rng, shard_model

logger = logging.getLogger(__name__)


class CompletionLoss(nn.Module):
    def __init__(self, model: QwenNTP) -> None:
        super().__init__()
        self.model = model
        self.config = model.qwen.config

    def add_model_tags(self, tags: list[str]) -> None:
        self.model.qwen.add_model_tags(tags)

    def forward(self, **batch: Unpack[PackedBatch]) -> dict[str, Tensor]:
        """Return scalar loss, action_loss, semantic_loss and accuracy tensors.

        Perception uses a global target-token mean. Policy uses per-action
        coefficients 1/(bin frame count * 16), SUM reduced and world scaled.
        """
        prompt = {field: batch[f"prompt_{field}"] for field in PackedPrompt.__annotations__}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = self.model(batch["frames"], prompt=prompt)
            weight = self.model.lm_head_weight()
            if isinstance(output, dict):
                hidden = output["action_hidden"]
                actions = batch["actions"][batch["seq_lens"].cumsum(0) - 1]
                selection = output.get("sequence_parallel_action_selection")
                if selection is not None:
                    actions = actions[selection]
                action_loss = hidden.sum() * 0.0
                semantic_hidden = output["semantic_hidden"]
                targets = output["semantic_targets"].long()
                semantic_sum = _chunked_linear_ce_sum(weight, semantic_hidden, targets)
                denominator = targets.new_tensor(targets.numel(), dtype=torch.float32)
                dist.all_reduce(denominator)
                semantic_loss = semantic_sum * dist.get_world_size() / denominator.clamp_min(1.0)
                loss = action_loss * 0.0 + semantic_loss
            else:
                hidden, actions = output, batch["actions"]
                token_ce = chunked_linear_ce(weight, hidden, self.model.action_targets(actions))
                # Each coefficient retains its original bin denominator after
                # redistribution. FSDP averages over every physical rank once.
                coefficients = batch["action_loss_coefficients"].float()
                if token_ce.shape != coefficients.shape:
                    raise ValueError("Action coefficients do not match supervised sites")
                action_loss = (token_ce.float() * coefficients).sum() * dist.get_world_size()
                semantic_loss = action_loss.new_zeros(())
                loss = action_loss
        with torch.no_grad():
            predictions = self.model.action_logits(hidden).argmax(dim=-1)
            accuracy = (predictions == actions).float().mean() if actions.numel() else hidden.new_zeros(())
        return {"loss": loss, "action_loss": action_loss,
                "semantic_loss": semantic_loss, "accuracy": accuracy}


class Trainer(SFTTrainer):
    def __init__(self, config: TrainConfig, model: QwenNTP, loader: ReferenceBinLoader, checkpoints: AsyncCheckpoint) -> None:
        """Retain TRL scheduling while preventing a second distributed wrapper."""
        self._loader = loader
        self._initial_update = checkpoints._completed
        self._metric_sums = None
        self._metric_steps = 0
        remaining = config.schedule.total_updates - self._initial_update
        if remaining <= 0:
            raise ValueError("Trainer requires at least one remaining update")
        optimizer = checkpoints.optimizer
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"]
        scheduler = LambdaLR(optimizer, lambda update: learning_rate_factor(update, config),
                             last_epoch=self._initial_update - 1)
        args = SFTConfig(
            output_dir=str(config.paths.resolve(config.paths.output)),
            max_steps=remaining, per_device_train_batch_size=1,
            gradient_accumulation_steps=1, max_grad_norm=config.optimizer.max_grad_norm,
            gradient_checkpointing=False, bf16=False, fp16=False,
            remove_unused_columns=False, dataset_kwargs={"skip_prepare_dataset": True},
            max_length=None, packing=False, padding_free=False, use_liger_kernel=False,
            loss_type="nll", eval_strategy="no", save_strategy="no", logging_strategy="no",
            report_to=[], disable_tqdm=True, seed=config.data.seed, data_seed=config.data.seed,
            ddp_find_unused_parameters=False, ddp_broadcast_buffers=True, ddp_static_graph=False,
        )
        wrapped = CompletionLoss(model)
        super().__init__(
            model=wrapped, args=args, processing_class=model.processor,
            train_dataset=Dataset.from_dict({"input_ids": [[0]], "labels": [[0]]}),
            data_collator=_unused_collator, optimizers=(optimizer, scheduler),
            callbacks=[_Lifecycle(config, model, checkpoints)],
        )
        self.model_accepts_loss_kwargs = False
        # Register the manually sharded model so Trainer skips a second DDP
        # wrapper. This preparation mode does not call model.eval().
        self.accelerator.prepare_model(wrapped, device_placement=False, evaluation_mode=True)
        wrapped.train()

        def clip(parameters, max_norm, norm_type=2):
            if norm_type != 2:
                raise ValueError("Training uses L2 gradient clipping")
            self.accelerator.unscale_gradients()
            return clip_grad_norm(parameters, max_norm)

        self.accelerator.clip_grad_norm_ = clip

    def get_train_dataloader(self):
        # Already balanced and rank-sharded. Accelerate must not shard it again.
        return self._loader

    def _prepare_input(self, data):
        if isinstance(data, Tensor):
            return data.to(device=self.args.device, non_blocking=True)
        return super()._prepare_input(data)

    def _prepare_inputs(self, inputs):
        return {name: self._prepare_input(value) for name, value in inputs.items()}

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        del num_items_in_batch
        outputs = model(**inputs)
        metrics = torch.stack([outputs[name] for name in ("action_loss", "semantic_loss", "accuracy")]).detach().float()
        if self._metric_sums is None:
            self._metric_sums = torch.zeros_like(metrics)
        self._metric_sums += metrics
        self._metric_steps += 1
        return (outputs["loss"], outputs) if return_outputs else outputs["loss"]

    def log(self, logs, start_time=None):
        if "loss" in logs and self._metric_steps:
            metrics = torch.cat([self._metric_sums,
                                 self._metric_sums.new_tensor([float(self._metric_steps)])])
            dist.all_reduce(metrics)
            values = metrics[:-1] / metrics[-1].clamp_min(1.0)
            logs.update(zip(("action_loss", "semantic_loss", "accuracy"), values.tolist()))
            self._metric_sums.zero_()
            self._metric_steps = 0
        logs["update"] = self._initial_update + self.state.global_step
        if dist.get_rank() == 0:
            logger.info("Training %s", logs)
        return super().log(logs, start_time)


class _Lifecycle(TrainerCallback):
    def __init__(self, config, model, checkpoints):
        self.config, self.model, self.checkpoints = config, model, checkpoints
        self.initial_update = checkpoints._completed

    def on_train_begin(self, args, state, control, **kwargs):
        # Trainer initialization seeds RNG again. Restore immediately before
        # consuming the first resumed batch.
        if self.checkpoints._restored_rng is not None:
            restore_rng(self.checkpoints._restored_rng)
        return control

    def on_step_end(self, args, state, control, **kwargs):
        update = self.initial_update + state.global_step
        checkpoint = self.config.checkpoint
        if self.checkpoints._pending is not None and self.checkpoints._pending.done():
            self.checkpoints.wait()
        if update % checkpoint.restart_every_updates == 0:
            self.checkpoints.save(update)
        if update % checkpoint.export_every_updates == 0:
            self.checkpoints.wait()
            export_weights(self.model, self.config.paths.resolve(self.config.paths.output) / f"step_{update:06d}")
        if update % self.config.log_interval == 0:
            control.should_log = True
        return control


def _unused_collator(_features):
    raise RuntimeError("Training uses the prebuilt reference-bin loader")


def chunked_linear_ce(weight: Tensor, hidden: Tensor, targets: Tensor) -> Tensor:
    """Return fp32 [K] full-vocabulary token losses without full-batch logits."""
    losses = []
    for start in range(0, targets.numel(), 128):
        losses.append(F.cross_entropy(F.linear(hidden[start:start + 128], weight),
                                      targets[start:start + 128], reduction="none").float())
    if not losses:
        return hidden.sum(dim=-1).float() + weight.reshape(-1)[0].float() * 0.0
    return torch.cat(losses)


def _chunked_linear_ce_sum(weight: Tensor, hidden: Tensor, targets: Tensor) -> Tensor:
    if not targets.numel():
        # Empty SP label shards still participate in the output-head backward.
        return hidden.sum() * 0.0 + weight.reshape(-1)[0] * 0.0
    loss = hidden.new_zeros((), dtype=torch.float32)
    for start in range(0, targets.numel(), 128):
        loss = loss + F.cross_entropy(F.linear(hidden[start:start + 128], weight),
                                      targets[start:start + 128], reduction="sum").float()
    return loss


def make_optimizer(model: QwenNTP, config: TrainConfig) -> Optimizer:
    """Fused AdamW, betas (0.9,0.999), with the model's parameter groups."""
    groups = model.optimizer_param_groups(config.optimizer)
    if not groups:
        raise ValueError("Model has no trainable parameters")
    for group in groups:
        group["initial_lr"] = group["lr"]
    return torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=config.optimizer.eps,
                             weight_decay=config.optimizer.weight_decay, fused=True)


def learning_rate_factor(update: int, config: TrainConfig) -> float:
    """Linear warmup followed by cosine decay to the configured minimum ratio."""
    schedule = config.schedule
    step = max(0, int(update))
    if schedule.warmup_updates and step < schedule.warmup_updates:
        return step / schedule.warmup_updates
    progress = min(1.0, max(0.0, (step - schedule.warmup_updates) /
                            (schedule.total_updates - schedule.warmup_updates)))
    return schedule.min_lr_ratio + (1.0 - schedule.min_lr_ratio) * 0.5 * (
        1.0 + math.cos(math.pi * progress)
    )


def _initialize(config: TrainConfig) -> tuple[int, torch.device]:
    rank, world, local_rank = (int(os.environ[name]) for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
    if world != config.parallel.world_size:
        raise ValueError(
            f"{config.model.role} training requires {config.parallel.world_size} physical ranks "
            f"for its reference-bin update membership; got {world}"
        )
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl", device_id=device, timeout=timedelta(hours=2))
    seed = config.data.seed + rank
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return rank, device


def train(config: TrainConfig) -> None:
    """Initialize, restore a complete restart if present, train and export."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    rank, device = _initialize(config)
    checkpoints = None
    try:
        vocabulary = Vocabulary(config.paths.root.expanduser())
        model = QwenNTP(config.model, vocabulary).to(device)
        prompt_builder = model.make_prompt_builder(config.data.sources[0].name)
        loader = ReferenceBinLoader(config, vocabulary, prompt_builder, model, rank=rank)
        context = install_parallel(model, config.parallel.sequence_parallel_size)
        compile_token_regions(model)
        shard_model(model, context=context)
        optimizer = make_optimizer(model, config)
        checkpoints = AsyncCheckpoint(config, model, optimizer, loader, vocabulary)
        completed = checkpoints.restore()
        if completed < config.schedule.total_updates:
            trainer = Trainer(config, model, loader, checkpoints)
            # The inherited Trainer loop owns backward, clipping, optimizer,
            # scheduler, zero_grad and callbacks, in that order.
            trainer.train()
            completed += trainer.state.global_step
        if completed != config.schedule.total_updates:
            raise RuntimeError(f"Training stopped at {completed} updates")
        checkpoints.save(completed)
        checkpoints.wait()
        export_weights(model, config.paths.resolve(config.paths.output) / "final")
    finally:
        if checkpoints is not None:
            try:
                checkpoints.wait()
            finally:
                checkpoints._executor.shutdown()
                dist.destroy_process_group(checkpoints._group)
        dist.destroy_process_group()
