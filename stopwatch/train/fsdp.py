"""Full-world FSDP2 sharding, state gathering and per-rank RNG capture."""

from __future__ import annotations

import random
from typing import TypedDict

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor

from stopwatch.model.parallel import SequenceParallelContext
from stopwatch.model.qwen import QwenNTP


class RNGState(TypedDict):
    python: tuple
    numpy: tuple
    torch: Tensor  # CPU uint8 generator state
    cuda: Tensor  # CPU uint8 state for this rank's CUDA device


def shard_model(model: QwenNTP, *, context: SequenceParallelContext | None) -> None:
    """Shard after loading and loader construction, before optimizer creation.

    Masters and reductions are fp32, gathered parameters bf16. Shard the
    decoder layers and then the root across the full physical world.
    """
    if not dist.is_initialized():
        raise ValueError("Sharding requires an initialized process group")
    if isinstance(model, FSDPModule):
        raise TypeError("Model is already sharded")
    for parameter in model.parameters():
        if parameter.requires_grad and parameter.dtype != torch.float32:
            raise ValueError("Training requires fp32 master parameters")
    for tensor in list(model.parameters()) + list(model.buffers()):
        dist.broadcast(tensor.detach(), src=0)
    frozen = {parameter for parameter in model.parameters() if not parameter.requires_grad}
    mesh = {} if context is None else {
        "mesh": context.mesh._flatten(mesh_dim_name="dp_shard_sp"),
    }
    precision = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=False,
    )
    for layer in model.qwen.model.language_model.layers:
        fully_shard(layer, mp_policy=precision, ignored_params=frozen,
                    reshard_after_forward=True, **mesh)
    fully_shard(model.qwen.model.visual, mp_policy=precision, ignored_params=frozen,
                reshard_after_forward=True, **mesh)
    # The loss reads the output projection after forward. Keep it gathered
    # until backward, while retaining full-world parameter and optimizer shards.
    fully_shard(model, mp_policy=precision, ignored_params=frozen,
                reshard_after_forward=False, **mesh)


def full_model_state(model: QwenNTP) -> dict[str, Tensor] | None:
    """Collective gather; only rank zero retains full CPU weights."""
    for module in model.modules():
        if isinstance(module, FSDPModule):
            module.reshard()
    rank = dist.get_rank()
    result = {} if rank == 0 else None
    for name, value in model.state_dict().items():
        full = value.full_tensor() if isinstance(value, DTensor) else value
        if rank == 0:
            result[name] = full.detach().cpu()
    return result


def rng_state() -> RNGState:
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(),
    }


def restore_rng(state: RNGState) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    torch.cuda.set_rng_state(state["cuda"].cpu())
