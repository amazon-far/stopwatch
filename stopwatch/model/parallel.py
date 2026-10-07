"""Deterministic FlashAttention 4, Ulysses exchanges and per-token compilation."""

from __future__ import annotations

import inspect
from collections.abc import Iterable
from functools import wraps
from itertools import pairwise
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributed import ProcessGroup
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import DTensor

if TYPE_CHECKING:
    from stopwatch.model.qwen import QwenNTP


class SequenceParallelContext:
    size: int
    sp_rank: int
    sp_group: ProcessGroup
    mesh: DeviceMesh

    def __init__(self, size: int) -> None:
        """Create sequence groups of two ranks across the distributed world."""
        world = dist.get_world_size()
        if size != 2 or world % size:
            raise ValueError("Sequence parallelism requires groups of two dividing the world size")
        self.size = size
        self.mesh = init_device_mesh("cuda", (world // size, size), mesh_dim_names=("dp", "sp"))
        self.sp_group = self.mesh["sp"].get_group()
        self.sp_rank = dist.get_rank(self.sp_group)

    def local_sequence(self, value: Tensor, *, dim: int = -1, fill: float = 0) -> Tensor:
        """Slice the packed token axis and pad its final shard."""
        result = value.narrow(dim, min(self.start, self.tokens), max(0, min(self.stop, self.tokens) - self.start))
        missing = self.chunk - result.shape[dim]
        if missing:
            shape = list(result.shape)
            shape[dim] = missing
            result = torch.cat((result, result.new_full(shape, fill)), dim=dim)
        return result.contiguous()

    def shard_loss_mask(self, mask: Tensor) -> tuple[Tensor, Tensor]:
        """Return local bool mask and bool selection over global supervised sites."""
        selected = torch.arange(self.tokens, device=mask.device)[mask.reshape(-1)]
        keep = (selected >= self.start) & (selected < self.stop)
        return self.local_sequence(mask, fill=False), keep

    def _shard_language(self, module, args, kwargs):
        embeddings, positions = kwargs["inputs_embeds"], kwargs["position_ids"]
        if embeddings.shape[0] != 1 or kwargs.get("use_cache") or kwargs.get("past_key_values") is not None:
            raise ValueError("Sequence parallelism requires uncached packed training")
        self.tokens = embeddings.shape[1]
        self.chunk = (self.tokens + self.size - 1) // self.size
        self.start, self.stop = self.sp_rank * self.chunk, (self.sp_rank + 1) * self.chunk
        if int(kwargs["cu_seq_lens_q"][-1]) != self.tokens:
            raise ValueError("Cumulative lengths do not cover the packed input")
        self.full_positions = positions[1:] if positions.shape[0] == 4 else positions
        kwargs["inputs_embeds"] = self.local_sequence(embeddings, dim=1)
        kwargs["position_ids"] = self.local_sequence(positions)
        mask = kwargs.get("visual_pos_masks")
        if mask is not None:
            local_mask, selection = self.shard_loss_mask(mask)
            kwargs["visual_pos_masks"] = local_mask
            if kwargs.get("deepstack_visual_embeds") is not None:
                kwargs["deepstack_visual_embeds"] = [x[selection] for x in kwargs["deepstack_visual_embeds"]]
        return args, kwargs

    def _global_rotary(self, module, args):
        return args[0], self.full_positions

    def _local_rotary(self, module, args, output):
        return tuple(self.local_sequence(value, dim=1) for value in output)


def install_parallel(model: QwenNTP, sequence_parallel_size: int) -> SequenceParallelContext | None:
    """Install deterministic FA4, SP vision partitioning and decoder exchanges."""
    _deterministic_fa4()
    if sequence_parallel_size == 1:
        return None
    from transformers.masking_utils import AttentionMaskInterface, flash_attention_mask
    from transformers.modeling_utils import AttentionInterface

    context = SequenceParallelContext(sequence_parallel_size)
    language = model.qwen.model.language_model
    if language.config.num_attention_heads % context.size or language.config.num_key_value_heads % context.size:
        raise ValueError("Query and KV heads must divide by sequence parallel size")
    AttentionInterface.register("stopwatch_ulysses_fa4", _attention)
    AttentionMaskInterface.register("stopwatch_ulysses_fa4", flash_attention_mask)
    language.config._attn_implementation = "stopwatch_ulysses_fa4"
    language.register_forward_pre_hook(context._shard_language, with_kwargs=True)
    language.rotary_emb.register_forward_pre_hook(context._global_rotary)
    language.rotary_emb.register_forward_hook(context._local_rotary)
    for layer in language.layers:
        layer.self_attn._sequence_parallel = context
    model._sequence_parallel = context
    _partition_vision(model, context)
    return context


def compile_token_regions(model: QwenNTP) -> None:
    """Compile norm/adjacent-linear and MLP regions with dynamic token counts."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb

    def compile_layer(layer):
        attention = layer.self_attn

        def projections(hidden):
            hidden = hidden.unsqueeze(0)
            normalized = layer.input_layernorm(hidden)
            shape = (*hidden.shape[:-1], -1, attention.head_dim)
            query = attention.q_norm(attention.q_proj(normalized).view(shape)).transpose(1, 2)
            key = attention.k_norm(attention.k_proj(normalized).view(shape)).transpose(1, 2)
            value = attention.v_proj(normalized).view(shape).transpose(1, 2)
            return query, key, value

        def output_and_mlp(residual, attended):
            residual, attended = residual.unsqueeze(0), attended.unsqueeze(0)
            hidden = attention.o_proj(attended) + residual
            return layer.mlp(layer.post_attention_layernorm(hidden)) + hidden

        project = torch.compile(projections, fullgraph=True, options={"max_autotune": False})
        finish = torch.compile(output_and_mlp, fullgraph=True, options={"max_autotune": False})

        @wraps(layer.forward)
        def forward(hidden_states, position_embeddings, attention_mask=None, position_ids=None,
                    past_key_values=None, use_cache=False, **kwargs):
            if past_key_values is not None or use_cache:
                raise ValueError("Compiled training regions do not accept a decode cache")
            hidden = hidden_states.reshape(-1, hidden_states.shape[-1]).contiguous()
            torch._dynamo.mark_dynamic(hidden, 0)
            query, key, value = project(hidden)
            query, key = apply_rotary_pos_emb(query, key, *position_embeddings)
            attended, _ = ALL_ATTENTION_FUNCTIONS[attention.config._attn_implementation](
                attention, query, key, value, attention_mask,
                dropout=attention.attention_dropout if attention.training else 0.0,
                scaling=attention.scaling, position_ids=position_ids, **kwargs,
            )
            attended = attended.reshape(hidden.shape[0], -1).contiguous()
            torch._dynamo.mark_dynamic(attended, 0)
            return finish(hidden, attended)
        return forward

    for layer in model.qwen.model.language_model.layers:
        layer.forward = compile_layer(layer)


def sequence_to_head(value: Tensor, group: ProcessGroup) -> Tensor:
    """Autograd exchange [L,H,D] to [L*SP,H/SP,D]."""
    return _SequenceToHead.apply(value, group)


def head_to_sequence(value: Tensor, group: ProcessGroup) -> Tensor:
    """Autograd inverse of sequence_to_head."""
    return _HeadToSequence.apply(value, group)


def clip_grad_norm(parameters: Iterable[nn.Parameter], max_norm: float) -> Tensor:
    """Return the fp32 pre-clip norm with full-world FSDP/SP accounting."""
    gradients = [p.grad for p in parameters if p.grad is not None]
    local = [g.to_local() if isinstance(g, DTensor) else g for g in gradients]
    squared = torch.zeros((), dtype=torch.float64, device=torch.device("cuda", torch.cuda.current_device()))
    for value in local:
        squared += torch.linalg.vector_norm(value, dtype=torch.float64).square()
    # Full-world FSDP owns a distinct gradient shard on every physical rank.
    dist.all_reduce(squared)
    norm = squared.sqrt().float()
    factor = (float(max_norm) / (norm + 1e-6)).clamp(max=1.0)
    with torch.no_grad():
        torch._foreach_mul_(local, factor)
    return norm


def _exchange_to_head(value, group):
    size = dist.get_world_size(group)
    length, heads, width = value.shape
    if heads % size:
        raise ValueError("Attention heads do not divide by sequence parallel size")
    send = value.reshape(length, size, heads // size, width).permute(1, 0, 2, 3).contiguous()
    receive = torch.empty_like(send)
    dist.all_to_all_single(receive, send, group=group)
    return receive.reshape(length * size, heads // size, width)


def _exchange_to_sequence(value, group):
    size = dist.get_world_size(group)
    length, heads, width = value.shape
    send = value.reshape(size, length // size, heads, width).contiguous()
    receive = torch.empty_like(send)
    dist.all_to_all_single(receive, send, group=group)
    return receive.permute(1, 0, 2, 3).reshape(length // size, heads * size, width)


class _SequenceToHead(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, group):
        ctx.group = group
        return _exchange_to_head(value, group)

    @staticmethod
    def backward(ctx, gradient):
        return _exchange_to_sequence(gradient, ctx.group), None


class _HeadToSequence(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, group):
        ctx.group = group
        return _exchange_to_sequence(value, group)

    @staticmethod
    def backward(ctx, gradient):
        return _exchange_to_head(gradient, ctx.group), None


def _attention(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kwargs):
    from flash_attn.cute import flash_attn_varlen_func
    from transformers.integrations.flash_attention import get_target_dtype

    context = module._sequence_parallel
    if dropout or query.shape[0] != 1:
        raise ValueError("Ulysses requires packed deterministic attention")
    dtype = get_target_dtype(query, module)
    tensors = [x.transpose(1, 2).squeeze(0) for x in (query, key, value)]
    if dtype is not None:
        tensors = [x.to(dtype) for x in tensors]
    q, k, v = [sequence_to_head(x, context.sp_group)[:context.tokens].contiguous() for x in tensors]
    output = flash_attn_varlen_func(
        q, k, v, cu_seqlens_q=kwargs["cu_seq_lens_q"], cu_seqlens_k=kwargs["cu_seq_lens_k"],
        max_seqlen_q=kwargs["max_length_q"], max_seqlen_k=kwargs["max_length_k"],
        softmax_scale=scaling, causal=True, num_splits=1,
    )
    if isinstance(output, tuple):
        output = output[0]
    output = F.pad(output, (0, 0, 0, 0, 0, context.chunk * context.size - context.tokens))
    return head_to_sequence(output, context.sp_group).unsqueeze(0), None


def _deterministic_fa4():
    import flash_attn.cute as attention
    from flash_attn.cute import interface

    def wrap(function, backward=False):
        signature = inspect.signature(function)

        @wraps(function)
        def call(*args, **kwargs):
            bound = signature.bind_partial(*args, **kwargs)
            if not backward:
                window = bound.arguments.get("window_size", (None, None))
                if window == (-1, 0):
                    raise ValueError("FA4 causal windows require (None, 0)")
                if bound.arguments.get("causal", False) and window in ((None, None), (-1, -1)):
                    bound.arguments["window_size"] = (None, 0)
            if "deterministic" in signature.parameters:
                bound.arguments["deterministic"] = True
            return function(*bound.args, **bound.kwargs)
        call._stopwatch_deterministic = True
        return call

    if not getattr(interface._flash_attn_bwd, "_stopwatch_deterministic", False):
        if "deterministic" not in inspect.signature(interface._flash_attn_bwd).parameters:
            raise RuntimeError("FA4 backward lacks deterministic support")
        interface._flash_attn_bwd = wrap(interface._flash_attn_bwd, backward=True)
    for name in ("flash_attn_func", "flash_attn_varlen_func"):
        function = getattr(attention, name)
        if not getattr(function, "_stopwatch_deterministic", False):
            setattr(attention, name, wrap(function))


class _GatherVision(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, counts, group):
        ctx.counts, ctx.group, ctx.dtype = counts, group, value.dtype
        ctx.rank = dist.get_rank(group)
        cap = max(counts)
        padded = F.pad(value, (0, 0, 0, 0, 0, cap - value.shape[0])).contiguous()
        output = value.new_empty((cap * len(counts), *value.shape[1:]))
        dist.all_gather_into_tensor(output, padded, group=group)
        return torch.cat([part[:count] for part, count in zip(output.split(cap), counts, strict=True)])

    @staticmethod
    def backward(ctx, gradient):
        cap = max(ctx.counts)
        pieces = [F.pad(part.float(), (0, 0, 0, 0, 0, cap - count))
                  for part, count in zip(gradient.split(ctx.counts), ctx.counts, strict=True)]
        output = gradient.new_empty((cap, *gradient.shape[1:]), dtype=torch.float32)
        dist.reduce_scatter_tensor(output, torch.cat(pieces).contiguous(), op=dist.ReduceOp.SUM, group=ctx.group)
        return output[:ctx.counts[ctx.rank]].to(ctx.dtype), None, None


def _partition_vision(model, context):
    visual = model.qwen.model.visual
    original = visual.forward

    def prepare(module, args, kwargs):
        pixels = kwargs["image_grid_thw"].prod(-1).tolist()
        rows = [count // visual.spatial_merge_size ** 2 for count in pixels]
        types = kwargs["mm_token_type_ids"].reshape(-1)
        positions = types.nonzero().flatten()
        if positions.numel() != sum(rows):
            raise ValueError("Vision ownership differs from image grids")
        ends = torch.tensor(rows, device=positions.device).cumsum(0)
        centers = (positions[ends - 1] + positions[ends - torch.tensor(rows, device=ends.device)]) / 2
        chunk = (types.numel() + context.size - 1) // context.size
        cuts = [0, *(int((centers < chunk * i).sum()) for i in range(1, context.size)), len(rows)]
        context.vision_plan = pixels, rows, cuts

    def forward(hidden_states, grid_thw, **kwargs):
        pixels, rows, cuts = context.vision_plan
        lo, hi = cuts[context.sp_rank:context.sp_rank + 2]
        counts = tuple(sum(rows[a:b]) for a, b in pairwise(cuts))
        first, last = (lo, hi) if hi > lo else (0, 1)
        output = original(hidden_states[sum(pixels[:first]):sum(pixels[:last])], grid_thw[first:last], **kwargs)
        features = torch.stack([output.pooler_output, *output.deepstack_features], dim=1)
        if hi == lo:
            features = features[:0]
        gathered = _GatherVision.apply(features, counts, context.sp_group)
        output.pooler_output = gathered[:, 0]
        output.deepstack_features = list(gathered[:, 1:].unbind(1))
        output.last_hidden_state = None
        return output

    model.qwen.model.register_forward_pre_hook(prepare, with_kwargs=True)
    visual.forward = forward
