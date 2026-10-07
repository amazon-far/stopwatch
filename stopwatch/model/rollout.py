"""Perception full prefill of the latest 128 frames and cached policy decode.

Slots receive resolved goals and advance only when active.
"""

from __future__ import annotations

import math
import weakref
from collections.abc import Sequence
from contextlib import contextmanager
from types import SimpleNamespace
from typing import NotRequired, TypedDict

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers.cache_utils import Cache, CacheLayerMixin

from stopwatch.model.parallel import _deterministic_fa4
from stopwatch.model.qwen import (
    PromptBuilder,
    QwenNTP,
    _decoder_kwargs,
    _SemTextTrie,
    patchify_images,
)


class ObservationBatch(TypedDict):
    rgb: Tensor  # uint8 [B,360,640,3]
    goal_name: list[str]  # exact resolved prompt goal for each slot
    goal_class_ids: list[frozenset[int]]  # accepted zero-based semantic IDs
    semantic_bow_ids: NotRequired[Tensor]  # policy int64 [B,16], one-based
    goal_distance: NotRequired[Tensor]  # policy float32 [B,1], NaN if absent


class Rollout:
    """Own slot-local history, vision features and key/value caches for one model."""

    predictions: list[str]

    def __init__(self, model: QwenNTP, *, num_environments: int, max_new_tokens: int = 192) -> None:
        from transformers.masking_utils import (
            AttentionMaskInterface,
            flash_attention_mask,
        )
        from transformers.modeling_utils import AttentionInterface

        self.model, self.num_environments, self.max_new_tokens = model, num_environments, max_new_tokens
        self.model.eval()
        if next(model.qwen.parameters()).dtype != torch.bfloat16:
            raise ValueError("Rollout requires bf16 release weights")
        if model._sequence_parallel is not None:
            raise ValueError("Rollout cannot use a sequence-parallel training model")
        self.builder = PromptBuilder.__new__(PromptBuilder)
        self.builder._initialize(model.config, model.vocabulary, None, model.processor)
        self.language = model.qwen.model.language_model
        _deterministic_fa4()
        AttentionInterface.register("stopwatch_rollout_fa4", _rollout_attention)
        AttentionMaskInterface.register("stopwatch_rollout_fa4", flash_attention_mask)
        self.language.config._attn_implementation = "stopwatch_rollout_fa4"
        self.trie = _SemTextTrie(model.processor.tokenizer, model.vocabulary.categories,
                                self.builder.distance_labels) if model.config.role == "perception" else None
        self.reset()

    @torch.no_grad()
    def step(self, observations: ObservationBatch, prev_actions: Tensor, masks: Tensor, *, active_indices: Sequence[int]) -> Tensor:
        """Return float32 [B,4] action logits; perception also fills predictions.

        prev_actions is int64 [B] or [B,1]. masks is float32 [B,1]; zero resets
        every history/cache for that slot before incorporating its RGB frame.
        Only active_indices advance or reset. Inactive slots retain their
        histories, caches and previous outputs in the full B-slot batch.
        Perception logits are ignored by the environment loop.
        """
        active = list(active_indices)
        if len(set(active)) != len(active) or any(i < 0 or i >= self.num_environments for i in active):
            raise ValueError("Invalid active rollout slots")
        rgb = observations["rgb"]
        if len(rgb) != self.num_environments or len(observations["goal_name"]) != len(rgb):
            raise ValueError("Rollout observations must retain all environment slots")
        if self._logits is None:
            self._logits = torch.zeros(len(rgb), 4, dtype=torch.float32, device=rgb.device)
        for index in active:
            slot = self._slots[index]
            if not bool(masks.reshape(-1)[index]):
                slot = self._slots[index] = _slot()
                self.predictions[index] = ""
                if self._policy_store is not None:
                    self._policy_store.lengths[index] = 0
            elif slot.frames:
                slot.actions.append(int(prev_actions.reshape(-1)[index]))
            frame = rgb[index].detach().clone()
            if self.model.config.role == "perception":
                slot.frames.append(frame)
                if len(slot.frames) > self.model.config.frame_window:
                    slot.frames.pop(0)
                    slot.actions.pop(0)
            else:
                slot.frames[:] = [frame]
                ids = observations["semantic_bow_ids"][index].detach().cpu()
                visible = any(int(value) - 1 in observations["goal_class_ids"][index] for value in ids if int(value))
                distance = float(observations["goal_distance"].reshape(-1)[index])
                slot.semantic.append(ids)
                slot.visible.append(visible)
                slot.distances.append(distance if visible and math.isfinite(distance) else None)
        if not active:
            return self._logits.clone()
        if self.model.config.role == "perception":
            logits, texts = self._perception(
                [observations["goal_name"][i] for i in active],
                [self._slots[i].frames for i in active], [self._slots[i].actions for i in active],
            )
            self._logits[active] = logits
            for index, text in zip(active, texts, strict=True):
                self.predictions[index] = text
        else:
            self._policy(observations["goal_name"], active)
        self._vision.prune([frame for slot in self._slots for frame in slot.frames])
        return self._logits.clone()

    def reset(self) -> None:
        """Clear all slot histories, grammar state, predictions and caches."""
        self._slots = [_slot() for _ in range(self.num_environments)]
        self.predictions = [""] * self.num_environments
        self._logits = None
        self._policy_store = self._window_store = self._window_graph = None
        self._vision = _VisionCache()

    def _prompt(self, goal, actions, semantic=None, visible=None, distances=None):
        if len(actions) + 1 > self.model.config.frame_window:
            raise ValueError("Rollout exceeded the model frame window")
        messages = self.builder._history(goal, [*actions, 0], semantic, visible, distances,
                                         self.builder.semantic_names)
        raw = self.builder._render(messages, generate=True)
        if raw[-len(self.builder.header_ids):] != self.builder.header_ids:
            raise RuntimeError("Rollout assistant header is not token-stable")
        return self.builder._expand(raw)[0]

    def _policy(self, goals, active):
        backbone, builder = self.model.qwen.model, self.builder
        device = self._slots[active[0]].frames[0].device
        if self._policy_store is None:
            names = sorted(builder.semantic_names, key=lambda n: len(builder._encode(" | " + n)), reverse=True)[:16]
            distance = max(builder.distance_labels, key=lambda n: len(builder._encode(n)))
            text = "Objects: " + " | ".join(names) + "\nGoal visible: false\nDistance: " + distance
            per_step = builder.visual_tokens_per_frame + 32 + len(builder._encode(text)) + 8
            capacity = (self.model.config.frame_window + 1) * per_step + self.max_new_tokens + 512
            self._policy_store = _Store(self.num_environments, capacity, len(self.language.layers))
        store = self._policy_store
        groups = {}
        for index in active:
            slot = self._slots[index]
            prompt = self._prompt(goals[index], slot.actions, slot.semantic, slot.visible, slot.distances)
            stable = prompt[:-len(builder.header_ids)]
            boundary = len(slot.stable)
            if stable[:boundary] != slot.stable:
                raise RuntimeError("Policy prompt history is not append-only")
            chunk = prompt[boundary:]
            plan = SimpleNamespace(index=index, chunk=chunk, stable=stable,
                stable_length=len(stable) - boundary, offset=boundary, mrope=slot.mrope)
            groups.setdefault((len(chunk), plan.stable_length), []).append(plan)
        for plans in groups.values():
            ids = torch.tensor([p.chunk for p in plans], dtype=torch.long, device=device)
            pixels, grid = patchify_images(torch.stack([self._slots[p.index].frames[-1] for p in plans]), self.model.config)
            types = ids.eq(self.model.qwen.config.image_token_id).int()
            local_positions, _ = backbone.get_rope_index(ids, mm_token_type_ids=types,
                image_grid_thw=grid, video_grid_thw=None, attention_mask=torch.ones_like(ids))
            positions = torch.cat((
                torch.tensor([p.offset for p in plans], device=device).view(1, -1, 1)
                + torch.arange(ids.shape[1], device=device).view(1, 1, -1),
                local_positions + torch.tensor([p.mrope for p in plans], device=device).view(1, -1, 1),
            ))
            if len(plans) == 1:
                plan = plans[0]
                cache = Cache(layers=[_RowWrite(store, i, plan.index) for i in range(store.num_layers)])
                mask = torch.ones(1, plan.offset + len(plan.chunk), dtype=torch.long, device=device)
                context = None
            else:
                cache = Cache(layers=[_BatchWrite(store, i, [p.index for p in plans], [p.offset for p in plans])
                                      for i in range(store.num_layers)])
                context = SimpleNamespace(capacity=store.capacity,
                    page_table=torch.tensor([p.index for p in plans], dtype=torch.int32, device=device).unsqueeze(1),
                    seqused=torch.tensor([p.offset + len(p.chunk) for p in plans], dtype=torch.int32, device=device),
                    cu_seqlens_q=torch.arange(0, (len(plans) + 1) * ids.shape[1], ids.shape[1], dtype=torch.int32, device=device),
                    max_seqlen_q=ids.shape[1])
                mask = None
            with _ragged(self.language, context):
                output = backbone(input_ids=None, inputs_embeds=self.model.qwen.get_input_embeddings()(ids),
                    position_ids=positions, attention_mask=mask, mm_token_type_ids=types,
                    pixel_values=pixels, image_grid_thw=grid, past_key_values=cache,
                    use_cache=True, return_dict=True)
            logits = self.model.action_logits(output.last_hidden_state[:, -1])
            for row, plan in enumerate(plans):
                slot = self._slots[plan.index]
                slot.stable = plan.stable
                slot.mrope = plan.mrope + int(local_positions[:, row, :plan.stable_length].max()) + 1
                store.lengths[plan.index] = len(plan.stable)
                self._logits[plan.index] = logits[row]

    def _perception(self, goals, histories, actions):
        if any(len(frames) != len(prior) + 1 for frames, prior in zip(histories, actions, strict=True)):
            raise ValueError("Perception frame and action histories differ")
        prompts = [self._prompt(goal, prior) for goal, prior in zip(goals, actions, strict=True)]
        frames = [frame for history in histories for frame in history]
        device = frames[0].device
        lengths = [len(row) for row in prompts]
        ids = torch.full((len(prompts), max(lengths)), self.builder.pad_token_id, dtype=torch.long, device=device)
        mask = torch.zeros_like(ids)
        for row, prompt in enumerate(prompts):
            ids[row, :len(prompt)] = torch.tensor(prompt, device=device)
            mask[row, :len(prompt)] = 1
        pixels, grid = patchify_images(torch.stack(frames), self.model.config)
        backbone = self.model.qwen.model
        types = ids.eq(self.model.qwen.config.image_token_id).int()
        embeddings = self.model.qwen.get_input_embeddings()(ids)
        positions = backbone.compute_3d_position_ids(input_ids=ids, inputs_embeds=embeddings,
            image_grid_thw=grid, video_grid_thw=None, attention_mask=mask,
            past_key_values=None, mm_token_type_ids=types)
        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + length)
        packed = min(lengths) != max(lengths)
        with self._vision.use(backbone, frames), _packed_prefill(self.language, mask, lengths, offsets, packed):
            output = backbone(input_ids=None, inputs_embeds=embeddings, position_ids=positions,
                attention_mask=mask, mm_token_type_ids=types, pixel_values=pixels,
                image_grid_thw=grid, use_cache=True, return_dict=True)
        if packed:
            hidden = output.last_hidden_state[0, torch.tensor(offsets[1:], device=device) - 1]
        else:
            hidden = output.last_hidden_state[torch.arange(len(prompts), device=device), mask.sum(1) - 1]
        store, graph = self._prepare_window(output.past_key_values, lengths, offsets if packed else None)
        next_positions = mask.sum(1) + backbone.rope_deltas.reshape(-1)
        generated = [[] for _ in prompts]
        states = [self.trie.initial_state() for _ in prompts]
        cue_offsets = [None] * len(prompts)
        active = [True] * len(prompts)
        logits = torch.zeros(len(prompts), 4, dtype=torch.float32, device=device)
        support_ids = torch.tensor(sorted(self.builder._support(False)), dtype=torch.long, device=device)
        support_map = {int(token): index for index, token in enumerate(support_ids.tolist())}
        weight = self.model.lm_head_weight()[support_ids]
        tokenizer = self.model.processor.tokenizer

        def append(tokens, advancing):
            graph.tokens.copy_(tokens[:, None])
            graph.positions[0, :, 0].copy_(store.cursor)
            graph.positions[1:, :, 0].copy_(next_positions[None, :])
            graph.graph.replay()
            next_positions.add_(torch.tensor(advancing, device=device, dtype=torch.long))
            return graph.hidden

        for _ in range(self.max_new_tokens):
            scores = F.linear(hidden, weight).float()
            tokens = torch.full((len(prompts),), self.builder.pad_token_id, dtype=torch.long, device=device)
            ended = []
            for row in range(len(prompts)):
                if not active[row]:
                    continue
                offset = cue_offsets[row]
                if offset is None:
                    offset = self.trie.cue_offset(states[row])
                if offset is not None:
                    token = self.trie.cue_ids[offset]
                    cue_offsets[row] = offset + 1
                else:
                    allowed = self.trie.allowed(states[row])
                    candidates = sorted(token for token in allowed if token in support_map)
                    if candidates:
                        columns = torch.tensor([support_map[t] for t in candidates], device=device)
                        token = candidates[int(scores[row, columns].argmax())]
                    else:
                        # Legal noncanonical spellings can leave the training token support.
                        full = F.linear(hidden[row:row + 1], self.model.lm_head_weight()).float()[0]
                        candidates = sorted(allowed)
                        token = candidates[int(full[candidates].argmax())]
                    states[row] = allowed[token]
                    cue_offsets[row] = self.trie.cue_offset(states[row])
                generated[row].append(token)
                tokens[row] = token
                if cue_offsets[row] == len(self.trie.cue_ids):
                    ended.append(row)
            hidden = append(tokens, active)
            for row in ended:
                logits[row] = self.model.action_logits(hidden[row:row + 1])[0]
                active[row] = False
            if not any(active):
                break
        if any(active):
            # Append the action cue after the token budget.
            remaining = [self.trie.cue_ids[cue_offsets[i] or 0:] if active[i] else () for i in range(len(prompts))]
            for index in range(max(map(len, remaining))):
                tokens = torch.full((len(prompts),), self.builder.pad_token_id, dtype=torch.long, device=device)
                advancing = [index < len(row) for row in remaining]
                for row, suffix in enumerate(remaining):
                    if advancing[row]:
                        tokens[row] = suffix[index]
                        generated[row].append(suffix[index])
                hidden = append(tokens, advancing)
                for row, suffix in enumerate(remaining):
                    if suffix and index + 1 == len(suffix):
                        logits[row] = self.model.action_logits(hidden[row:row + 1])[0]
        texts = [tokenizer.decode(row, skip_special_tokens=True, clean_up_tokenization_spaces=False) for row in generated]
        if any(not text.endswith(',"action":"') for text in texts):
            raise RuntimeError("Perception generation did not end at the action cue")
        return logits, texts

    def _prepare_window(self, cache, lengths, offsets):
        capacity = ((max(lengths) + self.max_new_tokens + len(self.trie.cue_ids) + 4 + 1023) // 1024) * 1024
        if self._window_store is None or self._window_store.num_envs != len(lengths) or self._window_store.capacity < capacity:
            self._window_store = _Store(len(lengths), capacity, len(cache.layers))
            self._window_graph = None
        store = self._window_store
        for index, layer in enumerate(cache.layers):
            store.ensure(index, layer.keys)
            for row, length in enumerate(lengths):
                source, start = (row, 0) if offsets is None else (0, offsets[row])
                store.keys[index][row, :, :length].copy_(layer.keys[source, :, start:start + length])
                store.values[index][row, :, :length].copy_(layer.values[source, :, start:start + length])
        store.lengths[:] = lengths
        store.sync()
        if self._window_graph is None:
            self._window_graph = _capture_decode(self.model.qwen.model, self.language, store)
        return store, self._window_graph


def _slot():
    return SimpleNamespace(frames=[], actions=[], semantic=[], visible=[], distances=[], stable=(), mrope=0)


class _VisionCache:
    def __init__(self):
        self.entries = {}

    def prune(self, frames):
        live = {id(frame) for frame in frames}
        self.entries = {key: entry for key, entry in self.entries.items() if key in live and entry[0]() is not None}

    @contextmanager
    def use(self, backbone, frames):
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            BaseModelOutputWithDeepstackFeatures,
        )

        original = backbone.get_image_features

        def features(pixel_values, image_grid_thw=None, **kwargs):
            sizes = image_grid_thw.prod(-1).tolist()
            patches = pixel_values.split(sizes)
            missing = [i for i, frame in enumerate(frames) if id(frame) not in self.entries or self.entries[id(frame)][0]() is not frame]
            if missing:
                output = original(torch.cat([patches[i] for i in missing]), image_grid_thw[missing], **kwargs)
                pooled = output.pooler_output
                deep = output.deepstack_features
                for i, frame_index in enumerate(missing):
                    self.entries[id(frames[frame_index])] = (weakref.ref(frames[frame_index]), pooled[i].detach().clone(),
                        tuple(part[i].detach().clone() for part in deep))
            ordered = [self.entries[id(frame)] for frame in frames]
            deep = [tuple(entry[2][i] for entry in ordered) for i in range(len(ordered[0][2]))]
            return BaseModelOutputWithDeepstackFeatures(pooler_output=tuple(entry[1] for entry in ordered),
                deepstack_features=deep)
        backbone.get_image_features = features
        try:
            yield
        finally:
            del backbone.get_image_features


@contextmanager
def _packed_prefill(language, mask, lengths, offsets, enabled):
    if not enabled:
        yield
        return
    original = language.forward
    had_forward = "forward" in language.__dict__
    valid = mask.bool()

    def forward(*args, **kwargs):
        kwargs["inputs_embeds"] = kwargs["inputs_embeds"][valid].unsqueeze(0)
        kwargs["position_ids"] = kwargs["position_ids"][:, valid].unsqueeze(1)
        if kwargs.get("visual_pos_masks") is not None:
            kwargs["visual_pos_masks"] = kwargs["visual_pos_masks"][valid].unsqueeze(0)
        kwargs["attention_mask"] = None
        return original(*args, **kwargs)
    language.forward = forward
    cumulative = torch.tensor(offsets, dtype=torch.int32, device=mask.device)
    try:
        with _decoder_kwargs(language, {"cu_seq_lens_q": cumulative, "cu_seq_lens_k": cumulative,
                                       "max_length_q": max(lengths), "max_length_k": max(lengths)}):
            yield
    finally:
        if had_forward:
            language.forward = original
        else:
            del language.forward


class _Store:
    """Fixed-address KV storage with row-local write and attention cursors."""

    def __init__(self, num_envs, capacity, num_layers):
        self.num_envs, self.capacity, self.num_layers = num_envs, capacity, num_layers
        self.keys, self.values = {}, {}
        self.lengths = [0] * num_envs
        self.cursor = None
        self.max_seqlen_q = 1

    def ensure(self, layer, key):
        if layer not in self.keys:
            shape = (self.num_envs, key.shape[1], self.capacity, key.shape[-1])
            self.keys[layer], self.values[layer] = key.new_zeros(shape), key.new_zeros(shape)
        if self.cursor is None:
            self.cursor = torch.zeros(self.num_envs, dtype=torch.int32, device=key.device)
            self.seqused = torch.zeros_like(self.cursor)
            self.page_table = torch.arange(self.num_envs, dtype=torch.int32, device=key.device).unsqueeze(1)
            self.cu_seqlens_q = torch.arange(self.num_envs + 1, dtype=torch.int32, device=key.device)

    def sync(self):
        self.cursor.copy_(torch.tensor(self.lengths, dtype=torch.int32, device=self.cursor.device))
        self.seqused.copy_(self.cursor + 1)

    def advance(self):
        self.cursor.add_(1)
        self.seqused.add_(1)


class _CacheLayer(CacheLayerMixin):
    is_sliding = False

    def __init__(self, store, layer):
        super().__init__()
        self.store, self.layer = store, layer
        self.is_initialized = True

    def lazy_initialization(self, key_states, value_states):
        self.store.ensure(self.layer, key_states)

    def get_mask_sizes(self, query_length):
        return self.store.capacity, 0

    def get_seq_length(self):
        return self.store.capacity

    def get_max_length(self):
        return self.store.capacity


class _RowWrite(_CacheLayer):
    def __init__(self, store, layer, row):
        super().__init__(store, layer)
        self.row = row

    def update(self, key_states, value_states, *args, **kwargs):
        store = self.store
        store.ensure(self.layer, key_states)
        start = store.lengths[self.row]
        end = start + key_states.shape[-2]
        if end > store.capacity:
            raise RuntimeError("Policy exceeded its KV capacity")
        keys, values = store.keys[self.layer], store.values[self.layer]
        keys[self.row:self.row + 1, :, start:end].copy_(key_states)
        values[self.row:self.row + 1, :, start:end].copy_(value_states)
        return keys[self.row:self.row + 1, :, :end], values[self.row:self.row + 1, :, :end]

    def get_seq_length(self):
        return self.store.lengths[self.row]

    def get_mask_sizes(self, query_length):
        return self.get_seq_length() + query_length, 0


class _BatchWrite(_CacheLayer):
    def __init__(self, store, layer, rows, starts):
        super().__init__(store, layer)
        self.rows, self.starts = rows, starts

    def update(self, key_states, value_states, *args, **kwargs):
        store = self.store
        store.ensure(self.layer, key_states)
        keys, values = store.keys[self.layer], store.values[self.layer]
        for row, (index, start) in enumerate(zip(self.rows, self.starts, strict=True)):
            end = start + key_states.shape[-2]
            if end > store.capacity:
                raise RuntimeError("Policy exceeded its KV capacity")
            keys[index, :, start:end].copy_(key_states[row])
            values[index, :, start:end].copy_(value_states[row])
        return keys, values


class _DecodeWrite(_CacheLayer):
    def update(self, key_states, value_states, *args, **kwargs):
        keys, values = self.store.keys[self.layer], self.store.values[self.layer]
        index = self.store.cursor.long().view(-1, 1, 1, 1).expand(-1, keys.shape[1], 1, keys.shape[-1])
        keys.scatter_(2, index, key_states.to(keys.dtype))
        values.scatter_(2, index, value_states.to(values.dtype))
        return keys, values


@contextmanager
def _ragged(language, context):
    previous = [getattr(layer.self_attn, "_rollout_context", None) for layer in language.layers]
    for layer in language.layers:
        layer.self_attn._rollout_context = context
    try:
        yield
    finally:
        for layer, value in zip(language.layers, previous, strict=True):
            layer.self_attn._rollout_context = value


def _capture_decode(backbone, language, store):
    device = store.cursor.device
    cache = Cache(layers=[_DecodeWrite(store, i) for i in range(store.num_layers)])
    tokens = torch.zeros(store.num_envs, 1, dtype=torch.long, device=device)
    positions = torch.zeros(4, store.num_envs, 1, dtype=torch.long, device=device)

    def run():
        hidden = backbone(input_ids=tokens, position_ids=positions, attention_mask=None,
            past_key_values=cache, use_cache=True, return_dict=True).last_hidden_state[:, -1]
        store.advance()
        return hidden

    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup), _ragged(language, store):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(warmup)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph), _ragged(language, store):
        hidden = run()
    store.sync()
    return SimpleNamespace(graph=graph, tokens=tokens, positions=positions, hidden=hidden)


def _rollout_attention(module, query, key, value, attention_mask, dropout=0.0, scaling=None,
                       is_causal=None, **kwargs):
    from transformers.integrations.flash_attention import (
        flash_attention_forward,
        get_target_dtype,
    )

    context = getattr(module, "_rollout_context", None)
    if context is None and query.shape[2] != 1:
        previous = module.config._attn_implementation
        module.config._attn_implementation = "flash_attention_4"
        try:
            return flash_attention_forward(module, query, key, value, attention_mask,
                dropout=dropout, scaling=scaling, is_causal=is_causal, **kwargs)
        finally:
            module.config._attn_implementation = previous
    from flash_attn.cute import flash_attn_func, flash_attn_varlen_func
    from transformers.modeling_flash_attention_utils import (
        _pad_input,
        _unpad_input,
        _upad_input,
    )

    query, key, value = [tensor.transpose(1, 2) for tensor in (query, key, value)]
    dtype = get_target_dtype(query, module)
    if dtype is not None:
        query, key, value = [tensor.to(dtype) for tensor in (query, key, value)]
    splits = 4 if key.shape[1] >= 8192 else 2 if key.shape[1] >= 2048 else 1
    if torch.cuda.get_device_capability(query.device)[0] < 10 or query.shape[-1] == 256:
        splits = 1
    causal = module.is_causal if is_causal is None else is_causal
    if context is not None:
        rows, length = query.shape[:2]
        output = flash_attn_varlen_func(query.reshape(rows * length, *query.shape[2:]), key, value,
            cu_seqlens_q=context.cu_seqlens_q, max_seqlen_q=context.max_seqlen_q,
            max_seqlen_k=context.capacity, seqused_k=context.seqused, page_table=context.page_table,
            softmax_scale=scaling, causal=causal, num_splits=splits)
        if isinstance(output, tuple):
            output = output[0]
        return output.reshape(rows, length, *output.shape[1:]), None
    if attention_mask is None:
        output = flash_attn_func(query, key, value, softmax_scale=scaling, causal=causal, num_splits=splits)
        return (output[0] if isinstance(output, tuple) else output), None
    query, key, value, indices, lengths, maxima = _upad_input(query, key, value, attention_mask, 1, _unpad_input)
    output = flash_attn_varlen_func(query, key, value, cu_seqlens_q=lengths[0], cu_seqlens_k=lengths[1],
        max_seqlen_q=maxima[0], max_seqlen_k=maxima[1], softmax_scale=scaling, causal=causal, num_splits=splits)
    if isinstance(output, tuple):
        output = output[0]
    return _pad_input(output, indices, query.shape[0], 1), None
