"""Native Qwen3-VL token prediction and worker-side prompt construction.

The pretrained vocabulary is unchanged. Training cross-entropy uses the full
language-model vocabulary; deployment gathers four existing action-token rows.
"""

from __future__ import annotations

import json
import math
from bisect import bisect_right
from collections.abc import Sequence
from contextlib import contextmanager
from itertools import groupby, pairwise
from pathlib import Path
from typing import TYPE_CHECKING, NotRequired, TypedDict

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import AutoModelForImageTextToText, AutoProcessor

from stopwatch.config import ModelConfig, OptimizerConfig, Source

if TYPE_CHECKING:
    from stopwatch.data.vocab import Vocabulary


ACTION_WORDS = ("Stop", "Forward", "Left", "Right")
POLICY_ACTION_TOKEN_IDS = (14215, 22164, 13727, 10083)
POLICY_STOP_RULE = "Stop when the goal distance is <0.15m."
_SYSTEM_PROMPT = (
    "You are a mobile robot navigating to the target object. "
    "Target object: {category}."
)
_PERCEPTION_PROMPT = (
    "You are a mobile robot navigating to the target object. "
    "Target object: {category}. Use the observation history for navigation. "
    "For the latest image only, predict visible objects. Output "
    '"target":"not seen" whenever the target is not visible in the latest '
    "image, even if it appeared earlier. When it is visible in the latest "
    'image, output the target\'s {distance_description} in "target". Return '
    "exactly one JSON object with no additional text using this field order: "
    "target (string), objects (array of strings), action (string). The objects "
    "value must contain at most 16 canonical category names sorted by decreasing "
    "visible image area. Set action to exactly one of: {actions}."
)

# All sites refer to predecessor hidden states.
TokenIds = tuple[int, ...]
Prompt = tuple[TokenIds, TokenIds, TokenIds, TokenIds]


class PackedPrompt(TypedDict):
    input_ids: Tensor
    attention_mask: Tensor
    action_mask: Tensor
    semantic_mask: Tensor
    semantic_targets: Tensor


class PromptNames(TypedDict):
    semantic_names: tuple[str, ...] | None
    goal_name: str | None


class PerceptionOutput(TypedDict):
    action_hidden: Tensor  # bf16 [A_local,D]
    semantic_hidden: Tensor  # bf16 [S_local,D]
    semantic_targets: Tensor  # int64 [S_local], native vocabulary IDs
    sequence_parallel_action_selection: NotRequired[Tensor]  # bool [E]


class OptimizerGroup(TypedDict):
    params: list[nn.Parameter]
    lr: float
    name: str
    weight_decay: NotRequired[float]


class PromptBuilder:
    """Weightless tokenizer with immutable canonical string tables."""

    pad_token_id: int

    def _initialize(self, config, vocabulary, source, processor):
        self.config, self.vocabulary, self.source = config, vocabulary, source
        self.processor = processor
        processor.tokenizer.padding_side = "right"
        self.pad_token_id = int(processor.tokenizer.pad_token_id)
        self.semantic_names = tuple(vocabulary.categories)
        self.image_token_id = int(processor.tokenizer.convert_tokens_to_ids("<|image_pad|>"))
        self.visual_tokens_per_frame = config.image_height // 32 * (config.image_width // 32)
        edges = config.distance_bin_edges
        self.distance_labels = (
            f"<{edges[0]:g}m",
            *(f"{a:g}-{b:g}m" for a, b in pairwise(edges)),
            f">{edges[-1]:g}m",
        )
        self.action_ids = _action_ids(processor, config.role == "perception")
        probe = [_message("system", "")]
        plain, headed = self._render(probe), self._render(probe, generate=True)
        if headed[:len(plain)] != plain:
            raise RuntimeError("Assistant header is not a stable token prefix")
        self.header_ids = headed[len(plain):]
        self.end_ids = self._encode("<|im_end|>")
        self._target_support = {}

    def _encode(self, text):
        return tuple(self.processor.tokenizer.encode(text, add_special_tokens=False))

    def _render(self, messages, *, generate=False):
        return self._encode(self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=generate,
        ))

    def _system(self, goal_name):
        if self.config.role == "policy":
            return _SYSTEM_PROMPT.format(category=goal_name) + " " + POLICY_STOP_RULE
        return _PERCEPTION_PROMPT.format(
            category=goal_name, actions=", ".join(ACTION_WORDS),
            distance_description="distance bin (" + ", ".join(self.distance_labels) + ")",
        )

    def _distance(self, distance):
        if not math.isfinite(distance) or distance < 0:
            raise ValueError("Goal distance must be finite and nonnegative")
        return self.distance_labels[bisect_right(self.config.distance_bin_edges, distance)]

    def _observation(self, ids=None, visible=None, distance=None, names=None):
        content = [{"type": "text", "text": "Observation:"}, {"type": "image"}]
        if ids is not None:
            objects = [names[int(value) - 1] for value in ids if int(value)]
            text = "Objects:" + (" " + " | ".join(objects) if objects else "")
            text += f"\nGoal visible: {'true' if visible else 'false'}"
            if not visible:
                text += "\nDistance: not seen"
            elif distance is not None and math.isfinite(distance):
                text += "\nDistance: " + self._distance(distance)
            content.append({"type": "text", "text": text})
        return content

    def _history(self, goal_name, actions, semantic=None, visible=None, distances=None, names=None):
        messages = [_message("system", self._system(goal_name))]
        for index in range(len(actions)):
            messages.append({"role": "user", "content": self._observation(
                None if semantic is None else semantic[index],
                None if visible is None else visible[index],
                None if distances is None else distances[index], names,
            )})
            if index + 1 < len(actions):
                messages.append(_message("assistant", " " + ACTION_WORDS[int(actions[index])]))
        return messages

    def _expand(self, ids, *site_lists):
        expanded, index_map = [], []
        for token in ids:
            expanded.extend([token] * (self.visual_tokens_per_frame if token == self.image_token_id else 1))
            index_map.append(len(expanded) - 1)
        return (tuple(expanded), *(tuple(index_map[i] for i in sites) for sites in site_lists))

    def _support(self, augmented):
        if augmented in self._target_support:
            return self._target_support[augmented]
        names = self.semantic_names
        if augmented:
            names = tuple(dict.fromkeys((*names, *(n for pool in self.vocabulary.member_pools for n in pool))))
        tokens = set()
        for value in ("not seen", *self.distance_labels):
            for objects in ([], [names[0]]):
                tokens.update(self._encode(_json({"target": value, "objects": objects, "action": "Stop"})))
        for name in names:
            tokens.update(self._encode(_json({
                "target": self.distance_labels[0], "objects": [names[0], name], "action": "Stop",
            })))
        tokens.difference_update(self.action_ids)
        self._target_support[augmented] = frozenset(tokens)
        return self._target_support[augmented]

    def build_prompt(self, goal: int, actions: Sequence[int], semantic: Tensor, *, goal_visible: Sequence[bool], goal_distances: Sequence[float | None], semantic_names: Sequence[str] | None = None, goal_name: str | None = None) -> Prompt:
        """Use T actions, T visibility flags and T optional distance values.

        semantic is int16 [16] for perception, [T,16] for policy. Perception
        reads only the final visibility/distance entry; preceding entries may
        be false/None. Earlier expert actions remain in both prompt histories.
        Per-sample semantic_names has 540 entries. goal_name overrides the
        resolved prompt goal.
        Neither argument mutates the builder or vocabulary.
        """
        if not 0 < len(actions) <= self.config.frame_window:
            raise ValueError("Prompt frame count is outside the configured window")
        if len(goal_visible) != len(actions) or len(goal_distances) != len(actions):
            raise ValueError("Prompt visibility and distances must cover every frame")
        if any(int(action) not in range(4) for action in actions):
            raise ValueError("Invalid navigation action")
        names = self.semantic_names if semantic_names is None else tuple(semantic_names)
        if len(names) != 540:
            raise ValueError("Per-sample semantic names must contain 540 entries")
        if goal_name is None:
            class_id = self.vocabulary.goal_to_class_ids[goal] if self.source == "ovon" else goal
            goal_name = names[class_id]
        messages = self._history(
            goal_name, actions,
            semantic if self.config.role == "policy" else None,
            goal_visible, goal_distances, names,
        )
        if self.config.role == "policy":
            raw = self._render(messages, generate=True)
            system = self._render(messages[:1])
            positions = [i for i in range(len(system), len(raw)) if raw[i] in self.action_ids]
            if tuple(raw[i] for i in positions) != tuple(self.action_ids[int(a)] for a in actions[:-1]):
                raise RuntimeError("Action tokens outside the assistant history")
            ids, sites = self._expand(raw, [*(i - 1 for i in positions), len(raw) - 1])
            return ids, sites, (), ()
        if tuple(semantic.shape) != (16,):
            raise ValueError("Perception target must have 16 semantic IDs")
        distance = goal_distances[-1]
        payload = {
            "target": "not seen" if distance is None else self._distance(distance),
            "objects": [names[int(value) - 1] for value in semantic if int(value)],
            "action": ACTION_WORDS[int(actions[-1])],
        }
        messages.append(_message("assistant", _json(payload)))
        raw = self._render(messages)
        headers = [i for i in range(len(raw)) if raw[i:i + len(self.header_ids)] == self.header_ids]
        start = headers[-1] + len(self.header_ids)
        end = _find(raw, self.end_ids, start)
        target = raw[start:end]
        suffix = self._encode('"}')
        action_index = len(target) - len(suffix) - 1
        if target[action_index] != self.action_ids[int(actions[-1])] or target[action_index + 1:] != suffix:
            raise RuntimeError("JSON action target is not one native token")
        payload_end = _semantic_end(self.processor.tokenizer, target, action_index)
        targets = target[:payload_end]
        if not set(targets).issubset(self._support(semantic_names is not None)):
            raise RuntimeError("Semantic target contains unsupported native tokens")
        ids, action_sites, semantic_sites = self._expand(
            raw, [start + action_index - 1], range(start - 1, start + payload_end - 1),
        )
        return ids, action_sites, semantic_sites, targets


def _message(role, text):
    return {"role": role, "content": [{"type": "text", "text": text}]}


def _json(payload):
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _find(ids, pattern, start):
    for index in range(start, len(ids) - len(pattern) + 1):
        if ids[index:index + len(pattern)] == pattern:
            return index
    raise RuntimeError(f"Missing prompt token span {pattern}")


def _semantic_end(tokenizer, target, action_index):
    cue = ',"action":"'
    for index in range(action_index - 1, -1, -1):
        text = tokenizer.decode(target[index:action_index], skip_special_tokens=False,
                                clean_up_tokenization_spaces=False)
        if text.endswith(cue):
            return index + int(len(text) > len(cue))
    raise RuntimeError("Missing semantic action cue")


def _action_ids(processor, perception):
    tokenizer = processor.tokenizer
    if perception:
        prefix = '{"objects":[],"goal_visible":false,"action":"'
    else:
        prefix = processor.apply_chat_template([
            _message("system", _SYSTEM_PROMPT.format(category="chair")),
            {"role": "user", "content": [{"type": "text", "text": "Observation:"}, {"type": "image"}]},
        ], tokenize=False, add_generation_prompt=True)
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    result = []
    for word in ACTION_WORDS:
        ids = tokenizer.encode(prefix + ("" if perception else " ") + word, add_special_tokens=False)
        if ids[:len(prefix_ids)] != prefix_ids or len(ids) != len(prefix_ids) + 1:
            raise ValueError("Actions must each be one native token in their prompt context")
        result.append(ids[-1])
    if len(set(result)) != 4 or (not perception and tuple(result) != POLICY_ACTION_TOKEN_IDS):
        raise ValueError("Unexpected native action-token vocabulary")
    return tuple(result)


@contextmanager
def _decoder_kwargs(language, kwargs):
    def inject(module, args, values):
        return args, {**values, **kwargs}
    hook = language.register_forward_pre_hook(inject, with_kwargs=True, prepend=True)
    try:
        yield
    finally:
        hook.remove()


class QwenNTP(nn.Module):
    def __init__(self, config: ModelConfig, vocabulary: Vocabulary, *, checkpoint: str | Path | None = None, data_root: Path = Path("data")) -> None:
        """Load stock AutoModelForImageTextToText and retain its pretrained head.

        Without checkpoint, load the pinned base in fp32 for training. With
        checkpoint, load local or Hub bf16 release weights for evaluation. Distributed
        optimizer restarts belong to AsyncCheckpoint, not this constructor.
        """
        super().__init__()
        self.config, self.vocabulary = config, vocabulary
        location = str(checkpoint) if checkpoint is not None else config.model_id
        if checkpoint is not None:
            path = Path(checkpoint).expanduser()
            for candidate in (data_root.expanduser() / path, path):
                if candidate.is_dir():
                    location = str(candidate.resolve())
                    break
        revision = {} if checkpoint is not None else {"revision": config.revision}
        self.processor = AutoProcessor.from_pretrained(location, **revision)
        self.processor.tokenizer.padding_side = "right"
        self.qwen = AutoModelForImageTextToText.from_pretrained(
            location, dtype=torch.bfloat16 if checkpoint is not None else torch.float32,
            attn_implementation="flash_attention_4", **revision,
        )
        if checkpoint is not None:
            # Cast rotary buffers together with the weights.
            self.qwen.to(dtype=torch.bfloat16)
        if self.qwen.config.model_type != "qwen3_vl":
            raise ValueError("The release requires the stock Qwen3-VL architecture")
        vision = self.qwen.config.vision_config
        if (vision.patch_size, vision.temporal_patch_size, vision.spatial_merge_size) != (16, 2, 2):
            raise ValueError("Unexpected Qwen vision patch geometry")
        for parameter in self.qwen.parameters():
            parameter.requires_grad_(False)
        for parameter in self.qwen.model.language_model.parameters():
            parameter.requires_grad_(True)
        self.qwen.lm_head.weight.requires_grad_(True)
        for name, parameter in self.qwen.model.visual.named_parameters():
            if name.startswith(("merger.", "deepstack_merger_list.")):
                parameter.requires_grad_(True)
        self.register_buffer("action_token_ids", torch.tensor(
            _action_ids(self.processor, config.role == "perception"), dtype=torch.long,
        ), persistent=False)
        self._sequence_parallel = None
        self.qwen.model.language_model.register_forward_pre_hook(_bf16_inputs, with_kwargs=True)
        _install_image_geometry(self.qwen.model)

    def forward(self, frames: Tensor, *, prompt: PackedPrompt) -> Tensor | PerceptionOutput:
        """Return policy bf16 [N,D] states or perception's selected-state mapping.

        Input prompt rows are padded on CPU. Remove padding internally, reset
        multimodal rotary positions per episode and retain episode boundaries.
        """
        ids, valid = prompt["input_ids"], prompt["attention_mask"].bool()
        pixels, grid = patchify_images(frames, self.config)
        types = ids.eq(self.qwen.config.image_token_id).int()
        if int(types.sum()) != len(frames) * self.config.image_width // 32 * (self.config.image_height // 32):
            raise ValueError("Prompt image placeholders do not match the frame count")
        positions = self.qwen.model.compute_3d_position_ids(
            input_ids=ids, inputs_embeds=None, image_grid_thw=grid,
            video_grid_thw=None, attention_mask=prompt["attention_mask"],
            past_key_values=None, mm_token_type_ids=types,
        )
        text_positions = torch.arange(ids.shape[1], device=ids.device).expand_as(ids)
        positions = torch.cat((text_positions[valid].view(1, 1, -1), positions[:, valid].unsqueeze(1)))
        lengths = valid.sum(1)
        cumulative = F.pad(lengths.cumsum(0), (1, 0)).to(torch.int32)
        packed_kwargs = {"cu_seq_lens_q": cumulative, "cu_seq_lens_k": cumulative,
                         "max_length_q": int(lengths.max()), "max_length_k": int(lengths.max())}
        packed_ids = ids[valid].unsqueeze(0)
        embeddings = self.qwen.get_input_embeddings()(packed_ids)
        with _decoder_kwargs(self.qwen.model.language_model, packed_kwargs):
            output = self.qwen.model(
                input_ids=None, inputs_embeds=embeddings, position_ids=positions,
                pixel_values=pixels, image_grid_thw=grid, mm_token_type_ids=types[valid].unsqueeze(0),
                use_cache=False, return_dict=True,
            ).last_hidden_state
        action_mask = prompt["action_mask"][valid].unsqueeze(0)
        semantic_mask = prompt["semantic_mask"][valid].unsqueeze(0)
        targets = prompt["semantic_targets"]
        selection = None
        if self._sequence_parallel is not None:
            action_mask, selection = self._sequence_parallel.shard_loss_mask(action_mask)
            semantic_mask, semantic_selection = self._sequence_parallel.shard_loss_mask(semantic_mask)
            targets = targets[semantic_selection]
        hidden = output[action_mask]
        if self.config.role == "policy":
            return hidden
        result = PerceptionOutput(
            action_hidden=hidden, semantic_hidden=output[semantic_mask], semantic_targets=targets,
        )
        if selection is not None:
            result["sequence_parallel_action_selection"] = selection
        return result

    def action_logits(self, hidden: Tensor) -> Tensor:
        """Return float32 [A,4] logits in STOP, FORWARD, LEFT, RIGHT order."""
        weight = self.lm_head_weight().index_select(0, self.action_token_ids)
        return F.linear(hidden.to(weight.dtype), weight).float()

    def action_targets(self, actions: Tensor) -> Tensor:
        """Map int64 Habitat action IDs to int64 native vocabulary token IDs."""
        if bool(((actions < 0) | (actions >= 4)).any()):
            raise ValueError("Action targets must be in 0..3")
        return self.action_token_ids[actions.long()]

    def lm_head_weight(self) -> Tensor:
        """Return the pretrained output projection, respecting tied embeddings."""
        return self.qwen.lm_head.weight

    def optimizer_param_groups(self, config: OptimizerConfig) -> list[OptimizerGroup]:
        """Group trainable mergers, language and head; exclude the frozen vision tower."""
        visual = {id(p) for p in self.qwen.model.visual.parameters()}
        no_decay = set()
        for module in self.modules():
            if isinstance(module, nn.Embedding) or type(module).__name__.lower().endswith("norm"):
                no_decay.update(id(p) for p in module.parameters(recurse=False))
        no_decay.update(id(p) for name, p in self.named_parameters() if name.rsplit(".", 1)[-1] == "bias")
        tiers = {"trunk": [], "encoder": []}
        for parameter in self.parameters():
            if parameter.requires_grad:
                name = "encoder" if id(parameter) in visual else "trunk"
                tiers[name].append(parameter)
        groups = []
        for (name, parameters), rate in zip(tiers.items(), (config.trunk_lr, config.encoder_lr)):
            for excluded in (False, True):
                selected = [p for p in parameters if (id(p) in no_decay) == excluded]
                if selected:
                    groups.append(OptimizerGroup(params=selected, lr=rate,
                        name=name + (".no_decay" if excluded else ""),
                        weight_decay=0.0 if excluded else config.weight_decay))
        return groups

    def make_prompt_builder(self, source: Source) -> PromptBuilder:
        """Copy tokenizer/config state to CPU without any model weights."""
        builder = PromptBuilder.__new__(PromptBuilder)
        builder._initialize(self.config, self.vocabulary, source, self.processor)
        return builder


def patchify_images(frames: Tensor, config: ModelConfig) -> tuple[Tensor, Tensor]:
    """Return float32 [N*grid_h*grid_w,1536] patches and int64 [N,3] grids."""
    height, width = config.image_height, config.image_width
    images = frames.permute(0, 3, 1, 2).float()
    if images.shape[-2:] != (height, width):
        images = F.interpolate(images, size=(height, width), mode="bicubic", align_corners=False, antialias=True)
    images = (images / 127.5 - 1.0).unsqueeze(1).repeat(1, 2, 1, 1, 1)
    count, gh, gw = len(frames), height // 16, width // 16
    patches = images.view(count, 1, 2, 3, gh // 2, 2, 16, gw // 2, 2, 16)
    patches = patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9).reshape(count * gh * gw, 1536)
    grid = torch.tensor([1, gh, gw], dtype=torch.long, device=frames.device).expand(count, -1).clone()
    return patches, grid


def _bf16_inputs(module, args, kwargs):
    embeddings = kwargs.get("inputs_embeds")
    if embeddings is not None and embeddings.is_cuda and torch.is_autocast_enabled():
        kwargs["inputs_embeds"] = embeddings.to(torch.get_autocast_dtype("cuda"))
    return args, kwargs


def _install_image_geometry(backbone):
    """Set rotary geometry for fixed-size, independent images."""
    visual = backbone.visual

    def vision_rotary(hidden_states, position_ids):
        # Preserve the bf16 frequency product and trigonometry.
        frequencies = (position_ids.unsqueeze(-1) * visual.rotary_pos_emb.inv_freq).flatten(1)
        angles = torch.cat((frequencies, frequencies), dim=-1)
        return angles.cos(), angles.sin()

    visual.rotary_pos_emb.forward = vision_rotary

    def get_rope_index(input_ids, mm_token_type_ids, image_grid_thw=None,
                       video_grid_thw=None, attention_mask=None, **kwargs):
        if video_grid_thw is not None:
            raise ValueError("Navigation prompts use independent images")
        grids = iter(image_grid_thw.tolist()) if image_grid_thw is not None else iter(())
        device = input_ids.device
        merge = backbone.config.vision_config.spatial_merge_size
        bases = {}
        positions = torch.zeros(3, *input_ids.shape, dtype=input_ids.dtype, device=device)
        deltas = []
        for row, tokens in enumerate(input_ids):
            types = mm_token_type_ids[row]
            keep = attention_mask[row].bool() if attention_mask is not None else torch.ones_like(tokens, dtype=torch.bool)
            types = types[keep]
            current, pieces = 0, []
            for modality, group in groupby(types.tolist()):
                length = sum(1 for _ in group)
                if modality == 0:
                    pieces.append(torch.arange(length, device=device).view(1, -1).expand(3, -1) + current)
                    current += length
                else:
                    grid = tuple(next(grids))
                    if grid not in bases:
                        time, height, width = grid[0], grid[1] // merge, grid[2] // merge
                        bases[grid] = torch.stack((
                            torch.arange(time, device=device).repeat_interleave(height * width),
                            torch.arange(height, device=device).repeat_interleave(width).repeat(time),
                            torch.arange(width, device=device).repeat(height * time),
                        ))
                    pieces.append(bases[grid] + current)
                    current += max(grid[1], grid[2]) // merge
            local = torch.cat(pieces, dim=1)
            positions[:, row, keep] = local
            deltas.append(local.max() + 1 - int(keep.sum()))
        return positions, torch.stack(deltas).unsqueeze(1)

    backbone.get_rope_index = get_rope_index


class _SemTextTrie:
    """Character automaton for canonical target-first JSON, including merged tokens.

    Training name augmentation never changes this evaluation grammar.
    """

    def __init__(self, tokenizer, categories, distance_labels):
        if len(set(categories)) != len(categories) or any(
            not name or name != name.strip() or not name.isascii()
            or any(char in name for char in '"\\\n') for name in categories
        ):
            raise ValueError("Canonical category names are invalid for the JSON trie")
        self.edges = []
        start = self._new()
        target = self._literal(start, '{"target":"')
        after_target = self._new()
        for literal in ("not seen", *distance_labels):
            state = self._literal(target, literal)
            self.edges[state]['"'] = after_target
        array = self._literal(after_target, ',"objects":[')
        root, after_element, after_array = self._new(), self._new(), self._new()
        self.edges[array]['"'], self.edges[array]["]"] = root, after_array
        self.subtree, self.terminals = {root: 0}, {}
        for index, name in enumerate(categories):
            bit, state = 1 << index, root
            self.subtree[root] |= bit
            for char in name:
                state = self._literal(state, char)
                self.subtree[state] = self.subtree.get(state, 0) | bit
            self.terminals[state] = index
            self.edges[state]['"'] = after_element
        quote = self._new()
        self.edges[after_element][","], self.edges[after_element]["]"] = quote, after_array
        self.edges[quote]['"'] = root
        self.after_element, self.handoff = after_element, after_array
        self.cue_ids = tuple(tokenizer.encode(',"action":"', add_special_tokens=False))
        boundaries, consumed = {}, 0
        cue_texts = [tokenizer.decode([token]) for token in self.cue_ids]
        for offset, text in enumerate(cue_texts):
            consumed += len(text)
            boundaries[consumed] = offset + 1
        self.cue_nodes, self.cue_offsets = set(), {}
        state = after_array
        for index, char in enumerate("".join(cue_texts)):
            state = self._literal(state, char)
            self.cue_nodes.add(state)
            if index + 1 in boundaries:
                self.cue_offsets[state] = boundaries[index + 1]
        special = set(tokenizer.all_special_ids)
        strings = tokenizer.batch_decode([[token] for token in range(len(tokenizer))],
            skip_special_tokens=False, clean_up_tokenization_spaces=False)
        self.buckets = {}
        for token, text in enumerate(strings):
            if token not in special and text and "\ufffd" not in text:
                self.buckets.setdefault(text[0], []).append((token, text))
        self.static = {}

    def _new(self):
        self.edges.append({})
        return len(self.edges) - 1

    def _literal(self, state, text):
        for char in text:
            if char not in self.edges[state]:
                self.edges[state][char] = self._new()
            state = self.edges[state][char]
        return state

    def initial_state(self):
        return 0, 0, 0

    def cue_offset(self, state):
        if state[0] == self.handoff:
            return 0
        return self.cue_offsets.get(state[0])

    def _walk(self, state, text):
        node, used, count = state
        for char in text:
            nxt = self.edges[node].get(char)
            if nxt is None:
                return None
            if char == '"' and node in self.terminals:
                bit = 1 << self.terminals[node]
                if used & bit:
                    return None
                used |= bit
                count += 1
            elif nxt in self.subtree:
                if not self.subtree[nxt] & ~used:
                    return None
            elif node == self.after_element and char == "," and count >= 16:
                return None
            node = nxt
        if node in self.cue_nodes and node not in self.cue_offsets:
            return None
        return node, used, count

    def allowed(self, state):
        node = state[0]
        if node not in self.static:
            candidates = []
            for first in self.edges[node]:
                for token, text in self.buckets.get(first, ()):
                    current = node
                    for char in text:
                        current = self.edges[current].get(char)
                        if current is None:
                            break
                    if current is not None:
                        candidates.append((token, text))
            self.static[node] = candidates
        allowed = {}
        for token, text in self.static[node]:
            landed = self._walk(state, text)
            if landed is not None:
                allowed[token] = landed
        if not allowed and self.cue_offset(state) is None:
            raise RuntimeError("Semantic trie has no legal token continuation")
        return allowed
