from __future__ import annotations
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import List, Type, Any, Literal, Dict
import math
from modules.devices import device

from ldm.models.diffusion.ddpm import DiffusionWrapper, LatentDiffusion
from ldm.modules.diffusionmodules.openaimodel import UNetModel
from ldm.modules.attention import CrossAttention, default, exists

import numba
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat

from .experiment import COCO80_LABELS
from .hook import ObjectHooker, AggregateHooker, UNetCrossAttentionLocator
from .utils import compute_token_merge_indices, PromptAnalyzer


__all__ = ['trace', 'DiffusionHeatMapHooker', 'HeatMap', 'MmDetectHeatMap']


class UNetForwardHooker(ObjectHooker[UNetModel]):
    def __init__(self, module: UNetModel, heat_maps: defaultdict(defaultdict)):
        super().__init__(module)
        self.all_heat_maps = []
        self.heat_maps = heat_maps

    def _hook_impl(self):
        self.monkey_patch('forward', self._forward)

    def _unhook_impl(self):
        pass

    def _forward(hk_self, self, *args, **kwargs):
        super_return = hk_self.monkey_super('forward', *args, **kwargs)
        hk_self.all_heat_maps.append(deepcopy(hk_self.heat_maps))
        hk_self.heat_maps.clear()

        return super_return


class HeatMap:
    def __init__(self, prompt_analyzer: PromptAnalyzer, prompt: str, heat_maps: torch.Tensor):
        self.prompt_analyzer = prompt_analyzer.create(prompt)
        self.heat_maps = heat_maps
        self.prompt = prompt

    def compute_word_heat_map(self, word: str, word_idx: int = None) -> torch.Tensor:
        merge_idxs, _ = self.prompt_analyzer.calc_word_indecies(word)
        if len(merge_idxs) == 0:
            return None
        
        return self.heat_maps[merge_idxs].mean(0)


class MmDetectHeatMap:
    def __init__(self, pred_file: str | Path, threshold: float = 0.95):
        @numba.njit
        def _compute_mask(masks: np.ndarray, bboxes: np.ndarray):
            x_any = np.any(masks, axis=1)
            y_any = np.any(masks, axis=2)
            num_masks = len(bboxes)

            for idx in range(num_masks):
                x = np.where(x_any[idx, :])[0]
                y = np.where(y_any[idx, :])[0]
                bboxes[idx, :4] = np.array([x[0], y[0], x[-1] + 1, y[-1] + 1], dtype=np.float32)

        pred_file = Path(pred_file)
        self.word_masks: Dict[str, torch.Tensor] = defaultdict(lambda: 0)
        bbox_result, masks = torch.load(pred_file)
        labels = [np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(bbox_result)]
        labels = np.concatenate(labels)
        bboxes = np.vstack(bbox_result)

        if masks is not None and bboxes[:, :4].sum() == 0:
            _compute_mask(masks, bboxes)
            scores = bboxes[:, -1]
            inds = scores > threshold
            labels = labels[inds]
            masks = masks[inds, ...]

            for lbl, mask in zip(labels, masks):
                self.word_masks[COCO80_LABELS[lbl]] |= torch.from_numpy(mask)

            self.word_masks = {k: v.float() for k, v in self.word_masks.items()}

    def compute_word_heat_map(self, word: str) -> torch.Tensor:
        return self.word_masks[word]


class DiffusionHeatMapHooker(AggregateHooker):
    def __init__(self, model: LatentDiffusion, heigth : int, width : int, context_size : int = 77, weighted: bool = False, layer_idx: int = None, head_idx: int = None):
        heat_maps = defaultdict(lambda: defaultdict(list)) # batch index, factor, attention
        modules = [UNetCrossAttentionHooker(x, heigth, width, heat_maps, context_size=context_size, weighted=weighted, head_idx=head_idx) for x in UNetCrossAttentionLocator().locate(model.model.diffusion_model, layer_idx)]
        self.forward_hook = UNetForwardHooker(model.model.diffusion_model, heat_maps)
        modules.append(self.forward_hook)
        
        self.height = heigth
        self.width = width
        self.model = model
        self.last_prompt = ''
        
        super().__init__(modules)

        

    @property
    def all_heat_maps(self):
        return self.forward_hook.all_heat_maps
    
    def reset(self):
        map(lambda module: module.reset(), self.module)
        return self.forward_hook.all_heat_maps.clear()

    def compute_global_heat_map(self, prompt_analyzer, prompt, batch_index, time_weights=None, time_idx=None, last_n=None, first_n=None, factors=None):
        # type: (PromptAnalyzer, str, int, int, int, int, int, List[float]) -> HeatMap
        """
        Compute the global heat map for the given prompt, aggregating across time (inference steps) and space (different
        spatial transformer block heat maps).

        Args:
            prompt: The prompt to compute the heat map for.
            time_weights: The weights to apply to each time step. If None, all time steps are weighted equally.
            time_idx: The time step to compute the heat map for. If None, the heat map is computed for all time steps.
                Mutually exclusive with `last_n` and `first_n`.
            last_n: The number of last n time steps to use. If None, the heat map is computed for all time steps.
                Mutually exclusive with `time_idx`.
            first_n: The number of first n time steps to use. If None, the heat map is computed for all time steps.
                Mutually exclusive with `time_idx`.
            factors: Restrict the application to heat maps with spatial factors in this set. If `None`, use all sizes.
        """
        if len(self.forward_hook.all_heat_maps) == 0:
            return None
        
        if time_weights is None:
            time_weights = [1.0] * len(self.forward_hook.all_heat_maps)

        time_weights = np.array(time_weights)
        time_weights /= time_weights.sum()
        all_heat_maps = self.forward_hook.all_heat_maps

        if time_idx is not None:
            heat_maps = [all_heat_maps[time_idx]]
        else:
            heat_maps = all_heat_maps[-last_n:] if last_n is not None else all_heat_maps
            heat_maps = heat_maps[:first_n] if first_n is not None else heat_maps
            

        if factors is None:
            factors = {1, 2, 4, 8, 16, 32}
        else:
            factors = set(factors)

        all_merges = []
        
        for batch_to_heat_maps in heat_maps:
            
            if not (batch_index in batch_to_heat_maps):
                continue    
            
            merge_list = []
                 
            factors_to_heat_maps = batch_to_heat_maps[batch_index]

            for k, heat_map in factors_to_heat_maps.items():
                # heat_map shape: (tokens, 1, height, width)
                # each v is a heat map tensor for a layer of factor size k across the tokens
                if k in factors:
                    merge_list.append(torch.stack(heat_map, 0).mean(0))

            if  len(merge_list) > 0:
               all_merges.append(merge_list)

        maps = torch.stack([torch.stack(x, 0) for x in all_merges], dim=0)
        maps = maps.sum(0).to(device).sum(2).sum(0)

        return HeatMap(prompt_analyzer, prompt, maps)


class UNetCrossAttentionHooker(ObjectHooker[CrossAttention]):
    def __init__(self, module: CrossAttention, img_height : int, img_width : int, heat_maps: defaultdict(defaultdict), context_size: int = 77, weighted: bool = False, head_idx: int = 0):
        super().__init__(module)
        self.heat_maps = heat_maps
        self.context_size = context_size
        self.weighted = weighted
        self.head_idx = head_idx
        self.img_height = img_height
        self.img_width =  img_width
        self.calledCount = 0
        
    def reset(self):
        self.heat_maps.clear()
        self.calledCount = 0
        
    @torch.no_grad()
    def _up_sample_attn(self, x, value, factor, method='bicubic'):
        # type: (torch.Tensor, torch.Tensor, int, Literal['bicubic', 'conv']) -> torch.Tensor
        # x shape: (heads, height * width, tokens)
        """
        Up samples the attention map in x using interpolation to the maximum size of (64, 64), as assumed in the Stable
        Diffusion model.

        Args:
            x (`torch.Tensor`): cross attention slice/map between the words and the tokens.
            value (`torch.Tensor`): the value tensor.
            method (`str`): the method to use; one of `'bicubic'` or `'conv'`.

        Returns:
            `torch.Tensor`: the up-sampled attention map of shape (tokens, 1, height, width).
        """
        weight = torch.full((factor, factor), 1 / factor ** 2, device=x.device)
        weight = weight.view(1, 1, factor, factor)
        
        h = int(math.sqrt ( (self.img_height * x.size(1)) / self.img_width))
        w = int(self.img_width * h / self.img_height)
        
        h_fix = w_fix = 64
        if h >= w:
            w_fix = int((w * h_fix) / h)
        else:
            h_fix = int((h * w_fix) / w)
                
        maps = []
        x = x.permute(2, 0, 1)
        value = value.permute(1, 0, 2)
        weights = 1

        with torch.cuda.amp.autocast(dtype=torch.float32):
            for map_ in x:
                map_ = map_.unsqueeze(1).view(map_.size(0), 1, h, w)

                if method == 'bicubic':
                    map_ = F.interpolate(map_, size=(h_fix, w_fix), mode='bicubic')
                    maps.append(map_.squeeze(1))
                else:
                    maps.append(F.conv_transpose2d(map_, weight, stride=factor).squeeze(1))

        if self.weighted:
            weights = value.norm(p=1, dim=-1, keepdim=True).unsqueeze(-1)

        maps = torch.stack(maps, 0)  # shape: (tokens, heads, height, width)
        
        if self.head_idx:
            maps = maps[:, self.head_idx:self.head_idx+1, :, :]

        return (weights * maps).sum(1, keepdim=True).cpu()
    
    def _forward(hk_self, self, x, context=None, mask=None):
        hk_self.calledCount += 1
        batch_size, sequence_length, _ = x.shape
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)
        
        dim = q.shape[-1]

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)
        
        out = hk_self._hooked_attention(self, q, k, v, batch_size, sequence_length, dim)
        
        return self.to_out(out)
    
    ### forward implemetation of diffuser CrossAttention
    # def forward(self, hidden_states, context=None, mask=None):
    #     batch_size, sequence_length, _ = hidden_states.shape

    #     query = self.to_q(hidden_states)
    #     context = context if context is not None else hidden_states
    #     key = self.to_k(context)
    #     value = self.to_v(context)

    #     dim = query.shape[-1]

    #     query = self.reshape_heads_to_batch_dim(query)
    #     key = self.reshape_heads_to_batch_dim(key)
    #     value = self.reshape_heads_to_batch_dim(value)

    #     # TODO(PVP) - mask is currently never used. Remember to re-implement when used

    #     # attention, what we cannot get enough of
    #     if self._use_memory_efficient_attention_xformers:
    #         hidden_states = self._memory_efficient_attention_xformers(query, key, value)
    #         # Some versions of xformers return output in fp32, cast it back to the dtype of the input
    #         hidden_states = hidden_states.to(query.dtype)
    #     else:
    #         if self._slice_size is None or query.shape[0] // self._slice_size == 1:
    #             hidden_states = self._attention(query, key, value)
    #         else:
    #             hidden_states = self._sliced_attention(query, key, value, sequence_length, dim)

    #     # linear proj
    #     hidden_states = self.to_out[0](hidden_states)
    #     # dropout
    #     hidden_states = self.to_out[1](hidden_states)
    #     return hidden_states

    def _hooked_attention(hk_self, self, query, key, value, batch_size, sequence_length, dim, use_context: bool = True):
        """
        Monkey-patched version of :py:func:`.CrossAttention._attention` to capture attentions and aggregate them.

        Args:
            hk_self (`UNetCrossAttentionHooker`): pointer to the hook itself.
            self (`CrossAttention`): pointer to the module.
            query (`torch.Tensor`): the query tensor.
            key (`torch.Tensor`): the key tensor.
            value (`torch.Tensor`): the value tensor.
            batch_size (`int`): the batch size
            use_context (`bool`): whether to check if the resulting attention slices are between the words and the image
        """
        batch_size_attention = query.shape[0]
        hidden_states = torch.zeros(
            (batch_size_attention, sequence_length, dim // self.heads), device=query.device, dtype=query.dtype
        )
        slice_size = hidden_states.shape[0] // batch_size # self._slice_size if self._slice_size is not None else hidden_states.shape[0]
        
        def calc_factor_base(w, h):
            z = max(w/64, h/64)
            factor_b = min(w, h) * z
            return factor_b
        
        factor_base = calc_factor_base(hk_self.img_width, hk_self.img_height)
                
        for batch_index in range(hidden_states.shape[0] // slice_size):
            start_idx = batch_index * slice_size
            end_idx = (batch_index + 1) * slice_size
            attn_slice = (
                    torch.einsum("b i d, b j d -> b i j", query[start_idx:end_idx], key[start_idx:end_idx]) * self.scale
            )
            factor = int(math.sqrt(factor_base // attn_slice.shape[1]))
            attn_slice = attn_slice.softmax(-1)
            hid_states = torch.einsum("b i j, b j d -> b i d", attn_slice, value[start_idx:end_idx])            
                
            if use_context and  hk_self.calledCount % 2 == 1 and attn_slice.shape[-1] == hk_self.context_size:    
                if factor >= 1:
                    factor //= 1
                    maps = hk_self._up_sample_attn(attn_slice, value, factor)
                    hk_self.heat_maps[batch_index][factor].append(maps)

            hidden_states[start_idx:end_idx] = hid_states

        # reshape hidden_states
        hidden_states = hk_self.reshape_batch_dim_to_heads(self, hidden_states)
        return hidden_states
    
    def reshape_batch_dim_to_heads(hk_self, self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
        return tensor

    def _hook_impl(self):
        self.monkey_patch('forward', self._forward)

    @property
    def num_heat_maps(self):
        return len(next(iter(self.heat_maps.values())))


trace: Type[DiffusionHeatMapHooker] = DiffusionHeatMapHooker
