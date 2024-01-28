from __future__ import annotations
from typing import List, Generic, TypeVar, Callable, Union, Any
import functools
import itertools
from ldm.modules.attention import SpatialTransformer

from ldm.modules.diffusionmodules.openaimodel import UNetModel
from ldm.modules.attention import CrossAttention
import torch.nn as nn


__all__ = ['ObjectHooker', 'ModuleLocator', 'AggregateHooker', 'UNetCrossAttentionLocator']


ModuleType = TypeVar('ModuleType')
ModuleListType = TypeVar('ModuleListType', bound=List)


class ModuleLocator(Generic[ModuleType]):
    def locate(self, model: nn.Module) -> List[ModuleType]:
        raise NotImplementedError


class ObjectHooker(Generic[ModuleType]):
    def __init__(self, module: ModuleType):
        self.module: ModuleType = module
        self.hooked = False
        self.old_state = dict()

    def __enter__(self):
        self.hook()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unhook()

    def hook(self):
        if self.hooked:
            raise RuntimeError('Already hooked module')

        self.old_state = dict()
        self.hooked = True
        self._hook_impl()

        return self

    def unhook(self):
        if not self.hooked:
            raise RuntimeError('Module is not hooked')

        for k, v in self.old_state.items():
            if k.startswith('old_fn_'):
                setattr(self.module, k[7:], v)

        self.hooked = False
        self._unhook_impl()

        return self

    def monkey_patch(self, fn_name, fn):
        self.old_state[f'old_fn_{fn_name}'] = getattr(self.module, fn_name)
        setattr(self.module, fn_name, functools.partial(fn, self.module))

    def monkey_super(self, fn_name, *args, **kwargs):
        return self.old_state[f'old_fn_{fn_name}'](*args, **kwargs)

    def _hook_impl(self):
        raise NotImplementedError

    def _unhook_impl(self):
        pass


class AggregateHooker(ObjectHooker[ModuleListType]):
    def _hook_impl(self):
        for h in self.module:
            h.hook()

    def _unhook_impl(self):
        for h in self.module:
            h.unhook()

    def register_hook(self, hook: ObjectHooker):
        self.module.append(hook)


class UNetCrossAttentionLocator(ModuleLocator[CrossAttention]):
    def locate(self, model: UNetModel, layer_idx: int) -> List[CrossAttention]:
        """
        Locate all cross-attention modules in a UNetModel.

        Args:
            model (`UNetModel`): The model to locate the cross-attention modules in.

        Returns:
            `List[CrossAttention]`: The list of cross-attention modules.
        """
        blocks = []
        
        for i, unet_block in enumerate(itertools.chain(model.input_blocks, [model.middle_block], model.output_blocks)):
            # if 'CrossAttn' in unet_block.__class__.__name__:
            if not layer_idx or i == layer_idx:
                for module in unet_block.modules():
                    if module.__class__.__name__ == "SpatialTransformer":
                        spatial_transformer = module
                        for basic_transformer_block in spatial_transformer.transformer_blocks:
                            blocks.append(basic_transformer_block.attn2)
                            

        return blocks
