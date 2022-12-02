from functools import lru_cache
from pathlib import Path
import random
import re

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
# import spacy
import torch
import torch.nn.functional as F

from ldm.modules.encoders.modules import FrozenCLIPEmbedder, FrozenOpenCLIPEmbedder
import open_clip.tokenizer


__all__ = ['expand_image', 'set_seed', 'compute_token_merge_indices', 'image_overlay_heat_map', 'plot_overlay_heat_map', 'plot_mask_heat_map']


def expand_image(im: torch.Tensor, h = 512, w = 512,  absolute: bool = False, threshold: float = None) -> torch.Tensor:

    im = im.unsqueeze(0).unsqueeze(0)
    im = F.interpolate(im.float().detach(), size=(h, w), mode='bicubic')

    if not absolute:
        im = (im - im.min()) / (im.max() - im.min() + 1e-8)

    if threshold:
        im = (im > threshold).float()

    # im = im.cpu().detach()

    return im.squeeze()

def image_overlay_heat_map(im, heat_map, word=None, out_file=None, crop=None, alpha=0.5):
    # type: (Image.Image | np.ndarray, torch.Tensor, str, Path, int, float) -> Image.Image

    # im = im.numpy().array()
        
    shape : torch.Size = heat_map.shape
    # heat_map = heat_map.unsqueeze(-1).expand(shape[0], shape[1], 3).clone()
    heat_map = _convert_heat_map_colors(heat_map)
    heat_map = heat_map.to('cpu').detach().numpy().copy().astype(np.uint8)
    heat_map_img = Image.fromarray(heat_map)
        
    return Image.blend(im, heat_map_img, alpha)
    

def _convert_heat_map_colors(heat_map : torch.Tensor):
    
    color_gradients = np.array([
        [0, 0.0, 0.0, 0.0],  # Black
        [0.25, 0.0, 0.0, 1.0],  # Blue
        [0.5, 0.0, 1.0, 0.0],  # Green
        [0.75, 1.0, 1.0, 0.0],  # Yellow
        [1.0, 1.0, 0.0, 0.0],  # Red
    ])
    
    def percentile(a, b, percentile):
        return ((1.0 - percentile) * a + percentile * b)
    
    def get_color(value):
        for idx, v in enumerate(color_gradients):
            if value <= v[0] * 255:
                if idx == 0:
                    return v[1:]
                else:
                    current = color_gradients[idx]
                    prev = color_gradients[idx-1]
                    p = (current[0] * 255 - value) / (current[0] * 255 - prev[0] * 255)
                    return percentile(current[1:],  prev[1:], p)
        
        return color_gradients[0][1:]
    
    color_map = np.array([ get_color(i) * 255 for i in range(256) ])
    color_map = torch.tensor(color_map, device=heat_map.device)
    
    heat_map = (heat_map * 255).long()
    
    return color_map[heat_map]

def plot_overlay_heat_map(im, heat_map, word=None, out_file=None, crop=None):
    # type: (Image.Image | np.ndarray, torch.Tensor, str, Path, int) -> None
    plt.clf()
    plt.rcParams.update({'font.size': 24})

    im = np.array(im)
    if crop is not None:
        heat_map = heat_map.squeeze()[crop:-crop, crop:-crop]
        im = im[crop:-crop, crop:-crop]

    plt.imshow(heat_map.squeeze().cpu().numpy(), cmap='jet')
    im = torch.from_numpy(im).float() / 255
    im = torch.cat((im, (1 - heat_map.unsqueeze(-1))), dim=-1)
    plt.imshow(im)

    if word is not None:
        plt.title(word)

    if out_file is not None:
        plt.savefig(out_file)


def plot_mask_heat_map(im: Image.Image, heat_map: torch.Tensor, threshold: float = 0.4):
    im = torch.from_numpy(np.array(im)).float() / 255
    mask = (heat_map.squeeze() > threshold).float()
    im = im * mask.unsqueeze(-1)
    plt.imshow(im)


def set_seed(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    gen = torch.Generator(device='cuda')
    gen.manual_seed(seed)

    return gen


def compute_token_merge_indices(model, prompt: str, word: str, word_idx: int = None):
        
    clip = None
    tokenize = None
    if type(model.cond_stage_model.wrapped) == FrozenCLIPEmbedder:
        clip : FrozenCLIPEmbedder = model.cond_stage_model.wrapped
        tokenize = clip.tokenizer.tokenize
    elif type(model.cond_stage_model.wrapped) == FrozenOpenCLIPEmbedder:
        clip : FrozenOpenCLIPEmbedder = model.cond_stage_model.wrapped
        tokenize = open_clip.tokenizer._tokenizer.encode
    else:
        assert False
        
    prompt = prompt.lower()
    escaped_prompt = re.sub(r"[\(\)\[\]]", "", prompt)
    escaped_prompt = re.sub(r":\d+\.*\d*", "", escaped_prompt)
    # escaped_prompt = re.sub(r"[_-]", " ", escaped_prompt)
    tokens : list = tokenize(escaped_prompt)
    word = word.lower()
    merge_idxs = []
    
    needles = tokenize(word)
        
    for i, token in enumerate(tokens):
        if needles[0] == token and len(needles) > 1:
            next = i + 1
            success = True
            for needle in needles[1:]:
                if next >= len(tokens) or needle != tokens[next]:
                    success = False
                    break
                next += 1
            
            # append consecutive indexes if all pass
            if success:
                merge_idxs.extend(list(range(i, next)))
            
        elif needles[0] == token:
            merge_idxs.append(i)
            
    idxs = []
    for x in merge_idxs:
        seq = (int)(x / 75)
        if seq == 0:
            idxs.append(x + 1) # padding
        else:
            idxs.append(x + 1 + seq*2) # If tokens exceed 75, they are split.

    return idxs

nlp = None


@lru_cache(maxsize=100000)
def cached_nlp(prompt: str, type='en_core_web_md'):
    global nlp

#    if nlp is None:
#       nlp = spacy.load(type)

    return nlp(prompt)
