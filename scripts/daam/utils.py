from __future__ import annotations
from itertools import chain
from functools import lru_cache
from pathlib import Path
import random
import re

from PIL import Image, ImageFont, ImageDraw
from fonts.ttf import Roboto
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
# import spacy
import torch
import torch.nn.functional as F
from modules.devices import dtype

from ldm.modules.encoders.modules import FrozenCLIPEmbedder, FrozenOpenCLIPEmbedder
import open_clip.tokenizer
from modules.sd_hijack_clip import FrozenCLIPEmbedderWithCustomWordsBase, FrozenCLIPEmbedderWithCustomWords
from modules.sd_hijack_open_clip import FrozenOpenCLIPEmbedderWithCustomWords
from modules.shared import opts

__all__ = ['expand_image', 'set_seed', 'escape_prompt', 'calc_context_size', 'compute_token_merge_indices', 'compute_token_merge_indices_with_tokenizer', 'image_overlay_heat_map', 'plot_overlay_heat_map', 'plot_mask_heat_map', 'PromptAnalyzer']

def expand_image(im: torch.Tensor, h = 512, w = 512,  absolute: bool = False, threshold: float = None) -> torch.Tensor:

    im = im.unsqueeze(0).unsqueeze(0)
    im = F.interpolate(im.float().detach(), size=(h, w), mode='bicubic')

    if not absolute:
        im = (im - im.min()) / (im.max() - im.min() + 1e-8)

    if threshold:
        im = (im > threshold).float()

    # im = im.cpu().detach()

    return im.squeeze()

def _write_on_image(img, caption, font_size = 32):
    ix,iy = img.size
    draw = ImageDraw.Draw(img)
    margin=2
    fontsize=font_size
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(Roboto, fontsize)
    text_height=iy-60
    tx = draw.textbbox((0,0),caption,font)
    draw.text((int((ix-tx[2])/2),text_height+margin),caption,(0,0,0),font=font)
    draw.text((int((ix-tx[2])/2),text_height-margin),caption,(0,0,0),font=font)
    draw.text((int((ix-tx[2])/2+margin),text_height),caption,(0,0,0),font=font)
    draw.text((int((ix-tx[2])/2-margin),text_height),caption,(0,0,0),font=font)
    draw.text((int((ix-tx[2])/2),text_height), caption,(255,255,255),font=font)
    return img

def image_overlay_heat_map(img, heat_map, word=None, out_file=None, crop=None, alpha=0.5, caption=None, image_scale=1.0):
    # type: (Image.Image | np.ndarray, torch.Tensor, str, Path, int, float, str, float) -> Image.Image
    assert(img is not None)

    if heat_map is not None:
        shape : torch.Size = heat_map.shape
        # heat_map = heat_map.unsqueeze(-1).expand(shape[0], shape[1], 3).clone()
        heat_map = _convert_heat_map_colors(heat_map)
        heat_map = heat_map.to('cpu').detach().numpy().copy().astype(np.uint8)
        heat_map_img = Image.fromarray(heat_map)

        img = Image.blend(img, heat_map_img, alpha)
    else:
        img = img.copy()

    if caption:
        img = _write_on_image(img, caption)

    if image_scale != 1.0:
        x, y = img.size
        size = (int(x * image_scale), int(y * image_scale))
        img = img.resize(size, Image.BICUBIC)

    return img


def _convert_heat_map_colors(heat_map : torch.Tensor):
    def get_color(value):
        return np.array(cm.turbo(value / 255)[0:3])

    color_map = np.array([ get_color(i) * 255 for i in range(256) ])
    color_map = torch.tensor(color_map, device=heat_map.device, dtype=dtype)

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

def calc_context_size(token_length : int):
    len_check = 0 if (token_length - 1) < 0 else token_length - 1
    return ((int)(len_check // 75) + 1) * 77

def escape_prompt(prompt):
    if type(prompt) is str:
        prompt = prompt.lower()
        prompt = re.sub(r"[\(\)\[\]]", "", prompt)
        prompt = re.sub(r":\d+\.*\d*", "", prompt)
        return prompt
    elif type(prompt) is list:
        prompt_new = []
        for i in range(len(prompt)):
            prompt_new.append(escape_prompt(prompt[i]))
        return prompt_new


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

    escaped_prompt = escape_prompt(prompt)
    # escaped_prompt = re.sub(r"[_-]", " ", escaped_prompt)
    tokens : list = tokenize(escaped_prompt)
    word = word.lower()
    merge_idxs = []

    needles = tokenize(word)

    if len(needles) == 0:
        return []

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

def compute_token_merge_indices_with_tokenizer(tokenizer, prompt: str, word: str, word_idx: int = None, limit : int = -1):

    escaped_prompt = escape_prompt(prompt)
    # escaped_prompt = re.sub(r"[_-]", " ", escaped_prompt)
    tokens : list = tokenizer.tokenize(escaped_prompt)
    word = word.lower()
    merge_idxs = []

    needles = tokenizer.tokenize(word)

    if len(needles) == 0:
        return []

    limit_count = 0
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
                if limit > 0:
                    limit_count += 1
                    if limit_count >= limit:
                        break

        elif needles[0] == token:
            merge_idxs.append(i)
            if limit > 0:
                limit_count += 1
                if limit_count >= limit:
                    break

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

class PromptAnalyzer:
    def __init__(self, clip : FrozenCLIPEmbedderWithCustomWordsBase, text : str):
        use_old = opts.use_old_emphasis_implementation
        assert not use_old, "use_old_emphasis_implementation is not supported"

        self.clip = clip
        self.id_start = clip.id_start
        self.id_end = clip.id_end
        self.is_open_clip = True if type(clip) == FrozenOpenCLIPEmbedderWithCustomWords else False
        self.used_custom_terms = []
        self.hijack_comments = []

        chunks, token_count = self.tokenize_line(text)

        self.token_count = token_count
        self.fixes = list(chain.from_iterable(chunk.fixes for chunk in chunks))
        self.context_size = calc_context_size(token_count)

        tokens = list(chain.from_iterable(chunk.tokens for chunk in chunks))
        multipliers = list(chain.from_iterable(chunk.multipliers for chunk in chunks))

        self.tokens = []
        self.multipliers = []
        for i in range(self.context_size // 77):
            self.tokens.extend([self.id_start] + tokens[i*75:i*75+75] + [self.id_end])
            self.multipliers.extend([1.0] + multipliers[i*75:i*75+75]+ [1.0])

    def create(self, text : str):
        return PromptAnalyzer(self.clip, text)

    def tokenize_line(self, line):
        chunks, token_count = self.clip.tokenize_line(line)
        return chunks, token_count

    def process_text(self, texts):
        batch_multipliers, remade_batch_tokens, used_custom_terms, hijack_comments, hijack_fixes, token_count = self.clip.process_text(texts)
        return batch_multipliers, remade_batch_tokens, used_custom_terms, hijack_comments, hijack_fixes, token_count

    def encode(self, text : str):
        return self.clip.tokenize([text])[0]

    def calc_word_indecies(self, word : str, limit : int = -1, start_pos = 0):
        word = word.lower()
        merge_idxs = []

        tokens = self.tokens
        needles = self.encode(word)

        limit_count = 0
        current_pos = 0
        for i, token in enumerate(tokens):
            current_pos = i
            if i < start_pos:
                continue

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
                    if limit > 0:
                        limit_count += 1
                        if limit_count >= limit:
                            break

            elif needles[0] == token:
                merge_idxs.append(i)
                if limit > 0:
                    limit_count += 1
                    if limit_count >= limit:
                        break

        return merge_idxs, current_pos