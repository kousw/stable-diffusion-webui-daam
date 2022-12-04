from __future__ import annotations
from collections import defaultdict
import os
import re
import traceback

import gradio as gr
import modules.images as images
import modules.scripts as scripts
import torch
from ldm.modules.encoders.modules import FrozenCLIPEmbedder, FrozenOpenCLIPEmbedder
import open_clip.tokenizer
from modules import script_callbacks
from modules.processing import (Processed, StableDiffusionProcessing, fix_seed,
                                process_images)
from modules.shared import cmd_opts, opts, state
import modules.shared as shared
from PIL import Image

from scripts.daam import trace, utils

before_image_saved_handler = None

class Script(scripts.Script):
    
    GRID_LAYOUT_AUTO = "Auto"
    GRID_LAYOUT_PREVENT_EMPTY = "Prevent Empty Spot"
    GRID_LAYOUT_BATCH_LENGTH_AS_ROW = "Batch Length As Row"
    

    def title(self):
        return "Daam script"

    def show(self, is_img2img):
        return True

    def ui(self, is_img2img):
                
        attention_texts = gr.Text(label='Attention texts for visualization. (comma separated)', value='')

        with gr.Row():
            hide_images = gr.Checkbox(label='Hide heatmap images', value=False)
            
            hide_caption = gr.Checkbox(label='Hide caption', value=False)
            
        with gr.Row():
            use_grid = gr.Checkbox(label='Use grid (output to grid dir)', value=False)
                
            grid_layouyt = gr.Dropdown(
                    [Script.GRID_LAYOUT_AUTO, Script.GRID_LAYOUT_PREVENT_EMPTY, Script.GRID_LAYOUT_BATCH_LENGTH_AS_ROW], label="Grid layout",
                    value=Script.GRID_LAYOUT_AUTO
                )
                
        with gr.Row():
            alpha = gr.Slider(label='Heatmap blend alpha', value=0.5, minimum=0, maximum=1, step=0.01)
        
            heatmap_image_scale = gr.Slider(label='Heatmap image scale', value=1.0, minimum=0.1, maximum=1, step=0.025)
        
        self.tracer = None
        
        return [attention_texts, hide_images, hide_caption, use_grid, grid_layouyt, alpha, heatmap_image_scale] 
    
    def run(self,
            p : StableDiffusionProcessing, 
            attention_texts : str, 
            hide_images : bool, 
            hide_caption : bool, 
            use_grid : bool, 
            grid_layouyt :str,
            alpha : float, 
            heatmap_image_scale : float):

        initial_info = None
        self.images = []
        self.hide_images = hide_images
        self.hide_caption = hide_caption
        self.alpha = alpha
        self.use_grid = use_grid
        self.grid_layouyt = grid_layouyt
        self.heatmap_image_scale = heatmap_image_scale
        self.grid_images = list()
        
        fix_seed(p)
        
        styled_prompt = shared.prompt_styles.apply_styles_to_prompt(p.prompt, p.styles)
        
        attentions = [ s.strip() for s in attention_texts.split(",") if s.strip() ]
        self.attentions = attentions
        
        clip = None
        tokenize = None
        if type(p.sd_model.cond_stage_model.wrapped) == FrozenCLIPEmbedder:
            clip : FrozenCLIPEmbedder = p.sd_model.cond_stage_model.wrapped
            tokenize = clip.tokenizer.tokenize
        elif type(p.sd_model.cond_stage_model.wrapped) == FrozenOpenCLIPEmbedder:
            clip : FrozenOpenCLIPEmbedder = p.sd_model.cond_stage_model.wrapped
            tokenize = open_clip.tokenizer._tokenizer.encode
        else:
            assert False
            
        tokens = tokenize(utils.escape_prompt(styled_prompt))
        len_check = 0 if (len(tokens) - 1) < 0 else len(tokens) - 1
        context_size = ((int)(len_check // 75) + 1) * 77
        
        print("daam run with context_size=", context_size)
        
        global before_image_saved_handler
        before_image_saved_handler = lambda params : self.before_image_saved(params)
                
        with torch.no_grad():
            with trace(p.sd_model, p.height, p.width, context_size) as tr:
                self.tracer = tr
                               
                proc = process_images(p)
                if initial_info is None:
                    initial_info = proc.info
                self.images  += proc.images        
                
                self.tracer = None        

        before_image_saved_handler = None
        
        if self.use_grid and len(self.grid_images) > 0:

            grid_layout = self.grid_layouyt
            if grid_layout == Script.GRID_LAYOUT_AUTO:
                if p.batch_size * p.n_iter == 1:
                    grid_layout = Script.GRID_LAYOUT_PREVENT_EMPTY
                else:
                    grid_layout = Script.GRID_LAYOUT_BATCH_LENGTH_AS_ROW
                       
            if grid_layout == Script.GRID_LAYOUT_PREVENT_EMPTY:
                grid_img = images.image_grid(self.grid_images)
            elif grid_layout == Script.GRID_LAYOUT_BATCH_LENGTH_AS_ROW:
                grid_img = images.image_grid(self.grid_images, batch_size=p.batch_size, rows=p.batch_size * p.n_iter)
            else:
                pass
            
            images.save_image(grid_img, p.outpath_grids, "grid_daam", grid=True, p=p)
            if not self.hide_images:
                self.images += [grid_img]

        processed = Processed(p, self.images, p.seed, initial_info)

        return processed
    
    def before_image_saved(self, params : script_callbacks.ImageSaveParams):               
        batch_pos = -1
        if params.p.batch_size > 1:
            match  = re.search(r"Batch pos: (\d+)", params.pnginfo['parameters'])
            if match:
                batch_pos = int(match.group(1))
        else:
            batch_pos = 0
            
        if batch_pos < 0:
            return        
        
        if self.tracer is not None and len(self.attentions) > 0:
            with torch.no_grad():
                styled_prompot = shared.prompt_styles.apply_styles_to_prompt(params.p.prompt, params.p.styles)
                global_heat_map = self.tracer.compute_global_heat_map(styled_prompot, batch_pos)                
                
                if global_heat_map is not None:
                    grid_images = []
                    for attention in self.attentions:
                                
                        img_size = params.image.size
                        caption = attention if not self.hide_caption else None
                        
                        heat_map = global_heat_map.compute_word_heat_map(attention)
                        if heat_map is None : print(f"No heatmaps for '{attention}'")
                        
                        heat_map_img = utils.expand_image(heat_map, img_size[1], img_size[0]) if heat_map is not None else None
                        img : Image.Image = utils.image_overlay_heat_map(params.image, heat_map_img, alpha=self.alpha, caption=caption, image_scale=self.heatmap_image_scale)
                        
                        fullfn_without_extension, extension = os.path.splitext(params.filename) 
                        full_filename = fullfn_without_extension + "_" + attention + extension
                        
                        if self.use_grid:
                            grid_images.append(img)
                        else:                  
                            img.save(full_filename)
                            
                            if not self.hide_images:
                                self.images += [img]
                    
                    if self.use_grid:
                        self.grid_images += grid_images
        
        # if it is last batch pos, clear heatmaps
        if batch_pos == params.p.batch_size - 1:
            self.tracer.reset()
            
        return

    def process(self, p, *args):
        return 

    def postprocess(self, *args):
        return


def handle_before_image_saved(params : script_callbacks.ImageSaveParams):
    
    if before_image_saved_handler is not None and callable(before_image_saved_handler):
        before_image_saved_handler(params)
   
    return
 
script_callbacks.on_before_image_saved(handle_before_image_saved)   
