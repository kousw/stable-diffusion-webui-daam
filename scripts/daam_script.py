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
from modules import script_callbacks, sd_hijack_clip, sd_hijack_open_clip
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
            
            dont_save_images = gr.Checkbox(label='Do not save heatmap images', value=False)
            
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
        
        return [attention_texts, hide_images, dont_save_images, hide_caption, use_grid, grid_layouyt, alpha, heatmap_image_scale] 
    
    def run(self,
            p : StableDiffusionProcessing, 
            attention_texts : str, 
            hide_images : bool, 
            dont_save_images : bool,
            hide_caption : bool, 
            use_grid : bool, 
            grid_layouyt :str,
            alpha : float, 
            heatmap_image_scale : float):
                
        assert opts.samples_save, "Cannot run Daam script. Enable 'Always save all generated images' setting."

        initial_info = None
        self.images = []
        self.hide_images = hide_images
        self.dont_save_images = dont_save_images
        self.hide_caption = hide_caption
        self.alpha = alpha
        self.use_grid = use_grid
        self.grid_layouyt = grid_layouyt
        self.heatmap_image_scale = heatmap_image_scale
        self.heatmap_images = list()
        
        fix_seed(p)
        
        styled_prompt = shared.prompt_styles.apply_styles_to_prompt(p.prompt, p.styles)
        
        attentions = [ s.strip() for s in attention_texts.split(",") if s.strip() ]
        self.attentions = attentions
        
        embedder = None
        if type(p.sd_model.cond_stage_model) == sd_hijack_clip.FrozenCLIPEmbedderWithCustomWords or \
            type(p.sd_model.cond_stage_model) == sd_hijack_open_clip.FrozenOpenCLIPEmbedderWithCustomWords:
            embedder = p.sd_model.cond_stage_model  
        else:
            assert False, f"Embedder '{type(p.sd_model.cond_stage_model)}' is not supported."
            
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
        context_size = utils.calc_context_size(len(tokens))
        
        prompt_analyzer = utils.PromptAnalyzer(embedder, styled_prompt)
        self.prompt_analyzer = prompt_analyzer
        context_size = prompt_analyzer.context_size
               
        print(f"daam run with context_size={prompt_analyzer.context_size}, token_count={prompt_analyzer.token_count}")
        # print(f"remade_tokens={prompt_analyzer.tokens}, multipliers={prompt_analyzer.multipliers}")
        # print(f"hijack_comments={prompt_analyzer.hijack_comments}, used_custom_terms={prompt_analyzer.used_custom_terms}")
        # print(f"fixes={prompt_analyzer.fixes}")
        
        if any(item[0] in attentions for item in self.prompt_analyzer.used_custom_terms):
            print("Embedding heatmap cannot be shown.")
            
        global before_image_saved_handler
        before_image_saved_handler = lambda params : self.before_image_saved(params)
                
        with torch.no_grad():
            with trace(p.sd_model, p.height, p.width, context_size) as tr:
                self.tracer = tr
                               
                processed = process_images(p)
                if initial_info is None:
                    initial_info = processed.info
                self.images  += processed.images        
                
                self.tracer = None        

        before_image_saved_handler = None
        
        # processed = Processed(p, self.images, p.seed, initial_info)
                
        if len(self.heatmap_images) > 0:
            
            if self.use_grid:

                grid_layout = self.grid_layouyt
                if grid_layout == Script.GRID_LAYOUT_AUTO:
                    if p.batch_size * p.n_iter == 1:
                        grid_layout = Script.GRID_LAYOUT_PREVENT_EMPTY
                    else:
                        grid_layout = Script.GRID_LAYOUT_BATCH_LENGTH_AS_ROW
                        
                if grid_layout == Script.GRID_LAYOUT_PREVENT_EMPTY:
                    grid_img = images.image_grid(self.heatmap_images)
                elif grid_layout == Script.GRID_LAYOUT_BATCH_LENGTH_AS_ROW:
                    grid_img = images.image_grid(self.heatmap_images, batch_size=p.batch_size, rows=p.batch_size * p.n_iter)
                else:
                    pass
                
                if not self.dont_save_images:
                    images.save_image(grid_img, p.outpath_grids, "grid_daam", grid=True, p=p)
                
                if not self.hide_images:
                    processed.images.insert(0, grid_img)
                    processed.index_of_first_image += 1
                    processed.infotexts.insert(0, processed.infotexts[0])
            
            else:
                if not self.hide_images:
                    processed.images[:0] = self.heatmap_images
                    processed.index_of_first_image += len(self.heatmap_images)
                    processed.infotexts[:0] = [processed.infotexts[0]] * len(self.heatmap_images)

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
                global_heat_map = self.tracer.compute_global_heat_map(self.prompt_analyzer, styled_prompot, batch_pos)              
                
                if global_heat_map is not None:
                    heatmap_images = []
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
                            heatmap_images.append(img)
                        else:
                            heatmap_images.append(img)
                            if not self.dont_save_images:               
                                img.save(full_filename)                            
                    
                    self.heatmap_images += heatmap_images
        
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
