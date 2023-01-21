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
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Group():
            with gr.Accordion("Attention Heatmap", open=False):
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

                with gr.Row():
                    trace_each_layers = gr.Checkbox(label = 'Trace each layers', value=False)

                    layers_as_row = gr.Checkbox(label = 'Use layers as row instead of Batch Length', value=False)
        
        
        self.tracers = None
        
        return [attention_texts, hide_images, dont_save_images, hide_caption, use_grid, grid_layouyt, alpha, heatmap_image_scale, trace_each_layers, layers_as_row] 
    
    def process(self, 
            p : StableDiffusionProcessing, 
            attention_texts : str, 
            hide_images : bool, 
            dont_save_images : bool,
            hide_caption : bool, 
            use_grid : bool, 
            grid_layouyt :str,
            alpha : float, 
            heatmap_image_scale : float,
            trace_each_layers : bool,
            layers_as_row: bool):
        
        self.enabled = False # in case the assert fails
        assert opts.samples_save, "Cannot run Daam script. Enable 'Always save all generated images' setting."

        self.images = []
        self.hide_images = hide_images
        self.dont_save_images = dont_save_images
        self.hide_caption = hide_caption
        self.alpha = alpha
        self.use_grid = use_grid
        self.grid_layouyt = grid_layouyt
        self.heatmap_image_scale = heatmap_image_scale
        self.heatmap_images = dict()

        self.attentions = [s.strip() for s in attention_texts.split(",") if s.strip()]
        self.enabled = len(self.attentions) > 0

        fix_seed(p)
        
    def process_batch(self,
            p : StableDiffusionProcessing, 
            attention_texts : str, 
            hide_images : bool, 
            dont_save_images : bool,
            hide_caption : bool, 
            use_grid : bool, 
            grid_layouyt :str,
            alpha : float, 
            heatmap_image_scale : float,
            trace_each_layers : bool,
            layers_as_row: bool,
            prompts,
            **kwargs):
                
        if not self.enabled:
            return
        
        styled_prompt = prompts[0]         
        
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
        
        if any(item[0] in self.attentions for item in self.prompt_analyzer.used_custom_terms):
            print("Embedding heatmap cannot be shown.")
            
        global before_image_saved_handler
        before_image_saved_handler = lambda params : self.before_image_saved(params)
                
        with torch.no_grad():
            # cannot trace the same block from two tracers
            if trace_each_layers:
                num_input = len(p.sd_model.model.diffusion_model.input_blocks)
                num_output = len(p.sd_model.model.diffusion_model.output_blocks)
                self.tracers = [trace(p.sd_model, p.height, p.width, context_size, layer_idx=i) for i in range(num_input + num_output + 1)]
                self.attn_captions = [f"IN{i:02d}" for i in range(num_input)] + ["MID"] + [f"OUT{i:02d}" for i in range(num_output)]
            else:
                self.tracers = [trace(p.sd_model, p.height, p.width, context_size)]
                self.attn_captions = [""]
        
            for tracer in self.tracers:
                tracer.hook()

    def postprocess(self, p, processed,
            attention_texts : str, 
            hide_images : bool, 
            dont_save_images : bool,
            hide_caption : bool, 
            use_grid : bool, 
            grid_layouyt :str,
            alpha : float, 
            heatmap_image_scale : float,
            trace_each_layers : bool,
            layers_as_row: bool,
            **kwargs):
        if self.enabled == False:
            return
        
        for trace in self.tracers:
            trace.unhook()
        self.tracers = None
        
        initial_info = None

        if initial_info is None:
            initial_info = processed.info
            
        self.images += processed.images

        global before_image_saved_handler
        before_image_saved_handler = None

        if layers_as_row:
            images_list = []
            for i in range(p.batch_size * p.n_iter):
                imgs = []
                for k in sorted(self.heatmap_images.keys()):
                    imgs += [self.heatmap_images[k][len(self.attentions)*i + j] for j in range(len(self.attentions))]
                images_list.append(imgs)
        else:
            images_list = [self.heatmap_images[k] for k in sorted(self.heatmap_images.keys())]

        for img_list in images_list:

            if img_list and self.use_grid:

                grid_layout = self.grid_layouyt
                if grid_layout == Script.GRID_LAYOUT_AUTO:
                    if p.batch_size * p.n_iter == 1:
                        grid_layout = Script.GRID_LAYOUT_PREVENT_EMPTY
                    else:
                        grid_layout = Script.GRID_LAYOUT_BATCH_LENGTH_AS_ROW
                        
                if grid_layout == Script.GRID_LAYOUT_PREVENT_EMPTY:
                    grid_img = images.image_grid(img_list)
                elif grid_layout == Script.GRID_LAYOUT_BATCH_LENGTH_AS_ROW:
                    if layers_as_row:
                        batch_size = len(self.attentions)
                        rows = len(self.heatmap_images)
                    else:
                        batch_size = p.batch_size
                        rows = p.batch_size * p.n_iter
                    grid_img = images.image_grid(img_list, batch_size=batch_size, rows=rows)
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
                    processed.images[:0] = img_list
                    processed.index_of_first_image += len(img_list)
                    processed.infotexts[:0] = [processed.infotexts[0]] * len(img_list)

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
        
        if self.tracers is not None and len(self.attentions) > 0:
            for i, tracer in enumerate(self.tracers):
                with torch.no_grad():
                    styled_prompot = shared.prompt_styles.apply_styles_to_prompt(params.p.prompt, params.p.styles)
                    try:
                        global_heat_map = tracer.compute_global_heat_map(self.prompt_analyzer, styled_prompot, batch_pos)              
                    except:
                        continue
                    
                    if i not in self.heatmap_images:
                        self.heatmap_images[i] = []
                    
                    if global_heat_map is not None:
                        heatmap_images = []
                        for attention in self.attentions:
                                    
                            img_size = params.image.size
                            caption = attention + (" " + self.attn_captions[i] if self.attn_captions[i] else "") if not self.hide_caption else None
                            
                            heat_map = global_heat_map.compute_word_heat_map(attention)
                            if heat_map is None : print(f"No heatmaps for '{attention}'")
                            
                            heat_map_img = utils.expand_image(heat_map, img_size[1], img_size[0]) if heat_map is not None else None
                            img : Image.Image = utils.image_overlay_heat_map(params.image, heat_map_img, alpha=self.alpha, caption=caption, image_scale=self.heatmap_image_scale)

                            fullfn_without_extension, extension = os.path.splitext(params.filename) 
                            full_filename = fullfn_without_extension + "_" + attention +  ("_" + self.attn_captions[i] if self.attn_captions[i] else "") + extension
                            
                            if self.use_grid:
                                heatmap_images.append(img)
                            else:
                                heatmap_images.append(img)
                                if not self.dont_save_images:               
                                    img.save(full_filename)                            
                        
                        self.heatmap_images[i] += heatmap_images
        
        self.heatmap_images = {j:self.heatmap_images[j] for j in self.heatmap_images.keys() if self.heatmap_images[j]}

        # if it is last batch pos, clear heatmaps
        if batch_pos == params.p.batch_size - 1:
            for tracer in self.tracers:
                tracer.reset()
            
        return


def handle_before_image_saved(params : script_callbacks.ImageSaveParams):
    
    if before_image_saved_handler is not None and callable(before_image_saved_handler):
        before_image_saved_handler(params)
   
    return
 
script_callbacks.on_before_image_saved(handle_before_image_saved)   
