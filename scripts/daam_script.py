import os

import gradio as gr
import modules.scripts as scripts
import torch
from ldm.modules.encoders.modules import FrozenCLIPEmbedder, FrozenOpenCLIPEmbedder
import open_clip.tokenizer
from modules import script_callbacks
from modules.processing import (Processed, StableDiffusionProcessing, fix_seed,
                                process_images)
from modules.shared import cmd_opts, opts, state
from PIL import Image

from scripts.daam import trace, utils


class Script(scripts.Script):

    def title(self):
        return "Daam script"

    def show(self, is_img2img):
        return True

    def ui(self, is_img2img):
        
        script_callbacks.on_before_image_saved(self.before_image_saved)
        
        attention_texts = gr.Text(label='Attention texts for visualization. (comma separated)', value='')

        hide_images = gr.Checkbox(label='Hide heatmap images', value=False)
        
        self.tracer = None
        self.attentions = []
        self.images = []
        self.hide_images = hide_images
        
        return [attention_texts, hide_images]
    
    def run(self, p : StableDiffusionProcessing, attention_texts : str, hide_images : bool):

        initial_info = None
        self.images = []
        self.hide_images = hide_images
        
        fix_seed(p)
        
        initial_prompt = p.prompt
        initial_negative_prompt = p.negative_prompt
        
        attentions = [ s.strip() for s in attention_texts.split(",") ]
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
            
        tokens = tokenize(p.prompt.lower())
        context_size = 77 if len(tokens) <= 75 else 154
        
        print("daam run")
        
        with torch.cuda.amp.autocast(dtype=torch.float16), torch.no_grad():
            with trace(p.sd_model, p.height, p.width, context_size) as tr:
                self.tracer = tr
                               
                proc = process_images(p)
                if initial_info is None:
                    initial_info = proc.info
                self.images  += proc.images        
                
                self.tracer = None        

        processed = Processed(p, self.images, p.seed, initial_info)

        return processed
    
    def before_image_saved(self, params : script_callbacks.ImageSaveParams):
        if self.tracer is not None and len(self.attentions) > 0:
            with torch.cuda.amp.autocast(dtype=torch.float16), torch.no_grad():                                
                global_heat_map = self.tracer.compute_global_heat_map(params.p.prompt)
                
                for attention in self.attentions:
                               
                    img_size = params.image.size
                    heat_map = utils.expand_image(global_heat_map.compute_word_heat_map(attention), img_size[1], img_size[0])
                    img : Image.Image = utils.image_overlay_heat_map(params.image, heat_map)
                    
                    fullfn_without_extension, extension = os.path.splitext(params.filename)
                
                    img.save(fullfn_without_extension + "_" + attention + extension)
                    
                    if not self.hide_images:
                        self.images += [img]
                
                self.tracer.clear_heat_map()

    def process(self, p, *args):
        return 

    def postprocess(self, *args):
        return

