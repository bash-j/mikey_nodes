import datetime
from fractions import Fraction
import json
from math import ceil
import os
import re

import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import torch

from comfy.model_management import unload_model, soft_empty_cache
import folder_paths

class EmptyLatentRatioSelector:
    ratio_sizes = ['1:1','5:4','4:3','3:2','16:9','21:9','4:5','3:4','2:3','9:16','9:21','5:7','7:5']
    ratio_dict = {'1:1': (1024, 1024),
              '5:4': (1152, 896),
              '4:3': (1152, 832),
              '3:2': (1216, 832),
              '16:9': (1344, 768),
              '21:9': (1536, 640),
              '4:5': (896, 1152),
              '3:4': (832, 1152),
              '2:3': (832, 1216),
              '9:16': (768, 1344),
              '9:21': (640, 1536),
              '5:7': (840, 1176),
              '7:5': (1176, 840),}

    @classmethod
    def INPUT_TYPES(s):
        return {'required': {'ratio_selected': (s.ratio_sizes,),
                             "batch_size": ("INT", {"default": 1, "min": 1, "max": 64})}}
    
    RETURN_TYPES = ('LATENT',)
    FUNCTION = 'generate'
    CATEGORY = 'sdxl'

    def generate(self, ratio_selected, batch_size=1):
        width = self.ratio_dict[ratio_selected][0]
        height = self.ratio_dict[ratio_selected][1]
        latent = torch.zeros([batch_size, 4, height // 8, width // 8])
        return ({"samples":latent}, )

class EmptyLatentRatioCustom:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "width": ("INT", {"default": 1024, "min": 1, "max": 8192, "step": 1}),
                              "height": ("INT", {"default": 1024, "min": 1, "max": 8192, "step": 1}),
                              "batch_size": ("INT", {"default": 1, "min": 1, "max": 64})}}
    
    RETURN_TYPES = ('LATENT',)
    FUNCTION = 'generate'
    CATEGORY = 'sdxl'

    def generate(self, width, height, batch_size=1):
        # solver
        w = 0
        h = 0
        for i in range(1, 256):
            for j in range(1, 256):
                if Fraction(8 * i, 8 * j) > Fraction(width, height) * 0.98 and Fraction(8 * i, 8 * j) < Fraction(width, height) and 8 * i * 8 * j <= 1024 * 1024:
                    if (ceil(8 * i / 64) * 64) * (ceil(8 * j / 64) * 64) <= 1024 * 1024:
                        w = ceil(8 * i / 64) * 64
                        h = ceil(8 * j / 64) * 64
                    elif (8 * i // 64 * 64) * (ceil(8 * j / 64) * 64) <= 1024 * 1024:
                        w = 8 * i // 64 * 64
                        h = ceil(8 * j / 64) * 64
                    elif (ceil(8 * i / 64) * 64) * (8 * j // 64 * 64) <= 1024 * 1024:
                        w = ceil(8 * i / 64) * 64
                        h = 8 * j // 64 * 64
                    else:
                        w = 8 * i // 64 * 64
                        h = 8 * j // 64 * 64
        latent = torch.zeros([batch_size, 4, h // 8, w // 8])
        return ({"samples":latent}, )

class SaveImagesMikey:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"

    @classmethod
    def INPUT_TYPES(s):
        return {"required": 
                    {"images": ("IMAGE", ),
                     "positive_prompt": ("STRING", {'default': 'Positive Prompt'}),
                     "negative_prompt": ("STRING", {'default': 'Negative Prompt'}),},
                     "filename_prefix": ("STRING", {"default": ""}),
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }

    RETURN_TYPES = ()
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "sdxl"

    def save_images(self, images, filename_prefix='', prompt=None, extra_pnginfo=None, positive_prompt='', negative_prompt=''):
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        results = list()
        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = PngInfo()
            pos_trunc = ''
            if prompt is not None:
                metadata.add_text("prompt", json.dumps(prompt))
            if extra_pnginfo is not None:
                for x in extra_pnginfo:
                    metadata.add_text(x, json.dumps(extra_pnginfo[x]))
            if positive_prompt:
                metadata.add_text("positive_prompt", json.dumps(positive_prompt))
                # replace any special characters with nothing and spaces with _
                clean_pos = re.sub(r'[^a-zA-Z0-9 ]', '', positive_prompt)
                pos_trunc = clean_pos.replace(' ', '_')[0:40]
            if negative_prompt:
                metadata.add_text("negative_prompt", json.dumps(negative_prompt))
            ts_str = datetime.datetime.now().strftime("%y%m%d%H%M%S")
            file = f"{ts_str}_{pos_trunc}_{filename}_{counter:05}_.png"
            img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=4)
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1

        return { "ui": { "images": results } }

class VAEDecode6GB:
    """ deprecated. update comfy to fix issue. """
    @classmethod
    def INPUT_TYPES(s):
        return {'required': {'vae': ('VAE',),
                             'samples': ('LATENT',)}}
    RETURN_TYPES = ('IMAGE',)
    FUNCTION = 'decode'
    CATEGORY = 'sdxl'

    def decode(self, vae, samples):
        unload_model()
        soft_empty_cache()
        return (vae.decode(samples['samples']), )

NODE_CLASS_MAPPINGS = {
    'Empty Latent Ratio Select SDXL': EmptyLatentRatioSelector,
    'Empty Latent Ratio Custom SDXL': EmptyLatentRatioCustom,
    'Save Image With Prompt Data': SaveImagesMikey,
    'VAE Decode 6GB SDXL (deprecated)': VAEDecode6GB,
}