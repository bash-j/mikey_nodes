from fractions import Fraction
from math import ceil

import torch

from comfy.model_management import unload_model, soft_empty_cache

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

NODE_CLASS_MAPPINGS = {
    'Empty Latent Ratio Select SDXL': EmptyLatentRatioSelector,
    'Empty Latent Ratio Custom SDXL': EmptyLatentRatioCustom,
}