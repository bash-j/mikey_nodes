import datetime
from fractions import Fraction
import json
from math import ceil, pow
import os
import re

import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import torch

from comfy.model_management import unload_model, soft_empty_cache
import comfy.utils
import folder_paths

def sdxl_size(width: int, height: int) -> (int, int):
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
        return w, h

def read_ratios():
    p = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(p, 'ratios.json')
    with open(file_path, 'r') as file:
        data = json.load(file)
    ratio_sizes = list(data['ratios'].keys())
    ratio_dict = data['ratios']
    # user_styles.json
    user_styles_path = os.path.join(folder_paths.base_path, 'user_ratios.json')
    # check if file exists
    if os.path.isfile(user_styles_path):
        # read json and update ratio_dict
        with open(user_styles_path, 'r') as file:
            user_data = json.load(file)
        for ratio in user_data['ratios']:
            ratio_dict[ratio] = user_data['ratios'][ratio]
            ratio_sizes.append(ratio)
    return ratio_sizes, ratio_dict

def read_styles():
    p = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(p, 'styles.json')
    with open(file_path, 'r') as file:
        data = json.load(file)
    # each style has a positive and negative key
    """ start of json styles.json looks like this:
    {
    "styles": {
        "none": {
        "positive": "",
        "negative": ""
        },
        "3d-model": {
        "positive": "3d model, polygons, mesh, textures, lighting, rendering",
        "negative": "2D representation, lack of depth and volume, no realistic rendering"
        },
    """
    styles = list(data['styles'].keys())
    pos_style = {}
    neg_style = {}
    for style in styles:
        pos_style[style] = data['styles'][style]['positive']
        neg_style[style] = data['styles'][style]['negative']
    # user_styles.json
    user_styles_path = os.path.join(folder_paths.base_path, 'user_styles.json')
    # check if file exists
    if os.path.isfile(user_styles_path):
        # read json and update pos_style and neg_style
        with open(user_styles_path, 'r') as file:
            user_data = json.load(file)
        for style in user_data['styles']:
            pos_style[style] = user_data['styles'][style]['positive']
            neg_style[style] = user_data['styles'][style]['negative']
            styles.append(style)
    return styles, pos_style, neg_style

def read_cluts():
    p = os.path.dirname(os.path.realpath(__file__))
    halddir = os.path.join(p, 'HaldCLUT')
    files = [os.path.join(halddir, f) for f in os.listdir(halddir) if os.path.isfile(os.path.join(halddir, f)) and f.endswith('.png')]
    return files

def apply_hald_clut(hald_img, img):
    hald_w, hald_h = hald_img.size
    clut_size = int(round(pow(hald_w, 1/3)))
    scale = (clut_size * clut_size - 1) / 255
    img = np.asarray(img)

    # Convert the HaldCLUT image to numpy array
    hald_img_array = np.asarray(hald_img)

    # If the HaldCLUT image is monochrome, duplicate its single channel to three
    if len(hald_img_array.shape) == 2:
        hald_img_array = np.stack([hald_img_array]*3, axis=-1)

    hald_img_array = hald_img_array.reshape(clut_size ** 6, 3)

    clut_r = np.rint(img[:, :, 0] * scale).astype(int)
    clut_g = np.rint(img[:, :, 1] * scale).astype(int)
    clut_b = np.rint(img[:, :, 2] * scale).astype(int)
    filtered_image = np.zeros((img.shape))
    filtered_image[:, :] = hald_img_array[clut_r + clut_size ** 2 * clut_g + clut_size ** 4 * clut_b]
    filtered_image = Image.fromarray(filtered_image.astype('uint8'), 'RGB')
    return filtered_image

def gamma_correction_pil(image, gamma):
    # Convert PIL Image to NumPy array
    img_array = np.array(image)
    # Normalization [0,255] -> [0,1]
    img_array = img_array / 255.0
    # Apply gamma correction
    img_corrected = np.power(img_array, gamma)
    # Convert corrected image back to original scale [0,1] -> [0,255]
    img_corrected = np.uint8(img_corrected * 255)
    # Convert NumPy array back to PIL Image
    corrected_image = Image.fromarray(img_corrected)
    return corrected_image

# Tensor to PIL
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

# PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

class HaldCLUT:
    @classmethod
    def INPUT_TYPES(s):
        s.haldclut_files = read_cluts()
        s.file_names = [os.path.basename(f) for f in s.haldclut_files]
        return {"required": {"image": ("IMAGE",),
                             "hald_clut": (s.file_names,),
                             "gamma_correction": (['True','False'],)}}

    RETURN_TYPES = ('IMAGE',)
    RETURN_NAMES = ('image,')
    FUNCTION = 'apply_haldclut'
    CATEGORY = 'Mikey/Image'
    OUTPUT_NODE = True

    def apply_haldclut(self, image, hald_clut, gamma_correction):
        hald_img = Image.open(self.haldclut_files[self.file_names.index(hald_clut)])
        img = tensor2pil(image)
        if gamma_correction == 'True':
            corrected_img = gamma_correction_pil(img, 1.0/2.2)
        else:
            corrected_img = img
        filtered_image = apply_hald_clut(hald_img, corrected_img).convert("RGB")
        return (pil2tensor(filtered_image), )

    @classmethod
    def IS_CHANGED(self, hald_clut):
        return (np.nan,)

class EmptyLatentRatioSelector:
    @classmethod
    def INPUT_TYPES(s):
        s.ratio_sizes, s.ratio_dict = read_ratios()
        return {'required': {'ratio_selected': (s.ratio_sizes,),
                             "batch_size": ("INT", {"default": 1, "min": 1, "max": 64})}}

    RETURN_TYPES = ('LATENT',)
    FUNCTION = 'generate'
    CATEGORY = 'Mikey/Latent'

    def generate(self, ratio_selected, batch_size=1):
        width = self.ratio_dict[ratio_selected]["width"]
        height = self.ratio_dict[ratio_selected]["height"]
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
    CATEGORY = 'Mikey/Latent'

    def generate(self, width, height, batch_size=1):
        # solver
        if width == 1 and height == 1:
            w, h = 1024, 1024
        if f'{width}:{height}' in EmptyLatentRatioSelector.ratio_dict:
            w, h = EmptyLatentRatioSelector.ratio_dict[f'{width}:{height}']
        else:
            w, h = sdxl_size(width, height)
        latent = torch.zeros([batch_size, 4, h // 8, w // 8])
        return ({"samples":latent}, )

class ResizeImageSDXL:
    crop_methods = ["disabled", "center"]
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "image": ("IMAGE",), "upscale_method": (s.upscale_methods,),
                              "crop": (s.crop_methods,)},
                "optional": { "mask": ("MASK", )}}

    RETURN_TYPES = ('IMAGE',)
    FUNCTION = 'upscale'
    CATEGORY = 'Mikey/Image'

    def upscale(self, image, upscale_method, width, height, crop):
        samples = image.movedim(-1,1)
        s = comfy.utils.common_upscale(samples, width, height, upscale_method, crop)
        s = s.movedim(1,-1)
        return (s,)

    def resize(self, image, upscale_method, crop):
        w, h = sdxl_size(image.shape[2], image.shape[1])
        print('Resizing image from {}x{} to {}x{}'.format(image.shape[2], image.shape[1], w, h))
        img = self.upscale(image, upscale_method, w, h, crop)[0]
        return (img, )

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
    CATEGORY = "Mikey/Image"

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
                pos_trunc = clean_pos.replace(' ', '_')[0:80]
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

class PromptWithStyle:
    #elrs = EmptyLatentRatioSelector()

    @classmethod
    def INPUT_TYPES(s):
        s.ratio_sizes, s.ratio_dict = read_ratios()
        s.styles, s.pos_style, s.neg_style = read_styles()
        return {"required": {"positive_prompt": ("STRING", {"multiline": True, 'default': 'Positive Prompt'}),
                             "negative_prompt": ("STRING", {"multiline": True, 'default': 'Negative Prompt'}),
                             "style": (s.styles,),
                             "ratio_selected": (s.ratio_sizes,),
                             "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
                             "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                             }
        }

    RETURN_TYPES = ('LATENT','STRING','STRING','STRING','STRING','INT','INT','INT','INT',)
    RETURN_NAMES = ('samples','positive_prompt_text_g','negative_prompt_text_g','positive_style_text_l',
                    'negative_style_text_l','width','height','refiner_width','refiner_height',)
    FUNCTION = 'start'
    CATEGORY = 'Mikey'

    def start(self, positive_prompt, negative_prompt, style, ratio_selected, batch_size, seed):
        # wildcards always have a __ prefix and __ suffix
        # regex to find all wildcards
        # path to wildcard folder with matching wildcard.txt files is the root_project_dir/wildcards
        wildcard_path = os.path.join(folder_paths.base_path, 'wildcards')
        wildcard_regex = r'__.*?__'
        pos_wildcard = ''
        pos_seed = seed
        for match in re.findall(wildcard_regex, positive_prompt):
            # check for matching file in wildcard folder
            if pos_wildcard == match:
                pos_seed += 1
            else:
                pos_seed = seed
            is_file = os.path.isfile(os.path.join(wildcard_path, match[2:-2] + '.txt'))
            if is_file:
                with open(os.path.join(wildcard_path, match[2:-2] + '.txt'), 'r') as file:
                    wildcard_lines = file.readlines()
                    # offset can be larger than the number of lines in the file, or it could even be a negative number
                    # starting with line 0, so we need to use modulo to wrap around
                    line_number = (pos_seed % len(wildcard_lines))
                    # only replace the first match so that duplicates can be read from the next line
                    positive_prompt = positive_prompt.replace(match, wildcard_lines[line_number].strip(), 1)
                    pos_wildcard = match
            else:
                print(f'Wildcard file {match[2:-2]}.txt not found in {wildcard_path}')
        neg_wildcard = ''
        neg_seed = seed
        for match in re.findall(wildcard_regex, negative_prompt):
            # check for matching file in wildcard folder
            if neg_wildcard == match:
                neg_seed += 1
            else:
                neg_seed = seed
            is_file = os.path.isfile(os.path.join(wildcard_path, match[2:-2] + '.txt'))
            if is_file:
                with open(os.path.join(wildcard_path, match[2:-2] + '.txt'), 'r') as file:
                    wildcard_lines = file.readlines()
                    # offset can be larger than the number of lines in the file, or it could even be a negative number
                    # starting with line 0, so we need to use modulo to wrap around
                    line_number = (neg_seed % len(wildcard_lines))
                    negative_prompt = negative_prompt.replace(match, wildcard_lines[line_number].strip(), 1)
                    neg_wildcard = match
            else:
                print(f'Wildcard file {match[2:-2]}.txt not found in {wildcard_path}')
        if '{prompt}' in self.pos_style[style]:
            positive_prompt = self.pos_style[style].replace('{prompt}', positive_prompt)
        if positive_prompt == '' or positive_prompt == 'Positive Prompt' or positive_prompt is None:
            pos_prompt = self.pos_style[style]
        else:
            pos_prompt = positive_prompt + ', ' + self.pos_style[style]
        if negative_prompt == '' or negative_prompt == 'Negative Prompt' or negative_prompt is None:
            neg_prompt = self.neg_style[style]
        else:
            neg_prompt = negative_prompt + ', ' + self.neg_style[style]
        width = self.ratio_dict[ratio_selected]["width"]
        height = self.ratio_dict[ratio_selected]["height"]
        latent = torch.zeros([batch_size, 4, height // 8, width // 8])
        refiner_width = width * 8
        refiner_height = height * 8
        return ({"samples":latent},
                str(pos_prompt),
                str(neg_prompt),
                str(self.pos_style[style]),
                str(self.neg_style[style]),
                width,
                height,
                refiner_width,
                refiner_height,)

class PromptWithSDXL:
    @classmethod
    def INPUT_TYPES(s):
        s.ratio_sizes, s.ratio_dict = read_ratios()
        return {"required": {"positive_prompt": ("STRING", {"multiline": True, 'default': 'Positive Prompt'}),
                             "negative_prompt": ("STRING", {"multiline": True, 'default': 'Negative Prompt'}),
                             "positive_style": ("STRING", {"multiline": True, 'default': 'Positive Style'}),
                             "negative_style": ("STRING", {"multiline": True, 'default': 'Negative Style'}),
                             "ratio_selected": (s.ratio_sizes,),
                             "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
                             "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff})
                             }
        }

    RETURN_TYPES = ('LATENT','STRING','STRING','STRING','STRING','INT','INT','INT','INT',)
    RETURN_NAMES = ('samples','positive_prompt_text_g','negative_prompt_text_g','positive_style_text_l',
                    'negative_style_text_l','width','height','refiner_width','refiner_height',)
    FUNCTION = 'start'
    CATEGORY = 'Mikey'

    def start(self, positive_prompt, negative_prompt, positive_style, negative_style, ratio_selected, batch_size, seed):
        # wildcards always have a __ prefix and __ suffix
        # regex to find all wildcards
        # path to wildcard folder with matching wildcard.txt files is the root_project_dir/wildcards
        wildcard_path = os.path.join(folder_paths.base_path, 'wildcards')
        wildcard_regex = r'__.*?__'
        pos_wildcard = ''
        pos_seed = seed
        for match in re.findall(wildcard_regex, positive_prompt):
            # check for matching file in wildcard folder
            if pos_wildcard == match:
                pos_seed += 1
            else:
                pos_seed = seed
            is_file = os.path.isfile(os.path.join(wildcard_path, match[2:-2] + '.txt'))
            if is_file:
                with open(os.path.join(wildcard_path, match[2:-2] + '.txt'), 'r') as file:
                    wildcard_lines = file.readlines()
                    # offset can be larger than the number of lines in the file, or it could even be a negative number
                    # starting with line 0, so we need to use modulo to wrap around
                    line_number = (pos_seed % len(wildcard_lines))
                    # only replace the first match so that duplicates can be read from the next line
                    positive_prompt = positive_prompt.replace(match, wildcard_lines[line_number].strip(), 1)
                    pos_wildcard = match
            else:
                print(f'Wildcard file {match[2:-2]}.txt not found in {wildcard_path}')
        neg_wildcard = ''
        neg_seed = seed
        for match in re.findall(wildcard_regex, negative_prompt):
            # check for matching file in wildcard folder
            if neg_wildcard == match:
                neg_seed += 1
            else:
                neg_seed = seed
            is_file = os.path.isfile(os.path.join(wildcard_path, match[2:-2] + '.txt'))
            if is_file:
                with open(os.path.join(wildcard_path, match[2:-2] + '.txt'), 'r') as file:
                    wildcard_lines = file.readlines()
                    # offset can be larger than the number of lines in the file, or it could even be a negative number
                    # starting with line 0, so we need to use modulo to wrap around
                    line_number = (neg_seed % len(wildcard_lines))
                    negative_prompt = negative_prompt.replace(match, wildcard_lines[line_number].strip(), 1)
                    neg_wildcard = match
            else:
                print(f'Wildcard file {match[2:-2]}.txt not found in {wildcard_path}')
        width = self.ratio_dict[ratio_selected]["width"]
        height = self.ratio_dict[ratio_selected]["height"]
        latent = torch.zeros([batch_size, 4, height // 8, width // 8])
        refiner_width = width * 8
        refiner_height = height * 8
        return ({"samples":latent},
                str(positive_prompt),
                str(negative_prompt),
                str(positive_style),
                str(negative_style),
                width,
                height,
                refiner_width,
                refiner_height,)


class VAEDecode6GB:
    """ deprecated. update comfy to fix issue. """
    @classmethod
    def INPUT_TYPES(s):
        return {'required': {'vae': ('VAE',),
                             'samples': ('LATENT',)}}
    RETURN_TYPES = ('IMAGE',)
    FUNCTION = 'decode'
    CATEGORY = 'Mikey/Latent'

    def decode(self, vae, samples):
        unload_model()
        soft_empty_cache()
        return (vae.decode(samples['samples']), )

NODE_CLASS_MAPPINGS = {
    'Empty Latent Ratio Select SDXL': EmptyLatentRatioSelector,
    'Empty Latent Ratio Custom SDXL': EmptyLatentRatioCustom,
    'Save Image With Prompt Data': SaveImagesMikey,
    'Resize Image for SDXL': ResizeImageSDXL,
    'Prompt With Style': PromptWithStyle,
    'Prompt With SDXL': PromptWithSDXL,
    'HaldCLUT': HaldCLUT,
    'VAE Decode 6GB SDXL (deprecated)': VAEDecode6GB,
}
## TODO
# Resize Image and return the new width and height
# SDXL Ultimate Upscaler?
