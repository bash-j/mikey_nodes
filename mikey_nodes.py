import datetime
from fractions import Fraction
import gc
import importlib.util
from itertools import product
import json
from math import ceil, pow, gcd
import os
import psutil
import random
import re
import sys
from textwrap import wrap

import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFilter, ImageChops, ImageFont
from PIL.PngImagePlugin import PngInfo
import torch
import torch.nn.functional as F
from tqdm import tqdm

import folder_paths
file_path = os.path.join(folder_paths.base_path, 'comfy_extras/nodes_clip_sdxl.py')
module_name = "nodes_clip_sdxl"
spec = importlib.util.spec_from_file_location(module_name, file_path)
module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = module
spec.loader.exec_module(module)
from nodes_clip_sdxl import CLIPTextEncodeSDXL, CLIPTextEncodeSDXLRefiner
file_path = os.path.join(folder_paths.base_path, 'comfy_extras/nodes_upscale_model.py')
module_name = "nodes_upscale_model"
spec = importlib.util.spec_from_file_location(module_name, file_path)
module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = module
spec.loader.exec_module(module)
from nodes_upscale_model import UpscaleModelLoader, ImageUpscaleWithModel
from comfy.model_management import soft_empty_cache, current_loaded_models
from nodes import LoraLoader, ConditioningAverage, common_ksampler, ImageScale, VAEEncode, VAEDecode
import comfy.utils

def find_latent_size(width: int, height: int, res: int = 1024) -> (int, int):
    best_w = 0
    best_h = 0
    target_ratio = Fraction(width, height)

    for i in range(1, 256):
        for j in range(1, 256):
            if Fraction(8 * i, 8 * j) > target_ratio * 0.98 and Fraction(8 * i, 8 * j) < target_ratio and 8 * i * 8 * j <= res * res:
                candidates = [
                    (ceil(8 * i / 64) * 64, ceil(8 * j / 64) * 64),
                    (8 * i // 64 * 64, ceil(8 * j / 64) * 64),
                    (ceil(8 * i / 64) * 64, 8 * j // 64 * 64),
                    (8 * i // 64 * 64, 8 * j // 64 * 64),
                ]
                for w, h in candidates:
                    if w * h > res * res:
                        continue
                    if w * h > best_w * best_h:
                        best_w, best_h = w, h
    return best_w, best_h

def find_tile_dimensions(width: int, height: int, multiplier: float, res: int) -> (int, int):
    new_width = width * multiplier // 8 * 8
    new_height = height * multiplier // 8 * 8
    width_multiples = round(new_width / res, 0)
    height_multiples = round(new_height / res, 0)
    tile_width = new_width / width_multiples // 1
    tile_height = new_height / height_multiples // 1
    return tile_width, tile_height

def find_tile_dimensions(width: int, height: int, multiplier: float, res: int) -> (int, int):
    new_width = int(width * multiplier) // 8 * 8
    new_height = int(height * multiplier) // 8 * 8

    width_multiples = max(1, new_width // res)
    height_multiples = max(1, new_height // res)

    tile_width = new_width // width_multiples
    tile_height = new_height // height_multiples

    return int(tile_width), int(tile_height)

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

def read_ratio_presets():
    p = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(p, 'ratio_presets.json')
    with open(file_path, 'r') as file:
        data = json.load(file)
    ratio_presets = list(data['ratio_presets'].keys())
    ratio_preset_dict = data['ratio_presets']
    # user_ratio_presets.json
    user_ratio_presets_path = os.path.join(folder_paths.base_path, 'user_ratio_presets.json')
    # check if file exists
    if os.path.isfile(user_ratio_presets_path):
        # read json and update ratio_dict
        with open(user_ratio_presets_path, 'r') as file:
            user_data = json.load(file)
        for ratio in user_data['ratio_presets']:
            ratio_preset_dict[ratio] = user_data['ratio_presets'][ratio]
            ratio_presets.append(ratio)
    # remove duplicate presets
    ratio_presets = sorted(list(set(ratio_presets)))
    return ratio_presets, ratio_preset_dict

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

#def read_ratio_presets():
#    file_path = os.path.join(folder_paths.base_path, 'user_ratio_presets.json')
#    if os.path.isfile(file_path):
#        with open(file_path, 'r') as file:
#            data = json.load(file)
#        ratio_presets = list(data['ratio_presets'].keys())
#        return ratio_presets, data['ratio_presets']
#    else:
#        return ['none'], {'none': None}

def find_and_replace_wildcards(prompt, offset_seed, debug=False):
    # wildcards use the __file_name__ syntax with optional |word_to_find
    wildcard_path = os.path.join(folder_paths.base_path, 'wildcards')
    wildcard_regex = r'(\[(\d+)\$\$)?__((?:[^|_]+_)*[^|_]+)((?:\|[^|]+)*)__\]?'
    match_strings = []
    random.seed(offset_seed)
    offset = offset_seed

    new_prompt = ''
    last_end = 0

    for m in re.finditer(wildcard_regex, prompt):
        full_match, lines_count_str, actual_match, words_to_find_str = m.groups()
        # Append everything up to this match
        new_prompt += prompt[last_end:m.start()]

    #for full_match, lines_count_str, actual_match, words_to_find_str in re.findall(wildcard_regex, prompt):
        words_to_find = words_to_find_str.split('|')[1:] if words_to_find_str else None
        if debug:
            print(f'Wildcard match: {actual_match}')
            print(f'Wildcard words to find: {words_to_find}')
        lines_to_insert = int(lines_count_str) if lines_count_str else 1
        if debug:
            print(f'Wildcard lines to insert: {lines_to_insert}')
        match_parts = actual_match.split('/')
        if len(match_parts) > 1:
            wildcard_dir = os.path.join(*match_parts[:-1])
            wildcard_file = match_parts[-1]
        else:
            wildcard_dir = ''
            wildcard_file = match_parts[0]
        search_path = os.path.join(wildcard_path, wildcard_dir)
        file_path = os.path.join(search_path, wildcard_file + '.txt')
        if not os.path.isfile(file_path) and wildcard_dir == '':
            file_path = os.path.join(wildcard_path, wildcard_file + '.txt')
        if os.path.isfile(file_path):
            store_offset = None
            if actual_match in match_strings:
                store_offset = offset
                offset = random.randint(0, 1000000)
            selected_lines = []
            with open(file_path, 'r', encoding='utf-8') as file:
                file_lines = file.readlines()
                num_lines = len(file_lines)
                if words_to_find:
                    for i in range(lines_to_insert):
                        start_idx = (offset + i) % num_lines
                        for j in range(num_lines):
                            line_number = (start_idx + j) % num_lines
                            line = file_lines[line_number].strip()
                            if any(re.search(r'\b' + re.escape(word) + r'\b', line, re.IGNORECASE) for word in words_to_find):
                                selected_lines.append(line)
                                break
                else:
                    start_idx = offset % num_lines
                    for i in range(lines_to_insert):
                        line_number = (start_idx + i) % num_lines
                        line = file_lines[line_number].strip()
                        selected_lines.append(line)
            if len(selected_lines) == 1:
                replacement_text = selected_lines[0]
            else:
                replacement_text = ','.join(selected_lines)
            new_prompt += replacement_text
            match_strings.append(actual_match)
            if store_offset is not None:
                offset = store_offset
                store_offset = None
            offset += lines_to_insert
            if debug:
                print('Wildcard prompt selected: ' + replacement_text)
        else:
            if debug:
                print(f'Wildcard file {wildcard_file}.txt not found in {search_path}')
        last_end = m.end()
    new_prompt += prompt[last_end:]
    return new_prompt


def strip_all_syntax(text):
    # replace any <lora:lora_name> with nothing
    text = re.sub(r'<lora:(.*?)>', '', text)
    # replace any <lora:lora_name:multiplier> with nothing
    text = re.sub(r'<lora:(.*?):(.*?)>', '', text)
    # replace any <style:style_name> with nothing
    text = re.sub(r'<style:(.*?)>', '', text)
    # replace any __wildcard_name__ with nothing
    text = re.sub(r'__(.*?)__', '', text)
    # replace any __wildcard_name|word__ with nothing
    text = re.sub(r'__(.*?)\|(.*?)__', '', text)
    # replace any [2$__wildcard__] with nothing
    text = re.sub(r'\[\d+\$(.*?)\]', '', text)
    # replace any [2$__wildcard|word__] with nothing
    text = re.sub(r'\[\d+\$(.*?)\|(.*?)\]', '', text)
    # replace double spaces with single spaces
    text = text.replace('  ', ' ')
    # replace double commas with single commas
    text = text.replace(',,', ',')
    # replace ` , ` with `, `
    text = text.replace(' , ', ', ')
    # replace leading and trailing spaces and commas
    text = text.strip(' ,')
    # clean up any < > [ ] or _ that are left over
    text = text.replace('<', '').replace('>', '').replace('[', '').replace(']', '').replace('_', '')
    return text

def add_metadata_to_dict(info_dict, **kwargs):
    for key, value in kwargs.items():
        if isinstance(value, (int, float, str)):
            if key not in info_dict:
                info_dict[key] = [value]
            else:
                info_dict[key].append(value)

def extract_and_load_loras(text, model, clip):
    # load loras detected in the prompt text
    # The text for adding LoRA to the prompt, <lora:filename:multiplier>, is only used to enable LoRA, and is erased from prompt afterwards
    # The multiplier is optional, and defaults to 1.0
    # We update the model and clip, and return the new model and clip with the lora prompt stripped from the text
    # If multiple lora prompts are detected we chain them together like: original clip > clip_with_lora1 > clip_with_lora2 > clip_with_lora3 > etc
    lora_re = r'<lora:(.*?)(?::(.*?))?>'
    # find all lora prompts
    lora_prompts = re.findall(lora_re, text)
    stripped_text = text
    # if we found any lora prompts
    if len(lora_prompts) > 0:
        # loop through each lora prompt
        for lora_prompt in lora_prompts:
            # get the lora filename
            lora_filename = lora_prompt[0]
            # check for file extension in filename
            if '.safetensors' not in lora_filename:
                lora_filename += '.safetensors'
            # get the lora multiplier
            lora_multiplier = float(lora_prompt[1]) if lora_prompt[1] != '' else 1.0
            print('Loading LoRA: ' + lora_filename + ' with multiplier: ' + str(lora_multiplier))
            # apply the lora to the clip using the LoraLoader.load_lora function
            # def load_lora(self, model, clip, lora_name, strength_model, strength_clip):
            # ...
            # return (model_lora, clip_lora)
            # apply the lora to the clip
            model, clip_lora = LoraLoader.load_lora(model, clip, lora_filename, lora_multiplier, lora_multiplier)
    # strip the lora prompts from the text
    stripped_text = re.sub(lora_re, '', stripped_text)
    return model, clip, stripped_text

def process_random_syntax(text, seed):
        # The syntax for a random number is <random:lower_bound:upper_bound>
        # For example, <random:-1:0.5> will generate a random number between -1 and 0.5
        random.seed(seed)
        random_re = r'<random:(-?\d*\.?\d+):(-?\d*\.?\d+)>'
        matches = re.findall(random_re, text)

        for match in matches:
            lower_bound, upper_bound = map(float, match)
            random_value = random.uniform(lower_bound, upper_bound)
            random_value = round(random_value, 4)
            # Replace the syntax with the generated number
            text = text.replace(f'<random:{lower_bound}:{upper_bound}>', str(random_value))

        return text

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

def tensor2numpy(image):
    # Convert tensor to numpy array and transpose dimensions from (C, H, W) to (H, W, C)
    return (255.0 * image.cpu().numpy().squeeze().transpose(1, 2, 0)).astype(np.uint8)

class WildcardProcessor:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"prompt": ("STRING", {"multiline": True, "placeholder": "Prompt Text"}),
                             "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff})}}

    RETURN_TYPES = ('STRING',)
    FUNCTION = 'process'
    CATEGORY = 'Mikey/Text'

    def process(self, prompt, seed):
        prompt = find_and_replace_wildcards(prompt, seed)
        return (prompt, )

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
        s.ratio_sizes, s.ratio_dict = read_ratios()
        return {"required": { "width": ("INT", {"default": 1024, "min": 1, "max": 8192, "step": 1}),
                              "height": ("INT", {"default": 1024, "min": 1, "max": 8192, "step": 1}),
                              "batch_size": ("INT", {"default": 1, "min": 1, "max": 64})}}

    RETURN_TYPES = ('LATENT',)
    FUNCTION = 'generate'
    CATEGORY = 'Mikey/Latent'

    def generate(self, width, height, batch_size=1):
        # solver
        if width == 1 and height == 1 or width == height:
            w, h = 1024, 1024
        if f'{width}:{height}' in self.ratio_dict:
            w, h = self.ratio_dict[f'{width}:{height}']
        else:
            w, h = find_latent_size(width, height)
        latent = torch.zeros([batch_size, 4, h // 8, w // 8])
        return ({"samples":latent}, )

class RatioAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        s.ratio_sizes, s.ratio_dict = read_ratios()
        default_ratio = s.ratio_sizes[0]
        # prepend 'custom' to ratio_sizes
        s.ratio_sizes.insert(0, 'custom')
        s.ratio_presets, s.ratio_config = read_ratio_presets()
        if 'none' not in s.ratio_presets:
            s.ratio_presets.append('none')
        return {"required": { "preset": (s.ratio_presets, {"default": "none"}),
                              "swap_axis": (['true','false'], {"default": 'false'}),
                              "select_latent_ratio": (s.ratio_sizes, {'default': default_ratio}),
                              "custom_latent_w": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1}),
                              "custom_latent_h": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1}),
                              "select_cte_ratio": (s.ratio_sizes, {'default': default_ratio}),
                              "cte_w": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1}),
                              "cte_h": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1}),
                              "cte_mult": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.01}),
                              "cte_res": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1}),
                              "cte_fit_size": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1}),
                              "select_target_ratio": (s.ratio_sizes, {'default': default_ratio}),
                              "target_w": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1}),
                              "target_h": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1}),
                              "target_mult": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.01}),
                              "target_res": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1}),
                              "target_fit_size": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1}),
                              "crop_w": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1}),
                              "crop_h": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1}),
                              "use_preset_seed": (['true','false'], {"default": 'false'}),
                              "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                              }}

    RETURN_TYPES = ('INT', 'INT', # latent
                    'INT', 'INT', # clip text encode
                    'INT', 'INT', # target
                    'INT', 'INT') # crop
    RETURN_NAMES = ('latent_w', 'latent_h',
                    'cte_w', 'cte_h',
                    'target_w', 'target_h',
                    'crop_w', 'crop_h')
    CATEGORY = 'Mikey/Utils'
    FUNCTION = 'calculate'

    def mult(self, width, height, mult):
        return int(width * mult), int(height * mult)

    def fit(self, width, height, fit_size):
        if width > height:
            return fit_size, int(height * fit_size / width)
        else:
            return int(width * fit_size / height), fit_size

    def res(self, width, height, res):
        return find_latent_size(width, height, res)

    def calculate(self, preset, swap_axis, select_latent_ratio, custom_latent_w, custom_latent_h,
                  select_cte_ratio, cte_w, cte_h, cte_mult, cte_res, cte_fit_size,
                  select_target_ratio, target_w, target_h, target_mult, target_res, target_fit_size,
                  crop_w, crop_h, use_preset_seed, seed):
        # check if use_preset_seed is true
        if use_preset_seed == 'true' and len(self.ratio_presets) > 1:
            # seed is a randomly generated number that can be much larger than the number of presets
            # we use the seed to select a preset
            offset = seed % len(self.ratio_presets - 1)
            presets = [p for p in self.ratio_presets if p != 'none']
            preset = presets[offset]
        # check if ratio preset is selected
        if preset != 'none':
            latent_width = self.ratio_config[preset]['custom_latent_w']
            latent_height = self.ratio_config[preset]['custom_latent_h']
            cte_w = self.ratio_config[preset]['cte_w']
            cte_h = self.ratio_config[preset]['cte_h']
            target_w = self.ratio_config[preset]['target_w']
            target_h = self.ratio_config[preset]['target_h']
            crop_w = self.ratio_config[preset]['crop_w']
            crop_h = self.ratio_config[preset]['crop_h']
            if swap_axis == 'true':
                latent_width, latent_height = latent_height, latent_width
                cte_w, cte_h = cte_h, cte_w
                target_w, target_h = target_h, target_w
                crop_w, crop_h = crop_h, crop_w
            """
            example user_ratio_presets.json
            {
                "ratio_presets": {
                    "all_1024": {
                        "custom_latent_w": 1024,
                        "custom_latent_h": 1024,
                        "cte_w": 1024,
                        "cte_h": 1024,
                        "target_w": 1024,
                        "target_h": 1024,
                        "crop_w": 0,
                        "crop_h": 0
                    },
                }
            }
            """
            return (latent_width, latent_height,
                    cte_w, cte_h,
                    target_w, target_h,
                    crop_w, crop_h)
        # if no preset is selected, check if custom latent ratio is selected
        if select_latent_ratio != 'custom':
            latent_width = self.ratio_dict[select_latent_ratio]["width"]
            latent_height = self.ratio_dict[select_latent_ratio]["height"]
        else:
            latent_width = custom_latent_w
            latent_height = custom_latent_h
        # check if cte ratio is selected
        if select_cte_ratio != 'custom':
            cte_w = self.ratio_dict[select_cte_ratio]["width"]
            cte_h = self.ratio_dict[select_cte_ratio]["height"]
        else:
            cte_w = cte_w
            cte_h = cte_h
        # check if cte_mult not 0
        if cte_mult != 0.0:
            cte_w, cte_h = self.mult(cte_w, cte_h, cte_mult)
        # check if cte_res not 0
        if cte_res != 0:
            cte_w, cte_h = self.res(cte_w, cte_h, cte_res)
        # check if cte_fit_size not 0
        if cte_fit_size != 0:
            cte_w, cte_h = self.fit(cte_w, cte_h, cte_fit_size)
        # check if target ratio is selected
        if select_target_ratio != 'custom':
            target_w = self.ratio_dict[select_target_ratio]["width"]
            target_h = self.ratio_dict[select_target_ratio]["height"]
        else:
            target_w = target_w
            target_h = target_h
        # check if target_mult not 0
        if target_mult != 0.0:
            target_w, target_h = self.mult(target_w, target_h, target_mult)
        # check if target_res not 0
        if target_res != 0:
            target_w, target_h = self.res(target_w, target_h, target_res)
        # check if target_fit_size not 0
        if target_fit_size != 0:
            target_w, target_h = self.fit(target_w, target_h, target_fit_size)
        return (latent_width, latent_height,
                cte_w, cte_h,
                target_w, target_h,
                crop_w, crop_h)

class PresetRatioSelector:
    @classmethod
    def INPUT_TYPES(s):
        s.ratio_presets, s.ratio_config = read_ratio_presets()
        return {"required": { "select_preset": (s.ratio_presets, {"default": "none"}),
                              "swap_axis": (['true','false'], {"default": 'false'}),
                              "use_preset_seed": (['true','false'], {"default": 'false'}),
                              "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff})}}

    RETURN_TYPES = ('INT', 'INT', # latent
                    'INT', 'INT', # clip text encode
                    'INT', 'INT', # target
                    'INT', 'INT') # crop
    RETURN_NAMES = ('latent_w', 'latent_h',
                    'cte_w', 'cte_h',
                    'target_w', 'target_h',
                    'crop_w', 'crop_h')
    CATEGORY = 'Mikey/Utils'
    FUNCTION = 'calculate'

    def calculate(self, select_preset, swap_axis, use_preset_seed, seed):
        # check if use_preset_seed is true
        if use_preset_seed == 'true' and len(self.ratio_presets) > 0:
            # seed is a randomly generated number that can be much larger than the number of presets
            # we use the seed to select a preset
            offset = seed % len(self.ratio_presets - 1)
            presets = [p for p in self.ratio_presets if p != 'none']
            select_preset = presets[offset]
        latent_width = self.ratio_config[select_preset]['custom_latent_w']
        latent_height = self.ratio_config[select_preset]['custom_latent_h']
        cte_w = self.ratio_config[select_preset]['cte_w']
        cte_h = self.ratio_config[select_preset]['cte_h']
        target_w = self.ratio_config[select_preset]['target_w']
        target_h = self.ratio_config[select_preset]['target_h']
        crop_w = self.ratio_config[select_preset]['crop_w']
        crop_h = self.ratio_config[select_preset]['crop_h']
        if swap_axis == 'true':
            latent_width, latent_height = latent_height, latent_width
            cte_w, cte_h = cte_h, cte_w
            target_w, target_h = target_h, target_w
            crop_w, crop_h = crop_h, crop_w
        return (latent_width, latent_height,
                cte_w, cte_h,
                target_w, target_h,
                crop_w, crop_h)

class INTtoSTRING:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"int_": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                             "use_commas": (['true','false'], {"default": 'false'})}}

    RETURN_TYPES = ('STRING',)
    FUNCTION = 'convert'
    CATEGORY = 'Mikey/Utils'

    def convert(self, int_, use_commas):
        if use_commas == 'true':
            return (f'{int_:,}', )
        else:
            return (f'{int_}', )

class FLOATtoSTRING:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"float_": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1000000.0}),
                             "use_commas": (['true','false'], {"default": 'false'})}}

    RETURN_TYPES = ('STRING',)
    FUNCTION = 'convert'
    CATEGORY = 'Mikey/Utils'

    def convert(self, float_, use_commas):
        if use_commas == 'true':
            return (f'{float_:,}', )
        else:
            return (f'{float_}', )

class ResizeImageSDXL:
    crop_methods = ["disabled", "center"]
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "image": ("IMAGE",), "upscale_method": (s.upscale_methods,),
                              "crop": (s.crop_methods,)}}

    RETURN_TYPES = ('IMAGE',)
    FUNCTION = 'resize'
    CATEGORY = 'Mikey/Image'

    def upscale(self, image, upscale_method, width, height, crop):
        samples = image.movedim(-1,1)
        s = comfy.utils.common_upscale(samples, width, height, upscale_method, crop)
        s = s.movedim(1,-1)
        return (s,)

    def resize(self, image, upscale_method, crop):
        w, h = find_latent_size(image.shape[2], image.shape[1])
        print('Resizing image from {}x{} to {}x{}'.format(image.shape[2], image.shape[1], w, h))
        img = self.upscale(image, upscale_method, w, h, crop)[0]
        return (img, )

class BatchResizeImageSDXL(ResizeImageSDXL):
    crop_methods = ["disabled", "center"]
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image_directory": ("STRING", {"multiline": False, "placeholder": "Image Directory"}),
                             "upscale_method": (s.upscale_methods,),
                             "crop": (s.crop_methods,)},}

    RETURN_TYPES = ('IMAGE',)
    RETURN_NAMES = ('image',)
    FUNCTION = 'batch'
    CATEGORY = 'Mikey/Image'
    OUTPUT_IS_LIST = (True, )

    def batch(self, image_directory, upscale_method, crop):
        if not os.path.exists(image_directory):
            raise Exception(f"Image directory {image_directory} does not exist")

        images = []
        for file in os.listdir(image_directory):
            if file.endswith('.png') or file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.webp') or file.endswith('.bmp') or file.endswith('.gif'):
                img = Image.open(os.path.join(image_directory, file))
                img = pil2tensor(img)
                # resize image
                img = self.resize(img, upscale_method, crop)[0]
                images.append(img)
        return (images,)

class BatchCropImage:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image_directory": ("STRING", {"multiline": False, "placeholder": "Image Directory"}),
                             "crop_amount": ("FLOAT", {"default": 0.05})}}

    RETURN_TYPES = ('IMAGE',)
    RETURN_NAMES = ('image',)
    FUNCTION = 'batch'
    CATEGORY = 'Mikey/Image'
    OUTPUT_IS_LIST = (True, )

    def batch(self, image_directory, crop_amount):
        if not os.path.exists(image_directory):
            raise Exception(f"Image directory {image_directory} does not exist")

        images = []
        for file in os.listdir(image_directory):
            if file.endswith('.png') or file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.webp') or file.endswith('.bmp') or file.endswith('.gif'):
                img = Image.open(os.path.join(image_directory, file))
                # resize image
                width, height = img.size
                pixels = int(width * crop_amount) // 8 * 8
                left = pixels
                upper = pixels
                right = width - pixels
                lower = height - pixels
                # Crop the image
                cropped_img = img.crop((left, upper, right, lower))
                img = pil2tensor(cropped_img)
                images.append(img)
        return (images,)

class BatchCropResizeInplace:
    crop_methods = ["disabled", "center"]
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image_directory": ("STRING", {"multiline": False, "placeholder": "Image Directory"}),
                             "subdirectories": (['true', 'false'], {"default": 'false'}),
                             "replace_original": (['true', 'false'], {"default": 'false'}),
                             "replace_suffix": ("STRING", {"default": "_cropped_resized"}),
                             "upscale_method": (s.upscale_methods,),
                             "crop": (s.crop_methods,),
                             "crop_amount": ("FLOAT", {"default": 0.05})}}

    RETURN_TYPES = ('STRING',)
    RETURN_NAMES = ('job_done',)
    FUNCTION = 'batch'
    CATEGORY = 'Mikey/Image'

    def crop(self, image, crop_amount):
        # resize image
        width, height = image.size
        pixels = int(width * crop_amount) // 8 * 8
        left = pixels
        upper = pixels
        right = width - pixels
        lower = height - pixels
        # Crop the image
        cropped_img = image.crop((left, upper, right, lower))
        return cropped_img

    def upscale(self, image, upscale_method, width, height, crop):
        samples = image.movedim(-1,1)
        s = comfy.utils.common_upscale(samples, width, height, upscale_method, crop)
        s = s.movedim(1,-1)
        return (s,)

    def resize(self, image, upscale_method, crop):
        image = pil2tensor(image)
        w, h = find_latent_size(image.shape[2], image.shape[1])
        img = self.upscale(image, upscale_method, w, h, crop)[0]
        img = tensor2pil(img)
        return img

    def get_files_from_directory(self, image_directory, subdirectories):
        if subdirectories == 'true':
            files = [os.path.join(root, name)
                    for root, dirs, files in os.walk(image_directory)
                    for name in files
                    if name.endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"))]
        else:
            files = [os.path.join(image_directory, f)
                     for f in os.listdir(image_directory)
                     if os.path.isfile(os.path.join(image_directory, f)) and f.endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"))]
        return files

    def batch(self, image_directory, subdirectories, replace_original, replace_suffix, upscale_method, crop, crop_amount):
        if not os.path.exists(image_directory):
            raise Exception(f"Image directory {image_directory} does not exist")

        files = self.get_files_from_directory(image_directory, subdirectories)

        for file in tqdm(files, desc='Processing images'):
            img = Image.open(file)
            # crop image
            if crop != 'disabled':
                img = self.crop(img, crop_amount)
            # resize image
            img = self.resize(img, upscale_method, crop)
            # save image
            if replace_original == 'true':
                img.save(file)
            else:
                filename, file_extension = os.path.splitext(file)
                img.save(filename + replace_suffix + file_extension)
        return (f'Job done, {len(files)} images processed',)


def get_save_image_path(filename_prefix, output_dir, image_width=0, image_height=0):
    def map_filename(filename):
        try:
            # Ignore files that are not images
            if not filename.endswith('.png'):
                return 0
            # Assuming filenames are in the format you provided,
            # the counter would be the second last item when splitting by '_'
            digits = int(filename.split('_')[-2])
        except:
            digits = 0
        return digits

    def compute_vars(input, image_width, image_height):
        input = input.replace("%width%", str(image_width))
        input = input.replace("%height%", str(image_height))
        return input

    filename_prefix = compute_vars(filename_prefix, image_width, image_height)

    subfolder = os.path.dirname(os.path.normpath(filename_prefix))
    filename = os.path.basename(os.path.normpath(filename_prefix))

    # Remove trailing period from filename, if present
    if filename.endswith('.'):
        filename = filename[:-1]

    full_output_folder = os.path.join(output_dir, subfolder)

    if os.path.commonpath((output_dir, os.path.abspath(full_output_folder))) != output_dir:
        print("Saving image outside the output folder is not allowed.")
        return {}

    try:
        counter = max(map(map_filename, os.listdir(full_output_folder)), default=0) + 1
    except FileNotFoundError:
        os.makedirs(full_output_folder, exist_ok=True)
        counter = 1
    return full_output_folder, filename, counter, subfolder, filename_prefix

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
        full_output_folder, filename, counter, subfolder, filename_prefix = get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        results = list()
        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = PngInfo()
            pos_trunc = ''
            if prompt is not None:
                metadata.add_text("prompt", json.dumps(prompt, ensure_ascii=False))
            if extra_pnginfo is not None:
                for x in extra_pnginfo:
                    metadata.add_text(x, json.dumps(extra_pnginfo[x], ensure_ascii=False))
            if positive_prompt:
                metadata.add_text("positive_prompt", json.dumps(positive_prompt, ensure_ascii=False))
                # replace any special characters with nothing and spaces with _
                clean_pos = re.sub(r'[^a-zA-Z0-9 ]', '', positive_prompt)
                pos_trunc = clean_pos.replace(' ', '_')[0:80]
            if negative_prompt:
                metadata.add_text("negative_prompt", json.dumps(negative_prompt, ensure_ascii=False))
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

class SaveImagesMikeyML:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"images": ("IMAGE", ),
                     'sub_directory': ("STRING", {'default': ''}),
                     "filename_text_1": ("STRING", {'default': 'Filename Text 1'}),
                     "filename_text_2": ("STRING", {'default': 'Filename Text 2'}),
                     "filename_text_3": ("STRING", {'default': 'Filename Text 3'}),
                     "filename_separator": ("STRING", {'default': '_'}),
                     "timestamp": (["true", "false"], {'default': 'true'}),
                     "counter_type": (["none", "folder", "filename"], {'default': 'folder'}),
                     "filename_text_1_pos": ("INT", {'default': 0}),
                     "filename_text_2_pos": ("INT", {'default': 2}),
                     "filename_text_3_pos": ("INT", {'default': 4}),
                     "timestamp_pos": ("INT", {'default': 1}),
                     "timestamp_type": (['job','save_time'], {'default': 'save_time'}),
                     "counter_pos": ("INT", {'default': 3}),
                     "extra_metadata": ("STRING", {'default': 'Extra Metadata'}),},
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "Mikey/Image"

    def _prepare_filename_texts(self, filename_text_1, filename_text_2, filename_text_3):
        # replace default values with empty strings
        filename_texts = [filename_text_1, filename_text_2, filename_text_3]
        default_texts = ['Filename Text 1', 'Filename Text 2', 'Filename Text 3']
        for i, text in enumerate(filename_texts):
            if text == default_texts[i]:
                filename_texts[i] = ''
            # replace any special characters with nothing
            filename_texts[i] = re.sub(r'[^a-zA-Z0-9 ]', '', filename_texts[i])
        # need to make sure the total filelength name is under 256 characters including the .png, separator, and counter
        # if the total length is over 256 characters, truncate the longest text to fit under 250 characters total length
        total_length = len(filename_texts[0]) + len(filename_texts[1]) + len(filename_texts[2]) + 5 + 5 + 12
        if total_length > 120:
            longest_text = max(filename_texts, key=len)
            longest_text_idx = filename_texts.index(longest_text)
            text_length_without_longest = total_length - len(longest_text)
            filename_texts[longest_text_idx] = longest_text[0:120 - text_length_without_longest]
        return filename_texts

    def _get_initial_counter(self, files, full_output_folder, counter_type, filename_separator, counter_pos, filename_texts):
        counter = 1
        if counter_type == "folder":
            if files:
                for f in files:
                    if filename_separator in f:
                        try:
                            counter = max(counter, int(f.split(filename_separator)[counter_pos]) + 1)
                        except:
                            counter = 1
                            break
            else:
                counter = 1
        elif counter_type == "filename":
            for f in files:
                f_split = f.split(filename_separator)
                # strip .png from strings
                f_split = [x.replace('.png', '') for x in f_split]
                matched_texts = all(
                    filename_texts[i] == f_split[i] for i in range(3) if filename_texts[i]
                )
                if matched_texts:
                    counter += 1
        return counter

    def _get_next_counter(self, full_output_folder, filename_base, counter):
        """Checks for the next available counter value."""
        while True:
            current_filename = filename_base.format(counter=f"{counter:05}")
            if not os.path.exists(os.path.join(full_output_folder, f"{current_filename}.png")):
                return counter
            counter += 1

    def save_images(self, images, sub_directory, filename_text_1, filename_text_2, filename_text_3,
                    filename_separator, timestamp, counter_type,
                    filename_text_1_pos, filename_text_2_pos, filename_text_3_pos,
                    timestamp_pos, timestamp_type, counter_pos, extra_metadata,
                    prompt=None, extra_pnginfo=None):
        positions = [filename_text_1_pos, filename_text_2_pos, filename_text_3_pos, timestamp_pos, counter_pos]
        if len(positions) != len(set(positions)):
            raise ValueError("Duplicate position numbers detected. Please ensure all position numbers are unique.")

        full_output_folder = os.path.join(self.output_dir, sub_directory)
        os.makedirs(full_output_folder, exist_ok=True)

        filename_texts = self._prepare_filename_texts(filename_text_1, filename_text_2, filename_text_3)

        if timestamp == 'true':
            ts = datetime.datetime.now().strftime("%y%m%d%H%M%S")
        else:
            ts = ''

        elements = {
            filename_text_1_pos: filename_texts[0],
            filename_text_2_pos: filename_texts[1],
            filename_text_3_pos: filename_texts[2],
            timestamp_pos: ts,
            counter_pos: 'counter' if counter_type != 'none' else None
        }

        # Construct initial filename without the counter
        sorted_elements = [elem for _, elem in sorted(elements.items()) if elem]
        filename_base = filename_separator.join(sorted_elements).replace('counter', '{counter}')

        # Get initial counter value
        files = os.listdir(full_output_folder)
        counter = self._get_initial_counter(files, full_output_folder, counter_type, filename_separator, counter_pos, filename_texts)

        results = list()
        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = PngInfo()
            if prompt is not None:
                metadata.add_text("prompt", json.dumps(prompt, ensure_ascii=False))
            if extra_pnginfo is not None:
                for x in extra_pnginfo:
                    metadata.add_text(x, json.dumps(extra_pnginfo[x], ensure_ascii=False))
            if extra_metadata:
                metadata.add_text("extra_metadata", json.dumps(extra_metadata, ensure_ascii=False))
            # Check and get the next available counter
            counter = self._get_next_counter(full_output_folder, filename_base, counter)
            current_filename = filename_base.format(counter=f"{counter:05}")
            if timestamp_type == 'save_time' and timestamp == 'true':
                current_timestamp = datetime.datetime.now().strftime("%y%m%d%H%M%S")
                current_filename = current_filename.replace(ts, current_timestamp)
                ts = current_timestamp

            img.save(os.path.join(full_output_folder, f"{current_filename}.png"), pnginfo=metadata, compress_level=4)
            results.append({
                "filename": f"{current_filename}.png",
                "subfolder": sub_directory,
                "type": self.type
            })
            counter += 1

        return {"ui": {"images": results}}

class SaveImageNoDisplay(SaveImagesMikeyML):
    # inherits from SaveImagesMikeyML
    # only difference is we are not going to output anything to the UI
    def __init__(self):
        super().__init__()

    RETURN_TYPES = ()
    FUNCTION = "save_images_no_display"
    OUTPUT_NODE = True
    CATEGORY = "Mikey/Image"

    def save_images_no_display(self, images, sub_directory, filename_text_1, filename_text_2, filename_text_3,
                    filename_separator, timestamp, counter_type,
                    filename_text_1_pos, filename_text_2_pos, filename_text_3_pos,
                    timestamp_pos, timestamp_type, counter_pos, extra_metadata,
                    prompt=None, extra_pnginfo=None):
        self.save_images(images, sub_directory, filename_text_1, filename_text_2, filename_text_3,
                    filename_separator, timestamp, counter_type,
                    filename_text_1_pos, filename_text_2_pos, filename_text_3_pos,
                    timestamp_pos, timestamp_type, counter_pos, extra_metadata,
                    prompt, extra_pnginfo)
        return (None,)

class AddMetaData:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",),
                             "label": ("STRING", {"multiline": False, "placeholder": "Label for metadata"}),
                             "text_value": ("STRING", {"multiline": True, "placeholder": "Text to add to metadata"})},
                "hidden": {"extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ('IMAGE',)
    FUNCTION = "add_metadata"
    CATEGORY = "Mikey/Meta"
    OUTPUT_NODE = True

    def add_metadata(self, image, label, text_value, prompt=None, extra_pnginfo=None):
        if extra_pnginfo is None:
            extra_pnginfo = {}
        if label in extra_pnginfo:
            extra_pnginfo[label] += ', ' + text_value
        else:
            extra_pnginfo[label] = text_value
        return (image,)

class SaveMetaData:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {'image': ('IMAGE',),
                             'filename_prefix': ("STRING", {"default": ""}),
                             'timestamp_prefix': (['true','false'], {'default':'true'}),
                             'counter': (['true','false'], {'default':'true'}),},
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},}

    RETURN_TYPES = ()
    FUNCTION = "save_metadata"
    CATEGORY = "Mikey/Meta"
    OUTPUT_NODE = True

    def save_metadata(self, image, filename_prefix, timestamp_prefix, counter, prompt=None, extra_pnginfo=None):
        # save metatdata to txt file
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, folder_paths.get_output_directory(), 1, 1)
        ts_str = datetime.datetime.now().strftime("%y%m%d%H%M")
        filen = ''
        if timestamp_prefix == 'true':
            filen += ts_str + '_'
        filen = filen + filename_prefix
        if counter == 'true':
            filen += '_' + str(counter)
        filename = filen + '.txt'
        file_path = os.path.join(full_output_folder, filename)
        with open(file_path, 'w') as file:
            for key, value in extra_pnginfo.items():
                file.write(f'{key}: {value}\n')
            for key, value in prompt.items():
                file.write(f'{key}: {value}\n')
        return {'save_metadata': {'filename': filename, 'subfolder': subfolder}}

class FileNamePrefix:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {'date': (['true','false'], {'default':'true'}),
                             'date_directory': (['true','false'], {'default':'true'}),
                             'custom_text': ('STRING', {'default': ''})}}

    RETURN_TYPES = ('STRING',)
    RETURN_NAMES = ('filename_prefix',)
    FUNCTION = 'get_filename_prefix'
    CATEGORY = 'Mikey/Meta'

    def get_filename_prefix(self, date, date_directory, custom_directory, custom_text):
        filename_prefix = ''
        if date_directory == 'true':
            ts_str = datetime.datetime.now().strftime("%y%m%d")
            filename_prefix += ts_str + '/'
        if date == 'true':
            ts_str = datetime.datetime.now().strftime("%y%m%d%H%M")
            filename_prefix += ts_str
        if custom_text != '':
            filename_prefix += '_' + custom_text
        return (filename_prefix,)

class PromptWithStyle:
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
    OUTPUT_NODE = True

    def start(self, positive_prompt, negative_prompt, style, ratio_selected, batch_size, seed):
        # process random syntax
        positive_prompt = process_random_syntax(positive_prompt, seed)
        negative_prompt = process_random_syntax(negative_prompt, seed)
        # process wildcards
        print('Positive Prompt Entered:', positive_prompt)
        pos_prompt = find_and_replace_wildcards(positive_prompt, seed, debug=True)
        print('Positive Prompt:', pos_prompt)
        print('Negative Prompt Entered:', negative_prompt)
        neg_prompt = find_and_replace_wildcards(negative_prompt, seed, debug=True)
        print('Negative Prompt:', neg_prompt)
        if pos_prompt != '' and pos_prompt != 'Positive Prompt' and pos_prompt is not None:
            if '{prompt}' in self.pos_style[style]:
                pos_prompt = self.pos_style[style].replace('{prompt}', pos_prompt)
            else:
                if self.pos_style[style]:
                    pos_prompt = pos_prompt + ', ' + self.pos_style[style]
        else:
            pos_prompt = self.pos_style[style]
        if neg_prompt != '' and neg_prompt != 'Negative Prompt' and neg_prompt is not None:
            if '{prompt}' in self.neg_style[style]:
                neg_prompt = self.neg_style[style].replace('{prompt}', neg_prompt)
            else:
                if self.neg_style[style]:
                    neg_prompt = neg_prompt + ', ' + self.neg_style[style]
        else:
            neg_prompt = self.neg_style[style]
        width = self.ratio_dict[ratio_selected]["width"]
        height = self.ratio_dict[ratio_selected]["height"]
        # calculate dimensions for target_width, target height (base) and refiner_width, refiner_height (refiner)
        ratio = min([width, height]) / max([width, height])
        target_width, target_height = (4096, 4096 * ratio // 8 * 8) if width > height else (4096 * ratio // 8 * 8, 4096)
        refiner_width = target_width
        refiner_height = target_height
        print('Width:', width, 'Height:', height,
              'Target Width:', target_width, 'Target Height:', target_height,
              'Refiner Width:', refiner_width, 'Refiner Height:', refiner_height)
        latent = torch.zeros([batch_size, 4, height // 8, width // 8])
        return ({"samples":latent},
                str(pos_prompt),
                str(neg_prompt),
                str(self.pos_style[style]),
                str(self.neg_style[style]),
                width,
                height,
                refiner_width,
                refiner_height,)

class PromptWithStyleV2:
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
                             "clip_base": ("CLIP",), "clip_refiner": ("CLIP",),
                             }
        }

    RETURN_TYPES = ('LATENT',
                    'CONDITIONING','CONDITIONING','CONDITIONING','CONDITIONING',
                    'STRING','STRING')
    RETURN_NAMES = ('samples',
                    'base_pos_cond','base_neg_cond','refiner_pos_cond','refiner_neg_cond',
                    'positive_prompt','negative_prompt')

    FUNCTION = 'start'
    CATEGORY = 'Mikey'

    def start(self, clip_base, clip_refiner, positive_prompt, negative_prompt, style, ratio_selected, batch_size, seed):
        """ get output from PromptWithStyle.start """
        (latent,
         pos_prompt, neg_prompt,
         pos_style, neg_style,
         width, height,
         refiner_width, refiner_height) = PromptWithStyle.start(self, positive_prompt,
                                                                negative_prompt,
                                                                style, ratio_selected,
                                                                batch_size, seed)
        # calculate dimensions for target_width, target height (base) and refiner_width, refiner_height (refiner)
        ratio = min([width, height]) / max([width, height])
        target_width, target_height = (4096, 4096 * ratio // 8 * 8) if width > height else (4096 * ratio // 8 * 8, 4096)
        refiner_width = target_width
        refiner_height = target_height
        print('Width:', width, 'Height:', height,
              'Target Width:', target_width, 'Target Height:', target_height,
              'Refiner Width:', refiner_width, 'Refiner Height:', refiner_height)
        # encode text
        sdxl_pos_cond = CLIPTextEncodeSDXL.encode(self, clip_base, width, height, 0, 0, target_width, target_height, pos_prompt, pos_style)[0]
        sdxl_neg_cond = CLIPTextEncodeSDXL.encode(self, clip_base, width, height, 0, 0, target_width, target_height, neg_prompt, neg_style)[0]
        refiner_pos_cond = CLIPTextEncodeSDXLRefiner.encode(self, clip_refiner, 6, refiner_width, refiner_height, pos_prompt)[0]
        refiner_neg_cond = CLIPTextEncodeSDXLRefiner.encode(self, clip_refiner, 2.5, refiner_width, refiner_height, neg_prompt)[0]
        # return
        return (latent,
                sdxl_pos_cond, sdxl_neg_cond,
                refiner_pos_cond, refiner_neg_cond,
                pos_prompt, neg_prompt)

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
    OUTPUT_NODE = True

    def start(self, positive_prompt, negative_prompt, positive_style, negative_style, ratio_selected, batch_size, seed):
        # process random syntax
        positive_prompt = process_random_syntax(positive_prompt, seed)
        negative_prompt = process_random_syntax(negative_prompt, seed)
        # process wildcards
        positive_prompt = find_and_replace_wildcards(positive_prompt, seed)
        negative_prompt = find_and_replace_wildcards(negative_prompt, seed)
        width = self.ratio_dict[ratio_selected]["width"]
        height = self.ratio_dict[ratio_selected]["height"]
        latent = torch.zeros([batch_size, 4, height // 8, width // 8])
        # calculate dimensions for target_width, target height (base) and refiner_width, refiner_height (refiner)
        ratio = min([width, height]) / max([width, height])
        target_width, target_height = (4096, 4096 * ratio // 8 * 8) if width > height else (4096 * ratio // 8 * 8, 4096)
        refiner_width = target_width
        refiner_height = target_height
        print('Width:', width, 'Height:', height,
              'Target Width:', target_width, 'Target Height:', target_height,
              'Refiner Width:', refiner_width, 'Refiner Height:', refiner_height)
        return ({"samples":latent},
                str(positive_prompt),
                str(negative_prompt),
                str(positive_style),
                str(negative_style),
                width,
                height,
                refiner_width,
                refiner_height,)

class PromptWithStyleV3:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        s.ratio_sizes, s.ratio_dict = read_ratios()
        s.styles, s.pos_style, s.neg_style = read_styles()
        s.fit = ['true','false']
        s.custom_size = ['true', 'false']
        return {"required": {"positive_prompt": ("STRING", {"multiline": True, 'default': 'Positive Prompt'}),
                             "negative_prompt": ("STRING", {"multiline": True, 'default': 'Negative Prompt'}),
                             "ratio_selected": (s.ratio_sizes,),
                             "custom_size": (s.custom_size, {"default": "false"}),
                             "fit_custom_size": (s.fit,),
                             "custom_width": ("INT", {"default": 1024, "min": 1, "max": 8192, "step": 1}),
                             "custom_height": ("INT", {"default": 1024, "min": 1, "max": 8192, "step": 1}),
                             "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
                             "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                             "target_mode": (["match", "2x", "4x", "2x90", "4x90",
                                              "2048","2048-90","4096", "4096-90"], {"default": "4x"}),
                             "base_model": ("MODEL",), "clip_base": ("CLIP",), "clip_refiner": ("CLIP",),
                             },
                "hidden": {"extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ('MODEL','LATENT',
                    'CONDITIONING','CONDITIONING','CONDITIONING','CONDITIONING',
                    'STRING','STRING')
    RETURN_NAMES = ('base_model','samples',
                    'base_pos_cond','base_neg_cond','refiner_pos_cond','refiner_neg_cond',
                    'positive_prompt','negative_prompt')

    FUNCTION = 'start'
    CATEGORY = 'Mikey'

    def extract_and_load_loras(self, text, model, clip):
        # load loras detected in the prompt text
        # The text for adding LoRA to the prompt, <lora:filename:multiplier>, is only used to enable LoRA, and is erased from prompt afterwards
        # The multiplier is optional, and defaults to 1.0
        # We update the model and clip, and return the new model and clip with the lora prompt stripped from the text
        # If multiple lora prompts are detected we chain them together like: original clip > clip_with_lora1 > clip_with_lora2 > clip_with_lora3 > etc
        lora_re = r'<lora:(.*?)(?::(.*?))?>'
        # find all lora prompts
        lora_prompts = re.findall(lora_re, text)
        stripped_text = text
        # if we found any lora prompts
        if len(lora_prompts) > 0:
            # loop through each lora prompt
            for lora_prompt in lora_prompts:
                # get the lora filename
                lora_filename = lora_prompt[0]
                # check for file extension in filename
                if '.safetensors' not in lora_filename:
                    lora_filename += '.safetensors'
                # get the lora multiplier
                lora_multiplier = float(lora_prompt[1]) if lora_prompt[1] != '' else 1.0
                print('Loading LoRA: ' + lora_filename + ' with multiplier: ' + str(lora_multiplier))
                # apply the lora to the clip using the LoraLoader.load_lora function
                # def load_lora(self, model, clip, lora_name, strength_model, strength_clip):
                # ...
                # return (model_lora, clip_lora)
                # apply the lora to the clip
                model, clip_lora = LoraLoader.load_lora(self, model, clip, lora_filename, lora_multiplier, lora_multiplier)
                stripped_text = stripped_text.replace(f'<lora:{lora_filename}:{lora_multiplier}>', '')
                stripped_text = stripped_text.replace(f'<lora:{lora_filename}>', '')
        return model, clip, stripped_text

    def parse_prompts(self, positive_prompt, negative_prompt, style, seed):
        positive_prompt = find_and_replace_wildcards(positive_prompt, seed, debug=True)
        negative_prompt = find_and_replace_wildcards(negative_prompt, seed, debug=True)
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
        return pos_prompt, neg_prompt

    def start(self, base_model, clip_base, clip_refiner, positive_prompt, negative_prompt, ratio_selected, batch_size, seed,
              custom_size='false', fit_custom_size='false', custom_width=1024, custom_height=1024, target_mode='match',
              extra_pnginfo=None):
        if extra_pnginfo is None:
            extra_pnginfo = {'PromptWithStyle': {}}

        prompt_with_style = extra_pnginfo.get('PromptWithStyle', {})

        add_metadata_to_dict(prompt_with_style, positive_prompt=positive_prompt, negative_prompt=negative_prompt,
                            ratio_selected=ratio_selected, batch_size=batch_size, seed=seed, custom_size=custom_size,
                            fit_custom_size=fit_custom_size, custom_width=custom_width, custom_height=custom_height,
                            target_mode=target_mode)

        if custom_size == 'true':
            if fit_custom_size == 'true':
                if custom_width == 1 and custom_height == 1:
                    width, height = 1024, 1024
                if custom_width == custom_height:
                    width, height = 1024, 1024
                if f'{custom_width}:{custom_height}' in self.ratio_dict:
                    width, height = self.ratio_dict[f'{custom_width}:{custom_height}']
                else:
                    width, height = find_latent_size(custom_width, custom_height)
            else:
                width, height = custom_width, custom_height
        else:
            width = self.ratio_dict[ratio_selected]["width"]
            height = self.ratio_dict[ratio_selected]["height"]

        latent = torch.zeros([batch_size, 4, height // 8, width // 8])
        print(batch_size, 4, height // 8, width // 8)
        # calculate dimensions for target_width, target height (base) and refiner_width, refiner_height (refiner)
        ratio = min([width, height]) / max([width, height])
        if target_mode == 'match':
            target_width, target_height = width, height
            refiner_width, refiner_height = width * 4, height * 4
            #refiner_width, refiner_height = (4096, 4096 * ratio // 8 * 8) if width > height else (4096 * ratio // 8 * 8, 4096)
        elif target_mode == '2x':
            target_width, target_height = width * 2, height * 2
            refiner_width, refiner_height = width * 4, height * 4
            #refiner_width, refiner_height = (4096, 4096 * ratio // 8 * 8) if width > height else (4096 * ratio // 8 * 8, 4096)
        elif target_mode == '4x':
            target_width, target_height = width * 4, height * 4
            refiner_width, refiner_height = width * 4, height * 4
            #refiner_width, refiner_height = (4096, 4096 * ratio // 8 * 8) if width > height else (4096 * ratio // 8 * 8, 4096)
        elif target_mode == '2x90':
            target_width, target_height = height * 2, width * 2
            refiner_width, refiner_height = width * 4, height * 4
            #refiner_width, refiner_height = (4096, 4096 * ratio // 8 * 8) if width > height else (4096 * ratio // 8 * 8, 4096)
        elif target_mode == '4x90':
            target_width, target_height = height * 4, width * 4
            refiner_width, refiner_height = width * 4, height * 4
            #refiner_width, refiner_height = (4096, 4096 * ratio // 8 * 8) if width > height else (4096 * ratio // 8 * 8, 4096)
        elif target_mode == '4096':
            target_width, target_height = (4096, 4096 * ratio // 8 * 8) if width > height else (4096 * ratio // 8 * 8, 4096)
            refiner_width, refiner_height = width * 4, height * 4
            #refiner_width, refiner_height = (4096, 4096 * ratio // 8 * 8) if width > height else (4096 * ratio // 8 * 8, 4096)
        elif target_mode == '4096-90':
            target_width, target_height = (4096, 4096 * ratio // 8 * 8) if width < height else (4096 * ratio // 8 * 8, 4096)
            refiner_width, refiner_height = width * 4, height * 4
            #refiner_width, refiner_height = (4096, 4096 * ratio // 8 * 8) if width > height else (4096 * ratio // 8 * 8, 4096)
        elif target_mode == '2048':
            target_width, target_height = (2048, 2048 * ratio // 8 * 8) if width > height else (2048 * ratio // 8 * 8, 2048)
            refiner_width, refiner_height = width * 4, height * 4
            #refiner_width, refiner_height = (4096, 4096 * ratio // 8 * 8) if width > height else (4096 * ratio // 8 * 8, 4096)
        elif target_mode == '2048-90':
            target_width, target_height = (2048, 2048 * ratio // 8 * 8) if width < height else (2048 * ratio // 8 * 8, 2048)
            refiner_width, refiner_height = width * 4, height * 4
            #refiner_width, refiner_height = (4096, 4096 * ratio // 8 * 8) if width > height else (4096 * ratio // 8 * 8, 4096)
        print('Width:', width, 'Height:', height,
              'Target Width:', target_width, 'Target Height:', target_height,
              'Refiner Width:', refiner_width, 'Refiner Height:', refiner_height)
        add_metadata_to_dict(prompt_with_style, width=width, height=height, target_width=target_width, target_height=target_height,
                             refiner_width=refiner_width, refiner_height=refiner_height, crop_w=0, crop_h=0)
        # process random syntax
        positive_prompt = process_random_syntax(positive_prompt, seed)
        negative_prompt = process_random_syntax(negative_prompt, seed)

        # check for $style in prompt, split the prompt into prompt and style
        user_added_style = False
        if '$style' in positive_prompt:
            self.styles.append('user_added_style')
            self.pos_style['user_added_style'] = positive_prompt.split('$style')[1].strip()
            self.neg_style['user_added_style'] = ''
            user_added_style = True
        if '$style' in negative_prompt:
            if 'user_added_style' not in self.styles:
                self.styles.append('user_added_style')
            self.neg_style['user_added_style'] = negative_prompt.split('$style')[1].strip()
            user_added_style = True
        if user_added_style:
            positive_prompt = positive_prompt.split('$style')[0].strip()
            if '$style' in negative_prompt:
                negative_prompt = negative_prompt.split('$style')[0].strip()
            positive_prompt = positive_prompt + '<style:user_added_style>'

        # first process wildcards
        positive_prompt_ = find_and_replace_wildcards(positive_prompt, seed, True)
        negative_prompt_ = find_and_replace_wildcards(negative_prompt, seed, True)
        add_metadata_to_dict(prompt_with_style, positive_prompt=positive_prompt_, negative_prompt=negative_prompt_)
        if len(positive_prompt_) != len(positive_prompt) or len(negative_prompt_) != len(negative_prompt):
            seed += random.randint(0, 1000000)
        positive_prompt = positive_prompt_
        negative_prompt = negative_prompt_
        # extract and load loras
        base_model, clip_base_pos, pos_prompt = self.extract_and_load_loras(positive_prompt, base_model, clip_base)
        base_model, clip_base_neg, neg_prompt = self.extract_and_load_loras(negative_prompt, base_model, clip_base)
        # find and replace style syntax
        # <style:style_name> will update the selected style
        style_re = r'<style:(.*?)>'
        pos_style_prompts = re.findall(style_re, pos_prompt)
        neg_style_prompts = re.findall(style_re, neg_prompt)
        # concat style prompts
        style_prompts = pos_style_prompts + neg_style_prompts
        print(style_prompts)
        base_pos_conds = []
        base_neg_conds = []
        refiner_pos_conds = []
        refiner_neg_conds = []
        if len(style_prompts) == 0:
            style_ = 'none'
            pos_prompt_, neg_prompt_ = self.parse_prompts(positive_prompt, negative_prompt, style_, seed)
            pos_style_, neg_style_ = pos_prompt_, neg_prompt_
            pos_prompt_, neg_prompt_ = strip_all_syntax(pos_prompt_), strip_all_syntax(neg_prompt_)
            pos_style_, neg_style_ = strip_all_syntax(pos_style_), strip_all_syntax(neg_style_)
            print("pos_prompt_", pos_prompt_)
            print("neg_prompt_", neg_prompt_)
            print("pos_style_", pos_style_)
            print("neg_style_", neg_style_)
            # encode text
            add_metadata_to_dict(prompt_with_style, style=style_, clip_g_positive=pos_prompt, clip_l_positive=pos_style_)
            add_metadata_to_dict(prompt_with_style, clip_g_negative=neg_prompt, clip_l_negative=neg_style_)
            sdxl_pos_cond = CLIPTextEncodeSDXL.encode(self, clip_base_pos, width, height, 0, 0, target_width, target_height, pos_prompt_, pos_style_)[0]
            sdxl_neg_cond = CLIPTextEncodeSDXL.encode(self, clip_base_neg, width, height, 0, 0, target_width, target_height, neg_prompt_, neg_style_)[0]
            refiner_pos_cond = CLIPTextEncodeSDXLRefiner.encode(self, clip_refiner, 6, refiner_width, refiner_height, pos_prompt_)[0]
            refiner_neg_cond = CLIPTextEncodeSDXLRefiner.encode(self, clip_refiner, 2.5, refiner_width, refiner_height, neg_prompt_)[0]
            return (base_model, {"samples":latent},
                    sdxl_pos_cond, sdxl_neg_cond,
                    refiner_pos_cond, refiner_neg_cond,
                    pos_prompt_, neg_prompt_, {'extra_pnginfo': extra_pnginfo})

        for style_prompt in style_prompts:
            """ get output from PromptWithStyle.start """
            # strip all style syntax from prompt
            style_ = style_prompt
            print(style_ in self.styles)
            if style_ not in self.styles:
                # try to match a key without being case sensitive
                style_search = next((x for x in self.styles if x.lower() == style_.lower()), None)
                # if there are still no matches
                if style_search is None:
                    print(f'Could not find style: {style_}')
                    style_ = 'none'
                    continue
                else:
                    style_ = style_search
            pos_prompt_ = re.sub(style_re, '', pos_prompt)
            neg_prompt_ = re.sub(style_re, '', neg_prompt)
            pos_prompt_, neg_prompt_ = self.parse_prompts(pos_prompt_, neg_prompt_, style_, seed)
            pos_style_, neg_style_ = str(self.pos_style[style_]), str(self.neg_style[style_])
            pos_prompt_, neg_prompt_ = strip_all_syntax(pos_prompt_), strip_all_syntax(neg_prompt_)
            pos_style_, neg_style_ = strip_all_syntax(pos_style_), strip_all_syntax(neg_style_)
            add_metadata_to_dict(prompt_with_style, style=style_, positive_prompt=pos_prompt_, negative_prompt=neg_prompt_,
                                 positive_style=pos_style_, negative_style=neg_style_)
            #base_model, clip_base_pos, pos_prompt_ = self.extract_and_load_loras(pos_prompt_, base_model, clip_base)
            #base_model, clip_base_neg, neg_prompt_ = self.extract_and_load_loras(neg_prompt_, base_model, clip_base)
            width_, height_ = width, height
            refiner_width_, refiner_height_ = refiner_width, refiner_height
            # encode text
            add_metadata_to_dict(prompt_with_style, style=style_, clip_g_positive=pos_prompt_, clip_l_positive=pos_style_)
            add_metadata_to_dict(prompt_with_style, clip_g_negative=neg_prompt_, clip_l_negative=neg_style_)
            base_pos_conds.append(CLIPTextEncodeSDXL.encode(self, clip_base_pos, width_, height_, 0, 0, target_width, target_height, pos_prompt_, pos_style_)[0])
            base_neg_conds.append(CLIPTextEncodeSDXL.encode(self, clip_base_neg, width_, height_, 0, 0, target_width, target_height, neg_prompt_, neg_style_)[0])
            refiner_pos_conds.append(CLIPTextEncodeSDXLRefiner.encode(self, clip_refiner, 6, refiner_width_, refiner_height_, pos_prompt_)[0])
            refiner_neg_conds.append(CLIPTextEncodeSDXLRefiner.encode(self, clip_refiner, 2.5, refiner_width_, refiner_height_, neg_prompt_)[0])
        # if none of the styles matched we will get an empty list so we need to check for that again
        if len(base_pos_conds) == 0:
            style_ = 'none'
            pos_prompt_, neg_prompt_ = self.parse_prompts(positive_prompt, negative_prompt, style_, seed)
            pos_style_, neg_style_ = pos_prompt_, neg_prompt_
            pos_prompt_, neg_prompt_ = strip_all_syntax(pos_prompt_), strip_all_syntax(neg_prompt_)
            pos_style_, neg_style_ = strip_all_syntax(pos_style_), strip_all_syntax(neg_style_)
            # encode text
            add_metadata_to_dict(prompt_with_style, style=style_, clip_g_positive=pos_prompt_, clip_l_positive=pos_style_)
            add_metadata_to_dict(prompt_with_style, clip_g_negative=neg_prompt_, clip_l_negative=neg_style_)
            sdxl_pos_cond = CLIPTextEncodeSDXL.encode(self, clip_base_pos, width, height, 0, 0, target_width, target_height, pos_prompt_, pos_style_)[0]
            sdxl_neg_cond = CLIPTextEncodeSDXL.encode(self, clip_base_neg, width, height, 0, 0, target_width, target_height, neg_prompt_, neg_style_)[0]
            refiner_pos_cond = CLIPTextEncodeSDXLRefiner.encode(self, clip_refiner, 6, refiner_width, refiner_height, pos_prompt_)[0]
            refiner_neg_cond = CLIPTextEncodeSDXLRefiner.encode(self, clip_refiner, 2.5, refiner_width, refiner_height, neg_prompt_)[0]
            return (base_model, {"samples":latent},
                    sdxl_pos_cond, sdxl_neg_cond,
                    refiner_pos_cond, refiner_neg_cond,
                    pos_prompt_, neg_prompt_, {'extra_pnginfo': extra_pnginfo})
        # loop through conds and add them together
        sdxl_pos_cond = base_pos_conds[0]
        weight = 1
        if len(base_pos_conds) > 1:
            for i in range(1, len(base_pos_conds)):
                weight += 1
                sdxl_pos_cond = ConditioningAverage.addWeighted(self, base_pos_conds[i], sdxl_pos_cond, 1 / weight)[0]
        sdxl_neg_cond = base_neg_conds[0]
        weight = 1
        if len(base_neg_conds) > 1:
            for i in range(1, len(base_neg_conds)):
                weight += 1
                sdxl_neg_cond = ConditioningAverage.addWeighted(self, base_neg_conds[i], sdxl_neg_cond, 1 / weight)[0]
        refiner_pos_cond = refiner_pos_conds[0]
        weight = 1
        if len(refiner_pos_conds) > 1:
            for i in range(1, len(refiner_pos_conds)):
                weight += 1
                refiner_pos_cond = ConditioningAverage.addWeighted(self, refiner_pos_conds[i], refiner_pos_cond, 1 / weight)[0]
        refiner_neg_cond = refiner_neg_conds[0]
        weight = 1
        if len(refiner_neg_conds) > 1:
            for i in range(1, len(refiner_neg_conds)):
                weight += 1
                refiner_neg_cond = ConditioningAverage.addWeighted(self, refiner_neg_conds[i], refiner_neg_cond, 1 / weight)[0]
        # return
        extra_pnginfo['PromptWithStyle'] = prompt_with_style
        return (base_model, {"samples":latent},
                sdxl_pos_cond, sdxl_neg_cond,
                refiner_pos_cond, refiner_neg_cond,
                pos_prompt_, neg_prompt_, {'extra_pnginfo': extra_pnginfo})

class LoraSyntaxProcessor:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "model": ("MODEL",),
                    "clip": ("CLIP",),
                    "text": ("STRING", {"multiline": True, "default": "<lora:filename:weight>"}),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff})
                    }
                }

    RETURN_TYPES = ('MODEL','CLIP','STRING','STRING')
    RETURN_NAMES = ('model','clip','text','unprocessed_text')
    FUNCTION = 'process'
    CATEGORY = 'Mikey/Lora'

    def process(self, model, clip, text, seed):
        # process random syntax
        text = process_random_syntax(text, seed)
        lora_re = r'<lora:(.*?)(?::(.*?))?>'
        # find all lora prompts
        lora_prompts = re.findall(lora_re, text)
        stripped_text = text
        # if we found any lora prompts
        if len(lora_prompts) > 0:
            # loop through each lora prompt
            for lora_prompt in lora_prompts:
                # get the lora filename
                lora_filename = lora_prompt[0]
                # check for file extension in filename
                if '.safetensors' not in lora_filename:
                    lora_filename += '.safetensors'
                # get the lora multiplier
                lora_multiplier = float(lora_prompt[1]) if lora_prompt[1] != '' else 1.0
                print('Loading LoRA: ' + lora_filename + ' with multiplier: ' + str(lora_multiplier))
                model, clip_lora = LoraLoader.load_lora(self, model, clip, lora_filename, lora_multiplier, lora_multiplier)
        # strip lora syntax from text
        stripped_text = re.sub(lora_re, '', stripped_text)
        return (model, clip_lora, stripped_text,  text, )

class WildcardAndLoraSyntaxProcessor:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "model": ("MODEL",),
                    "clip": ("CLIP",),
                    "text": ("STRING", {"multiline": True, "default": "<lora:filename:weight>"}),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    }
                }

    RETURN_TYPES = ('MODEL','CLIP','STRING','STRING')
    RETURN_NAMES = ('model','clip','text','unprocessed_text')
    FUNCTION = 'process'
    CATEGORY = 'Mikey/Lora'

    def extract_and_load_loras(self, text, model, clip):
        # load loras detected in the prompt text
        # The text for adding LoRA to the prompt, <lora:filename:multiplier>, is only used to enable LoRA, and is erased from prompt afterwards
        # The multiplier is optional, and defaults to 1.0
        # We update the model and clip, and return the new model and clip with the lora prompt stripped from the text
        # If multiple lora prompts are detected we chain them together like: original clip > clip_with_lora1 > clip_with_lora2 > clip_with_lora3 > etc
        lora_re = r'<lora:(.*?)(?::(.*?))?>'
        # find all lora prompts
        lora_prompts = re.findall(lora_re, text)
        stripped_text = text
        # if we found any lora prompts
        if len(lora_prompts) > 0:
            # loop through each lora prompt
            for lora_prompt in lora_prompts:
                # get the lora filename
                lora_filename = lora_prompt[0]
                # check for file extension in filename
                if '.safetensors' not in lora_filename:
                    lora_filename += '.safetensors'
                # get the lora multiplier
                lora_multiplier = float(lora_prompt[1]) if lora_prompt[1] != '' else 1.0
                print('Loading LoRA: ' + lora_filename + ' with multiplier: ' + str(lora_multiplier))
                # apply the lora to the clip using the LoraLoader.load_lora function
                # def load_lora(self, model, clip, lora_name, strength_model, strength_clip):
                # ...
                # return (model_lora, clip_lora)
                # apply the lora to the clip
                model, clip_lora = LoraLoader.load_lora(self, model, clip, lora_filename, lora_multiplier, lora_multiplier)
        # strip lora syntax from text
        stripped_text = re.sub(lora_re, '', stripped_text)
        return model, clip, stripped_text

    def process(self, model, clip, text, seed):
        # first process wildcards
        text_ = find_and_replace_wildcards(text, seed, True)
        if len(text_) != len(text):
            seed = random.randint(0, 1000000)
        else:
            seed = 0
        # process random syntax
        text_ = process_random_syntax(text_, seed)
        # extract and load loras
        model, clip, stripped_text = self.extract_and_load_loras(text_, model, clip)
        # process wildcards again
        stripped_text = find_and_replace_wildcards(stripped_text, seed, True)
        return (model, clip, stripped_text, text_, )

class StyleConditioner:
    @classmethod
    def INPUT_TYPES(s):
        s.styles, s.pos_style, s.neg_style = read_styles()
        return {"required": {"style": (s.styles,),"strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1}),
                             "positive_cond_base": ("CONDITIONING",), "negative_cond_base": ("CONDITIONING",),
                             "positive_cond_refiner": ("CONDITIONING",), "negative_cond_refiner": ("CONDITIONING",),
                             "base_clip": ("CLIP",), "refiner_clip": ("CLIP",),
                             }
        }

    RETURN_TYPES = ('CONDITIONING','CONDITIONING','CONDITIONING','CONDITIONING',)
    RETURN_NAMES = ('base_pos_cond','base_neg_cond','refiner_pos_cond','refiner_neg_cond',)
    FUNCTION = 'add_style'
    CATEGORY = 'Mikey/Conditioning'

    def add_style(self, style, strength, positive_cond_base, negative_cond_base, positive_cond_refiner, negative_cond_refiner, base_clip, refiner_clip):
        pos_prompt = self.pos_style[style]
        neg_prompt = self.neg_style[style]
        pos_prompt = pos_prompt.replace('{prompt}', '')
        neg_prompt = neg_prompt.replace('{prompt}', '')
        if style == 'none':
            return (positive_cond_base, negative_cond_base, positive_cond_refiner, negative_cond_refiner,)
        # encode the style prompt
        positive_cond_base_new = CLIPTextEncodeSDXL.encode(self, base_clip, 1024, 1024, 0, 0, 1024, 1024, pos_prompt, pos_prompt)[0]
        negative_cond_base_new = CLIPTextEncodeSDXL.encode(self, base_clip, 1024, 1024, 0, 0, 1024, 1024, neg_prompt, neg_prompt)[0]
        positive_cond_refiner_new = CLIPTextEncodeSDXLRefiner.encode(self, refiner_clip, 6, 4096, 4096, pos_prompt)[0]
        negative_cond_refiner_new = CLIPTextEncodeSDXLRefiner.encode(self, refiner_clip, 2.5, 4096, 4096, neg_prompt)[0]
        # average the style prompt with the existing conditioning
        positive_cond_base = ConditioningAverage.addWeighted(self, positive_cond_base_new, positive_cond_base, strength)[0]
        negative_cond_base = ConditioningAverage.addWeighted(self, negative_cond_base_new, negative_cond_base, strength)[0]
        positive_cond_refiner = ConditioningAverage.addWeighted(self, positive_cond_refiner_new, positive_cond_refiner, strength)[0]
        negative_cond_refiner = ConditioningAverage.addWeighted(self, negative_cond_refiner_new, negative_cond_refiner, strength)[0]

        return (positive_cond_base, negative_cond_base, positive_cond_refiner, negative_cond_refiner,)

def calculate_image_complexity(image):
    pil_image = tensor2pil(image)
    np_image = np.array(pil_image)

    # 1. Convert image to grayscale for edge detection
    gray_pil = ImageOps.grayscale(pil_image)
    gray = np.array(gray_pil)

    # 2. Edge Detection using simple difference method
    # Edge Detection using simple difference method
    diff_x = np.diff(gray, axis=1)
    diff_y = np.diff(gray, axis=0)

    # Ensure same shape
    min_shape = (min(diff_x.shape[0], diff_y.shape[0]),
                min(diff_x.shape[1], diff_y.shape[1]))

    diff_x = diff_x[:min_shape[0], :min_shape[1]]
    diff_y = diff_y[:min_shape[0], :min_shape[1]]

    magnitude = np.sqrt(diff_x**2 + diff_y**2)

    threshold = 30  # threshold value after which we consider a pixel as an edge
    edge_density = np.sum(magnitude > threshold) / magnitude.size

    # 3. Color Variability
    hsv = np_image / 255.0  # Normalize
    hsv = np.dstack((hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]))
    hue_std = np.std(hsv[:, :, 0])
    saturation_std = np.std(hsv[:, :, 1])
    value_std = np.std(hsv[:, :, 2])

    # 4. Entropy
    hist = np.histogram(gray, bins=256, range=(0,256), density=True)[0]
    entropy = -np.sum(hist * np.log2(hist + np.finfo(float).eps))

    # Compute a combined complexity score. Adjust the weights if necessary.
    complexity = edge_density + hue_std + saturation_std + value_std + entropy

    return complexity

class MikeySampler:
    @classmethod
    def INPUT_TYPES(s):

        return {"required": {"base_model": ("MODEL",), "refiner_model": ("MODEL",), "samples": ("LATENT",), "vae": ("VAE",),
                             "positive_cond_base": ("CONDITIONING",), "negative_cond_base": ("CONDITIONING",),
                             "positive_cond_refiner": ("CONDITIONING",), "negative_cond_refiner": ("CONDITIONING",),
                             "model_name": (folder_paths.get_filename_list("upscale_models"), ),
                             "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                             "upscale_by": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                             "hires_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),}}

    RETURN_TYPES = ('LATENT',)
    FUNCTION = 'run'
    CATEGORY = 'Mikey/Sampling'

    def adjust_start_step(self, image_complexity, hires_strength=1.0):
        image_complexity /= 24
        if image_complexity > 1:
            image_complexity = 1
        image_complexity = min([0.55, image_complexity]) * hires_strength
        return min([16, 16 - int(round(image_complexity * 16,0))])

    def run(self, seed, base_model, refiner_model, vae, samples, positive_cond_base, negative_cond_base,
            positive_cond_refiner, negative_cond_refiner, model_name, upscale_by=1.0, hires_strength=1.0,
            upscale_method='normal'):
        image_scaler = ImageScale()
        vaeencoder = VAEEncode()
        vaedecoder = VAEDecode()
        uml = UpscaleModelLoader()
        upscale_model = uml.load_model(model_name)[0]
        iuwm = ImageUpscaleWithModel()
        # common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=1.0,
        # disable_noise=False, start_step=None, last_step=None, force_full_denoise=False)
        # step 1 run base model
        sample1 = common_ksampler(base_model, seed, 25, 6.5, 'dpmpp_2s_ancestral', 'simple', positive_cond_base, negative_cond_base, samples,
                                  start_step=0, last_step=18, force_full_denoise=False)[0]
        # step 2 run refiner model
        sample2 = common_ksampler(refiner_model, seed, 30, 3.5, 'dpmpp_2m', 'simple', positive_cond_refiner, negative_cond_refiner, sample1,
                                  disable_noise=True, start_step=21, force_full_denoise=True)[0]
        # step 3 upscale
        if upscale_by == 0:
            return sample2
        pixels = vaedecoder.decode(vae, sample2)[0]
        org_width, org_height = pixels.shape[2], pixels.shape[1]
        img = iuwm.upscale(upscale_model, image=pixels)[0]
        upscaled_width, upscaled_height = int(org_width * upscale_by // 8 * 8), int(org_height * upscale_by // 8 * 8)
        img = image_scaler.upscale(img, 'nearest-exact', upscaled_width, upscaled_height, 'center')[0]
        if hires_strength == 0:
            return (vaeencoder.encode(vae, img)[0],)
        # Adjust start_step based on complexity
        image_complexity = calculate_image_complexity(img)
        print('Image Complexity:', image_complexity)
        start_step = self.adjust_start_step(image_complexity, hires_strength)
        # encode image
        latent = vaeencoder.encode(vae, img)[0]
        # step 3 run base model
        out = common_ksampler(base_model, seed, 16, 9.5, 'dpmpp_2m_sde', 'karras', positive_cond_base, negative_cond_base, latent,
                                start_step=start_step, force_full_denoise=True)
        return out

class MikeySamplerBaseOnly:
    @classmethod
    def INPUT_TYPES(s):

        return {"required": {"base_model": ("MODEL",), "samples": ("LATENT",),
                             "positive_cond_base": ("CONDITIONING",), "negative_cond_base": ("CONDITIONING",),
                             "vae": ("VAE",),
                             "model_name": (folder_paths.get_filename_list("upscale_models"), ),
                             "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                             "upscale_by": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                             "hires_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                             'smooth_step': ("INT", {"default": 0, "min": -1, "max": 100})}}

    RETURN_TYPES = ('LATENT',)
    FUNCTION = 'run'
    CATEGORY = 'Mikey/Sampling'

    def adjust_start_step(self, image_complexity, hires_strength=1.0):
        image_complexity /= 24
        if image_complexity > 1:
            image_complexity = 1
        image_complexity = min([0.55, image_complexity]) * hires_strength
        return min([31, 31 - int(round(image_complexity * 31,0))])

    def run(self, seed, base_model, vae, samples, positive_cond_base, negative_cond_base,
            model_name, upscale_by=1.0, hires_strength=1.0, upscale_method='normal', smooth_step=0):
        image_scaler = ImageScale()
        vaeencoder = VAEEncode()
        vaedecoder = VAEDecode()
        uml = UpscaleModelLoader()
        upscale_model = uml.load_model(model_name)[0]
        iuwm = ImageUpscaleWithModel()
        # common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=1.0,
        # disable_noise=False, start_step=None, last_step=None, force_full_denoise=False)
        # step 1 run base model low cfg
        sample1 = common_ksampler(base_model, seed, 30, 5, 'dpmpp_3m_sde_gpu', 'exponential', positive_cond_base, negative_cond_base, samples,
                                  start_step=0, last_step=14, force_full_denoise=False)[0]
        # step 2 run base model high cfg
        sample2 = common_ksampler(base_model, seed+1, 31 + smooth_step, 9.5, 'dpmpp_3m_sde_gpu', 'exponential', positive_cond_base, negative_cond_base, sample1,
                                  disable_noise=True, start_step=15, force_full_denoise=True)[0]
        if upscale_by == 0:
            return sample2
        # step 3 upscale
        pixels = vaedecoder.decode(vae, sample2)[0]
        org_width, org_height = pixels.shape[2], pixels.shape[1]
        img = iuwm.upscale(upscale_model, image=pixels)[0]
        upscaled_width, upscaled_height = int(org_width * upscale_by // 8 * 8), int(org_height * upscale_by // 8 * 8)
        img = image_scaler.upscale(img, 'nearest-exact', upscaled_width, upscaled_height, 'center')[0]
        if hires_strength == 0:
            return (vaeencoder.encode(vae, img)[0],)
        # Adjust start_step based on complexity
        image_complexity = calculate_image_complexity(img)
        print('Image Complexity:', image_complexity)
        start_step = self.adjust_start_step(image_complexity, hires_strength)
        # encode image
        latent = vaeencoder.encode(vae, img)[0]
        # step 3 run base model
        out = common_ksampler(base_model, seed, 31, 9.5, 'dpmpp_3m_sde_gpu', 'exponential', positive_cond_base, negative_cond_base, latent,
                                start_step=start_step, force_full_denoise=True)
        return out

def match_histograms(source, reference):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image
    """
    src_img = source.convert('YCbCr')
    ref_img = reference.convert('YCbCr')
    src_y, src_cb, src_cr = src_img.split()
    ref_y, ref_cb, ref_cr = ref_img.split()

    src_values = np.asarray(src_y).flatten()
    ref_values = np.asarray(ref_y).flatten()

    # Compute CDFs
    src_cdf, bin_centers = np.histogram(src_values, bins=256, density=True, range=(0, 256))
    src_cdf = np.cumsum(src_cdf)
    ref_cdf, _ = np.histogram(ref_values, bins=256, density=True, range=(0, 256))
    ref_cdf = np.cumsum(ref_cdf)

    # Create a mapping from source values to reference values
    interp_values = np.interp(src_cdf, ref_cdf, bin_centers[:-1])

    # Map the source image to use the new pixel values
    matched = np.interp(src_values, bin_centers[:-1], interp_values).reshape(src_y.size[::-1])
    matched_img = Image.fromarray(np.uint8(matched))

    # Merge channels back
    matched_img = Image.merge('YCbCr', (matched_img, src_cb, src_cr)).convert('RGB')
    return matched_img


def split_image(img):
    """Generate tiles for a given image."""
    tile_width, tile_height = 1024, 1024
    width, height = img.width, img.height

    # Determine the number of tiles needed
    num_tiles_x = ceil(width / tile_width)
    num_tiles_y = ceil(height / tile_height)

    # If width or height is an exact multiple of the tile size, add an additional tile for overlap
    if width % tile_width == 0:
        num_tiles_x += 1
    if height % tile_height == 0:
        num_tiles_y += 1

    # Calculate the overlap
    overlap_x = (num_tiles_x * tile_width - width) / (num_tiles_x - 1)
    overlap_y = (num_tiles_y * tile_height - height) / (num_tiles_y - 1)
    if overlap_x < 256:
        num_tiles_x += 1
        overlap_x = (num_tiles_x * tile_width - width) / (num_tiles_x - 1)
    if overlap_y < 256:
        num_tiles_y += 1
        overlap_y = (num_tiles_y * tile_height - height) / (num_tiles_y - 1)

    tiles = []

    for i in range(num_tiles_y):
        for j in range(num_tiles_x):
            x_start = j * tile_width - j * overlap_x
            y_start = i * tile_height - i * overlap_y

            # Correct for potential float precision issues
            x_start = round(x_start)
            y_start = round(y_start)

            # Crop the tile from the image
            tile_img = img.crop((x_start, y_start, x_start + tile_width, y_start + tile_height))
            tiles.append(((x_start, y_start, x_start + tile_width, y_start + tile_height), tile_img))

    return tiles

def stitch_images(upscaled_size, tiles):
    """Stitch tiles together to create the final upscaled image with overlaps."""
    width, height = upscaled_size
    result = torch.zeros((3, height, width))

    # We assume tiles come in the format [(coordinates, tile), ...]
    sorted_tiles = sorted(tiles, key=lambda x: (x[0][1], x[0][0]))  # Sort by upper then left

    # Variables to keep track of the current row's starting point
    current_row_upper = None

    for (left, upper, right, lower), tile in sorted_tiles:

        # Check if we're starting a new row
        if current_row_upper != upper:
            current_row_upper = upper
            first_tile_in_row = True
        else:
            first_tile_in_row = False

        tile_width = right - left
        tile_height = lower - upper
        feather = tile_width // 8  # Assuming feather size is consistent with the example

        mask = torch.ones(tile.shape[0], tile.shape[1], tile.shape[2])

        if not first_tile_in_row:  # Left feathering for tiles other than the first in the row
            for t in range(feather):
                mask[:, :, t:t+1] *= (1.0 / feather) * (t + 1)

        if upper != 0:  # Top feathering for all tiles except the first row
            for t in range(feather):
                mask[:, t:t+1, :] *= (1.0 / feather) * (t + 1)

        # Apply the feathering mask
        tile = tile.squeeze(0).squeeze(0)  # Removes first two dimensions
        tile_to_add = tile.permute(2, 0, 1)
        # Use the mask to correctly feather the new tile on top of the existing image
        combined_area = tile_to_add * mask.unsqueeze(0) + result[:, upper:lower, left:right] * (1.0 - mask.unsqueeze(0))
        result[:, upper:lower, left:right] = combined_area

    # Expand dimensions to get (1, 3, height, width)
    tensor_expanded = result.unsqueeze(0)

    # Permute dimensions to get (1, height, width, 3)
    tensor_final = tensor_expanded.permute(0, 2, 3, 1)
    return tensor_final

def ai_upscale(tile, base_model, vae, seed, positive_cond_base, negative_cond_base, start_step=11, use_complexity_score='true'):
    """Upscale a tile using the AI model."""
    vaedecoder = VAEDecode()
    vaeencoder = VAEEncode()
    tile = pil2tensor(tile)
    complexity = calculate_image_complexity(tile)
    print('Tile Complexity:', complexity)
    if use_complexity_score == 'true':
        if complexity < 8:
            start_step = 15
        if complexity < 6.5:
            start_step = 18
    encoded_tile = vaeencoder.encode(vae, tile)[0]
    tile = common_ksampler(base_model, seed, 20, 7, 'dpmpp_3m_sde_gpu', 'exponential',
                           positive_cond_base, negative_cond_base, encoded_tile,
                           start_step=start_step, force_full_denoise=True)[0]
    tile = vaedecoder.decode(vae, tile)[0]
    return tile

def run_tiler(enlarged_img, base_model, vae, seed, positive_cond_base, negative_cond_base, denoise=0.25, use_complexity_score='true'):
    # Split the enlarged image into overlapping tiles
    tiles = split_image(enlarged_img)

    # Resample each tile using the AI model
    start_step = int(20 - (20 * denoise))
    resampled_tiles = [(coords, ai_upscale(tile, base_model, vae, seed, positive_cond_base, negative_cond_base, start_step, use_complexity_score)) for coords, tile in tiles]

    # Stitch the tiles to get the final upscaled image
    result = stitch_images(enlarged_img.size, resampled_tiles)

    return result

class MikeySamplerTiled:
    @classmethod
    def INPUT_TYPES(s):

        return {"required": {"base_model": ("MODEL",), "refiner_model": ("MODEL",), "samples": ("LATENT",), "vae": ("VAE",),
                             "positive_cond_base": ("CONDITIONING",), "negative_cond_base": ("CONDITIONING",),
                             "positive_cond_refiner": ("CONDITIONING",), "negative_cond_refiner": ("CONDITIONING",),
                             "model_name": (folder_paths.get_filename_list("upscale_models"), ),
                             "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                             "upscale_by": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                             "tiler_denoise": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.05}),
                             "tiler_model": (["base", "refiner"], {"default": "base"}),}}

    RETURN_TYPES = ('IMAGE', 'IMAGE',)
    RETURN_NAMES = ('tiled_image', 'upscaled_image',)
    FUNCTION = 'run'
    CATEGORY = 'Mikey/Sampling'

    def phase_one(self, base_model, refiner_model, samples, positive_cond_base, negative_cond_base,
                  positive_cond_refiner, negative_cond_refiner, upscale_by, model_name, seed, vae):
        image_scaler = ImageScale()
        vaedecoder = VAEDecode()
        uml = UpscaleModelLoader()
        upscale_model = uml.load_model(model_name)[0]
        iuwm = ImageUpscaleWithModel()
        # step 1 run base model
        sample1 = common_ksampler(base_model, seed, 30, 6.5, 'dpmpp_3m_sde_gpu', 'exponential', positive_cond_base, negative_cond_base, samples,
                                  start_step=0, last_step=14, force_full_denoise=False)[0]
        # step 2 run refiner model
        sample2 = common_ksampler(refiner_model, seed, 32, 3.5, 'dpmpp_3m_sde_gpu', 'exponential', positive_cond_refiner, negative_cond_refiner, sample1,
                                  disable_noise=True, start_step=15, force_full_denoise=True)[0]
        # step 3 upscale image using a simple AI image upscaler
        pixels = vaedecoder.decode(vae, sample2)[0]
        org_width, org_height = pixels.shape[2], pixels.shape[1]
        img = iuwm.upscale(upscale_model, image=pixels)[0]
        upscaled_width, upscaled_height = int(org_width * upscale_by // 8 * 8), int(org_height * upscale_by // 8 * 8)
        img = image_scaler.upscale(img, 'nearest-exact', upscaled_width, upscaled_height, 'center')[0]
        return img, upscaled_width, upscaled_height

    def run(self, seed, base_model, refiner_model, vae, samples, positive_cond_base, negative_cond_base,
            positive_cond_refiner, negative_cond_refiner, model_name, upscale_by=1.0, tiler_denoise=0.25,
            upscale_method='normal', tiler_model='base'):
        # phase 1: run base, refiner, then upscaler model
        img, upscaled_width, upscaled_height = self.phase_one(base_model, refiner_model, samples, positive_cond_base, negative_cond_base,
                                                              positive_cond_refiner, negative_cond_refiner, upscale_by, model_name, seed, vae)
        # phase 2: run tiler
        img = tensor2pil(img)
        if tiler_model == 'base':
            tiled_image = run_tiler(img, base_model, vae, seed, positive_cond_base, negative_cond_base, tiler_denoise)
        else:
            tiled_image = run_tiler(img, refiner_model, vae, seed, positive_cond_refiner, negative_cond_refiner, tiler_denoise)
        return (tiled_image, img)

class MikeySamplerTiledAdvanced:
    @classmethod
    def INPUT_TYPES(s):

        return {"required": {"base_model": ("MODEL",),
                             "refiner_model": ("MODEL",),
                             "samples": ("LATENT",), "vae": ("VAE",),
                             "positive_cond_base": ("CONDITIONING",), "negative_cond_base": ("CONDITIONING",),
                             "positive_cond_refiner": ("CONDITIONING",), "negative_cond_refiner": ("CONDITIONING",),
                             "model_name": (folder_paths.get_filename_list("upscale_models"), ),
                             "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                             "denoise_image": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
                             "steps": ("INT", {"default": 30, "min": 1, "max": 1000}),
                             "smooth_step": ("INT", {"default": 1, "min": -1, "max": 100}),
                             "cfg": ("FLOAT", {"default": 6.5, "min": 0.0, "max": 1000.0, "step": 0.1}),
                             "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                             "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                             "upscale_by": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                             "tiler_denoise": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.05}),
                             "tiler_model": (["base", "refiner"], {"default": "base"}),
                             "use_complexity_score": (['true','false'], {"default": 'true'}),},
                "optional": {"image_optional": ("IMAGE",),}}

    RETURN_TYPES = ('IMAGE', 'IMAGE',)
    RETURN_NAMES = ('tiled_image', 'upscaled_image',)
    FUNCTION = 'run'
    CATEGORY = 'Mikey/Sampling'

    #def phase_one(self, base_model, refiner_model, samples, positive_cond_base, negative_cond_base,
    #              positive_cond_refiner, negative_cond_refiner, upscale_by, model_name, seed, vae):
    # updated phase_one
    def phase_one(self, base_model, refiner_model, samples, positive_cond_base, negative_cond_base,
                  positive_cond_refiner, negative_cond_refiner, upscale_by, model_name, seed, vae, denoise_image,
                  steps, smooth_step, cfg, sampler_name, scheduler):
        image_scaler = ImageScale()
        vaedecoder = VAEDecode()
        uml = UpscaleModelLoader()
        upscale_model = uml.load_model(model_name)[0]
        iuwm = ImageUpscaleWithModel()
        # step 1 run base model
        start_step = int(steps - (steps * denoise_image))
        if start_step > steps // 2:
            last_step = steps - 1
        else:
            # last step should be 1/2 of steps - 1 step
            if start_step % 2 == 0:
                last_step = steps // 2 - 1
            else:
                last_step = steps // 2
        print(f'base model start_step: {start_step}, last_step: {last_step}')
        sample1 = common_ksampler(base_model, seed, steps, cfg, sampler_name, scheduler, positive_cond_base, negative_cond_base, samples,
                                  start_step=start_step, last_step=last_step, force_full_denoise=False)[0]
        # step 2 run refiner model
        start_step = last_step + 1
        total_steps = steps + smooth_step
        print(f'refiner model start_step: {start_step}, last_step: {total_steps}')
        sample2 = common_ksampler(refiner_model, seed, total_steps, cfg, sampler_name, scheduler, positive_cond_refiner, negative_cond_refiner, sample1,
                                  disable_noise=True, start_step=start_step, force_full_denoise=True)[0]
        # step 3 upscale image using a simple AI image upscaler
        pixels = vaedecoder.decode(vae, sample2)[0]
        org_width, org_height = pixels.shape[2], pixels.shape[1]
        img = iuwm.upscale(upscale_model, image=pixels)[0]
        upscaled_width, upscaled_height = int(org_width * upscale_by // 8 * 8), int(org_height * upscale_by // 8 * 8)
        img = image_scaler.upscale(img, 'nearest-exact', upscaled_width, upscaled_height, 'center')[0]
        return img, upscaled_width, upscaled_height

    #def run(self, seed, base_model, refiner_model, vae, samples, positive_cond_base, negative_cond_base,
    #        positive_cond_refiner, negative_cond_refiner, model_name, upscale_by=1.0, tiler_denoise=0.25,
    #        upscale_method='normal', tiler_model='base'):
    # updated run
    def run(self, seed, base_model, refiner_model, vae, samples, positive_cond_base, negative_cond_base,
            positive_cond_refiner, negative_cond_refiner, model_name, upscale_by=1.0, tiler_denoise=0.25,
            upscale_method='normal', tiler_model='base', denoise_image=0.25, steps=30, smooth_step=0, cfg=6.5,
            sampler_name='dpmpp_3m_sde_gpu', scheduler='exponential', use_complexity_score='true', image_optional=None):
        # if image not none replace samples with decoded image
        if image_optional is not None:
            vaeencoder = VAEEncode()
            samples = vaeencoder.encode(vae, image_optional)[0]
        # phase 1: run base, refiner, then upscaler model
        img, upscaled_width, upscaled_height = self.phase_one(base_model, refiner_model, samples, positive_cond_base, negative_cond_base,
                                                              positive_cond_refiner, negative_cond_refiner, upscale_by, model_name, seed, vae, denoise_image,
                                                              steps, smooth_step, cfg, sampler_name, scheduler)
        # phase 2: run tiler
        img = tensor2pil(img)
        if tiler_model == 'base':
            tiled_image = run_tiler(img, base_model, vae, seed, positive_cond_base, negative_cond_base, tiler_denoise, use_complexity_score)
        else:
            tiled_image = run_tiler(img, refiner_model, vae, seed, positive_cond_refiner, negative_cond_refiner, tiler_denoise, use_complexity_score)
        return (tiled_image, img)

class MikeySamplerTiledBaseOnly(MikeySamplerTiled):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"base_model": ("MODEL",), "samples": ("LATENT",),
                             "positive_cond_base": ("CONDITIONING",), "negative_cond_base": ("CONDITIONING",),
                             "vae": ("VAE",),
                             "model_name": (folder_paths.get_filename_list("upscale_models"), ),
                             "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                             "upscale_by": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                             "tiler_denoise": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.05}),}}

    RETURN_TYPES = ('IMAGE',)
    RETURN_NAMES = ('image',)

    def phase_one(self, base_model, samples, positive_cond_base, negative_cond_base,
                  upscale_by, model_name, seed, vae):
        image_scaler = ImageScale()
        vaedecoder = VAEDecode()
        uml = UpscaleModelLoader()
        upscale_model = uml.load_model(model_name)[0]
        iuwm = ImageUpscaleWithModel()
        # step 1 run base model low cfg
        sample1 = common_ksampler(base_model, seed, 30, 5, 'dpmpp_3m_sde_gpu', 'exponential', positive_cond_base, negative_cond_base, samples,
                                  start_step=0, last_step=14, force_full_denoise=False)[0]
        # step 2 run base model high cfg
        sample2 = common_ksampler(base_model, seed+1, 32, 9.5, 'dpmpp_3m_sde_gpu', 'exponential', positive_cond_base, negative_cond_base, sample1,
                                  disable_noise=True, start_step=15, force_full_denoise=True)[0]
        # step 3 upscale image using a simple AI image upscaler
        pixels = vaedecoder.decode(vae, sample2)[0]
        org_width, org_height = pixels.shape[2], pixels.shape[1]
        img = iuwm.upscale(upscale_model, image=pixels)[0]
        upscaled_width, upscaled_height = int(org_width * upscale_by // 8 * 8), int(org_height * upscale_by // 8 * 8)
        img = image_scaler.upscale(img, 'nearest-exact', upscaled_width, upscaled_height, 'center')[0]
        return img, upscaled_width, upscaled_height

    def adjust_start_step(self, image_complexity, hires_strength=1.0):
        image_complexity /= 24
        if image_complexity > 1:
            image_complexity = 1
        image_complexity = min([0.55, image_complexity]) * hires_strength
        return min([32, 32 - int(round(image_complexity * 32,0))])

    def run(self, seed, base_model, vae, samples, positive_cond_base, negative_cond_base,
            model_name, upscale_by=1.0, tiler_denoise=0.25,
            upscale_method='normal'):
        # phase 1: run base, refiner, then upscaler model
        img, upscaled_width, upscaled_height = self.phase_one(base_model, samples, positive_cond_base, negative_cond_base,
                                                              upscale_by, model_name, seed, vae)
        print('img shape: ', img.shape)
        # phase 2: run tiler
        img = tensor2pil(img)
        tiled_image = run_tiler(img, base_model, vae, seed, positive_cond_base, negative_cond_base, tiler_denoise)
        #final_image = pil2tensor(tiled_image)
        return (tiled_image,)

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
        positive_prompt = find_and_replace_wildcards(positive_prompt, seed)
        negative_prompt = find_and_replace_wildcards(negative_prompt, seed)
        width = self.ratio_dict[ratio_selected]["width"]
        height = self.ratio_dict[ratio_selected]["height"]
        latent = torch.zeros([batch_size, 4, height // 8, width // 8])
        refiner_width = width * 4
        refiner_height = height * 4
        return ({"samples":latent},
                str(positive_prompt),
                str(negative_prompt),
                str(positive_style),
                str(negative_style),
                width,
                height,
                refiner_width,
                refiner_height,)

class UpscaleTileCalculator:
    @classmethod
    def INPUT_TYPES(s):
        return {'required': {'image': ('IMAGE',),
                            # 'upscale_by': ('FLOAT', {'default': 1.0, 'min': 0.1, 'max': 10.0, 'step': 0.1}),
                             'tile_resolution': ('INT', {'default': 512, 'min': 1, 'max': 8192, 'step': 8})}}

    RETURN_TYPES = ('IMAGE', 'INT', 'INT')
    RETURN_NAMES = ('image', 'tile_width', 'tile_height')
    FUNCTION = 'calculate'
    CATEGORY = 'Mikey/Image'

    def upscale(self, image, upscale_method, width, height, crop):
        samples = image.movedim(-1,1)
        s = comfy.utils.common_upscale(samples, width, height, upscale_method, crop)
        s = s.movedim(1,-1)
        return (s,)

    def resize(self, image, width, height, upscale_method, crop):
        w, h = find_latent_size(image.shape[2], image.shape[1])
        print('Resizing image from {}x{} to {}x{}'.format(image.shape[2], image.shape[1], w, h))
        img = self.upscale(image, upscale_method, w, h, crop)[0]
        return (img, )

    def calculate(self, image, tile_resolution):
        width, height = image.shape[2], image.shape[1]
        tile_width, tile_height = find_tile_dimensions(width, height, 1.0, tile_resolution)
        print('Tile width: ' + str(tile_width), 'Tile height: ' + str(tile_height))
        return (image, tile_width, tile_height)

class IntegerAndString:
    @classmethod
    def INPUT_TYPES(s):
        return {'required': {'seed': ('INT', {'default': 0, 'min': 0, 'max': 0xffffffffffffffff})}}

    RETURN_TYPES = ('INT','STRING')
    RETURN_NAMES = ('seed','seed_string')
    FUNCTION = 'output'
    CATEGORY = 'Mikey/Utils'

    def output(self, seed):
        seed_string = str(seed)
        return (seed, seed_string,)

class ImageCaption:

    @classmethod
    def INPUT_TYPES(cls):
        # check if path exists
        if os.path.exists(os.path.join(folder_paths.base_path, 'fonts')):
            cls.font_dir = os.path.join(folder_paths.base_path, 'fonts')
            cls.font_files = [os.path.join(cls.font_dir, f) for f in os.listdir(cls.font_dir) if os.path.isfile(os.path.join(cls.font_dir, f))]
            cls.font_file_names = [os.path.basename(f) for f in cls.font_files]
            return {'required': {'image': ('IMAGE',),
                        'font': (cls.font_file_names, {'default': cls.font_file_names[0]}),
                        'caption': ('STRING', {'multiline': True, 'default': 'Caption'})}}
        else:
            cls.font_dir = None
            cls.font_files = None
            cls.font_file_names = None
            return {'required': {'image': ('IMAGE',),
                    'font': ('STRING', {'default': 'Path to font file'}),
                    'caption': ('STRING', {'multiline': True, 'default': 'Caption'})}}


    RETURN_TYPES = ('IMAGE',)
    RETURN_NAMES = ('image',)
    FUNCTION = 'caption'
    CATEGORY = 'Mikey/Image'

    def wrap_text(self, text, font, max_width):
        """Wrap text to fit inside a specified width when rendered."""
        wrapped_lines = []
        for line in text.split('\n'):
            # Split lines by spaces to avoid breaking words
            words = line.split(' ')
            new_line = words[0]
            for word in words[1:]:
                # If line can fit the word, add it
                if font.getsize(new_line + ' ' + word)[0] <= max_width:
                    new_line += ' ' + word
                else:
                    wrapped_lines.append(new_line)
                    new_line = word
            wrapped_lines.append(new_line)
        return wrapped_lines

    def caption(self, image, font, caption):
        # Convert tensor to PIL image
        orig_image = tensor2pil(image)
        width, height = orig_image.size

        # Set up the font
        if self.font_dir is None:
            font_file = font
            if not os.path.isfile(font_file):
                raise Exception('Font file does not exist: ' + font_file)
        else:
            font_file = os.path.join(self.font_dir, font)
        font = ImageFont.truetype(font_file, 32)

        # Wrap the text
        max_width = width
        wrapped_lines = self.wrap_text(caption, font, max_width)

        # Calculate height needed for wrapped text
        wrapped_text_height = len(wrapped_lines) * font.getsize('A')[1]  # Estimate using height of letter 'A'
        caption_height = wrapped_text_height + 25  # A little buffer for better visual appeal

        # Create the caption bar
        text_image = Image.new('RGB', (width, caption_height), (0, 0, 0))
        draw = ImageDraw.Draw(text_image)

        y_position = (caption_height - wrapped_text_height) // 2
        for line in wrapped_lines:
            text_width, text_height = font.getsize(line)
            x_position = (width - text_width) // 2
            draw.text((x_position, y_position), line, (255, 255, 255), font=font)
            y_position += text_height

        # Combine the images
        combined_image = Image.new('RGB', (width, height + caption_height), (0, 0, 0))
        combined_image.paste(text_image, (0, height))
        combined_image.paste(orig_image, (0, 0))

        return (pil2tensor(combined_image),)

def tensor2pil_alpha(tensor):
    # convert a PyTorch tensor to a PIL Image object
    # assumes tensor is a 4D tensor with shape (batch_size, channels, height, width)
    # returns a PIL Image object with mode 'RGBA'
    tensor = tensor.squeeze(0)  # remove batch dimension
    tensor = tensor.permute(1, 2, 0)
    if tensor.shape[2] == 1:
        tensor = torch.cat([tensor, tensor, tensor], dim=2)
    elif tensor.shape[2] == 3:
        tensor = torch.cat([tensor, torch.ones_like(tensor[:, :, :1])], dim=2)
    tensor = tensor.mul(255).clamp(0, 255).byte()
    pil_image = Image.fromarray(tensor.numpy(), mode='RGBA')
    return pil_image

def checkerboard_border(image, border_width, border_color):
    # create a checkerboard pattern with fixed size
    pattern_size = (image.shape[2] + border_width * 2, image.shape[1] + border_width * 2)
    checkerboard = Image.new('RGB', pattern_size, border_color)
    for i in range(0, pattern_size[0], border_width):
        for j in range(0, pattern_size[1], border_width):
            box = (i, j, i + border_width, j + border_width)
            if (i // border_width + j // border_width) % 2 == 0:
                checkerboard.paste(Image.new('RGB', (border_width, border_width), 'white'), box)
            else:
                checkerboard.paste(Image.new('RGB', (border_width, border_width), 'black'), box)

    # resize the input image to fit inside the checkerboard pattern
    orig_image = tensor2pil(image)

    # paste the input image onto the checkerboard pattern
    checkerboard.paste(orig_image, (border_width, border_width))

    return pil2tensor(checkerboard)[None, :, :, :]

class ImageBorder:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'image': ('IMAGE',),
                    'border_width': ('INT', {'default': 10, 'min': 0, 'max': 1000}),
                    'border_color': ('STRING', {'default': 'black'})}}

    RETURN_TYPES = ('IMAGE',)
    RETURN_NAMES = ('image',)
    FUNCTION = 'border'
    CATEGORY = 'Mikey/Image'

    def border(self, image, border_width, border_color):
        # Convert tensor to PIL image
        orig_image = tensor2pil(image)
        width, height = orig_image.size
        # Create the border
        if border_color == 'checkerboard':
            return checkerboard_border(image, border_width, 'black')
        # check for string containing a tuple
        if border_color.startswith('(') and border_color.endswith(')'):
            border_color = border_color[1:-1]
            border_color = tuple(map(int, border_color.split(',')))
        border_image = Image.new('RGB', (width + border_width * 2, height + border_width * 2), border_color)
        border_image.paste(orig_image, (border_width, border_width))

        return (pil2tensor(border_image),)

class TextCombinations2:
    texts = ['text1', 'text2', 'text1 + text2']
    outputs = ['output1','output2']

    @classmethod
    def generate_combinations(cls, texts, outputs):
        operations = []
        for output1, output2 in product(texts, repeat=len(outputs)):
            operation = f"{output1} to {outputs[0]}, {output2} to {outputs[1]}"
            operations.append(operation)
        return operations

    @classmethod
    def INPUT_TYPES(cls):
        cls.operations = cls.generate_combinations(cls.texts, cls.outputs)
        return {'required': {'text1': ('STRING', {'multiline': True, 'default': 'Text 1'}),
                             'text2': ('STRING', {'multiline': True, 'default': 'Text 2'}),
                             'operation': (cls.operations, {'default':cls.operations[0]}),
                             'delimiter': ('STRING', {'default': ' '}),
                             'use_seed': (['true','false'], {'default': 'false'}),
                             'seed': ('INT', {'default': 0, 'min': 0, 'max': 0xffffffffffffffff})}}

    RETURN_TYPES = ('STRING','STRING')
    RETURN_NAMES = ('output1','output2')
    FUNCTION = 'mix'
    CATEGORY = 'Mikey/Text'

    def mix(self, text1, text2, operation, delimiter, use_seed, seed):
        text_dict = {'text1': text1, 'text2': text2}
        if use_seed == 'true' and len(self.operations) > 0:
            offset = seed % len(self.operations)
            operation = self.operations[offset]

        # Parsing the operation string
        ops = operation.split(", ")
        output_texts = [op.split(" to ")[0] for op in ops]

        # Generate the outputs
        outputs = []

        for output_text in output_texts:
            # Split the string by '+' to identify individual text components
            components = output_text.split(" + ")

            # Generate the final string for each output
            final_output = delimiter.join(eval(comp, {}, text_dict) for comp in components)

            outputs.append(final_output)

        return tuple(outputs)

class TextCombinations3:
    texts = ['text1', 'text2', 'text3', 'text1 + text2', 'text1 + text3', 'text2 + text3', 'text1 + text2 + text3']
    outputs = ['output1','output2','output3']

    @classmethod
    def generate_combinations(cls, texts, outputs):
        operations = []
        for output1, output2, output3 in product(texts, repeat=len(outputs)):
            operation = f"{output1} to {outputs[0]}, {output2} to {outputs[1]}, {output3} to {outputs[2]}"
            operations.append(operation)
        return operations

    @classmethod
    def INPUT_TYPES(cls):
        cls.operations = cls.generate_combinations(cls.texts, cls.outputs)
        return {'required': {'text1': ('STRING', {'multiline': True, 'default': 'Text 1'}),
                             'text2': ('STRING', {'multiline': True, 'default': 'Text 2'}),
                             'text3': ('STRING', {'multiline': True, 'default': 'Text 3'}),
                             'operation': (cls.operations, {'default':cls.operations[0]}),
                             'delimiter': ('STRING', {'default': ' '}),
                             'use_seed': (['true','false'], {'default': 'false'}),
                             'seed': ('INT', {'default': 0, 'min': 0, 'max': 0xffffffffffffffff})}}

    RETURN_TYPES = ('STRING','STRING','STRING')
    RETURN_NAMES = ('output1','output2','output3')
    FUNCTION = 'mix'
    CATEGORY = 'Mikey/Text'

    def mix(self, text1, text2, text3, operation, delimiter, use_seed, seed):
        text_dict = {'text1': text1, 'text2': text2, 'text3': text3}
        if use_seed == 'true' and len(self.operations) > 0:
            offset = seed % len(self.operations)
            operation = self.operations[offset]

        # Parsing the operation string
        ops = operation.split(", ")
        output_texts = [op.split(" to ")[0] for op in ops]

        # Generate the outputs
        outputs = []

        for output_text in output_texts:
            # Split the string by '+' to identify individual text components
            components = output_text.split(" + ")

            # Generate the final string for each output
            final_output = delimiter.join(eval(comp, {}, text_dict) for comp in components)

            outputs.append(final_output)

        return tuple(outputs)

class Text2InputOr3rdOption:
    @classmethod
    def INPUT_TYPES(s):
        return {'required': {'text_a': ('STRING', {'multiline': True, 'default': 'Text A'}),
                             'text_b': ('STRING', {'multiline': True, 'default': 'Text B'}),
                             'text_c': ('STRING', {'multiline': True, 'default': 'Text C'}),
                             'use_text_c_for_both': (['true','false'], {'default': 'false'}),}}

    RETURN_TYPES = ('STRING','STRING',)
    RETURN_NAMES = ('text_a','text_b',)
    FUNCTION = 'output'
    CATEGORY = 'Mikey/Text'

    def output(self, text_a, text_b, text_c, use_text_c_for_both):
        if use_text_c_for_both == 'true':
            return (text_c, text_c)
        else:
            return (text_a, text_b)

class FreeMemory:
    @classmethod
    def INPUT_TYPES(s):
        return {'required': {'image': ('IMAGE',),}}

    RETURN_TYPES = ('IMAGE',)
    RETURN_NAMES = ('image',)
    FUNCTION = 'cleanup'
    CATEGORY = 'Mikey/Utils'

    def cleanup(self, image):
        global current_loaded_models
        to_unload = []
        for i in range(len(current_loaded_models)):
            to_unload = [i] + to_unload
        for i in to_unload:
            print("unload model", i)
            m = current_loaded_models.pop(i)
            m.model.model.cpu()
            m.model.model = None
            m.model = None
            m = None
            del m
            gc.collect()
            torch.cuda.empty_cache()
        current_loaded_models = []
        soft_empty_cache()
        gc.collect()
        return (image,)

NODE_CLASS_MAPPINGS = {
    'Wildcard Processor': WildcardProcessor,
    'Empty Latent Ratio Select SDXL': EmptyLatentRatioSelector,
    'Empty Latent Ratio Custom SDXL': EmptyLatentRatioCustom,
    'PresetRatioSelector': PresetRatioSelector,
    'Ratio Advanced': RatioAdvanced,
    'Int to String': INTtoSTRING,
    'Float to String': FLOATtoSTRING,
    'Save Image With Prompt Data': SaveImagesMikey,
    'Save Images Mikey': SaveImagesMikeyML,
    'Save Images No Display': SaveImageNoDisplay,
    'Resize Image for SDXL': ResizeImageSDXL,
    'Upscale Tile Calculator': UpscaleTileCalculator,
    'Batch Resize Image for SDXL': BatchResizeImageSDXL,
    'Batch Crop Image': BatchCropImage,
    'Batch Crop Resize Inplace': BatchCropResizeInplace,
    'Prompt With Style': PromptWithStyle,
    'Prompt With Style V2': PromptWithStyleV2,
    'Prompt With Style V3': PromptWithStyleV3,
    'LoraSyntaxProcessor': LoraSyntaxProcessor,
    'WildcardAndLoraSyntaxProcessor': WildcardAndLoraSyntaxProcessor,
    'Prompt With SDXL': PromptWithSDXL,
    'Style Conditioner': StyleConditioner,
    'Mikey Sampler': MikeySampler,
    'MikeySamplerTiledAdvanced': MikeySamplerTiledAdvanced,
    'Mikey Sampler Base Only': MikeySamplerBaseOnly,
    'Mikey Sampler Tiled': MikeySamplerTiled,
    'Mikey Sampler Tiled Base Only': MikeySamplerTiledBaseOnly,
    'AddMetaData': AddMetaData,
    'SaveMetaData': SaveMetaData,
    'HaldCLUT ': HaldCLUT,
    'Seed String': IntegerAndString,
    'Image Caption': ImageCaption,
    'ImageBorder': ImageBorder,
    'TextCombinations': TextCombinations2,
    'TextCombinations3': TextCombinations3,
    'Text2InputOr3rdOption': Text2InputOr3rdOption,
    'FreeMemory': FreeMemory,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    'Wildcard Processor': 'Wildcard Processor (Mikey)',
    'Empty Latent Ratio Select SDXL': 'Empty Latent Ratio Select SDXL (Mikey)',
    'Empty Latent Ratio Custom SDXL': 'Empty Latent Ratio Custom SDXL (Mikey)',
    'PresetRatioSelector': 'Preset Ratio Selector (Mikey)',
    'Ratio Advanced': 'Ratio Advanced (Mikey)',
    'Int to String': 'Int to String (Mikey)',
    'Float to String': 'Float to String (Mikey)',
    'Save Images With Prompt Data': 'Save Image With Prompt Data (Mikey)',
    'Save Images Mikey': 'Save Images Mikey (Mikey)',
    'Save Images No Display': 'Save Images No Display (Mikey)',
    'Resize Image for SDXL': 'Resize Image for SDXL (Mikey)',
    'Batch Crop Image': 'Batch Crop Image (Mikey)',
    'Upscale Tile Calculator': 'Upscale Tile Calculator (Mikey)',
    'Batch Resize Image for SDXL': 'Batch Resize Image for SDXL (Mikey)',
    'Batch Crop Resize Inplace': 'Batch Crop Resize Inplace (Mikey)',
    'Prompt With Style V3': 'Prompt With Style (Mikey)',
    'LoraSyntaxProcessor': 'Lora Syntax Processor (Mikey)',
    'WildcardAndLoraSyntaxProcessor': 'Wildcard And Lora Syntax Processor (Mikey)',
    'Prompt With Style': 'Prompt With Style V1 (Mikey)',
    'Prompt With Style V2': 'Prompt With Style V2 (Mikey)',
    'Prompt With SDXL': 'Prompt With SDXL (Mikey)',
    'Style Conditioner': 'Style Conditioner (Mikey)',
    'Mikey Sampler': 'Mikey Sampler',
    'Mikey Sampler Base Only': 'Mikey Sampler Base Only',
    'Mikey Sampler Tiled': 'Mikey Sampler Tiled',
    'MikeySamplerTiledAdvanced': 'Mikey Sampler Tiled Advanced',
    'Mikey Sampler Tiled Base Only': 'Mikey Sampler Tiled Base Only',
    'AddMetaData': 'AddMetaData (Mikey)',
    'SaveMetaData': 'SaveMetaData (Mikey)',
    'HaldCLUT': 'HaldCLUT (Mikey)',
    'Seed String': 'Seed String (Mikey)',
    'Image Caption': 'Image Caption (Mikey)',
    'ImageBorder': 'Image Border (Mikey)',
    'TextCombinations': 'Text Combinations 2 (Mikey)',
    'TextCombinations3': 'Text Combinations 3 (Mikey)',
    'Text2InputOr3rdOption': 'Text 2 Inputs Or 3rd Option Instead (Mikey)',
    'FreeMemory': 'Free CPU Memory (Mikey)'
}
