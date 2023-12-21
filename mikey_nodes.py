import datetime
from fractions import Fraction
import gc
import hashlib
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

import html
import requests
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
from comfy.model_management import soft_empty_cache, free_memory, get_torch_device, current_loaded_models, load_model_gpu
from nodes import LoraLoader, ConditioningAverage, common_ksampler, ImageScale, ImageScaleBy, VAEEncode, VAEDecode
import comfy.utils
from comfy_extras.chainner_models import model_loading
from comfy import model_management, model_base

def calculate_file_hash(file_path):
    # open the file in binary mode
    with open(file_path, 'rb') as f:
        # read the file in chunks to avoid loading the whole file into memory
        chunk_size = 4096
        hash_object = hashlib.sha256()
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hash_object.update(chunk)
    # return the hexadecimal representation of the hash
    return hash_object.hexdigest()

def get_cached_file_hashes():
    # load the cached file hashes from the JSON file
    cache_file_path = os.path.join(folder_paths.base_path, 'file_hashes.json')
    if os.path.exists(cache_file_path):
        with open(cache_file_path, 'r') as f:
            return json.load(f)
    else:
        return {}

def get_file_hash(file_path):
    # check if the file hash is already cached
    cached_file_hashes = get_cached_file_hashes()
    file_name = os.path.basename(file_path)
    if file_name in cached_file_hashes:
        return cached_file_hashes[file_name]
    else:
        # calculate the file hash and cache it
        file_hash = calculate_file_hash(file_path)[:10]
        cache_file_hash(file_path, file_hash)
        return file_hash

def cache_file_hash(file_path, file_hash):
    # update the cached file hashes dictionary and save to the JSON file
    cache_file_path = os.path.join(folder_paths.base_path, 'file_hashes.json')
    cached_file_hashes = get_cached_file_hashes()
    cached_file_hashes[os.path.basename(file_path)] = file_hash
    with open(cache_file_path, 'w') as f:
        json.dump(cached_file_hashes, f)

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

def search_and_replace(text, extra_pnginfo, prompt):
    if extra_pnginfo is None or prompt is None:
        return text
    # if %date: in text, then replace with date
    #print(text)
    if '%date:' in text:
        for match in re.finditer(r'%date:(.*?)%', text):
            date_match = match.group(1)
            cursor = 0
            date_pattern = ''
            now = datetime.datetime.now()

            pattern_map = {
                'yyyy': now.strftime('%Y'),
                'yy': now.strftime('%y'),
                'MM': now.strftime('%m'),
                'M': now.strftime('%m').lstrip('0'),
                'dd': now.strftime('%d'),
                'd': now.strftime('%d').lstrip('0'),
                'hh': now.strftime('%H'),
                'h': now.strftime('%H').lstrip('0'),
                'mm': now.strftime('%M'),
                'm': now.strftime('%M').lstrip('0'),
                'ss': now.strftime('%S'),
                's': now.strftime('%S').lstrip('0')
            }

            sorted_keys = sorted(pattern_map.keys(), key=len, reverse=True)

            while cursor < len(date_match):
                replaced = False
                for key in sorted_keys:
                    if date_match.startswith(key, cursor):
                        date_pattern += pattern_map[key]
                        cursor += len(key)
                        replaced = True
                        break
                if not replaced:
                    date_pattern += date_match[cursor]
                    cursor += 1

            text = text.replace('%date:' + match.group(1) + '%', date_pattern)
    # Parse JSON if they are strings
    if isinstance(extra_pnginfo, str):
        extra_pnginfo = json.loads(extra_pnginfo)
    if isinstance(prompt, str):
        prompt = json.loads(prompt)

    # Map from "Node name for S&R" to id in the workflow
    node_to_id_map = {}
    try:
        for node in extra_pnginfo['workflow']['nodes']:
            node_name = node['properties'].get('Node name for S&R')
            node_id = node['id']
            node_to_id_map[node_name] = node_id
    except:
        return text

    # Find all patterns in the text that need to be replaced
    patterns = re.findall(r"%([^%]+)%", text)
    for pattern in patterns:
        # Split the pattern to get the node name and widget name
        node_name, widget_name = pattern.split('.')

        # Find the id for this node name
        node_id = node_to_id_map.get(node_name)
        if node_id is None:
            print(f"No node with name {node_name} found.")
            # check if user entered id instead of node name
            if node_name in node_to_id_map.values():
                node_id = node_name
            else:
                continue

        # Find the value of the specified widget in prompt JSON
        prompt_node = prompt.get(str(node_id))
        if prompt_node is None:
            print(f"No prompt data for node with id {node_id}.")
            continue

        widget_value = prompt_node['inputs'].get(widget_name)
        if widget_value is None:
            print(f"No widget with name {widget_name} found for node {node_name}.")
            continue

        # Replace the pattern in the text
        text = text.replace(f"%{pattern}%", str(widget_value))

    return text

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
    #print('checking for random syntax')
    random.seed(seed)
    random_re = r'<random:(-?\d*\.?\d+):(-?\d*\.?\d+)>'
    matches = re.finditer(random_re, text)

    # Create a list to hold the new segments of text
    new_text_list = []
    last_end = 0

    # Iterate through matches
    for match in matches:
        lower_bound, upper_bound = map(float, match.groups())
        random_value = random.uniform(lower_bound, upper_bound)
        random_value = round(random_value, 4)

        # Append text up to the match and the generated number
        new_text_list.append(text[last_end:match.start()])
        new_text_list.append(str(random_value))

        # Update the index of the last match end
        last_end = match.end()

    # Append remaining text after the last match
    new_text_list.append(text[last_end:])

    # Combine the list into a single string
    new_text = ''.join(new_text_list)

    #print(new_text)
    return new_text

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

# create a wrapper function that can apply a function to multiple images in a batch
# while passing all other arguments to the function
def apply_to_batch(func):
    def wrapper(self, image, *args, **kwargs):
        images = []
        for img in image:
            images.append(func(self, img, *args, **kwargs))
        batch_tensor = torch.cat(images, dim=0)
        return (batch_tensor, )
    return wrapper

class WildcardProcessor:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"prompt": ("STRING", {"multiline": True, "placeholder": "Prompt Text"}),
                             "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff})},
                "hidden": {"prompt_": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},}

    RETURN_TYPES = ('STRING',)
    FUNCTION = 'process'
    CATEGORY = 'Mikey/Text'

    def process(self, prompt, seed, prompt_=None, extra_pnginfo=None):
        if prompt_ is None:
            prompt_ = {}
        if extra_pnginfo is None:
            extra_pnginfo = {}
        prompt = search_and_replace(prompt, extra_pnginfo, prompt_)
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

    @apply_to_batch
    def apply_haldclut(self, image, hald_clut, gamma_correction):
        hald_img = Image.open(self.haldclut_files[self.file_names.index(hald_clut)])
        img = tensor2pil(image)
        if gamma_correction == 'True':
            corrected_img = gamma_correction_pil(img, 1.0/2.2)
        else:
            corrected_img = img
        filtered_image = apply_hald_clut(hald_img, corrected_img).convert("RGB")
        #return (pil2tensor(filtered_image), )
        return pil2tensor(filtered_image)

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
                              },
                "hidden": {"unique_id": "UNIQUE_ID", "extra_pnginfo": "EXTRA_PNGINFO", "prompt": "PROMPT"}}

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
                  crop_w, crop_h, use_preset_seed, seed, unique_id=None, extra_pnginfo=None, prompt=None):
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
        prompt.get(str(unique_id))['inputs']['output_latent_w'] = str(latent_width)
        prompt.get(str(unique_id))['inputs']['output_latent_h'] = str(latent_height)
        prompt.get(str(unique_id))['inputs']['output_cte_w'] = str(cte_w)
        prompt.get(str(unique_id))['inputs']['output_cte_h'] = str(cte_h)
        prompt.get(str(unique_id))['inputs']['output_target_w'] = str(target_w)
        prompt.get(str(unique_id))['inputs']['output_target_h'] = str(target_h)
        prompt.get(str(unique_id))['inputs']['output_crop_w'] = str(crop_w)
        prompt.get(str(unique_id))['inputs']['output_crop_h'] = str(crop_h)
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
                              "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff})},
                "hidden": {"unique_id": "UNIQUE_ID", "extra_pnginfo": "EXTRA_PNGINFO", "prompt": "PROMPT"}}

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

    def calculate(self, select_preset, swap_axis, use_preset_seed, seed, unique_id=None, extra_pnginfo=None, prompt=None):
        # check if use_preset_seed is true
        if use_preset_seed == 'true' and len(self.ratio_presets) > 0:
            # seed is a randomly generated number that can be much larger than the number of presets
            # we use the seed to select a preset
            len_presets = len(self.ratio_presets)
            offset = seed % (len_presets - 1)
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
        prompt.get(str(unique_id))['inputs']['output_latent_w'] = str(latent_width)
        prompt.get(str(unique_id))['inputs']['output_latent_h'] = str(latent_height)
        prompt.get(str(unique_id))['inputs']['output_cte_w'] = str(cte_w)
        prompt.get(str(unique_id))['inputs']['output_cte_h'] = str(cte_h)
        prompt.get(str(unique_id))['inputs']['output_target_w'] = str(target_w)
        prompt.get(str(unique_id))['inputs']['output_target_h'] = str(target_h)
        prompt.get(str(unique_id))['inputs']['output_crop_w'] = str(crop_w)
        prompt.get(str(unique_id))['inputs']['output_crop_h'] = str(crop_h)
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

class RangeFloat:
    # using the seed value as the step in a range
    # generate a list of numbers from start to end with a step value
    # then select the number at the offset value
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"start": ("FLOAT", {"default": 0, "min": 0, "step": 0.0001, "max": 0xffffffffffffffff}),
                             "end": ("FLOAT", {"default": 0, "min": 0, "step": 0.0001, "max": 0xffffffffffffffff}),
                             "step": ("FLOAT", {"default": 0, "min": 0, "step": 0.0001, "max": 0xffffffffffffffff}),
                             "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff})}}

    RETURN_TYPES = ('FLOAT','STRING',)
    FUNCTION = 'generate'
    CATEGORY = 'Mikey/Utils'

    def generate(self, start, end, step, seed):
        range_ = np.arange(start, end, step)
        list_of_numbers = list(range_)
        # offset
        offset = seed % len(list_of_numbers)
        return (list_of_numbers[offset], f'{list_of_numbers[offset]}',)

class RangeInteger:
    # using the seed value as the step in a range
    # generate a list of numbers from start to end with a step value
    # then select the number at the offset value
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"start": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                             "end": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                             "step": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                             "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff})}}

    RETURN_TYPES = ('INT','STRING',)
    FUNCTION = 'generate'
    CATEGORY = 'Mikey/Utils'

    def generate(self, start, end, step, seed):
        range_ = np.arange(start, end, step)
        list_of_numbers = list(range_)
        # offset
        offset = seed % len(list_of_numbers)
        return (list_of_numbers[offset], f'{list_of_numbers[offset]}',)

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
        #print('Resizing image from {}x{} to {}x{}'.format(image.shape[2], image.shape[1], w, h))
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
                             "crop_amount": ("FLOAT", {"default": 0.05})},
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},}

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

    def batch(self, image_directory, subdirectories, replace_original, replace_suffix, upscale_method, crop, crop_amount,
              prompt, extra_pnginfo):
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
                replace_suffix = search_and_replace(replace_suffix, extra_pnginfo, prompt)
                filename, file_extension = os.path.splitext(file)
                img.save(filename + replace_suffix + file_extension)
        return (f'Job done, {len(files)} images processed',)

class BatchLoadImages:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image_directory": ("STRING", {"multiline": False, "placeholder": "Image Directory"}),
                             "subdirectories": (['true', 'false'], {"default": 'false'})}}

    RETURN_TYPES = ('IMAGE',)
    RETURN_NAMES = ('image',)
    FUNCTION = 'batch'
    CATEGORY = 'Mikey/Image'
    OUTPUT_IS_LIST = (True, )

    def batch(self, image_directory, subdirectories):
        if not os.path.exists(image_directory):
            raise Exception(f"Image directory {image_directory} does not exist")

        images = []
        for file in os.listdir(image_directory):
            if file.endswith('.png') or file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.webp') or file.endswith('.bmp') or file.endswith('.gif'):
                img = Image.open(os.path.join(image_directory, file))
                img = pil2tensor(img)
                images.append(img)
        #print(f'Loaded {len(images)} images')
        return (images,)

class BatchLoadTxtPrompts:
    # reads all the txt files in a directory and returns a list of strings
    # which can be used as prompts to generate images
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"text_directory": ("STRING", {"multiline": False, "placeholder": "Text Directory"}),
                             "subdirectories": (['true', 'false'], {"default": 'false'})}}

    RETURN_TYPES = ('STRING',)
    RETURN_NAMES = ('string',)
    FUNCTION = 'batch'
    CATEGORY = 'Mikey/Text'
    OUTPUT_IS_LIST = (True, )

    def batch(self, text_directory, subdirectories):
        if not os.path.exists(text_directory):
            raise Exception(f"Text directory {text_directory} does not exist")

        strings = []
        for file in os.listdir(text_directory):
            if file.endswith('.txt'):
                with open(os.path.join(text_directory, file), 'r') as f:
                    strings.append(f.read())
        #print(f'Loaded {len(strings)} strings')
        return (strings,)

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
        #print("Saving image outside the output folder is not allowed.")
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
                     "negative_prompt": ("STRING", {'default': 'Negative Prompt'}),
                     "filename_prefix": ("STRING", {"default": ""}),
                     "parameters": ("STRING", {"default": ""}),},
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }

    RETURN_TYPES = ()
    #RETURN_NAMES = ('filename',)
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "Mikey/Image"

    #@classmethod
    #def IS_CHANGED(self, images):
    #    return (np.nan,)

    def save_images(self, images, filename_prefix='', parameters='', prompt=None, extra_pnginfo=None, positive_prompt='', negative_prompt=''):
        filename_prefix = search_and_replace(filename_prefix, extra_pnginfo, prompt)
        full_output_folder, filename, counter, subfolder, filename_prefix = get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
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
                    if x == 'parameters':
                        # encode text as utf-8
                        text = extra_pnginfo[x].encode('utf-8').decode('utf-8')
                        metadata.add_text(x, text)
                    elif x == 'workflow':
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))
                    elif x == 'prompt':
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))
                    else:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x], ensure_ascii=False))
            if positive_prompt:
                #metadata.add_text("positive_prompt", json.dumps(positive_prompt, ensure_ascii=False))
                metadata.add_text("positive_prompt", positive_prompt)
                # replace any special characters with nothing and spaces with _
                clean_pos = re.sub(r'[^a-zA-Z0-9 ]', '', positive_prompt)
                pos_trunc = clean_pos.replace(' ', '_')[0:80]
            if negative_prompt:
                #metadata.add_text("negative_prompt", json.dumps(negative_prompt, ensure_ascii=False))
                metadata.add_text("negative_prompt", negative_prompt)
            if filename_prefix != '':
                clean_filename_prefix = re.sub(r'[^a-zA-Z0-9 _-]', '', filename_prefix)
                metadata.add_text("filename_prefix", json.dumps(clean_filename_prefix, ensure_ascii=False))
                file = f"{clean_filename_prefix[:75]}_{counter:05}_.png"
            else:
                ts_str = datetime.datetime.now().strftime("%y%m%d%H%M%S")
                file = f"{ts_str}_{pos_trunc}_{filename}_{counter:05}_.png"
            if parameters:
                metadata.add_text("parameters", parameters)
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

    def _prepare_filename_texts(self, filename_text_1, filename_text_2, filename_text_3, extra_pnginfo, prompt):
        # replace default values with empty strings
        filename_texts = [filename_text_1, filename_text_2, filename_text_3]
        default_texts = ['Filename Text 1', 'Filename Text 2', 'Filename Text 3']
        for i, text in enumerate(filename_texts):
            if text == default_texts[i]:
                filename_texts[i] = ''
            # use search and replace
            filename_texts[i] = search_and_replace(text, extra_pnginfo, prompt)
            # replace any special characters with nothing
            #filename_texts[i] = re.sub(r'[^a-zA-Z0-9 _-]', '', filename_texts[i])
            # replace only characters that are not allowed in filenames
            filename_texts[i] = re.sub(r'[<>:"/\\|?*]', '', filename_texts[i])
            # remove non ascii characters
            filename_texts[i] = filename_texts[i].encode('ascii', 'ignore').decode('ascii')

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
        sub_directory = search_and_replace(sub_directory, extra_pnginfo, prompt)
        # strip special characters from sub_directory
        #sub_directory = re.sub(r'[^a-zA-Z0-9 _/\\]', '', sub_directory)
        # replace only characters that are not allowed in filenames
        sub_directory = re.sub(r'[<>:"|?*]', '', sub_directory)
        # remove non ascii characters
        sub_directory = sub_directory.encode('ascii', 'ignore').decode('ascii')
        full_output_folder = os.path.join(self.output_dir, sub_directory)
        os.makedirs(full_output_folder, exist_ok=True)

        filename_texts = self._prepare_filename_texts(filename_text_1, filename_text_2, filename_text_3, extra_pnginfo, prompt)

        if timestamp == 'true':
            ts = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
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
        if counter_type != 'none':
            counter = self._get_initial_counter(files, full_output_folder, counter_type, filename_separator, counter_pos, filename_texts)
        else:
            counter = 0

        results = list()
        for ix, image in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = PngInfo()
            if prompt is not None:
                metadata.add_text("prompt", json.dumps(prompt))
            if extra_pnginfo is not None:
                for x in extra_pnginfo:
                    if x == 'parameters':
                        # encode text as utf-8
                        text = extra_pnginfo[x].encode('utf-8').decode('utf-8')
                        metadata.add_text(x, text)
                    elif x == 'workflow':
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))
                    elif x == 'prompt':
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))
                    else:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x], ensure_ascii=False))
            if extra_metadata:
                #metadata.add_text("extra_metadata", json.dumps(extra_metadata, ensure_ascii=False))
                metadata.add_text("extra_metadata", extra_metadata)
            # Check and get the next available counter
            if counter_type != 'none':
                counter = self._get_next_counter(full_output_folder, filename_base, counter)
                current_filename = filename_base.format(counter=f"{counter:05}")
            else:
                current_filename = filename_base
            if timestamp_type == 'save_time' and timestamp == 'true':
                current_timestamp = datetime.datetime.now().strftime("%y%m%d%H%M%S")
                current_filename = current_filename.replace(ts, current_timestamp)
                ts = current_timestamp
            if ix > 0 and counter_type == 'none':
                current_filename = current_filename.replace(ts, ts + f'_{ix:02}')
            img.save(os.path.join(full_output_folder, f"{current_filename}.png"), pnginfo=metadata, compress_level=4)
            results.append({
                "filename": f"{current_filename}.png",
                "subfolder": sub_directory,
                "type": self.type
            })
            if counter_type != 'none':
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

class SaveImageIfTrue:
    # only saves image if save condition input is 1
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",),
                             "save_condition": ("INT", {"default": 0, "min": 0, "max": 1}),
                             "filename_prefix": ("STRING", {"default": ""})},
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }

    RETURN_TYPES = ()
    FUNCTION = "save_image_if_true"
    OUTPUT_NODE = True
    CATEGORY = "Mikey/Image"

    def save_image_if_true(self, image, save_condition, filename_prefix, prompt=None, extra_pnginfo=None):
        if save_condition == 1:
            # use SaveImagesMikey class
            save_images = SaveImagesMikey()
            result = save_images.save_images(image, filename_prefix, prompt, extra_pnginfo, positive_prompt='', negative_prompt='')
            return result
        else:
            return {'save_image_if_true': {'filename': '', 'subfolder': ''}}

class AddMetaData:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",),
                             "label": ("STRING", {"multiline": False, "placeholder": "Label for metadata"}),
                             "text_value": ("STRING", {"multiline": True, "placeholder": "Text to add to metadata"})},
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ('IMAGE',)
    FUNCTION = "add_metadata"
    CATEGORY = "Mikey/Meta"
    OUTPUT_NODE = True

    def add_metadata(self, image, label, text_value, prompt=None, extra_pnginfo=None):
        label = search_and_replace(label, extra_pnginfo, prompt)
        text_value = search_and_replace(text_value, extra_pnginfo, prompt)
        if extra_pnginfo is None:
            extra_pnginfo = {}
        if label in extra_pnginfo:
            extra_pnginfo[label] += ', ' + text_value
        else:
            extra_pnginfo[label] = text_value
        return (image,)

class SearchAndReplace:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"text": ("STRING", {"multiline": False, "placeholder": "Text to search and replace"}),
                             "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),},
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},}

    RETURN_TYPES = ('STRING',)
    FUNCTION = "search_and_replace"
    CATEGORY = "Mikey/Utils"

    def search_and_replace(self, text, seed, prompt=None, extra_pnginfo=None):
        result = search_and_replace(text, extra_pnginfo, prompt)
        s = seed + 1
        return (result,)

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
        filename_prefix = search_and_replace(filename_prefix, extra_pnginfo, prompt)
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
                             'custom_directory': ('STRING', {'default': ''}),
                             'custom_text': ('STRING', {'default': ''})},
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},}

    RETURN_TYPES = ('STRING',)
    RETURN_NAMES = ('filename_prefix',)
    FUNCTION = 'get_filename_prefix'
    CATEGORY = 'Mikey/Meta'

    def get_filename_prefix(self, date, date_directory, custom_directory, custom_text,
                            prompt=None, extra_pnginfo=None):
        filename_prefix = ''
        if custom_directory:
            custom_directory = search_and_replace(custom_directory, extra_pnginfo, prompt)
            filename_prefix += custom_directory + '/'
        if date_directory == 'true':
            ts_str = datetime.datetime.now().strftime("%y%m%d")
            filename_prefix += ts_str + '/'
        if date == 'true':
            ts_str = datetime.datetime.now().strftime("%y%m%d%H%M%S")
            filename_prefix += ts_str
        if custom_text != '':
            custom_text = search_and_replace(custom_text, extra_pnginfo, prompt)
            # remove invalid characters from filename
            custom_text = re.sub(r'[<>:"/\\|?*]', '', custom_text)
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
                             },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ('LATENT','STRING','STRING','STRING','STRING','INT','INT','INT','INT',)
    RETURN_NAMES = ('samples','positive_prompt_text_g','negative_prompt_text_g','positive_style_text_l',
                    'negative_style_text_l','width','height','refiner_width','refiner_height',)
    FUNCTION = 'start'
    CATEGORY = 'Mikey'
    OUTPUT_NODE = True

    def start(self, positive_prompt, negative_prompt, style, ratio_selected, batch_size, seed,
              prompt=None, extra_pnginfo=None):
        # use search and replace
        positive_prompt = search_and_replace(positive_prompt, extra_pnginfo, prompt)
        negative_prompt = search_and_replace(negative_prompt, extra_pnginfo, prompt)
        # process random syntax
        positive_prompt = process_random_syntax(positive_prompt, seed)
        negative_prompt = process_random_syntax(negative_prompt, seed)
        # process wildcards
        #print('Positive Prompt Entered:', positive_prompt)
        pos_prompt = find_and_replace_wildcards(positive_prompt, seed, debug=True)
        #print('Positive Prompt:', pos_prompt)
        #print('Negative Prompt Entered:', negative_prompt)
        neg_prompt = find_and_replace_wildcards(negative_prompt, seed, debug=True)
        #print('Negative Prompt:', neg_prompt)
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
        #print('Width:', width, 'Height:', height,
        #      'Target Width:', target_width, 'Target Height:', target_height,
        #      'Refiner Width:', refiner_width, 'Refiner Height:', refiner_height)
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
        #print('Width:', width, 'Height:', height,
        #     'Target Width:', target_width, 'Target Height:', target_height,
        #     'Refiner Width:', refiner_width, 'Refiner Height:', refiner_height)
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
                             },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ('LATENT','STRING','STRING','STRING','STRING','INT','INT','INT','INT',)
    RETURN_NAMES = ('samples','positive_prompt_text_g','negative_prompt_text_g','positive_style_text_l',
                    'negative_style_text_l','width','height','refiner_width','refiner_height',)
    FUNCTION = 'start'
    CATEGORY = 'Mikey'
    OUTPUT_NODE = True

    def start(self, positive_prompt, negative_prompt, positive_style, negative_style, ratio_selected, batch_size, seed,
              prompt=None, extra_pnginfo=None):
        # search and replace
        positive_prompt = search_and_replace(positive_prompt, extra_pnginfo, prompt)
        negative_prompt = search_and_replace(negative_prompt, extra_pnginfo, prompt)
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
        #print('Width:', width, 'Height:', height,
        #      'Target Width:', target_width, 'Target Height:', target_height,
        #      'Refiner Width:', refiner_width, 'Refiner Height:', refiner_height)
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
                "hidden": {"unique_id": "UNIQUE_ID", "extra_pnginfo": "EXTRA_PNGINFO", "prompt": "PROMPT"},
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
              unique_id=None, extra_pnginfo=None, prompt=None):
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
        #print(batch_size, 4, height // 8, width // 8)
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
        #print('Width:', width, 'Height:', height,
        #      'Target Width:', target_width, 'Target Height:', target_height,
        #      'Refiner Width:', refiner_width, 'Refiner Height:', refiner_height)
        add_metadata_to_dict(prompt_with_style, width=width, height=height, target_width=target_width, target_height=target_height,
                             refiner_width=refiner_width, refiner_height=refiner_height, crop_w=0, crop_h=0)
        # search and replace
        positive_prompt = search_and_replace(positive_prompt, extra_pnginfo, prompt)
        negative_prompt = search_and_replace(negative_prompt, extra_pnginfo, prompt)

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
        #print(style_prompts)
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
            #print("pos_prompt_", pos_prompt_)
            #print("neg_prompt_", neg_prompt_)
            #print("pos_style_", pos_style_)
            #print("neg_style_", neg_style_)
            # encode text
            add_metadata_to_dict(prompt_with_style, style=style_, clip_g_positive=pos_prompt, clip_l_positive=pos_style_)
            add_metadata_to_dict(prompt_with_style, clip_g_negative=neg_prompt, clip_l_negative=neg_style_)
            sdxl_pos_cond = CLIPTextEncodeSDXL.encode(self, clip_base_pos, width, height, 0, 0, target_width, target_height, pos_prompt_, pos_style_)[0]
            sdxl_neg_cond = CLIPTextEncodeSDXL.encode(self, clip_base_neg, width, height, 0, 0, target_width, target_height, neg_prompt_, neg_style_)[0]
            refiner_pos_cond = CLIPTextEncodeSDXLRefiner.encode(self, clip_refiner, 6, refiner_width, refiner_height, pos_prompt_)[0]
            refiner_neg_cond = CLIPTextEncodeSDXLRefiner.encode(self, clip_refiner, 2.5, refiner_width, refiner_height, neg_prompt_)[0]
            prompt.get(str(unique_id))['inputs']['output_positive_prompt'] = pos_prompt_
            prompt.get(str(unique_id))['inputs']['output_negative_prompt'] = neg_prompt_
            prompt.get(str(unique_id))['inputs']['output_latent_width'] = width
            prompt.get(str(unique_id))['inputs']['output_latent_height'] = height
            prompt.get(str(unique_id))['inputs']['output_target_width'] = target_width
            prompt.get(str(unique_id))['inputs']['output_target_height'] = target_height
            prompt.get(str(unique_id))['inputs']['output_refiner_width'] = refiner_width
            prompt.get(str(unique_id))['inputs']['output_refiner_height'] = refiner_height
            prompt.get(str(unique_id))['inputs']['output_crop_w'] = 0
            prompt.get(str(unique_id))['inputs']['output_crop_h'] = 0
            return (base_model, {"samples":latent},
                    sdxl_pos_cond, sdxl_neg_cond,
                    refiner_pos_cond, refiner_neg_cond,
                    pos_prompt_, neg_prompt_, {'extra_pnginfo': extra_pnginfo})

        for style_prompt in style_prompts:
            """ get output from PromptWithStyle.start """
            # strip all style syntax from prompt
            style_ = style_prompt
            #print(style_ in self.styles)
            if style_ not in self.styles:
                # try to match a key without being case sensitive
                style_search = next((x for x in self.styles if x.lower() == style_.lower()), None)
                # if there are still no matches
                if style_search is None:
                    #print(f'Could not find style: {style_}')
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
            prompt.get(str(unique_id))['inputs']['output_positive_prompt'] = pos_prompt_
            prompt.get(str(unique_id))['inputs']['output_negative_prompt'] = neg_prompt_
            prompt.get(str(unique_id))['inputs']['output_latent_width'] = width
            prompt.get(str(unique_id))['inputs']['output_latent_height'] = height
            prompt.get(str(unique_id))['inputs']['output_target_width'] = target_width
            prompt.get(str(unique_id))['inputs']['output_target_height'] = target_height
            prompt.get(str(unique_id))['inputs']['output_refiner_width'] = refiner_width
            prompt.get(str(unique_id))['inputs']['output_refiner_height'] = refiner_height
            prompt.get(str(unique_id))['inputs']['output_crop_w'] = 0
            prompt.get(str(unique_id))['inputs']['output_crop_h'] = 0
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
        prompt.get(str(unique_id))['inputs']['output_positive_prompt'] = pos_prompt_
        prompt.get(str(unique_id))['inputs']['output_negative_prompt'] = neg_prompt_
        prompt.get(str(unique_id))['inputs']['output_latent_width'] = width
        prompt.get(str(unique_id))['inputs']['output_latent_height'] = height
        prompt.get(str(unique_id))['inputs']['output_target_width'] = target_width
        prompt.get(str(unique_id))['inputs']['output_target_height'] = target_height
        prompt.get(str(unique_id))['inputs']['output_refiner_width'] = refiner_width
        prompt.get(str(unique_id))['inputs']['output_refiner_height'] = refiner_height
        prompt.get(str(unique_id))['inputs']['output_crop_w'] = 0
        prompt.get(str(unique_id))['inputs']['output_crop_h'] = 0
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
                    },
                "hidden": {"extra_pnginfo": "EXTRA_PNGINFO", "prompt": "PROMPT"},
                }

    RETURN_TYPES = ('MODEL','CLIP','STRING','STRING')
    RETURN_NAMES = ('model','clip','text','unprocessed_text')
    FUNCTION = 'process'
    CATEGORY = 'Mikey/Lora'

    def process(self, model, clip, text, seed, extra_pnginfo=None, prompt=None):
        # process random syntax
        text = process_random_syntax(text, seed)
        # search and replace
        text = search_and_replace(text, extra_pnginfo, prompt)
        lora_re = r'<lora:(.*?)(?::(.*?))?>'
        # find all lora prompts
        lora_prompts = re.findall(lora_re, text)
        stripped_text = text
        # if we found any lora prompts
        clip_lora = clip
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
                    },
                "hidden": {"extra_pnginfo": "EXTRA_PNGINFO", "prompt": "PROMPT"},
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
        clip_lora = clip
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

    def process(self, model, clip, text, seed, extra_pnginfo=None, prompt=None):
        # search and replace
        text = search_and_replace(text, extra_pnginfo, prompt)
        # process random syntax
        text = process_random_syntax(text, seed)
        # process wildcards
        text_ = find_and_replace_wildcards(text, seed, True)
        if len(text_) != len(text):
            seed = random.randint(0, 1000000)
        else:
            seed = 0
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
                             "use_seed": (['true','false'], {'default': 'false'}),
                             "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                             }
        }

    RETURN_TYPES = ('CONDITIONING','CONDITIONING','CONDITIONING','CONDITIONING','STRING',)
    RETURN_NAMES = ('base_pos_cond','base_neg_cond','refiner_pos_cond','refiner_neg_cond','style_str',)
    FUNCTION = 'add_style'
    CATEGORY = 'Mikey/Conditioning'

    def add_style(self, style, strength, positive_cond_base, negative_cond_base,
                  positive_cond_refiner, negative_cond_refiner, base_clip, refiner_clip,
                  use_seed, seed):
        if use_seed == 'true' and len(self.styles) > 0:
            offset = seed % len(self.styles)
            style = self.styles[offset]
        pos_prompt = self.pos_style[style]
        neg_prompt = self.neg_style[style]
        pos_prompt = pos_prompt.replace('{prompt}', '')
        neg_prompt = neg_prompt.replace('{prompt}', '')
        if style == 'none':
            return (positive_cond_base, negative_cond_base, positive_cond_refiner, negative_cond_refiner, style, )
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

        return (positive_cond_base, negative_cond_base, positive_cond_refiner, negative_cond_refiner, style, )

class StyleConditionerBaseOnly:
    @classmethod
    def INPUT_TYPES(s):
        s.styles, s.pos_style, s.neg_style = read_styles()
        return {"required": {"style": (s.styles,),"strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1}),
                             "positive_cond_base": ("CONDITIONING",), "negative_cond_base": ("CONDITIONING",),
                             "base_clip": ("CLIP",),
                             "use_seed": (['true','false'], {'default': 'false'}),
                             "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                             }
        }

    RETURN_TYPES = ('CONDITIONING','CONDITIONING','STRING',)
    RETURN_NAMES = ('base_pos_cond','base_neg_cond','style_str',)
    FUNCTION = 'add_style'
    CATEGORY = 'Mikey/Conditioning'

    def add_style(self, style, strength, positive_cond_base, negative_cond_base,
                  base_clip,
                  use_seed, seed):
        if use_seed == 'true' and len(self.styles) > 0:
            offset = seed % len(self.styles)
            style = self.styles[offset]
        pos_prompt = self.pos_style[style]
        neg_prompt = self.neg_style[style]
        pos_prompt = pos_prompt.replace('{prompt}', '')
        neg_prompt = neg_prompt.replace('{prompt}', '')
        if style == 'none':
            return (positive_cond_base, negative_cond_base, style, )
        # encode the style prompt
        positive_cond_base_new = CLIPTextEncodeSDXL.encode(self, base_clip, 1024, 1024, 0, 0, 1024, 1024, pos_prompt, pos_prompt)[0]
        negative_cond_base_new = CLIPTextEncodeSDXL.encode(self, base_clip, 1024, 1024, 0, 0, 1024, 1024, neg_prompt, neg_prompt)[0]
        # average the style prompt with the existing conditioning
        positive_cond_base = ConditioningAverage.addWeighted(self, positive_cond_base_new, positive_cond_base, strength)[0]
        negative_cond_base = ConditioningAverage.addWeighted(self, negative_cond_base_new, negative_cond_base, strength)[0]
        return (positive_cond_base, negative_cond_base, style, )

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
                             "upscale_by": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
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
                                  disable_noise=True, start_step=21, force_full_denoise=True)
        # step 3 upscale
        if upscale_by == 0:
            return sample2
        else:
            sample2 = sample2[0]
        pixels = vaedecoder.decode(vae, sample2)[0]
        org_width, org_height = pixels.shape[2], pixels.shape[1]
        img = iuwm.upscale(upscale_model, image=pixels)[0]
        upscaled_width, upscaled_height = int(org_width * upscale_by // 8 * 8), int(org_height * upscale_by // 8 * 8)
        img = image_scaler.upscale(img, 'nearest-exact', upscaled_width, upscaled_height, 'center')[0]
        if hires_strength == 0:
            return (vaeencoder.encode(vae, img)[0],)
        # Adjust start_step based on complexity
        image_complexity = calculate_image_complexity(img)
        #print('Image Complexity:', image_complexity)
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
                             "upscale_by": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
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
                                  disable_noise=True, start_step=15, force_full_denoise=True)
        if upscale_by == 0:
            return sample2
        else:
            sample2 = sample2[0]
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
        #print('Image Complexity:', image_complexity)
        start_step = self.adjust_start_step(image_complexity, hires_strength)
        # encode image
        latent = vaeencoder.encode(vae, img)[0]
        # step 3 run base model
        out = common_ksampler(base_model, seed, 31, 9.5, 'dpmpp_3m_sde_gpu', 'exponential', positive_cond_base, negative_cond_base, latent,
                                start_step=start_step, force_full_denoise=True)
        return out

class MikeySamplerBaseOnlyAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        s.image_scaler = ImageScaleBy()
        s.upscale_models = folder_paths.get_filename_list("upscale_models")
        s.all_upscale_models = s.upscale_models + s.image_scaler.upscale_methods
        try:
            default_model = 'lanczos' #'4x-UltraSharp.pth' if '4x-UltraSharp.pth' in s.upscale_models else s.upscale_models[0]
            um = (s.all_upscale_models, {'default': default_model})
        except:
            um = (folder_paths.get_filename_list("upscale_models"), )
        return {"required": {"base_model": ("MODEL",),
                             "positive_cond_base": ("CONDITIONING",),
                             "negative_cond_base": ("CONDITIONING",),
                             "samples": ("LATENT",),
                             "vae": ("VAE",),
                             "add_noise": (["enable","disable"], {"default": "enable"}),
                             "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                             "steps": ("INT", {"default": 31, "min": 1, "max": 1000}),
                             'smooth_step': ("INT", {"default": 0, "min": -1, "max": 100}),
                             "cfg_1": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 100.0, "step": 0.1}),
                             "cfg_2": ("FLOAT", {"default": 9.5, "min": 0.1, "max": 100.0, "step": 0.1}),
                             "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {'default': 'dpmpp_3m_sde_gpu'}),
                             "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {'default': 'exponential'}),
                             "upscale_model": um,
                             "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                             "upscale_by": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                             "hires_denoise": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.01}),
                             "hires_steps": ("INT", {"default": 31, "min": 1, "max": 1000}),
                             }}

    RETURN_TYPES = ('LATENT',)
    FUNCTION = 'run'
    CATEGORY = 'Mikey/Sampling'

    def run(self, seed, base_model, positive_cond_base, negative_cond_base,
            samples, vae, add_noise, denoise, steps, cfg_1, cfg_2, sampler_name,
            scheduler, upscale_model, upscale_by, hires_denoise, hires_steps, smooth_step):
        image_scaler = ImageScale()
        vaeencoder = VAEEncode()
        vaedecoder = VAEDecode()
        uml = UpscaleModelLoader()
        if upscale_model in image_scaler.upscale_methods:
            upscale_model = upscale_model
        else:
            upscale_model = uml.load_model(upscale_model)[0]
        iuwm = ImageUpscaleWithModel()
        # step 1 run base model low cfg
        start_step = int(steps - (steps * denoise))
        if start_step > steps // 2:
            last_step = steps - 1
        else:
            if start_step % 2 == 0:
                last_step = steps // 2 - 1
            else:
                last_step = steps // 2
        #print(f'base model start_step: {start_step}, last_step: {last_step}')
        sample1 = common_ksampler(base_model, seed, steps, cfg_1, sampler_name, scheduler,
                                  positive_cond_base, negative_cond_base, samples,
                                  start_step=start_step, last_step=last_step, force_full_denoise=False)[0]
        # step 2 run base model high cfg
        start_step = last_step + 1
        total_steps = steps + smooth_step
        sample2 = common_ksampler(base_model, seed+1, total_steps, cfg_2, sampler_name, scheduler,
                                  positive_cond_base, negative_cond_base, sample1,
                                  disable_noise=True, start_step=start_step, force_full_denoise=True)
        if upscale_by == 0:
            return sample2
        else:
            sample2 = sample2[0]
        # step 3 upscale
        pixels = vaedecoder.decode(vae, sample2)[0]
        org_width, org_height = pixels.shape[2], pixels.shape[1]
        if isinstance(upscale_model, str):
            img = self.image_scaler.upscale(pixels, upscale_model, upscale_by)[0]
        else:
            img = iuwm.upscale(upscale_model, image=pixels)[0]
        upscaled_width, upscaled_height = int(org_width * upscale_by // 8 * 8), int(org_height * upscale_by // 8 * 8)
        img = image_scaler.upscale(img, 'nearest-exact', upscaled_width, upscaled_height, 'center')[0]
        if hires_denoise == 0:
            return (vaeencoder.encode(vae, img)[0],)
        # encode image
        latent = vaeencoder.encode(vae, img)[0]
        # step 3 run base model
        start_step = int(hires_steps - (hires_steps * hires_denoise))
        out = common_ksampler(base_model, seed, hires_steps, cfg_2, sampler_name, scheduler,
                              positive_cond_base, negative_cond_base, latent,
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
    #print('Tile Complexity:', complexity)
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

def ai_upscale_adv(tile, base_model, vae, seed, cfg, sampler_name, scheduler, positive_cond_base, negative_cond_base, start_step=11, end_step=20):
    """Upscale a tile using the AI model."""
    vaedecoder = VAEDecode()
    vaeencoder = VAEEncode()
    tile = pil2tensor(tile)
    #print('Tile Complexity:', complexity)
    encoded_tile = vaeencoder.encode(vae, tile)[0]
    tile = common_ksampler(base_model, seed, end_step, cfg, sampler_name, scheduler,
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

def run_tiler_for_steps(enlarged_img, base_model, vae, seed, cfg, sampler_name, scheduler, positive_cond_base, negative_cond_base, steps=20, denoise=0.25):
    # Split the enlarged image into overlapping tiles
    tiles = split_image(enlarged_img)

    # Resample each tile using the AI model
    start_step = int(steps - (steps * denoise))
    end_step = steps
    resampled_tiles = [(coords, ai_upscale_adv(tile, base_model, vae, seed, cfg, sampler_name, scheduler,
                                               positive_cond_base, negative_cond_base, start_step, end_step)) for coords, tile in tiles]

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
                             "denoise_image": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
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
        #print(f'base model start_step: {start_step}, last_step: {last_step}')
        sample1 = common_ksampler(base_model, seed, steps, cfg, sampler_name, scheduler, positive_cond_base, negative_cond_base, samples,
                                  start_step=start_step, last_step=last_step, force_full_denoise=False)[0]
        # step 2 run refiner model
        start_step = last_step + 1
        total_steps = steps + smooth_step
        #print(f'refiner model start_step: {start_step}, last_step: {total_steps}')
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

class MikeySamplerTiledAdvancedBaseOnly:
    # there is no step skipped, so no smooth steps are required
    # also no refiner for this
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"base_model": ("MODEL",),
                             "samples": ("LATENT",), "vae": ("VAE",),
                             "positive_cond_base": ("CONDITIONING",), "negative_cond_base": ("CONDITIONING",),
                             "model_name": (folder_paths.get_filename_list("upscale_models"), ),
                             "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                             "denoise_image": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                             "steps": ("INT", {"default": 30, "min": 1, "max": 1000}),
                             "cfg": ("FLOAT", {"default": 6.5, "min": 0.0, "max": 1000.0, "step": 0.1}),
                             "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                             "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                             "upscale_by": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                             "tiler_denoise": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.05}),},
                "optional": {"image_optional": ("IMAGE",),}}

    RETURN_TYPES = ('IMAGE', )
    RETURN_NAMES = ('output_image', )
    FUNCTION = 'run'
    CATEGORY = 'Mikey/Sampling'

    def phase_one(self, base_model, samples, positive_cond_base, negative_cond_base,
                    upscale_by, model_name, seed, vae, denoise_image,
                    steps, cfg, sampler_name, scheduler):
            image_scaler = ImageScale()
            vaedecoder = VAEDecode()
            uml = UpscaleModelLoader()
            upscale_model = uml.load_model(model_name)[0]
            iuwm = ImageUpscaleWithModel()
            # step 1 run base model
            start_step = int(steps - (steps * denoise_image))
            #print(f'base model start_step: {start_step}, last_step: {last_step}')
            sample1 = common_ksampler(base_model, seed, steps, cfg, sampler_name, scheduler, positive_cond_base, negative_cond_base, samples,
                                    start_step=start_step, last_step=steps, force_full_denoise=False)[0]
            # step 3 upscale image using a simple AI image upscaler
            pixels = vaedecoder.decode(vae, sample1)[0]
            org_width, org_height = pixels.shape[2], pixels.shape[1]
            img = iuwm.upscale(upscale_model, image=pixels)[0]
            upscaled_width, upscaled_height = int(org_width * upscale_by // 8 * 8), int(org_height * upscale_by // 8 * 8)
            img = image_scaler.upscale(img, 'nearest-exact', upscaled_width, upscaled_height, 'center')[0]
            return img, upscaled_width, upscaled_height

    def upscale_image(self, samples, vae,
                    upscale_by, model_name):
            image_scaler = ImageScale()
            vaedecoder = VAEDecode()
            uml = UpscaleModelLoader()
            upscale_model = uml.load_model(model_name)[0]
            iuwm = ImageUpscaleWithModel()
            # step 3 upscale image using a simple AI image upscaler
            pixels = vaedecoder.decode(vae, samples)[0]
            org_width, org_height = pixels.shape[2], pixels.shape[1]
            img = iuwm.upscale(upscale_model, image=pixels)[0]
            upscaled_width, upscaled_height = int(org_width * upscale_by // 8 * 8), int(org_height * upscale_by // 8 * 8)
            img = image_scaler.upscale(img, 'nearest-exact', upscaled_width, upscaled_height, 'center')[0]
            return img, upscaled_width, upscaled_height

    def run(self, seed, base_model, vae, samples, positive_cond_base, negative_cond_base,
            model_name, upscale_by=2.0, tiler_denoise=0.4,
            upscale_method='normal', denoise_image=1.0, steps=30, cfg=6.5,
            sampler_name='dpmpp_sde_gpu', scheduler='karras', image_optional=None):
        # if image not none replace samples with decoded image
        if image_optional is not None:
            vaeencoder = VAEEncode()
            samples = vaeencoder.encode(vae, image_optional)[0]
        if denoise_image > 0:
            # phase 1: run base, refiner, then upscaler model
            img, upscaled_width, upscaled_height = self.phase_one(base_model, samples, positive_cond_base, negative_cond_base,
                                                                upscale_by, model_name, seed, vae, denoise_image,
                                                                steps, cfg, sampler_name, scheduler)
            img = tensor2pil(img)
        else:
            img = self.upscale_image(samples, vae, upscale_by, model_name)
            img = tensor2pil(img)
        # phase 2: run tiler
        tiled_image = run_tiler_for_steps(img, base_model, vae, seed, cfg, sampler_name, scheduler, positive_cond_base, negative_cond_base, steps, tiler_denoise)
        return (tiled_image, )

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
        #print('img shape: ', img.shape)
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
                             "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                             "output_option": (['prompt -> clip_g and style -> clip_l',
                                                'prompt+style -> clip_g and prompt+style -> clip_l',
                                                'prompt+style -> clip_g and style -> clip_l',
                                                'prompt -> clip_g and prompt+style -> clip_l',
                                                'prompt+style -> clip_g and prompt -> clip_l'],
                                               {"default": 'prompt -> clip_g and style -> clip_l'}),},
                "hidden": {"extra_pnginfo": "EXTRA_PNGINFO", "prompt": "PROMPT"},
        }

    RETURN_TYPES = ('LATENT','STRING','STRING','STRING','STRING','INT','INT','INT','INT',)
    RETURN_NAMES = ('samples','positive_prompt_text_g','negative_prompt_text_g','positive_style_text_l',
                    'negative_style_text_l','width','height','refiner_width','refiner_height',)
    FUNCTION = 'start'
    CATEGORY = 'Mikey'

    def start(self, positive_prompt, negative_prompt, positive_style, negative_style, ratio_selected, batch_size, seed, output_option,
              extra_pnginfo, prompt):
        # search and replace
        positive_prompt = search_and_replace(positive_prompt, extra_pnginfo, prompt)
        negative_prompt = search_and_replace(negative_prompt, extra_pnginfo, prompt)
        positive_style = search_and_replace(positive_style, extra_pnginfo, prompt)
        negative_style = search_and_replace(negative_style, extra_pnginfo, prompt)
        # wildcards
        positive_prompt = find_and_replace_wildcards(positive_prompt, seed)
        negative_prompt = find_and_replace_wildcards(negative_prompt, seed)
        positive_style = find_and_replace_wildcards(positive_style, seed)
        negative_style = find_and_replace_wildcards(negative_style, seed)
        # latent dimensions
        width = self.ratio_dict[ratio_selected]["width"]
        height = self.ratio_dict[ratio_selected]["height"]
        latent = torch.zeros([batch_size, 4, height // 8, width // 8])
        refiner_width = width * 4
        refiner_height = height * 4
        if output_option == 'prompt -> clip_g and style -> clip_l':
            positive_clip_g = positive_prompt
            negative_clip_g = negative_prompt
            positive_clip_l = positive_style
            negative_clip_l = negative_style
        elif output_option == 'prompt+style -> clip_g and prompt+style -> clip_l':
            positive_clip_g = positive_prompt + ', ' + positive_style
            negative_clip_g = negative_prompt + ', ' + negative_style
            positive_clip_l = positive_prompt + ', ' + positive_style
            negative_clip_l = negative_prompt + ', ' + negative_style
        elif output_option == 'prompt+style -> clip_g and style -> clip_l':
            positive_clip_g = positive_prompt + ', ' + positive_style
            negative_clip_g = negative_prompt + ', ' + negative_style
            positive_clip_l = positive_style
            negative_clip_l = negative_style
        elif output_option == 'prompt -> clip_g and prompt+style -> clip_l':
            positive_clip_g = positive_prompt
            negative_clip_g = negative_prompt
            positive_clip_l = positive_prompt + ', ' + positive_style
            negative_clip_l = negative_prompt + ', ' + negative_style
        elif output_option == 'prompt+style -> clip_g and prompt -> clip_l':
            positive_clip_g = positive_prompt + ', ' + positive_style
            negative_clip_g = negative_prompt + ', ' + negative_style
            positive_clip_l = positive_prompt
            negative_clip_l = negative_prompt
        return ({"samples":latent},
                str(positive_clip_g),
                str(negative_clip_g),
                str(positive_clip_l),
                str(negative_clip_l),
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
        #print('Resizing image from {}x{} to {}x{}'.format(image.shape[2], image.shape[1], w, h))
        img = self.upscale(image, upscale_method, w, h, crop)[0]
        return (img, )

    def calculate(self, image, tile_resolution):
        width, height = image.shape[2], image.shape[1]
        tile_width, tile_height = find_tile_dimensions(width, height, 1.0, tile_resolution)
        #print('Tile width: ' + str(tile_width), 'Tile height: ' + str(tile_height))
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
                        'caption': ('STRING', {'multiline': True, 'default': 'Caption'})},
                    "hidden": {"extra_pnginfo": "EXTRA_PNGINFO", "prompt": "PROMPT"}}
        else:
            cls.font_dir = None
            cls.font_files = None
            cls.font_file_names = None
            return {'required': {'image': ('IMAGE',),
                        'font': ('STRING', {'default': 'Path to font file'}),
                        'caption': ('STRING', {'multiline': True, 'default': 'Caption'})},
                    "hidden": {"extra_pnginfo": "EXTRA_PNGINFO", "prompt": "PROMPT"}}


    RETURN_TYPES = ('IMAGE',)
    RETURN_NAMES = ('image',)
    FUNCTION = 'caption'
    CATEGORY = 'Mikey/Image'

    def get_text_size(self, font, text):
        """
        Get width and height of a text string with given font.

        Parameters:
            font (ImageFont.FreeTypeFont): A font object.
            text (str): Text to measure.

        Returns:
            (int, int): Width and height of the text.
        """
        # Get the bounding box of the text
        left, top, right, bottom = font.getbbox(text)

        # Calculate width and height of the bounding box
        width = right - left
        height = bottom - top

        return width, height

    def wrap_text(self, text, font, max_width):
        """Wrap text to fit inside a specified width when rendered."""
        wrapped_lines = []
        for line in text.split('\n'):
            words = line.split(' ')
            new_line = words[0]
            for word in words[1:]:
                # Ensure that the width value is an integer
                if int(font.getlength(new_line + ' ' + word)) <= max_width:
                    new_line += ' ' + word
                else:
                    wrapped_lines.append(new_line)
                    new_line = word
            wrapped_lines.append(new_line)
        return wrapped_lines

    @apply_to_batch
    def caption(self, image, font, caption, extra_pnginfo=None, prompt=None):
        if extra_pnginfo is None:
            extra_pnginfo = {}
        # search and replace
        caption = search_and_replace(caption, extra_pnginfo, prompt)
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
        _, text_height = self.get_text_size(font, "Hg")  # Height of a tall character
        wrapped_text_height = len(wrapped_lines) * text_height
        padding = 15  # Adequate padding considering ascenders and descenders
        caption_height = wrapped_text_height + padding * 2  # Additional space above and below text

        # Create the caption bar
        text_image = Image.new('RGB', (width, caption_height), (0, 0, 0))
        draw = ImageDraw.Draw(text_image)

        line_spacing = 5  # Adjust to desired spacing

        # Start y_position a bit higher
        #y_position = (caption_height - wrapped_text_height - (line_spacing * (len(wrapped_lines) - 1))) // 2
        y_position = padding

        for line in wrapped_lines:
            # try/except block is removed since getsize() is not used anymore
            text_width = font.getlength(line)  # It should return a float, so ensure that x_position is an integer.
            x_position = (width - int(text_width)) // 2
            draw.text((x_position, y_position), line, (255, 255, 255), font=font)

            _, text_height = self.get_text_size(font, line)  # Calculate text height
            y_position += text_height + line_spacing # Increment y position by text height and line spacing

        # Combine the images
        combined_image = Image.new('RGB', (width, height + caption_height + line_spacing), (0, 0, 0))
        combined_image.paste(text_image, (0, height))
        combined_image.paste(orig_image, (0, 0))

        #return (pil2tensor(combined_image),)
        return pil2tensor(combined_image)

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

    def blur_border(self, image, border_width):
        # enlarge image and blur to create a border
        # that has similar colors to the image edges
        # scale factor is image with border added
        scale_factor = (image.width + border_width * 2) / image.width
        border_image = image.resize((int(image.width * scale_factor), int(image.height * scale_factor)))
        border_image = border_image.filter(ImageFilter.GaussianBlur(radius=border_width * 0.5))
        # paste image
        border_image.paste(image, (border_width, border_width))
        return pil2tensor(border_image)[None, :, :, :]

    @apply_to_batch
    def border(self, image, border_width, border_color):
        # Convert tensor to PIL image
        orig_image = tensor2pil(image)
        width, height = orig_image.size
        # Create the border
        if border_color == 'checkerboard':
            return checkerboard_border(image, border_width, 'black')
        if border_color == 'blur':
            return self.blur_border(orig_image, border_width)
        # check for string containing a tuple
        if border_color.startswith('(') and border_color.endswith(')'):
            border_color = border_color[1:-1]
            border_color = tuple(map(int, border_color.split(',')))
        border_image = Image.new('RGB', (width + border_width * 2, height + border_width * 2), border_color)
        border_image.paste(orig_image, (border_width, border_width))

        #return (pil2tensor(border_image),)
        return pil2tensor(border_image)

class ImagePaste:
    # takes 2 images, background image and foreground image with transparency areas
    # and pastes the foreground image over the background image
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'background_image': ('IMAGE',),
                             'foreground_image': ('IMAGE',),
                             'x_position': ('INT', {'default': 0, 'min': -10000, 'max': 10000}),
                             'y_position': ('INT', {'default': 0, 'min': -10000, 'max': 10000})}}

    RETURN_TYPES = ('IMAGE',)
    RETURN_NAMES = ('image',)
    FUNCTION = 'paste'
    CATEGORY = 'Mikey/Image'

    def tensor2pil(self, image):
        image_np = np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
        if image_np.shape[0] == 4:  # Check for an alpha channel
            return Image.fromarray(image_np.transpose(1, 2, 0), 'RGBA')
        else:
            return Image.fromarray(image_np.transpose(1, 2, 0), 'RGB')

    def paste(self, background_image, foreground_image, x_position, y_position):
        # Convert tensor to PIL image
        background_image = tensor2pil(background_image)
        foreground_image = tensor2pil(foreground_image)  # Using same function for now

        # Check if the images have alpha channel and create mask
        if foreground_image.mode != 'RGBA':
            foreground_image = foreground_image.convert('RGBA')

        # Separate the alpha channel and use it as mask
        r, g, b, alpha = foreground_image.split()

        # paste the foreground image onto the background image
        background_image.paste(foreground_image, (x_position, y_position), mask=alpha)

        return (pil2tensor(background_image),)

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
                             'seed': ('INT', {'default': 0, 'min': 0, 'max': 0xffffffffffffffff})},
                "hidden": {"extra_pnginfo": "EXTRA_PNGINFO", "prompt": "PROMPT"}}

    RETURN_TYPES = ('STRING','STRING')
    RETURN_NAMES = ('output1','output2')
    FUNCTION = 'mix'
    CATEGORY = 'Mikey/Text'

    def mix(self, text1, text2, operation, delimiter, use_seed, seed, extra_pnginfo, prompt):
        # search and replace
        text1 = search_and_replace(text1, extra_pnginfo, prompt)
        text2 = search_and_replace(text2, extra_pnginfo, prompt)

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
                             'seed': ('INT', {'default': 0, 'min': 0, 'max': 0xffffffffffffffff})},
                "hidden": {"extra_pnginfo": "EXTRA_PNGINFO", "prompt": "PROMPT"}}

    RETURN_TYPES = ('STRING','STRING','STRING')
    RETURN_NAMES = ('output1','output2','output3')
    FUNCTION = 'mix'
    CATEGORY = 'Mikey/Text'

    def mix(self, text1, text2, text3, operation, delimiter, use_seed, seed, extra_pnginfo, prompt):
        # search and replace
        text1 = search_and_replace(text1, extra_pnginfo, prompt)
        text2 = search_and_replace(text2, extra_pnginfo, prompt)
        text3 = search_and_replace(text3, extra_pnginfo, prompt)

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
                             'use_text_c_for_both': (['true','false'], {'default': 'false'}),},
                "hidden": {"extra_pnginfo": "EXTRA_PNGINFO", "prompt": "PROMPT"}}

    RETURN_TYPES = ('STRING','STRING',)
    RETURN_NAMES = ('text_a','text_b',)
    FUNCTION = 'output'
    CATEGORY = 'Mikey/Text'

    def output(self, text_a, text_b, text_c, use_text_c_for_both, extra_pnginfo, prompt):
        # search and replace
        text_a = search_and_replace(text_a, extra_pnginfo, prompt)
        text_b = search_and_replace(text_b, extra_pnginfo, prompt)
        text_c = search_and_replace(text_c, extra_pnginfo, prompt)
        if use_text_c_for_both == 'true':
            return (text_c, text_c)
        else:
            return (text_a, text_b)

class CheckpointLoaderSimpleMikey:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
                             },
                "hidden": {"unique_id": "UNIQUE_ID", "extra_pnginfo": "EXTRA_PNGINFO", "prompt": "PROMPT"}}

    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "STRING", "STRING")
    RETURN_NAMES = ("model", "clip", "vae", "ckpt_name", "ckpt_hash")
    FUNCTION = "load_checkpoint"

    CATEGORY = "Mikey/Loaders"

    def load_checkpoint(self, ckpt_name, output_vae=True, output_clip=True, unique_id=None, extra_pnginfo=None, prompt=None):
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
        # get hash using python, there is no function in the comfy library
        hash = get_file_hash(ckpt_path)[:10]
        # just return the filename, not path
        ckpt_name = os.path.basename(ckpt_name)
        prompt.get(str(unique_id))['inputs']['output_ckpt_hash'] = hash
        prompt.get(str(unique_id))['inputs']['output_ckpt_name'] = ckpt_name
        return out[:3] + (ckpt_name, hash)

class TextPreserve:
    @classmethod
    def INPUT_TYPES(s):
        return {'required': {'text': ('STRING', {'multiline': True, 'default': 'Input Text Here', 'dynamicPrompts': False}),
                             'result_text': ('STRING', {'multiline': True, 'default': 'Result Text Here (will be replaced)'})},
                "hidden": {"unique_id": "UNIQUE_ID", "extra_pnginfo": "EXTRA_PNGINFO", "prompt": "PROMPT"}}

    RETURN_TYPES = ('STRING',)
    RETURN_NAMES = ('text',)
    FUNCTION = 'process'
    OUTPUT_NODE = True

    CATEGORY = 'Mikey/Text'

    def process(self, text, result_text, unique_id=None, extra_pnginfo=None, prompt=None):
        preserve_text = text
        # search and replace
        text = search_and_replace(text, extra_pnginfo, prompt)
        # wildcard sytax is {like|this}
        # select a random word from the | separated list
        wc_re = re.compile(r'{([^}]+)}')
        def repl(m):
            return random.choice(m.group(1).split('|'))
        for m in wc_re.finditer(text):
            text = text.replace(m.group(0), repl(m))
        prompt.get(str(unique_id))['inputs']['text'] = preserve_text
        for i, node_dict in enumerate(extra_pnginfo['workflow']['nodes']):
            if node_dict['id'] == int(unique_id):
                node_dict['widgets_values'] = [preserve_text, text]
                extra_pnginfo['workflow']['nodes'][i] = node_dict
        prompt.get(str(unique_id))['inputs']['result_text'] = text
        return (text,)

class OobaPrompt:
    @classmethod
    def INPUT_TYPES(s):
        return {'required': {'input_prompt': ('STRING', {'multiline': True, 'default': 'Prompt Text Here', 'dynamicPrompts': False}),
                             'mode': (['prompt', 'style', 'descriptor', 'character', 'negative', 'custom'], {'default': 'prompt'}),
                             'custom_history': ('STRING', {'multiline': False, 'default': 'path to history.json', 'dynamicPrompts': True}),
                             'seed': ('INT', {'default': 0, 'min': 0, 'max': 0xffffffffffffffff}),},
                "hidden": {"unique_id": "UNIQUE_ID", "extra_pnginfo": "EXTRA_PNGINFO", "prompt": "PROMPT"}}

    RETURN_TYPES = ('STRING',)
    RETURN_NAMES = ('text',)
    FUNCTION = 'process'

    CATEGORY = 'Mikey/AI'

    def history(self, mode, custom_history):
        if mode == 'prompt':
            return {
                "internal": [
                    [
                        "<|BEGIN-VISIBLE-CHAT|>",
                        "How can I help you today?"
                    ],
                    [
                        "I say something like 'blonde woman' and you respond with a single prompt which I can use as prompts for an AI txt2image model. Your response to blonde woman would be something like 'Blonde woman wearing a patterned orange, white, and blue sundress, smiling, on a sunny day with a blue sky, surrounded by buildings and palm trees'. It describes the image with lots of details so the AI model will understand what it needs to generate.",
                        "Sure thing! Let's begin. What is your first prompt?"
                    ],
                    [
                        "futuristic man",
                        "Man with short black hair in a highly detailed silver and black mechanical exoskeleton suit, holding a futuristic rifle, standing in a futuristic city with tall buildings and flying cars, with a futuristic cityscape in the background."
                    ],
                    [
                        "a black cat",
                        "A black cat with green eyes, sitting on a wooden table, there is a vase of flowers on the table, the cat is looking at the flowers. Sunlight is streaming in through a window, illuminating the cat and the flowers."
                    ],
                    [
                        "toaster oven",
                        "A toaster oven with a slice of bread inside, the bread is toasted and has a smiley face burned into it. The toaster oven is sitting on a kitchen counter next to a coffee maker and a microwave. The kitchen counter is made of granite and has a sink in it. There is a window above the sink with a view of a city skyline."
                    ],
                    [
                        "war",
                        "A soldier wearing a gas mask and a helmet, holding a rifle, standing in a trench. There is barbed wire in front of the trench. The sky is dark and cloudy. There is a tank in the background."
                    ],
                    [
                        "portrait of a man",
                        "A portrait of a man in his 30s with short brown hair and a beard. He is wearing a suit and tie. He is smiling. He is standing in front of a brick wall with graffiti on it."
                    ],
                    [
                        "a friendly ogre",
                        "A friendly ogre with green skin and a big smile. He is wearing a red and white striped shirt and blue overalls. He is holding a pitchfork. He is standing in a field of flowers. There is a rainbow in the sky."
                    ],
                    [
                        "puppy",
                        "A cute pug dog wearing a bowtie and reading a book. The book is titled 'How to be a good dog'. The dog is sitting on a couch in a living room. There is a fireplace in the background."
                    ],
                    [
                        "apples",
                        "Red apples stacked in a wooden crate. The crate is sitting on a wooden table in the kitchen inside a rustic farm house with old appliances. The walls are decorated with family photos. There is a window in the background with a view of the farm."
                    ],
                    [
                        "working woman",
                        "A woman is working on a laptop computer. She is wearing a white shirt and black pants. She is sitting at a desk in a modern office with trendy furniture. She has a cup of coffee on the desk next to her."
                    ]
                ],
                "visible": [
                    [
                        "",
                        "How can I help you today?"
                    ],
                ]
            }
        elif mode == 'style':
            return {
                "internal": [
                    [
                        "<|BEGIN-VISIBLE-CHAT|>",
                        "How can I help you today?"
                    ],
                    [
                        "I say something like 'painting' and you respond with a single prompt which I can use as prompts for an AI txt2image model. Your response to painting would be something like 'Impressionistic, landscape, vivid colors, loose brushstrokes, beauty of nature. Inspired by Claude Monet'. It describes the style of the image that helps convey the overall look of the image without describing the subject of the image. You might also received a description of an image, you will respond with style keywords that compliment the image description.",
                        "Sure thing! Let's begin. What is your first prompt?"
                    ],
                    [
                        "painting",
                        "Realism, oil painting, dark shadows, bright highlights, focus on capturing light and texture. Inspired by Caravaggio's chiaroscuro technique"
                    ],
                    [
                        "Impressionism",
                        "Impressionism. Soft brushstrokes, fleeting moments of light and color. Landscapes, flowers, people in motion"
                    ],
                    [
                        "painting",
                        "Impressionistic, landscape, vivid colors, loose brushstrokes, beauty of nature. Inspired by Claude Monet"
                    ],
                    [
                        "The 2015 Audi R8 Spyder sports car is parked on a driveway outside a luxurious mansion. The car is painted in a metallic grey color. The top is down revealing the sleek interior. The car is surrounded by lush greenery and palm trees.",
                        "Landscape photography, vibrant saturation, dramatic shadows, golden hour lighting, inspired by the work of Peter Lik."
                    ],
                    [
                        "abstract",
                        "Abstract expressionism, bold brushstrokes, vivid colors"
                    ],
                    [
                        "water",
                        "Water sculpture, fluid dynamics, abstract representation, in the style of Yves Klein"
                    ],
                    [
                        "A MacBook Pro open and displaying an email message. The keyboard is illuminated and the trackpad is being used. A man is sitting at a wooden desk in a cozy home office. There is a plant in the corner and sunlight coming in from a nearby window.",
                        "Still life, muted colors, soft lighting, in the style of Henri Matisse"
                    ],
                    [
                        "Surrealism",
                        "dreamlike imagery, unexpected juxtapositions, symbolic elements. features distorted or unusual forms, animals or objects transformed into otherworldly shapes"
                    ],
                    [
                        "Art Nouveau",
                        "Art Nouveau style, flowing lines, organic shapes, muted color palette, decorative elements, floral motifs."
                    ],
                    [
                        "photo",
                        "Black and white photograph, shot using a large format camera with slow shutter speeds. Grainy texture and high contrast. Influenced by the works of Edward Hopper."
                    ],
                    [
                        "photo",
                        "Color photograph, Soft focus, muted colors, romantic atmosphere, in the style of Edward Weston."
                    ],
                    [
                        "film",
                        "Long shot, film still, cinematic lighting, gritty realism, inspired by the works of Gus Van Sant"
                    ],
                    [
                        "movie",
                        "Filmic storytelling, dreamlike imagery, surreal elements, poetic narratives, reminiscent of the works of David Lynch."
                    ]
                ],
                "visible": [
                    [
                        "",
                        "How can I help you today?"
                    ]
                ]
            }
        elif mode == 'descriptor':
            return {
                "internal": [
                    [
                        "<|BEGIN-VISIBLE-CHAT|>",
                        "How can I help you today?"
                    ],
                    [
                        "I say something like 'color' and you respond with a single prompt which I can use to build a prompt for an AI txt2image model. Your response to color would be something like 'red'. It is a very short description to add dynamic variety to the prompt.",
                        "Sure thing! Let's begin. What is your first prompt?"
                    ],
                    [
                        "color",
                        "burnt sienna"
                    ],
                    [
                        "hair color",
                        "platinum blonde"
                    ],
                    [
                        "metal",
                        "rusted iron"
                    ],
                    [
                        "weather",
                        "bright and sunny"
                    ],
                    [
                        "time of day",
                        "crack of dawn"
                    ],
                    [
                        "man",
                        "tall and slender man with wide shoulders in his 30s"
                    ],
                    [
                        "ethnicity",
                        "Vietnamese"
                    ],
                    [
                        "occupation",
                        "Heavy diesel mechanic"
                    ],
                    [
                        "art style",
                        "crystal cubism"
                    ],
                    [
                        "artist",
                        "Camille Pissarro"
                    ],
                    [
                        "movie director",
                        "David Lynch"
                    ]
                ],
                "visible": [
                    [
                        "",
                        "How can I help you today?"
                    ]
                ]
            }
        elif mode == 'character':
            return {
                "internal": [
                    [
                        "<|BEGIN-VISIBLE-CHAT|>",
                        "How can I help you today?"
                    ],
                    [
                        "When a user requests a character description, generate a detailed description of the character as you would expect from the writer George R. R. Martin. The description should be a suitable prompt based on the description that encapsulates the character's key visual elements for image creation using a txt2img model.",
                        "Sure thing! Let's begin. What is your first prompt?"
                    ],
                    [
                        "jolly cartoon octopus",
                        "A cartoon octopus with a jolly demeanor in a sunshiny yellow and orange color scheme, wearing a small top hat and a polka-dotted bow tie, surrounded by intricate sculptures made of its own ink, in the midst of an undersea setting that hints at festivity and mirth.",
                    ],
                    [
                        "a dapper young lady spy who wears suits",
                        "A young 1920s lady spy with chestnut hair in a bob cut, piercing emerald eyes, wearing a tailored charcoal pinstripe suit with a white shirt and a silk tie, holding a silver cigarette case, exuding an aura of mystery and sophistication against a backdrop of Parisian nightlife.",
                    ],
                    [
                        "a charming robot",
                        "A charming robot with a sleek chrome body, art deco design elements, azure glowing eyes, and a gentle smile. Wearing a holographic bow tie and a vest painted to look like a dapper suit, engaged in performing a magic trick with a playful, inquisitive expression in an urban park setting.",
                    ],
                    [
                        "cartoon ant",
                        "A cartoon ant with a vibrant blue exoskeleton, oversized round eyes full of curiosity, wearing a leaf-green vest and a tiny fabric cap. Exhibiting an expression of wonder and excitement, amidst a backdrop of an underground ant colony bustling with activity.",
                    ],
                    [
                        "a cyberpunk gnome",
                        "A cyberpunk gnome with pale skin and cybernetic jade eyes, wearing a long tattered coat with circuitry patches and a sleek metallic pointed helmet. Surrounded by holographic screens and neon lights in a dim, cluttered workshop filled with futuristic gadgets and data screens.",
                    ]
                ],
                "visible": [
                    [
                        "",
                        "How can I help you today?"
                    ]
                ]
            }
        elif mode == 'negative':
            return {
                "internal": [
                    [
                        "<|BEGIN-VISIBLE-CHAT|>",
                        "How can I help you today?"
                    ],
                    [
                        "I provide a prompt for a text to image AI model, and you will respond with a prompt that helps to improve the quality of the image." \
                            "The prompt contains styles you don't want to see in the image. For example if the user asks for a photo, the your response would contain terms such as 'cartoon, illustration, 3d render'."\
                                "The word 'no' is forbidden!. Do not repeat yourself, do not provide similar but different words. Do not provide sentences, just words separated by commas. Keep it short.",
                        "Sure thing! Let's begin. What is your first prompt?"
                    ],
                    [
                        "breathtaking image of a woman in a red dress. award-winning, professional, highly detailed",
                        "ugly, deformed, noisy, blurry, distorted, grainy, plain background, monochrome, signature, watermark."
                    ],
                    [
                        "concept art. digital artwork, illustrative, painterly, matte painting, highly detailed",
                        "photo, photorealistic, realism, ugly, signature."
                    ],
                    [
                        "3d model, unreal engine, textures, volumetric lighting, raytraced rendering, subject centered",
                        "watermark, 2D, low poly, polygon mesh, cartoon, trypophobia, miniature, traditional art, logo."
                    ],
                    [
                        "comic book art, onomatopoeic graphics, halftone dots, color separation, bold lines, reminiscent of art from Marvel, DC, Dark Horse, Image, and other comic book publishers",
                        "photo, photorealistic, realism, ugly, signature."
                    ],
                    [
                        "dark fantasy art, gothic themes, macabre, horror, dark, gloomy, sinister, ominous, eerie",
                        "watermark, bright colors, cheerful, happy, cute, cartoon, anime, digital art, 3d render, photograph, 2d art."
                    ],
                    [
                        "charcoal drawing, rich blacks, contrasting whites, smudging, erasure, hatching, cross-hatching, tonal values, rough texture",
                        "photograph, painting, watercolor, colorful, 3d render, cartoon, anime, digital art."
                    ],
                    [
                        "stunning photograph of a tiger in a tropical rainforest",
                        "black and white, cartoon, illustration, low resolution, noisy, blurry, desaturated, ugly, deformed."
                    ]
                ],
                "visible": [
                    [
                        "",
                        "How can I help you today?"
                    ]
                ]
            }
        elif mode == 'custom':
            # open json file that is in the custom_history path
            try:
                history = json.load(open(custom_history))
                return history
            except:
                raise Exception('Error loading custom history file')

    def api_request(self, prompt, seed, mode, custom_history):
        # check if json file in root comfy directory called oooba.json
        history = self.history(mode, custom_history)
        if mode == 'prompt':
            prompt = f'{prompt}, describe in detail.'
        if mode == 'descriptor':
            # use seed to add a bit more randomness to the prompt
            spice = ['a', 'the', 'this', 'that',
                     'an exotic', 'an interesting', 'a colorful', 'a vibrant', 'get creative!', 'think outside the box!', 'a rare',
                     'a standard', 'a typical', 'a common', 'a normal', 'a regular', 'a usual', 'an ordinary',
                     'a unique', 'a one of a kind', 'a special', 'a distinctive', 'a peculiar', 'a remarkable', 'a noteworthy',
                     'popular in the victorian era', 'popular in the 1920s', 'popular in the 1950s', 'popular in the 1980s',
                     'popular in the 1990s', 'popular in the 2000s', 'popular in the 2010s', 'popular in the 2020s',
                     'popular in asia', 'popular in europe', 'popular in north america', 'popular in south america',
                     'popular in africa', 'popular in australia', 'popular in the middle east', 'popular in the pacific islands',
                     'popular with young people', 'popular with the elderly', 'trending on social media', 'popular on tiktok',
                     'trending on pinterest', 'popular on instagram', 'popular on facebook', 'popular on twitter',
                     'popular on reddit', 'popular on youtube', 'popular on tumblr', 'popular on snapchat',
                     'popular on linkedin', 'popular on twitch', 'popular on discord',
                     'unusual example of', 'a classic', 'an underrated', 'an innovative','a historical', 'a modern', 'a contemporary',
                     'a futuristic', 'a traditional', 'an eco-friendly', 'a controversial', 'a political', 'a religious',
                     'a spiritual', 'a philosophical', 'a scientific']
            random.seed(seed)
            prompt = f'{random.choice(spice)} {prompt}'
        request = {
            'user_input': prompt,
            'max_new_tokens': 250,
            'auto_max_new_tokens': False,
            'max_tokens_second': 0,
            'history': history,
            'mode': 'instruct',
             'regenerate': False,
            '_continue': False,
            'preset': 'StarChat',
            'seed': seed,
        }
        HOST = 'localhost:5000'
        URI = f'http://{HOST}/api/v1/chat'
        try:
            response = requests.post(URI, json=request, timeout=20)
        except requests.exceptions.ConnectionError:
            raise Exception('Are you running oobabooga with API enabled?')

        if response.status_code == 200:
            result = response.json()['results'][0]['history']['visible'][-1][1]
            result = html.unescape(result)  # decode URL encoded special characters
            return result
        else:
            return 'Error'

    def process(self, input_prompt, mode, custom_history, seed, prompt=None, unique_id=None, extra_pnginfo=None):
        # search and replace
        input_prompt = search_and_replace(input_prompt, extra_pnginfo, prompt)
        # wildcard sytax is {like|this}
        # select a random word from the | separated list
        wc_re = re.compile(r'{([^}]+)}')
        def repl(m):
            return random.choice(m.group(1).split('|'))
        for m in wc_re.finditer(input_prompt):
            input_prompt = input_prompt.replace(m.group(0), repl(m))
        result = self.api_request(input_prompt, seed, mode, custom_history)
        prompt.get(str(unique_id))['inputs']['output_text'] = result
        return (result,)

class WildcardOobaPrompt:
    # processes wildcard syntax
    # and also processes a llm sytax using the oobaprompt class
    @classmethod
    def INPUT_TYPES(s):
        return {'required': {'input_prompt': ('STRING', {'multiline': True, 'default': 'Prompt Text Here', 'dynamicPrompts': False}),
                             #'mode': (['prompt', 'style', 'descriptor', 'custom'], {'default': 'prompt'}),
                             #'custom_history': ('STRING', {'multiline': False, 'default': 'path to history.json', 'dynamicPrompts': True}),
                             'seed': ('INT', {'default': 0, 'min': 0, 'max': 0xffffffffffffffff})},
                "hidden": {"unique_id": "UNIQUE_ID", "extra_pnginfo": "EXTRA_PNGINFO", "prompt": "PROMPT"}}

    RETURN_TYPES = ('STRING',)
    RETURN_NAMES = ('text',)
    FUNCTION = 'process'
    OUTPUT_NODE = True
    CATEGORY = 'Mikey/AI'

    def process(self, input_prompt, seed, prompt=None, unique_id=None, extra_pnginfo=None):
        # search and replace
        input_prompt = search_and_replace(input_prompt, extra_pnginfo, prompt)
        # wildcard sytax is {like|this}
        # select a random word from the | separated list
        wc_re = re.compile(r'{([^}]+)}')
        def repl(m):
            return random.choice(m.group(1).split('|'))
        for m in wc_re.finditer(input_prompt):
            input_prompt = input_prompt.replace(m.group(0), repl(m))
        # process wildcards
        input_prompt = find_and_replace_wildcards(input_prompt, seed, debug=True)
        # check if llm syntax is in the prompt
        # examples <llm:prompt:prompt text here>, <llm:style:prompt text here>, <llm:descriptor:prompt text here>
        # example for custom history file <llm:path/to/history.json:prompt text here>
        llm_re = re.compile(r'<llm:(.*?):(.*?)>')
        for m in llm_re.finditer(input_prompt):
            mode = m.group(1)
            if '.json' in mode:
                custom_history = mode
                mode = 'custom'
            else:
                custom_history = None
            prompt_text = m.group(2)
            ooba = OobaPrompt()
            result = ooba.api_request(prompt_text, seed, mode, custom_history)
            input_prompt = input_prompt.replace(m.group(0), result)
        prompt.get(str(unique_id))['inputs']['output_text'] = input_prompt
        return (input_prompt,)

# same function as oobaprompt but using the LM Studio API
class LMStudioPrompt:
    @classmethod
    def INPUT_TYPES(s):
        return {'required': {'input_prompt': ('STRING', {'multiline': True, 'default': 'Prompt Text Here', 'dynamicPrompts': False}),
                             'mode': (['prompt', 'style', 'descriptor', 'character', 'custom'], {'default': 'prompt'}),
                             'custom_history': ('STRING', {'multiline': False, 'default': 'path to history.json', 'dynamicPrompts': True}),
                             'server_address': ('STRING', {'default': 'localhost'}),
                             'server_port': ('INT', {'default': 1234, 'min': 0, 'max': 65535}),
                             'seed': ('INT', {'default': 0, 'min': 0, 'max': 0xffffffffffffffff}),},
                "hidden": {"unique_id": "UNIQUE_ID", "extra_pnginfo": "EXTRA_PNGINFO", "prompt": "PROMPT"}}

    RETURN_TYPES = ('STRING',)
    RETURN_NAMES = ('text',)
    FUNCTION = 'process'
    OUTPUT_NODE = True
    CATEGORY = 'Mikey/AI'

    def history(self, mode, custom_history):
        if mode == 'prompt':
            return {
                "messages": [
                    { "role": "system", "content": "You are a helpful assistant." },
                    { "role": "user", "content": "I say something like 'blonde woman' and you respond with a single prompt which I can use as prompts for an AI txt2image model. Your response to blonde woman would be something like 'Blonde woman wearing a patterned orange, white, and blue sundress, smiling, on a sunny day with a blue sky, surrounded by buildings and palm trees'. It describes the image with lots of details so the AI model will understand what it needs to generate."},
                    { "role": "assistant", "content": "Sure thing! Let's begin. What is your first prompt?" },
                    { "role": "user", "content": "futuristic man"},
                    { "role": "assistant", "content": "Man with short black hair in a highly detailed silver and black mechanical exoskeleton suit, holding a futuristic rifle, standing in a futuristic city with tall buildings and flying cars, with a futuristic cityscape in the background."},
                    { "role": "user", "content": "a black cat"},
                    { "role": "assistant", "content": "A black cat with green eyes, sitting on a wooden table, there is a vase of flowers on the table, the cat is looking at the flowers. Sunlight is streaming in through a window, illuminating the cat and the flowers."},
                    { "role": "user", "content": "toaster oven"},
                    { "role": "assistant", "content": "A toaster oven with a slice of bread inside, the bread is toasted and has a smiley face burned into it. The toaster oven is sitting on a kitchen counter next to a coffee maker and a microwave. The kitchen counter is made of granite and has a sink in it. There is a window above the sink with a view of a city skyline."},
                    { "role": "user", "content": "war"},
                    { "role": "assistant", "content": "A soldier wearing a gas mask and a helmet, holding a rifle, standing in a trench. There is barbed wire in front of the trench. The sky is dark and cloudy. There is a tank in the background."},
                    { "role": "user", "content": "portrait of a man"},
                    { "role": "assistant", "content": "A portrait of a man in his 30s with short brown hair and a beard. He is wearing a suit and tie. He is smiling. He is standing in front of a brick wall with graffiti on it."},
                    { "role": "user", "content": "a friendly ogre"},
                    { "role": "assistant", "content": "A friendly ogre with green skin and a big smile. He is wearing a red and white striped shirt and blue overalls. He is holding a pitchfork. He is standing in a field of flowers. There is a rainbow in the sky."},
                    { "role": "user", "content": "puppy"},
                    { "role": "assistant", "content": "A cute pug dog wearing a bowtie and reading a book. The book is titled 'How to be a good dog'. The dog is sitting on a couch in a living room. There is a fireplace in the background."},
                    { "role": "user", "content": "apples"},
                    { "role": "assistant", "content": "Red apples stacked in a wooden crate. The crate is sitting on a wooden table in the kitchen inside a rustic farm house with old appliances. The walls are decorated with family photos. There is a window in the background with a view of the farm."},
                    { "role": "user", "content": "working woman"},
                    { "role": "assistant", "content": "A woman is working on a laptop computer. She is wearing a white shirt and black pants. She is sitting at a desk in a modern office with trendy furniture. She has a cup of coffee on the desk next to her."}
                ]
            }
        elif mode == 'style':
            return {
                "messages": [
                    { "role": "system", "content": "You are a helpful assistant." },
                    { "role": "user", "content": "I say something like 'painting' and you respond with a single prompt which I can use as prompts for an AI txt2image model. Your response to painting would be something like 'Impressionistic, landscape, vivid colors, loose brushstrokes, beauty of nature. Inspired by Claude Monet'. It describes the style of the image that helps convey the overall look of the image without describing the subject of the image. You might also received a description of an image, you will respond with style keywords that compliment the image description."},
                    { "role": "assistant", "content": "Sure thing! Let's begin. What is your first prompt?" },
                    { "role": "user", "content": "painting"},
                    { "role": "assistant", "content": "Realism, oil painting, dark shadows, bright highlights, focus on capturing light and texture. Inspired by Caravaggio's chiaroscuro technique"},
                    { "role": "user", "content": "Impressionism"},
                    { "role": "assistant", "content": "Impressionism. Soft brushstrokes, fleeting moments of light and color. Landscapes, flowers, people in motion"},
                    { "role": "user", "content": "painting"},
                    { "role": "assistant", "content": "Impressionistic, landscape, vivid colors, loose brushstrokes, beauty of nature. Inspired by Claude Monet"},
                    { "role": "user", "content": "The 2015 Audi R8 Spyder sports car is parked on a driveway outside a luxurious mansion. The car is painted in a metallic grey color. The top is down revealing the sleek interior. The car is surrounded by lush greenery and palm trees."},
                    { "role": "assistant", "content": "Landscape photography, vibrant saturation, dramatic shadows, golden hour lighting, inspired by the work of Peter Lik."},
                    { "role": "user", "content": "abstract"},
                    { "role": "assistant", "content": "Abstract expressionism, bold brushstrokes, vivid colors"},
                    { "role": "user", "content": "water"},
                    { "role": "assistant", "content": "Water sculpture, fluid dynamics, abstract representation, in the style of Yves Klein"},
                    { "role": "user", "content": "A MacBook Pro open and displaying an email message. The keyboard is illuminated and the trackpad is being used. A man is sitting at a wooden desk in a cozy home office. There is a plant in the corner and sunlight coming in from a nearby window."},
                    { "role": "assistant", "content": "Still life, muted colors, soft lighting, in the style of Henri Matisse"},
                    { "role": "user", "content": "Surrealism"},
                    { "role": "assistant", "content": "dreamlike imagery, unexpected juxtapositions, symbolic elements. features distorted or unusual forms, animals or objects transformed into otherworldly shapes"},
                    { "role": "user", "content": "Art Nouveau"},
                    { "role": "assistant", "content": "Art Nouveau style, flowing lines, organic shapes, muted color palette, decorative elements, floral motifs."},
                    { "role": "user", "content": "photo"},
                    { "role": "assistant", "content": "Black and white photograph, shot using a large format camera with slow shutter speeds. Grainy texture and high contrast. Influenced by the works of Edward Hopper."},
                    { "role": "user", "content": "photo"},
                    { "role": "assistant", "content": "Color photograph, Soft focus, muted colors, romantic atmosphere, in the style of Edward Weston."},
                    { "role": "user", "content": "film"},
                    { "role": "assistant", "content": "Long shot, film still, cinematic lighting, gritty realism, inspired by the works of Gus Van Sant"},
                    { "role": "user", "content": "movie"},
                    { "role": "assistant", "content": "Filmic storytelling, dreamlike imagery, surreal elements, poetic narratives, reminiscent of the works of David Lynch."}
                ]
            }
        elif mode == 'descriptor':
            return {
                "messages": [
                    { "role": "system", "content": "You are a helpful assistant." },
                    { "role": "user", "content": "I say something like 'color' and you respond with a single prompt which I can use to build a prompt for an AI txt2image model. Your response to color would be something like 'red'. It is a very short description to add dynamic variety to the prompt."},
                    { "role": "assistant", "content": "Sure thing! Let's begin. What is your first prompt?" },
                    { "role": "user", "content": "color"},
                    { "role": "assistant", "content": "burnt sienna"},
                    { "role": "user", "content": "hair color"},
                    { "role": "assistant", "content": "platinum blonde"},
                    { "role": "user", "content": "metal"},
                    { "role": "assistant", "content": "rusted iron"},
                    { "role": "user", "content": "weather"},
                    { "role": "assistant", "content": "bright and sunny"},
                    { "role": "user", "content": "time of day"},
                    { "role": "assistant", "content": "crack of dawn"},
                    { "role": "user", "content": "man"},
                    { "role": "assistant", "content": "tall and slender man with wide shoulders in his 30s"},
                    { "role": "user", "content": "ethnicity"},
                    { "role": "assistant", "content": "Vietnamese"},
                    { "role": "user", "content": "occupation"},
                    { "role": "assistant", "content": "Heavy diesel mechanic"},
                    { "role": "user", "content": "art style"},
                    { "role": "assistant", "content": "crystal cubism"},
                    { "role": "user", "content": "artist"},
                    { "role": "assistant", "content": "Camille Pissarro"},
                    { "role": "user", "content": "movie director"},
                    { "role": "assistant", "content": "David Lynch"}
                ]
            }
        elif mode == 'character':
            return {
                "messages": [
                    { "role": "system", "content": "You are a helpful assistant." },
                    { "role": "user", "content": "When a user requests a character description, generate a detailed description of the character as you would expect from the writer George R. R. Martin. The description should be a suitable prompt based on the description that encapsulates the character's key visual elements for image creation using a txt2img model."},
                    { "role": "assistant", "content": "Sure thing! Let's begin. What is your first prompt?" },
                    { "role": "user", "content": "jolly cartoon octopus"},
                    { "role": "assistant", "content": "A cartoon octopus with a jolly demeanor in a sunshiny yellow and orange color scheme, wearing a small top hat and a polka-dotted bow tie, surrounded by intricate sculptures made of its own ink, in the midst of an undersea setting that hints at festivity and mirth."},
                    { "role": "user", "content": "a dapper young lady spy who wears suits"},
                    { "role": "assistant", "content": "A young 1920s lady spy with chestnut hair in a bob cut, piercing emerald eyes, wearing a tailored charcoal pinstripe suit with a white shirt and a silk tie, holding a silver cigarette case, exuding an aura of mystery and sophistication against a backdrop of Parisian nightlife."},
                    { "role": "user", "content": "a charming robot"},
                    { "role": "assistant", "content": "A charming robot with a sleek chrome body, art deco design elements, azure glowing eyes, and a gentle smile. Wearing a holographic bow tie and a vest painted to look like a dapper suit, engaged in performing a magic trick with a playful, inquisitive expression in an urban park setting."},
                    { "role": "user", "content": "cartoon ant"},
                    { "role": "assistant", "content": "A cartoon ant with a vibrant blue exoskeleton, oversized round eyes full of curiosity, wearing a leaf-green vest and a tiny fabric cap. Exhibiting an expression of wonder and excitement, amidst a backdrop of an underground ant colony bustling with activity."},
                    { "role": "user", "content": "a cyberpunk gnome"},
                    { "role": "assistant", "content": "A cyberpunk gnome with pale skin and cybernetic jade eyes, wearing a long tattered coat with circuitry patches and a sleek metallic pointed helmet. Surrounded by holographic screens and neon lights in a dim, cluttered workshop filled with futuristic gadgets and data screens."}
                ]
            }
        elif mode == 'custom':
            # open json file that is in the custom_history path
            try:
                history = json.load(open(custom_history))
                return history
            except:
                raise Exception('Error loading custom history file')

    def api_request(self, prompt, server_address, server_port, seed, mode, custom_history):
        # check if json file in root comfy directory called oooba.json
        history = self.history(mode, custom_history)
        if mode == 'prompt':
            prompt = f'{prompt}, describe in detail.'
        if mode == 'descriptor':
            # use seed to add a bit more randomness to the prompt
            spice = ['a', 'the', 'this', 'that',
                     'an exotic', 'an interesting', 'a colorful', 'a vibrant', 'get creative!', 'think outside the box!', 'a rare',
                     'a standard', 'a typical', 'a common', 'a normal', 'a regular', 'a usual', 'an ordinary',
                     'a unique', 'a one of a kind', 'a special', 'a distinctive', 'a peculiar', 'a remarkable', 'a noteworthy',
                     'popular in the victorian era', 'popular in the 1920s', 'popular in the 1950s', 'popular in the 1980s',
                     'popular in the 1990s', 'popular in the 2000s', 'popular in the 2010s', 'popular in the 2020s',
                     'popular in asia', 'popular in europe', 'popular in north america', 'popular in south america',
                     'popular in africa', 'popular in australia', 'popular in the middle east', 'popular in the pacific islands',
                     'popular with young people', 'popular with the elderly', 'trending on social media', 'popular on tiktok',
                     'trending on pinterest', 'popular on instagram', 'popular on facebook', 'popular on twitter',
                     'popular on reddit', 'popular on youtube', 'popular on tumblr', 'popular on snapchat',
                     'popular on linkedin', 'popular on twitch', 'popular on discord',
                     'unusual example of', 'a classic', 'an underrated', 'an innovative','a historical', 'a modern', 'a contemporary',
                     'a futuristic', 'a traditional', 'an eco-friendly', 'a controversial', 'a political', 'a religious',
                     'a spiritual', 'a philosophical', 'a scientific']
            random.seed(seed)
            prompt = f'{random.choice(spice)} {prompt}'
        """
        example curl request to LM Studio
        curl http://localhost:1234/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{
        "messages": [
            { "role": "system", "content": "Always answer in rhymes." },
            { "role": "user", "content": "Introduce yourself." }
        ],
        "temperature": 0.7,
        "max_tokens": -1,
        "stream": false
        }'
        """
        #prompt_prefix = "\\n<|user|>\\n"
        #prompt_suffix = "\\n<|assistant|>\\n"
        #prompt = prompt_prefix + prompt + prompt_suffix
        history['messages'].append({'role': 'user', 'content': prompt})
        request = {
            'messages': history['messages'],
            'temperature': 0.2,
            'top_p': 0.95,
            'presence_penalty': 0.0,
            'frequency_penalty': 0.0,
            'max_tokens': 1200,
            'stream': False,
            'seed': seed,
        }
        HOST = f'{server_address}:{server_port}'
        URI = f'http://{HOST}/v1/chat/completions'

        try:
            response = requests.post(URI, json=request, timeout=60)
        except requests.exceptions.ConnectionError:
            raise Exception('Are you running LM Studio with server running?')

        if response.status_code == 200:
            # response is in openai format
            result = response.json()['choices'][0]['message']['content']
            result = html.unescape(result)  # decode URL encoded special characters
            return result
        else:
            return 'Error'

    def process(self, input_prompt, mode, custom_history, server_address, server_port, seed, prompt=None, unique_id=None, extra_pnginfo=None):
        # search and replace
        input_prompt = search_and_replace(input_prompt, extra_pnginfo, prompt)
        # wildcard sytax is {like|this}
        # select a random word from the | separated list
        wc_re = re.compile(r'{([^}]+)}')
        def repl(m):
            return random.choice(m.group(1).split('|'))
        for m in wc_re.finditer(input_prompt):
            input_prompt = input_prompt.replace(m.group(0), repl(m))
        result = self.api_request(input_prompt, server_address, server_port, seed, mode, custom_history)
        prompt.get(str(unique_id))['inputs']['output_text'] = result
        return (result,)

class EvalFloats:
    # takes two float inputs and a text widget the user can type a formula for values a and b to calculate
    # then returns the result as the output
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'a': ('FLOAT', {'default': 0.0}),
                             'b': ('FLOAT', {'default': 0.0}),
                             'formula': ('STRING', {'multiline': False, 'default': 'a + b'})}}

    RETURN_TYPES = ('FLOAT',)
    RETURN_NAMES = ('result_float','result_int','result_str')
    FUNCTION = 'process'
    CATEGORY = 'Mikey/Math'

    def process(self, a, b, formula):
        # eval formula
        formula = formula.replace('a', str(a))
        formula = formula.replace('b', str(b))
        result = eval(formula)
        return (result, int(result), str(result))

class ImageOverlay:
    # overlay foreground image on top of background image
    # automatically fill or crop foreground image to match background image size
    # automatically resize foreground image to match background image size
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'background_image': ('IMAGE', {'default': None}),
                             'foreground_image': ('IMAGE', {'default': None}),
                             'opacity': ('FLOAT', {'default': 1.0, 'min': 0.0, 'max': 1.0, 'step': 0.01})}}

    RETURN_TYPES = ('IMAGE',)
    RETURN_NAMES = ('result_img',)
    FUNCTION = 'overlay'
    CATEGORY = 'Mikey/Image'

    def overlay(self, background_image, foreground_image, opacity):
        background_image = tensor2pil(background_image)
        foreground_image = tensor2pil(foreground_image)
        # Ensure images are in RGB mode and resize foreground to match background
        background_image = background_image.convert('RGB')
        foreground_image = foreground_image.convert('RGB')
        # create a cropped image from the foreground image with the same dimensions as the background image
        cropped_fg = Image.new('RGB', (background_image.size[0], background_image.size[1]))
        # paste the foreground image into the center of the cropped image
        cropped_fg.paste(foreground_image, (int((background_image.size[0] - foreground_image.size[0]) / 2), int((background_image.size[1] - foreground_image.size[1]) / 2)))

        # Convert images to NumPy arrays
        bg_array = np.array(background_image, dtype=np.float32) / 255
        fg_array = np.array(cropped_fg, dtype=np.float32) / 255

        ## Calculate Overlay blend
        mask = bg_array < 0.5
        overlay = np.zeros_like(bg_array)
        overlay[mask] = 2 * bg_array[mask] * fg_array[mask]
        overlay[~mask] = 1 - 2 * (1 - bg_array[~mask]) * (1 - fg_array[~mask])

        ## Apply opacity
        result = (1 - opacity) * bg_array + opacity * overlay

        ## Convert the result to uint8 and back to an Image
        result_img = Image.fromarray((result * 255).astype(np.uint8))
        result_img = pil2tensor(result_img)
        return result_img,

class CinematicLook:
    # combine function from ImageOverlay and HALDClut to create a cinematic look
    @classmethod
    def INPUT_TYPES(s):
        s.haldclut_files = read_cluts()
        s.file_names = [os.path.basename(f) for f in s.haldclut_files]
        return {'required': {'image': ('IMAGE', {'default': None}),
                             'look': (['modern','retro','clipped','broadcast','black and white','black and white - warm'],)}}

    RETURN_TYPES = ('IMAGE',)
    RETURN_NAMES = ('result_img',)
    FUNCTION = 'cinematic_look'
    CATEGORY = 'Mikey/Image'

    def apply_haldclut(self, image, hald_clut, gamma_correction):
        hald_img = Image.open(self.haldclut_files[self.file_names.index(hald_clut)])
        img = tensor2pil(image)
        if gamma_correction == 'True':
            corrected_img = gamma_correction_pil(img, 1.0/2.2)
        else:
            corrected_img = img
        filtered_image = apply_hald_clut(hald_img, corrected_img).convert("RGB")
        return filtered_image

    @apply_to_batch
    def cinematic_look(self, image, look):
        # load haldclut
        if look == 'modern':
            image = self.apply_haldclut(image, 'modern.png', 'False')
        elif look == 'retro':
            image = self.apply_haldclut(image, 'retro.png', 'False')
        elif look == 'clipped':
            image = self.apply_haldclut(image, 'clipped.png', 'False')
        elif look == 'broadcast':
            image = self.apply_haldclut(image, 'broadcast.png', 'False')
        elif look == 'black and white':
            image = self.apply_haldclut(image, 'bw.png', 'False')
        elif look == 'black and white - warm':
            image = self.apply_haldclut(image, 'bw_warm.png', 'False')
        p = os.path.dirname(os.path.realpath(__file__))
        if look in ['black and white',]:
            noise_img = os.path.join(p, 'noise_bw.png')
        else:
            noise_img = os.path.join(p, 'noise.png')
        # load noise image
        noise = Image.open(noise_img)
        IO = ImageOverlay()
        image = pil2tensor(image)
        noise = pil2tensor(noise)
        if look == 'modern':
            image = IO.overlay(image, noise, 0.3)[0]
        if look == 'retro':
            image = IO.overlay(image, noise, 0.4)[0]
        if look == 'clipped':
            image = IO.overlay(image, noise, 0.25)[0]
        if look == 'broadcast':
            image = IO.overlay(image, noise, 0.3)[0]
        if look == 'black and white':
            image = IO.overlay(image, noise, 0.25)[0]
        if look == 'black and white - warm':
            image = IO.overlay(image, noise, 0.25)[0]
        return image

    #def apply_cinematic_look(self, image, look):
    #    # image can be 1 or more images if batch size > 1
    #    images = []
    #    for img in image:
    #        images.append(self.cinematic_look(img, look))
    #    batch_tensor = torch.cat(images, dim=0)
    #    return (batch_tensor, )

NODE_CLASS_MAPPINGS = {
    'Wildcard Processor': WildcardProcessor,
    'Empty Latent Ratio Select SDXL': EmptyLatentRatioSelector,
    'Empty Latent Ratio Custom SDXL': EmptyLatentRatioCustom,
    'PresetRatioSelector': PresetRatioSelector,
    'Ratio Advanced': RatioAdvanced,
    'Int to String': INTtoSTRING,
    'Float to String': FLOATtoSTRING,
    'Range Float': RangeFloat,
    'Range Integer': RangeInteger,
    'Save Image With Prompt Data': SaveImagesMikey,
    'Save Images Mikey': SaveImagesMikeyML,
    'Save Images No Display': SaveImageNoDisplay,
    'Save Image If True': SaveImageIfTrue,
    'Resize Image for SDXL': ResizeImageSDXL,
    'Upscale Tile Calculator': UpscaleTileCalculator,
    'Batch Resize Image for SDXL': BatchResizeImageSDXL,
    'Batch Crop Image': BatchCropImage,
    'Batch Crop Resize Inplace': BatchCropResizeInplace,
    'Batch Load Images': BatchLoadImages,
    'Prompt With Style': PromptWithStyle,
    'Prompt With Style V2': PromptWithStyleV2,
    'Prompt With Style V3': PromptWithStyleV3,
    'LoraSyntaxProcessor': LoraSyntaxProcessor,
    'WildcardAndLoraSyntaxProcessor': WildcardAndLoraSyntaxProcessor,
    'Prompt With SDXL': PromptWithSDXL,
    'Style Conditioner': StyleConditioner,
    'Style Conditioner Base Only': StyleConditionerBaseOnly,
    'Mikey Sampler': MikeySampler,
    'MikeySamplerTiledAdvanced': MikeySamplerTiledAdvanced,
    'MikeySamplerTiledAdvancedBaseOnly': MikeySamplerTiledAdvancedBaseOnly,
    'Mikey Sampler Base Only': MikeySamplerBaseOnly,
    'Mikey Sampler Base Only Advanced': MikeySamplerBaseOnlyAdvanced,
    'Mikey Sampler Tiled': MikeySamplerTiled,
    'Mikey Sampler Tiled Base Only': MikeySamplerTiledBaseOnly,
    'AddMetaData': AddMetaData,
    'SaveMetaData': SaveMetaData,
    'SearchAndReplace': SearchAndReplace,
    'FileNamePrefix': FileNamePrefix,
    'HaldCLUT ': HaldCLUT,
    'Seed String': IntegerAndString,
    'Image Caption': ImageCaption,
    'ImageBorder': ImageBorder,
    'ImagePaste': ImagePaste,
    'TextCombinations': TextCombinations2,
    'TextCombinations3': TextCombinations3,
    'Text2InputOr3rdOption': Text2InputOr3rdOption,
    'Checkpoint Loader Simple Mikey': CheckpointLoaderSimpleMikey,
    'TextPreserve': TextPreserve,
    'OobaPrompt': OobaPrompt,
    'WildcardOobaPrompt': WildcardOobaPrompt,
    'LMStudioPrompt': LMStudioPrompt,
    'EvalFloats': EvalFloats,
    'ImageOverlay': ImageOverlay,
    'CinematicLook': CinematicLook
}

NODE_DISPLAY_NAME_MAPPINGS = {
    'Wildcard Processor': 'Wildcard Processor (Mikey)',
    'Empty Latent Ratio Select SDXL': 'Empty Latent Ratio Select SDXL (Mikey)',
    'Empty Latent Ratio Custom SDXL': 'Empty Latent Ratio Custom SDXL (Mikey)',
    'PresetRatioSelector': 'Preset Ratio Selector (Mikey)',
    'Ratio Advanced': 'Ratio Advanced (Mikey)',
    'Int to String': 'Int to String (Mikey)',
    'Float to String': 'Float to String (Mikey)',
    'Range Float': 'Range Float (Mikey)',
    'Range Integer': 'Range Integer (Mikey)',
    'Save Images With Prompt Data': 'Save Image With Prompt Data (Mikey)',
    'Save Images Mikey': 'Save Images Mikey (Mikey)',
    'Save Images No Display': 'Save Images No Display (Mikey)',
    'Save Image If True': 'Save Image If True (Mikey)',
    'Resize Image for SDXL': 'Resize Image for SDXL (Mikey)',
    'Batch Crop Image': 'Batch Crop Image (Mikey)',
    'Upscale Tile Calculator': 'Upscale Tile Calculator (Mikey)',
    'Batch Resize Image for SDXL': 'Batch Resize Image for SDXL (Mikey)',
    'Batch Crop Resize Inplace': 'Batch Crop Resize Inplace (Mikey)',
    'Batch Load Images': 'Batch Load Images (Mikey)',
    'Prompt With Style V3': 'Prompt With Style (Mikey)',
    'LoraSyntaxProcessor': 'Lora Syntax Processor (Mikey)',
    'WildcardAndLoraSyntaxProcessor': 'Wildcard And Lora Syntax Processor (Mikey)',
    'Prompt With Style': 'Prompt With Style V1 (Mikey)',
    'Prompt With Style V2': 'Prompt With Style V2 (Mikey)',
    'Prompt With SDXL': 'Prompt With SDXL (Mikey)',
    'Style Conditioner': 'Style Conditioner (Mikey)',
    'Style Conditioner Base Only': 'Style Conditioner Base Only (Mikey)',
    'Mikey Sampler': 'Mikey Sampler',
    'Mikey Sampler Base Only': 'Mikey Sampler Base Only',
    'Mikey Sampler Base Only Advanced': 'Mikey Sampler Base Only Advanced',
    'Mikey Sampler Tiled': 'Mikey Sampler Tiled',
    'MikeySamplerTiledAdvanced': 'Mikey Sampler Tiled Advanced',
    'MikeySamplerTiledAdvancedBaseOnly': 'Mikey Sampler Tiled Advanced Base Only',
    'Mikey Sampler Tiled Base Only': 'Mikey Sampler Tiled Base Only',
    'AddMetaData': 'AddMetaData (Mikey)',
    'SaveMetaData': 'SaveMetaData (Mikey)',
    'SearchAndReplace': 'Search And Replace (Mikey)',
    'FileNamePrefix': 'File Name Prefix (Mikey)',
    'HaldCLUT': 'HaldCLUT (Mikey)',
    'Seed String': 'Seed String (Mikey)',
    'Image Caption': 'Image Caption (Mikey)',
    'ImageBorder': 'Image Border (Mikey)',
    'ImagePaste': 'Image Paste (Mikey)',
    'TextCombinations': 'Text Combinations 2 (Mikey)',
    'TextCombinations3': 'Text Combinations 3 (Mikey)',
    'Text2InputOr3rdOption': 'Text 2 Inputs Or 3rd Option Instead (Mikey)',
    'Checkpoint Loader Simple Mikey': 'Checkpoint Loader Simple (Mikey)',
    'TextPreserve': 'Text Preserve (Mikey)',
    'OobaPrompt': 'OobaPrompt (Mikey)',
    'WildcardOobaPrompt': 'Wildcard OobaPrompt (Mikey)',
    'LMStudioPrompt': 'LM Studio Prompt (Mikey)',
    'EvalFloats': 'Eval Floats (Mikey)',
    'ImageOverlay': 'Image Overlay (Mikey)',
    'CinematicLook': 'Cinematic Look (Mikey)'
}
