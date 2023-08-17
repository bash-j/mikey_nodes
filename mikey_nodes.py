import datetime
from fractions import Fraction
import importlib.util
import json
from math import ceil, pow, gcd
import os
import random
import re
import sys

import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFilter
from PIL.PngImagePlugin import PngInfo
import torch
import torch.nn.functional as F

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
from comfy.model_management import soft_empty_cache
from nodes import LoraLoader, ConditioningAverage, common_ksampler, ImageScale, VAEEncode, VAEDecode
import comfy.utils
from comfy_extras.chainner_models import model_loading
from comfy import model_management

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
            stripped_text = stripped_text.replace(f'<lora:{lora_filename}:{lora_multiplier}>', '')
    return model, clip, stripped_text

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
            for f in files:
                if filename_separator in f:
                    counter = max(counter, int(f.split(filename_separator)[counter_pos]) + 1)
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
        # first process wildcards
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
        pixels = vaedecoder.decode(vae, sample2)[0]
        org_width, org_height = pixels.shape[2], pixels.shape[1]
        img = iuwm.upscale(upscale_model, image=pixels)[0]
        upscaled_width, upscaled_height = int(org_width * upscale_by // 8 * 8), int(org_height * upscale_by // 8 * 8)
        img = image_scaler.upscale(img, 'nearest-exact', upscaled_width, upscaled_height, 'center')[0]
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

    def divide_into_tiles_with_padding(self, image, tile_width, tile_height, padding=64):
        tiles = []
        positions = []

        width, height = image.size

        width_overflow = width % tile_width
        height_overflow = height % tile_height

        width_adjustment = width_overflow // (width // tile_width)
        height_adjustment = height_overflow // (height // tile_height)

        x_adjusted, y_adjusted = 0, 0

        for y in range(0, height, tile_height):
            x_adjusted = 0
            if y_adjusted < height_overflow:
                tile_height_adjusted = tile_height + height_adjustment
                y_adjusted += 1
            else:
                tile_height_adjusted = tile_height

            for x in range(0, width, tile_width):
                # Determine the adjustment based on the current iteration
                if x_adjusted < width_overflow:
                    tile_width_adjusted = tile_width + width_adjustment
                    x_adjusted += 1
                else:
                    tile_width_adjusted = tile_width

                # Define box with selective padding
                left_padding = padding if x != 0 else 0
                upper_padding = padding if y != 0 else 0
                right_padding = padding if x + tile_width_adjusted < width else 0
                lower_padding = padding if y + tile_height_adjusted < height else 0

                left = max(0, x - left_padding)
                upper = max(0, y - upper_padding)
                right = min(width, x + tile_width_adjusted + right_padding)
                lower = min(height, y + tile_height_adjusted + lower_padding)

                tile = image.crop((left, upper, right, lower))

                # Resize the cropped tile to maintain uniform tile dimensions
                new_width = tile_width + left_padding + right_padding
                new_height = tile_height + upper_padding + lower_padding
                tile = tile.resize((new_width, new_height))

                tiles.append(tile)
                positions.append((x, y))

        return tiles, positions

    def divide_into_tiles_with_offset(self, image, tile_width, tile_height, padding=64, offset=None):
        tiles = []
        positions = []

        width, height = image.size

        # If offset isn't given, just use the tile width/height as usual (i.e., no overlap).
        if offset is None:
            offset = tile_width  # For the x axis
            offset_y = tile_height  # For the y axis
        else:
            offset_y = offset  # If offset is given, use it for both axes

        for y in range(0, height - tile_height + 1, offset_y):  # Subtract tile height to ensure last tile doesn't exceed image bounds
            for x in range(0, width - tile_width + 1, offset):  # Similarly subtract tile width here
                left_padding = padding if x != 0 else 0
                upper_padding = padding if y != 0 else 0
                right_padding = padding if x + tile_width < width else 0
                lower_padding = padding if y + tile_height < height else 0

                left = max(0, x - left_padding)
                upper = max(0, y - upper_padding)
                right = min(width, x + tile_width + right_padding)
                lower = min(height, y + tile_height + lower_padding)

                tile = image.crop((left, upper, right, lower))

                new_width = tile_width + left_padding + right_padding
                new_height = tile_height + upper_padding + lower_padding
                tile = tile.resize((new_width, new_height))

                tiles.append(tile)
                positions.append((x, y))

        return tiles, positions

    def crop_tile_with_padding(self, base_image, tile, position, padding=64):
        # can't crop off every side or you will end up with a smaller tile than you started with
        # padding is not added to every side in the first place
        x, y = position
        left_padding = padding if x != 0 else 0
        upper_padding = padding if y != 0 else 0
        right_padding = padding if x + tile.width > base_image.width else 0
        lower_padding = padding if y + tile.height > base_image.height else 0

        cropped_tile = tile.crop((left_padding, upper_padding, tile.width - right_padding, tile.height - lower_padding))
        return cropped_tile

    def feather_padded_tile(self, base_image, tile, position, padding=64, width=16):
        x, y = position

        # Check for each side if it should be feathered
        left_feather = x != 0
        right_feather = x + tile.width - padding * 2 < base_image.width
        top_feather = y != 0
        bottom_feather = y + tile.height - padding * 2 < base_image.height

        tile = tile.convert("RGBA")
        mask = Image.new('L', tile.size, 255)
        draw = ImageDraw.Draw(mask)

        # Horizontal gradient
        for x in range(width):
            gradient_value = int(255 * (x / width))
            if left_feather:
                draw.line([(x, 0), (x, tile.height)], fill=gradient_value)
            if right_feather:
                draw.line([(tile.width - x - 1, 0), (tile.width - x - 1, tile.height)], fill=gradient_value)

        # Vertical gradient
        for y in range(width):
            gradient_value = int(255 * (y / width))
            if top_feather:
                draw.line([(0, y), (tile.width, y)], fill=gradient_value)
            if bottom_feather:
                draw.line([(0, tile.height - y - 1), (tile.width, tile.height - y - 1)], fill=gradient_value)

        tile.putalpha(mask)
        return tile

    def overlay_tiles(self, base_image, tile, position, padding=64, feathering_width=16):
        """
        Overlays a tile on top of a base image.
        The function assumes PIL.Image objects.
        """
        x, y = position

        # Define crop boundaries based on the position of the tile.
        left_padding = padding if x != 0 else 0
        upper_padding = padding if y != 0 else 0
        right_padding = padding if x + tile.width > base_image.width else 0
        lower_padding = padding if y + tile.height > base_image.height else 0

        cropped_tile = tile.crop((left_padding, upper_padding, tile.width - right_padding, tile.height - lower_padding))
        # feather cropped tile
        cropped_tile = self.feather_padded_tile(base_image, cropped_tile, position, padding=padding, width=feathering_width)
        # paste cropped tile that used to be padded onto base image
        base_image.paste(cropped_tile, position, cropped_tile)
        return base_image

    def overlay_offset_tiles(self, base_image, tiles, positions, padding=64, feathering_width=32):
        """
        Overlays a list of tiles on top of a base image.
        Assumes tiles have an offset and can overlap.
        The function assumes PIL.Image objects.
        """
        for tile, position in zip(tiles, positions):
            # Process each tile as before
            cropped_tile = self.crop_tile_with_padding(base_image, tile, position, padding=padding)
            feathered_tile = self.feather_padded_tile(base_image, cropped_tile, position, padding=padding, width=feathering_width)

            # Paste feathered tile onto the base image
            base_image.paste(feathered_tile, position, feathered_tile)

        return base_image

    def phase_one(self, base_model, refiner_model, samples, positive_cond_base, negative_cond_base,
                  positive_cond_refiner, negative_cond_refiner, upscale_by, model_name, seed, vae):
        image_scaler = ImageScale()
        vaedecoder = VAEDecode()
        uml = UpscaleModelLoader()
        upscale_model = uml.load_model(model_name)[0]
        iuwm = ImageUpscaleWithModel()
        # step 1 run base model
        sample1 = common_ksampler(base_model, seed, 25, 6.5, 'dpmpp_2s_ancestral', 'simple', positive_cond_base, negative_cond_base, samples,
                                  start_step=0, last_step=18, force_full_denoise=False)[0]
        # step 2 run refiner model
        sample2 = common_ksampler(refiner_model, seed, 30, 3.5, 'dpmpp_2m', 'simple', positive_cond_refiner, negative_cond_refiner, sample1,
                                  disable_noise=True, start_step=21, force_full_denoise=True)[0]
        # step 3 upscale image using a simple AI image upscaler
        pixels = vaedecoder.decode(vae, sample2)[0]
        org_width, org_height = pixels.shape[2], pixels.shape[1]
        img = iuwm.upscale(upscale_model, image=pixels)[0]
        upscaled_width, upscaled_height = int(org_width * upscale_by // 8 * 8), int(org_height * upscale_by // 8 * 8)
        img = image_scaler.upscale(img, 'nearest-exact', upscaled_width, upscaled_height, 'center')[0]
        return img, upscaled_width, upscaled_height

    def tiler(self, base_model, refiner_model, vae, img, positive_cond_base, negative_cond_base,
              positive_cond_refiner, negative_cond_refiner, seed, upscaled_width, upscaled_height,
              tiler_denoise, tiler_model, tiler_mode='padding', offset_amount=1.3):
        vaeencoder = VAEEncode()
        vaedecoder = VAEDecode()
        # Tiled upscaler logic  (more advanced upscaling method)
        pil_img = tensor2pil(img)
        tile_width, tile_height = find_tile_dimensions(upscaled_width, upscaled_height, 1.0, 1024)
        if tiler_mode == 'padding':
            tiles, positions = self.divide_into_tiles_with_padding(pil_img, tile_width, tile_height, 64)
        else:
            tiles, positions = self.divide_into_tiles_with_offset(pil_img, tile_width, tile_height, 64, offset=int(tile_width // offset_amount))
        # Phase 1: Encoding the tiles
        latent_tiles = []
        for tile in tiles:
            tile_img = pil2tensor(tile)
            tile_latent = vaeencoder.encode(vae, tile_img)[0]
            latent_tiles.append(tile_latent)
        # Phase 2: Sampling using the encoded latents
        start_step = int(20 - (20 * tiler_denoise))
        resampled_tiles = []
        if tiler_model == 'base':
            for tile_latent in latent_tiles:
                tile_resampled = common_ksampler(base_model, seed, 20, 7, 'dpmpp_2m_sde', 'karras',
                                                positive_cond_base, negative_cond_base, tile_latent,
                                                start_step=start_step, force_full_denoise=True)[0]
                resampled_tiles.append(tile_resampled)
        else:
            for tile_latent in latent_tiles:
                tile_resampled = common_ksampler(refiner_model, seed, 20, 7, 'dpmpp_2m_sde', 'karras',
                                                positive_cond_refiner, negative_cond_refiner, tile_latent,
                                                start_step=start_step, force_full_denoise=True)[0]
                resampled_tiles.append(tile_resampled)
        # Phase 3: Decoding the sampled tiles and feathering
        processed_tiles = []
        for tile_resampled, original_tile, position in zip(resampled_tiles, tiles, positions):
            # Decode the tile
            tile_img = vaedecoder.decode(vae, tile_resampled)[0]
            tile_pil = tensor2pil(tile_img)
            # Histogram match with original tile
            matched_tile = match_histograms(tile_pil, original_tile)
            processed_tiles.append(matched_tile)
        # stitch the tiles back together with overlay
        #white_img = Image.new('RGB', (upscaled_width, upscaled_height), (255, 255, 255))
        if tiler_mode == 'padding':
            final_image = pil_img
            for tile, position in zip(processed_tiles, positions):
                final_image = self.overlay_tiles(final_image, tile, position, 64)
            # second pass
            final_image = match_histograms(pil_img, final_image)
            for tile, position in zip(processed_tiles, positions):
                final_image = self.overlay_tiles(final_image, tile, position, 64)
            final_image = pil2tensor(final_image)
        else:
            final_image = pil_img
            final_image = self.overlay_offset_tiles(final_image, processed_tiles, positions)
            # second pass
            final_image = match_histograms(pil_img, final_image)
            final_image = self.overlay_offset_tiles(final_image, processed_tiles, positions)
            final_image = pil2tensor(final_image)
        return final_image

    def run(self, seed, base_model, refiner_model, vae, samples, positive_cond_base, negative_cond_base,
            positive_cond_refiner, negative_cond_refiner, model_name, upscale_by=1.0, tiler_denoise=0.25,
            upscale_method='normal', tiler_model='base'):
        # phase 1: run base, refiner, then upscaler model
        img, upscaled_width, upscaled_height = self.phase_one(base_model, refiner_model, samples, positive_cond_base, negative_cond_base,
                                                              positive_cond_refiner, negative_cond_refiner, upscale_by, model_name, seed, vae)
        # phase 2: run tiler
        tiled_image = self.tiler(base_model, refiner_model, vae, img, positive_cond_base, negative_cond_base,
                                 positive_cond_refiner, negative_cond_refiner, seed, upscaled_width, upscaled_height,
                                 tiler_denoise, tiler_model, tiler_mode='offset', offset_amount=1)
        tiled_image = self.tiler(base_model, refiner_model, vae, tiled_image, positive_cond_base, negative_cond_base,
                                 positive_cond_refiner, negative_cond_refiner, seed, upscaled_width, upscaled_height,
                                 .4, tiler_model, tiler_mode='offset', offset_amount=2)
        tiled_image = self.tiler(base_model, refiner_model, vae, tiled_image, positive_cond_base, negative_cond_base,
                                 positive_cond_refiner, negative_cond_refiner, seed, upscaled_width, upscaled_height,
                                 .2, tiler_model, tiler_mode='offset', offset_amount=1)
        return (tiled_image, img)

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
    CATEGORY = 'Mikey'

    def output(self, seed):
        seed_string = str(seed)
        return (seed, seed_string,)

NODE_CLASS_MAPPINGS = {
    'Wildcard Processor': WildcardProcessor,
    'Empty Latent Ratio Select SDXL': EmptyLatentRatioSelector,
    'Empty Latent Ratio Custom SDXL': EmptyLatentRatioCustom,
    'Save Image With Prompt Data': SaveImagesMikey,
    'Save Images Mikey': SaveImagesMikeyML,
    'Resize Image for SDXL': ResizeImageSDXL,
    'Upscale Tile Calculator': UpscaleTileCalculator,
    'Batch Resize Image for SDXL': BatchResizeImageSDXL,
    'Batch Crop Image': BatchCropImage,
    'Prompt With Style': PromptWithStyle,
    'Prompt With Style V2': PromptWithStyleV2,
    'Prompt With Style V3': PromptWithStyleV3,
    'Prompt With SDXL': PromptWithSDXL,
    'Style Conditioner': StyleConditioner,
    'Mikey Sampler': MikeySampler,
    'Mikey Sampler Tiled': MikeySamplerTiled,
    'AddMetaData': AddMetaData,
    'SaveMetaData': SaveMetaData,
    'HaldCLUT ': HaldCLUT,
    'Seed String': IntegerAndString,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    'Wildcard Processor': 'Wildcard Processor (Mikey)',
    'Empty Latent Ratio Select SDXL': 'Empty Latent Ratio Select SDXL (Mikey)',
    'Empty Latent Ratio Custom SDXL': 'Empty Latent Ratio Custom SDXL (Mikey)',
    'Save Images With Prompt Data': 'Save Image With Prompt Data (Mikey)',
    'Save Images Mikey': 'Save Images Mikey (Mikey)',
    'Resize Image for SDXL': 'Resize Image for SDXL (Mikey)',
    'Batch Crop Image': 'Batch Crop Image (Mikey)',
    'Upscale Tile Calculator': 'Upscale Tile Calculator (Mikey)',
    'Batch Resize Image for SDXL': 'Batch Resize Image for SDXL (Mikey)',
    'Prompt With Style V3': 'Prompt With Style (Mikey)',
    'Prompt With Style': 'Prompt With Style V1 (Mikey)',
    'Prompt With Style V2': 'Prompt With Style V2 (Mikey)',
    'Prompt With SDXL': 'Prompt With SDXL (Mikey)',
    'Style Conditioner': 'Style Conditioner (Mikey)',
    'Mikey Sampler': 'Mikey Sampler',
    'Mikey Sampler Tiled': 'Mikey Sampler Tiled',
    'AddMetaData': 'AddMetaData (Mikey)',
    'SaveMetaData': 'SaveMetaData (Mikey)',
    'HaldCLUT': 'HaldCLUT (Mikey)',
    'Seed String': 'Seed String (Mikey)',
}
