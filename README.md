# Mikey Nodes

This repository contains custom nodes for ComfyUI.

## Prompt With Style V3

![image](https://github.com/bash-j/mikey_nodes/assets/3195567/b39f6693-d0c4-479d-9c8f-530dfb67e7e4)

This node allows you to prompt with a style and loras. You can add a lora with `<lora:lora_name>` or `<lora:lora_name:0.8>` syntax for a weight of 0.8 as an example.

You can add styles by using the `<style:style_name>` syntax.

Some of the styles built in are:
`SAI-Enhance, SAI-Anime, SAI-Photographic, SAI-Digital art, SAI-Comic book, SAI-Fantasy art, SAI-Analog film, SAI-Neonpunk, SAI-Isometric, SAI-Lowpoly, SAI-Origami, SAI-Line art, SAI-Craft clay, SAI-Cinematic, SAI-3d-model, SAI-pixel art, SAI-Texture, photographic, deep-field, analog-film, cinematic, red, documentary, nature-photography, editorial, polaroid, 3d-model, low-poly, modeling-compound, diorama, miniatures, abstract, anime, comic-book, digital-art, fantasy-art, dark-fantasy, oil-painting, watercolor, charcoal, color-pencil, crayon, cross-stitch, felt, origami, scrapbook, isometric, line-art, vector-art, neon-punk, pixel-art, tile-texture, lithography, mosaic, woodblock-print, sticker, stained-glass, tattoo, statue, album-art, wes-anderson, vaporwave, clarendon, gingham, juno, lark, nonagon, kaleidoscope, haunting, glam, mecha, padam, collage, paper-mache, macrame, quilling, pottery, batik, felting, marquetry, wildstyle-graffiti, bubble-graffiti, stencil-graffiti, throw-up-graffiti, tag-graffiti`

Wildcards are supported using the `__word__` syntax. Folder location is comfyui/wildcards You can select more than 1 line from the wildcard by using the syntax `[2$$__wildcard__]` for 2 lines in this example.

There are outputs for image size and prompts for the clip conditioners.

Example workflow in the prompt_with_styles.json file.

You can add your own styles to a file named user_styles.json in the root ComfyUI folder. You have to create this file yourself.

The format needs to look like the following example:

```json
{
    "styles": {
        "newspaper": {
        "positive": "newspaper comic strip, panels, black and white",
        "negative": "photograph, color"
        }
    }
}
```

You can also add your own custom ratios to a file named user_ratios.json in the root ComfyUI folder. You have to create this file yourself.

The format needs to look like the following example:

```json
{
    "ratios": {
        "test": {"width": 1024, "height":  1024}
    }
}
```

## Empty Latent Ratio Select SDXL

This node allows you to select from a range of different aspect ratios.

## Empty Latent Ratio Custom SDXL

This node allows you to enter your own aspect ratio or image size from wherever, and it will make it fit under 1024x1024 resolution.

## Resize Image for SDXL

This node allows you to resize an image to fit into 1024x1024 resolution. Good for Img2Img workflows.

## Save Image With Prompt Data

This node allows you to save an image with the prompt data in the filename.

The filename will start with a datestamp, then part of the positive prompt.

It will also save the positive prompt and negative prompt to the png data.

## VAE Decode 6GB (deprecated)

This node is a bandaid fix for Mikey's 3060 6GB graphics card to keep VRAM usage below 6GB.

You shouldn't need this anymore since there was an update to comfyui to fix the issue.

## Installation

To use these nodes, simply open a terminal in ComfyUI/custom_nodes/ and run:

`git clone https://github.com/bash-j/mikey_nodes`

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
