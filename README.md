# Mikey Nodes

This repository contains custom nodes for ComfyUI.

## Prompt With Style
![image](https://github.com/bash-j/mikey_nodes/assets/3195567/0048faa8-5f46-4d92-8e38-e7ff603007da)

Example workflow in the prompt_with_styles.json file and mikey_node_workflow.json

This node allows you to prompt with a style and loras. You can add a lora with `<lora:lora_name>` or `<lora:lora_name:0.8>` syntax for a weight of 0.8 as an example.

You can add styles by using the `<style:style_name>` syntax.

Some of the styles built in are:
`SAI-Enhance, SAI-Anime, SAI-Photographic, SAI-Digital art, SAI-Comic book, SAI-Fantasy art, SAI-Analog film, SAI-Neonpunk, SAI-Isometric, SAI-Lowpoly, SAI-Origami, SAI-Line art, SAI-Craft clay, SAI-Cinematic, SAI-3d-model, SAI-pixel art, SAI-Texture, photographic, deep-field, analog-film, cinematic, red, documentary, nature-photography, editorial, polaroid, 3d-model, low-poly, modeling-compound, diorama, miniatures, abstract, anime, comic-book, digital-art, fantasy-art, dark-fantasy, oil-painting, watercolor, charcoal, color-pencil, crayon, cross-stitch, felt, origami, scrapbook, isometric, line-art, vector-art, neon-punk, pixel-art, tile-texture, lithography, mosaic, woodblock-print, sticker, stained-glass, tattoo, statue, album-art, wes-anderson, vaporwave, clarendon, gingham, juno, lark, nonagon, kaleidoscope, haunting, glam, mecha, padam, collage, paper-mache, macrame, quilling, pottery, batik, felting, marquetry, wildstyle-graffiti, bubble-graffiti, stencil-graffiti, throw-up-graffiti, tag-graffiti`

There is a V1 and V2 Prompt with Style node with different functionality, style selection instead of style syntax, no clip conditioning. Check them out if you like to break your workflow up into more nodes.
![image](https://github.com/bash-j/mikey_nodes/assets/3195567/03ca6183-0144-4533-a46a-c7accb8d9ec7)

### Wildcards

Wildcards are supported using the `__word__` syntax. Folder location is comfyui/wildcards You can select more than 1 line from the wildcard by using the syntax `[2$$__wildcard__]` for 2 lines in this example. You can also add a word to search for in the wildcard file e.g. `__wildcard|word__`

### Styles

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

### Image Size Ratios

You can also add your own custom ratios to a file named user_ratios.json in the root ComfyUI folder. You have to create this file yourself.

The format needs to look like the following example:

```json
{
    "ratios": {
        "test": {"width": 1024, "height":  1024}
    }
}
```

## Mikey Sampler
![image](https://github.com/bash-j/mikey_nodes/assets/3195567/beb24edb-4655-4d00-933a-e3fa2304aef2)

Example workflow: mikey_node_workflow.json

This is a sampler node as a companion to the Prompt with Style node, to allow for very straightforward image generation with SDXL.

The node uses a base -> refiner -> upscale with model -> base to produce the final image. It has an inbuilt image complexity detection function to make sure there aren't too many steps run on the final base sampler which can scramble faces and other simple areas on a large image. 

model_name is for the image uspcale model. These can be found at [OpenModelDB](https://openmodeldb.info/) and place them in the `ComfyUI\models\upscale_models` directory. I recommend lollipop as it provides a good balanced image that is not too sharp. If you want lots of sharp details, then try the NMKD Superscale or 4x-UltraSharp.

The seed is used by the samplers.

The upscale_by setting is how large you want the final image to be before it runs the last base sampler over the image. A value of 1 will output the original image size. 2 will be 2x the size of the original.

The highres_strength setting allows you to control this last stage and how much detail it adds. Setting of 1 is the sweet spot I found didn't scramble faces in a few test images. Dial it down if you are still getting scrambled images, or turn it up if you want to add more detail. 

## Style Conditioner
![image](https://github.com/bash-j/mikey_nodes/assets/3195567/b742b8a4-6ab3-4311-b278-db8bda66b5ce)

This chonky node is for injecting the style prompt into your conditioner by averaging the conditioner passed through the node.

A strength value of 0.5 would be even split between the conditioner passed in and the style selected.

## Empty Latent Ratio Select SDXL
![image](https://github.com/bash-j/mikey_nodes/assets/3195567/610dae5e-7c86-45fb-893d-06f7ccfe242e)

This node allows you to select from a range of different aspect ratios.

## Empty Latent Ratio Custom SDXL
![image](https://github.com/bash-j/mikey_nodes/assets/3195567/784d7b92-dfc7-4f79-9c81-272605ab7934)

This node allows you to enter your own aspect ratio or image size from wherever, and it will make it fit under 1024x1024 resolution.

## Resize Image for SDXL
![image](https://github.com/bash-j/mikey_nodes/assets/3195567/9febf7d4-31dd-4078-9ebf-03d3236f7291)

This node allows you to resize an image to fit into 1024x1024 resolution. Good for Img2Img workflows.

## Batch Resize Image for SDXL
![image](https://github.com/bash-j/mikey_nodes/assets/3195567/fb5ee831-9895-4901-91fa-fd0771f91a1d)

Given a path to a folder containing images, it will resize all images in the folder to fit the 1024^2 resolution and feed into the workflow. Careful of folders with lots of images!

## Save Image With Prompt Data

This node allows you to save an image with the prompt data in the filename.

The filename will start with a datestamp, then part of the positive prompt.

It will also save the positive prompt and negative prompt to the png data.

## Upscale Tile Calculator
![image](https://github.com/bash-j/mikey_nodes/assets/3195567/6f550f38-3e6a-4659-87fb-92deb5d98460)

Made for the [Ultimate SD Upscaler](https://github.com/ssitu/ComfyUI_UltimateSDUpscale) to help calcuate tile size to evenly fit inside the larger image and not get weird edges.

The resolution is what is suitable for the model you are using, 1.5: 512, 2.1: 768, SDXL Base: 1024, SDXL Refiner: 768

## Wildcard Processor
![image](https://github.com/bash-j/mikey_nodes/assets/3195567/ce62e6d7-1982-4c53-bde1-7569c94322e3)

This node is for building a string using your wildcard files.

## Add Metadata

![image](https://github.com/bash-j/mikey_nodes/assets/3195567/dbe07727-3b4e-473a-bdde-eb0d8ccb7e64)

I want to give a shout-out to masslevel on discord for the great idea to have nodes for metadata.

This node lets you add more metadata to the image that is passed through this node in your workflow. The metadata should be written to the image file in the Save Image node.

The label is the key or label you want to give to the metadata, and the text box is for entering a value. Convert these to inputs if you want to pipe in string values from other nodes.

## Save Metadata

This node should pick up the info stored in the prompt and extra_pnginfo data and save it to a txt file

## HaldCLUT

This will apply a HaldCLUT to an image to change the colors, which tend to imitate the look of the film or filter. I have included some in this package, but you can find more png files at [rawtherapee.com](http://rawtherapee.com/shared/HaldCLUT.zip)

## VAE Decode 6GB (deprecated)

This node is a bandaid fix for Mikey's 3060 6GB graphics card to keep VRAM usage below 6GB.

You shouldn't need this anymore since there was an update to comfyui to fix the issue.

## Installation

To use these nodes, simply open a terminal in ComfyUI/custom_nodes/ and run:

`git clone https://github.com/bash-j/mikey_nodes`

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
