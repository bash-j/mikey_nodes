# Mikey Nodes

This repository contains custom nodes for ComfyUI.

## Prompt With Style

This node allows you to prompt with a style. You can also select the aspect ratio.

There are outputs for image size and prompts for the clip conditioners.

Example workflow in the prompt_with_styles.json file.

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
