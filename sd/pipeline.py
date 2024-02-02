import torch
import numpy as np
from tqdm import tqdm



WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8


def generate(
    prompt : str,
    uncond_prompt : str,    #Negative prompt
    input_image=None,       #image to image-to image generation
    strength=0.8,           #attention to starting image
    do_cfg=True,
    cfg_scale=7.5,          #weight of prompt(number from 1 to 14)
    sampler_name="ddpm",
    n_inference_steps=50,
    models={},              #pretrained model link
    seed=None,
    device=None,
    idle_device=None,
    tokenizer=None,
):
