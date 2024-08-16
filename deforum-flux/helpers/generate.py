# Standard library imports
import os

# Related third-party imports
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext
from einops import rearrange, repeat
from .load_images import load_img, prepare_mask, prepare_overlay_mask

import argparse
import sys
import os
import re
import time
from glob import iglob
from io import BytesIO

import torch
from dataclasses import dataclass

from einops import rearrange
from PIL import ExifTags, Image
from torchvision import transforms
from transformers import pipeline
from src.flux.sampling import (
    denoise, 
    get_noise, 
    get_schedule, 
    prepare, 
    unpack
)
from src.flux.util import (
    configs, 
    load_ae, 
    load_clip,
    load_flow_model, 
    load_t5
)




def uint_number(datum, number):
    if number == 8:
        datum = Image.fromarray(datum.astype(np.uint8))
    elif number == 32:
        datum = datum.astype(np.float32)
    else:
        datum = datum.astype(np.uint16)
    return datum

def add_noise(sample: torch.Tensor, noise_amt: float) -> torch.Tensor:
    return sample + torch.randn(sample.shape, device=sample.device) * noise_amt

def generate(args, root, frame=0, return_latent=False, return_sample=False, return_c=False):
    seed_everything(args.seed)
    os.makedirs(args.outdir, exist_ok=True)


    batch_size = args.n_samples
    precision_scope = autocast if args.precision == "autocast" else nullcontext
    torch_device = torch.device(root.device) if isinstance(root.device, str) else root.device
    
    if batch_size > 1: 
        NotImplemented()
        
    # cond prompts
    cond_prompt = args.cond_prompt
    assert cond_prompt is not None
    cond_data = [batch_size * [cond_prompt]]

    # uncond prompts
    uncond_prompt = args.uncond_prompt
    assert uncond_prompt is not None
    uncond_data = [batch_size * [uncond_prompt]]
    
    init_latent = None
    mask_image = None
    init_image = None
    
    if args.init_latent is not None:
        init_latent = args.init_latent.to(torch.bfloat16)
    elif args.init_sample is not None:
        # with precision_scope(torch_device):
        init_latent = root.model.ae.encode(args.init_sample.to(torch.float32))
    elif args.use_init and args.init_image != None and args.init_image != '':
        
        init_image, mask_image = load_img(
            args.init_image, 
            shape=(args.W, args.H),  
            use_alpha_as_mask=args.use_alpha_as_mask
        )
        if args.add_init_noise:
            init_image = add_noise(init_image,args.init_noise)
            
        init_image = init_image.to(torch_device)
        init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
        # with precision_scope(torch_device):
        init_latent = root.model.ae.encode(init_image.to(torch.float32))

    if not args.use_init and args.strength > 0 and args.strength_0_no_init:
        #print("\nNo init image, but strength > 0. Strength has been auto set to 0, since use_init is False.")
        #print("If you want to force strength > 0 with no init, please set strength_0_no_init to False.\n")
        args.strength = 0

    results = []
    with torch.no_grad():
        # with precision_scope("cuda"):

        for cond_prompts, uncond_prompts in zip(cond_data,uncond_data):
            
            if isinstance(cond_prompts, tuple):
                cond_prompts = list(cond_prompts)
            if isinstance(uncond_prompts, tuple):
                uncond_prompts = list(uncond_prompts)

            x = get_noise(
                1,
                args.H,
                args.W, 
                device=torch_device,
                dtype=torch.bfloat16,
                seed=args.seed,
            )

            # divide pixel space by 16**2 to acocunt for latent space conversion
            timesteps = get_schedule(
                args.steps,
                (x.shape[-1] * x.shape[-2]) // 4,
                # shift=(not is_schnell),
                shift=True,
            )

            if isinstance(init_latent, torch.Tensor):
                t_idx = int((1 - args.strength) * args.steps)
                t = timesteps[t_idx]
                timesteps = timesteps[t_idx:]
                x = t * x + (1.0 - t) * init_latent.to(x.dtype)

            
            if args.init_c is None: 
                inp = prepare(t5=root.model.t5, clip=root.model.clip, img=x, prompt=cond_prompts[0])
            else: 
                inp = args.init_c
                
            # denoise initial noise
            samples = denoise(root.model.dit, **inp, timesteps=timesteps, guidance=args.scale).to(torch.float32)

            # decode latents to pixel space
            samples = unpack(samples, args.H, args.W)

            if return_latent:
                results.append(samples.clone())

            # with torch.autocast(device_type=torch_device.type, dtype=torch.bfloat16):
            x_samples = root.model.ae.decode(samples.to(torch.float32))


            if args.use_mask and args.overlay_mask:
                # Overlay the masked image after the image is generated
                if args.init_sample_raw is not None:
                    img_original = args.init_sample_raw
                elif init_image is not None:
                    img_original = init_image
                else:
                    raise Exception("Cannot overlay the masked image without an init image to overlay")

                if args.mask_sample is None or args.using_vid_init:
                    args.mask_sample = prepare_overlay_mask(args, root, img_original.shape)

                x_samples = img_original * args.mask_sample + x_samples * ((args.mask_sample * -1.0) + 1)

            if return_sample:
                results.append(x_samples.clone())

            if return_c:
                results.append(inp)

            if args.bit_depth_output == 8:
                exponent_for_rearrange = 1
            elif args.bit_depth_output == 32:
                exponent_for_rearrange = 0
            else:
                exponent_for_rearrange = 2

            # bring into np format 
            x_samples = x_samples.clamp(-1, 1)
            x_samples = rearrange(x_samples[0], "c h w -> h w c")

            image = (127.5 * (x_samples + 1.0)).cpu().byte().numpy()
            image = uint_number(image, args.bit_depth_output)
            results.append(image)
    return results
