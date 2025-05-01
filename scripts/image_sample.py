"""
Generate a large batch of image samples from a model and save them as individual image files.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch as th
import torch.distributed as dist
from PIL import Image
from tqdm import tqdm

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(logger.get_dir(), "image_samples")
    
    if dist.get_rank() == 0:
        os.makedirs(output_dir, exist_ok=True)
        logger.log(f"saving images to {output_dir}")

    logger.log("sampling...")
    all_images = []
    all_labels = []
    sample_count = 0

    with tqdm(total=args.num_samples, desc="Generating samples", disable=dist.get_rank() != 0) as pbar:
        while sample_count < args.num_samples:
            model_kwargs = {}
            if args.class_cond:
                classes = th.randint(
                    low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
                )
                model_kwargs["y"] = classes
            
            sample_fn = (
                diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
            )
            
            sample = sample_fn(
                model,
                (args.batch_size, 3, args.image_size, args.image_size),
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                progress=True,
            )
            
            sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
            sample = sample.permute(0, 2, 3, 1)
            sample = sample.contiguous()

            gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_samples, sample)
            
            gathered_labels = None
            if args.class_cond:
                gathered_labels = [th.zeros_like(classes) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_labels, classes)

            if dist.get_rank() == 0:
                images_to_process = 0
                for i, batch_images in enumerate(gathered_samples):
                    batch_numpy = batch_images.cpu().numpy()
                    batch_labels = None
                    if args.class_cond and gathered_labels:
                        batch_labels = gathered_labels[i].cpu().numpy()
                    
                    for j, image_array in enumerate(batch_numpy):
                        if sample_count >= args.num_samples:
                            break

                        img = Image.fromarray(image_array)
                        
                        if args.class_cond and batch_labels is not None:
                            img_class = batch_labels[j]
                            filename = f"sample_{sample_count:05d}_class_{img_class}.png"
                        else:
                            filename = f"sample_{sample_count:05d}.png"
                        
                        img_path = os.path.join(output_dir, filename)
                        img.save(img_path)
                        
                        sample_count += 1
                        images_to_process += 1
                    
                pbar.update(images_to_process)
        
    if args.save_npz and dist.get_rank() == 0:
        all_images = []
        all_labels = []
        
        for i in range(min(args.num_samples, sample_count)):
            if args.class_cond:
                for class_id in range(NUM_CLASSES):
                    path = os.path.join(output_dir, f"sample_{i:05d}_class_{class_id}.png")
                    if os.path.exists(path):
                        img = np.array(Image.open(path))
                        all_images.append(img)
                        all_labels.append(class_id)
                        break
            else:
                path = os.path.join(output_dir, f"sample_{i:05d}.png")
                if os.path.exists(path):
                    img = np.array(Image.open(path))
                    all_images.append(img)
        
        if all_images:
            arr = np.stack(all_images, axis=0)
            shape_str = "x".join([str(x) for x in arr.shape])
            out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
            logger.log(f"saving to {out_path}")
            
            if args.class_cond and all_labels:
                label_arr = np.array(all_labels)
                np.savez(out_path, arr, label_arr)
            else:
                np.savez(out_path, arr)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=30000,
        batch_size=8,
        image_size=128,
        use_ddim=False,
        model_path="",
        output_dir="./image_samples",
        save_npz=False,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()