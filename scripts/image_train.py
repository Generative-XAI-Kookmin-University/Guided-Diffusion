"""
Train a diffusion model on images.
"""
import torch
import argparse
import datetime

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop
from guided_diffusion.flaw_highlighter import FlawHighlighter
import wandb


def main():
    args = create_argparser().parse_args()
    wandb.init(project="guided-diffusion", config=args_to_dict(args, model_and_diffusion_defaults().keys()))

    dist_util.setup_dist()
    current_date = datetime.datetime.now().strftime("%y%m%d")
    log_dir = f"logs-{current_date}"
    logger.configure(log_dir)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    params = {
    'nc' : 3,
    'ndf' : 32, 
    }
    flaw_highlighter = FlawHighlighter(params)
    if args.fh_ckpt_path:
        print('loading flaw highlighter checkpoint...')
        fh_ckpt = torch.load(args.fh_ckpt_path)
        flaw_highlighter.load_state_dict(fh_ckpt['model_state_dict'])

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        image_size=args.image_size,
        first_process=args.first_process,
        FH = flaw_highlighter,
        fam_cycle = args.fam_cycle,
        fam_noise_w=args.fam_noise_w,
        fam_attn_w=args.fam_attn_w
    ).run_loop(num_iterations=args.num_iterations)


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=8,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10000,
        save_interval=50000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        num_iterations=500000,
        first_process=250000,
        fam_cycle=100,
        fam_noise_w=0.01,
        fam_attn_w=0.025,
        fh_ckpt_path='./ckpt/FH_best_9.pth'
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
