import copy
import functools
import os

import blobfile as bf
import torch as th
import torch.distributed as dist
from torch import nn
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torchvision.transforms.functional import normalize, resize
from torch.optim import AdamW
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from .respace import SpacedDiffusion, space_timesteps

from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler
import wandb
from tqdm.auto import tqdm

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        image_size=128,
        first_process=250000,
        FH=None,
        fam_cycle=100,
        fam_noise_w=0.01,
        fam_attn_w=0.025
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.image_size = image_size
        self.first_process = first_process
        self.FH = FH
        self.fam_cycle = fam_cycle
        self.fam_noise_w = fam_noise_w
        self.fam_attn_w = fam_attn_w

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def generate_fam(self):

        if len(self.ema_params) > 0:
            ema_rate = self.ema_rate[-1]
            ema_params = self.ema_params[-1]
            logger.log(f"Using EMA model with rate {ema_rate} for sampling")

            ema_model = copy.deepcopy(self.model)
            ema_state_dict = self.mp_trainer.master_params_to_state_dict(ema_params)
            ema_model.load_state_dict(ema_state_dict)

            ema_model.eval()
            if th.cuda.is_available():
                ema_model = ema_model.cuda()
            
            sampling_model = ema_model
        else:
            logger.log("Using current model for sampling")
            self.model.eval()
            sampling_model = self.model

        sampling_diffusion = SpacedDiffusion(
        use_timesteps=space_timesteps(1000, [250]),  # Properly space 250 steps within the original 1000
        betas=self.diffusion.betas,
        model_mean_type=self.diffusion.model_mean_type,
        model_var_type=self.diffusion.model_var_type,
        loss_type=self.diffusion.loss_type,
        rescale_timesteps=self.diffusion.rescale_timesteps
        )

        with th.no_grad():
            sample_shape = (self.batch_size, 3, self.image_size, self.image_size)
            device = dist_util.dev()

            generated_images = sampling_diffusion.ddim_sample_loop(
                model=sampling_model,
                shape=sample_shape,
                noise=None,
                clip_denoised=True,
                denoised_fn=None,
                cond_fn=None,
                model_kwargs=None,
                device=device,
                progress=True,
                eta=0.0,
            )

            generated_images = (generated_images + 1) / 2

        if len(self.ema_params) > 0:
            del ema_model
        else:
            self.model.train()


        flaw_maps = []
        for idx, img in enumerate(generated_images):
            input_tensor = normalize(resize(img, [self.image_size, self.image_size]), [0.45, 0.45, 0.45], [0.25, 0.25, 0.25])
            input_tensor = input_tensor.unsqueeze(0).to(device)
            input_tensor.requires_grad = True

            target_layer = []
            for layer in self.FH.features:
                if isinstance(layer, nn.Conv2d):
                    target_layer = [layer]
            
            self.FH.eval()
            cam = GradCAM(model=self.FH, target_layers=target_layer)
            cam.batch_size = 1
            grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(0)])
            grayscale_cam_img = grayscale_cam[0]
            
            flaw_maps.append(th.tensor(grayscale_cam_img).to(device))

        fam = th.mean(th.stack(flaw_maps), dim=0)

        return fam

    def run_loop(self, num_iterations):
        self.num_iterations = num_iterations

        with tqdm(total=num_iterations, initial=(self.step + self.resume_step),
                desc="Training", dynamic_ncols=True) as pbar:
            while self.step + self.resume_step < self.num_iterations:
                batch, cond = next(self.data)

                # base phase
                if self.step + self.resume_step < self.first_process:
                    self.run_step(batch, cond)
                # refinement phase
                else:
                    if (self.step + self.resume_step) % self.fam_cycle == 0: # cycle
                        self.current_fam = self.generate_fam() # flaw activation map generation

                    self.run_step(batch, cond, self.current_fam, self.fam_noise_w, self.fam_attn_w)


                if (self.step + self.resume_step) % self.log_interval == 0:
                    logger.dumpkvs()
                if (self.step + self.resume_step) % self.save_interval == 0:
                    self.save()
                    # Run for a finite amount of time in integration tests.
                    if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                        return
                self.step += 1

                pbar.update(1)
                pbar.set_postfix(loss=logger.getkvs().get("loss", 0), 
                                step=(self.step + self.resume_step),
                                samples=(self.step + self.resume_step) * self.global_batch)
        # Save the last checkpoint if it wasn't already saved.
        if ((self.step + self.resume_step) - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond, fam=None, fam_noise_w=0.01, fam_attn_w=0.025):
        self.forward_backward(batch, cond, fam, fam_noise_w, fam_attn_w)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, cond, fam=None, fam_noise_w=0.01, fam_attn_w=0.025):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
                fam=fam,
                fam_noise_w=fam_noise_w,
                fam_attn_w=fam_attn_w
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            self.mp_trainer.backward(loss)

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

        if wandb.run is not None:
            metrics = {"step": self.step + self.resume_step,
                    "samples": (self.step + self.resume_step + 1) * self.global_batch}
            for k, v in logger.getkvs().items():
                metrics[k] = v
            wandb.log(metrics)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        dist.barrier()


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
