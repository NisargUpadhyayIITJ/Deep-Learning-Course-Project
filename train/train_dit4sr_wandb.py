#!/usr/bin/env python
# coding=utf-8
# DiT4SR Training Script with WandB Integration
# Enhanced version of train_dit4sr.py with comprehensive monitoring

import argparse
import copy
import logging
import math
import os
import shutil
import sys
sys.path.append(os.getcwd())
from pathlib import Path
from datetime import datetime

import accelerate
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from tqdm.auto import tqdm

import diffusers
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    StableDiffusion3ControlNetPipeline,
    StableDiffusion3Pipeline,
)
from model_dit4sr.transformer_sd3 import SD3Transformer2DModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3, free_memory, cast_training_params
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.torch_utils import is_compiled_module

from dataloaders.paired_dataset_sd3_latent import PairedCaptionDataset

# Will error if the minimal version of diffusers is not installed
check_min_version("0.30.0.dev0")

logger = get_logger(__name__)


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="DiT4SR Training with WandB Integration")
    
    # Model arguments
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--transformer_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained transformer model. If not specified, initializes from pretrained_model_name_or_path.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files (e.g., 'fp16').",
    )
    
    # Output and logging
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments/dit4sr_wandb",
        help="The output directory for checkpoints and logs.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="TensorBoard/WandB log directory.",
    )
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    
    # Training parameters
    parser.add_argument("--resolution", type=int, default=512, help="Resolution for input images.")
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size per device.")
    parser.add_argument("--num_train_epochs", type=int, default=1000)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total training steps. Overrides num_train_epochs if set.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Use gradient checkpointing to save memory.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        help="Scale learning rate by batch size and gradient accumulation steps.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
        help="Learning rate scheduler type.",
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Warmup steps for LR scheduler.")
    parser.add_argument("--lr_num_cycles", type=int, default=1, help="Cycles for cosine_with_restarts scheduler.")
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power for polynomial scheduler.")
    
    # Optimizer
    parser.add_argument("--use_8bit_adam", action="store_true", help="Use 8-bit Adam from bitsandbytes.")
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping.")
    
    # Flow matching parameters
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="logit_normal",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap"],
    )
    parser.add_argument("--logit_mean", type=float, default=0.0)
    parser.add_argument("--logit_std", type=float, default=1.0)
    parser.add_argument("--mode_scale", type=float, default=1.29)
    parser.add_argument("--precondition_outputs", type=int, default=1)
    
    # Checkpointing
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=1000,
        help="Save checkpoint every X steps.",
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=5,
        help="Max number of checkpoints to keep.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint or 'latest' to resume from last checkpoint.",
    )
    
    # WandB and logging
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        choices=["tensorboard", "wandb", "all"],
        help="Logging platform.",
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="dit4sr-training",
        help="WandB project name.",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="WandB run name. Auto-generated if not specified.",
    )
    parser.add_argument(
        "--log_grad_norm",
        action="store_true",
        help="Log gradient norms to wandb.",
    )
    parser.add_argument(
        "--log_every_n_steps",
        type=int,
        default=10,
        help="Log metrics every N steps.",
    )
    
    # Validation
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=500,
        help="Run validation every X steps.",
    )
    parser.add_argument(
        "--validation_dir",
        type=str,
        default=None,
        help="Path to validation dataset (optional).",
    )
    parser.add_argument(
        "--num_validation_samples",
        type=int,
        default=4,
        help="Number of validation samples to log.",
    )
    
    # Data
    parser.add_argument("--root_folders", type=str, default='', help="Training data folder.")
    parser.add_argument("--null_text_ratio", type=float, default=0.2, help="Ratio of null text prompts.")
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument('--trainable_modules', nargs='*', type=str, default=["control"])
    
    # Mixed precision and hardware
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
    )
    parser.add_argument("--allow_tf32", action="store_true", help="Allow TF32 on Ampere GPUs.")
    parser.add_argument("--set_grads_to_none", action="store_true")
    
    # Misc
    parser.add_argument("--max_sequence_length", type=int, default=77)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_token", type=str, default=None)
    parser.add_argument("--hub_model_id", type=str, default=None)

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.resolution % 8 != 0:
        raise ValueError("`--resolution` must be divisible by 8.")

    return args


class EMALoss:
    """Exponential Moving Average for loss tracking."""
    def __init__(self, decay=0.99):
        self.decay = decay
        self.value = None
    
    def update(self, loss):
        if self.value is None:
            self.value = loss
        else:
            self.value = self.decay * self.value + (1 - self.decay) * loss
        return self.value


def compute_gradient_norm(model):
    """Compute total gradient norm."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5


def main(args):
    # Validate args
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "Cannot use both --report_to=wandb and --hub_token (security risk)."
        )

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        raise ValueError("bf16 not supported on MPS. Use fp16 instead.")

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Check wandb availability
    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Install wandb: `pip install wandb`")
        import wandb

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # Set seed
    if args.seed is not None:
        set_seed(args.seed)

    # Create output directory
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, 
                exist_ok=True, 
                token=args.hub_token
            ).repo_id

    # Load scheduler and models
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    if args.transformer_model_name_or_path is not None:
        transformer = SD3Transformer2DModel.from_pretrained_local(
            args.transformer_model_name_or_path, subfolder="transformer", 
            revision=args.revision, variant=args.variant
        )
    else:
        transformer = SD3Transformer2DModel.from_pretrained_local(
            args.pretrained_model_name_or_path, subfolder="transformer", 
            revision=args.revision, variant=args.variant
        )

    # Freeze all parameters first
    transformer.requires_grad_(False)

    # Unfreeze trainable modules
    trainable_param_count = 0
    for name, params in transformer.named_parameters():
        if any(trainable_module in name for trainable_module in tuple(args.trainable_modules)):
            logger.info(f'{name} will be optimized.')
            params.requires_grad = True
            trainable_param_count += params.numel()
    
    logger.info(f"Total trainable parameters: {trainable_param_count:,}")

    # Unwrap model helper
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # Custom save/load hooks for checkpointing
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                i = len(weights) - 1
                while len(weights) > 0:
                    weights.pop()
                    model = models[i]
                    sub_dir = "transformer"
                    model.save_pretrained(os.path.join(output_dir, sub_dir))
                    i -= 1

        def load_model_hook(models, input_dir):
            model = models.pop()
            load_model = SD3Transformer2DModel.from_pretrained(
                input_dir, subfolder="transformer"
            )
            model.register_to_config(**load_model.config)
            model.load_state_dict(load_model.state_dict())
            del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # TF32 for Ampere GPUs
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Scale learning rate
    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * 
            args.train_batch_size * accelerator.num_processes
        )

    # Optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("Install bitsandbytes: `pip install bitsandbytes`")
        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    params_to_optimize = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Weight dtype for mixed precision
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    transformer.to(accelerator.device, dtype=weight_dtype)

    # Keep trainable params in fp32 for stability
    if args.mixed_precision == "fp16":
        models = [transformer]
        cast_training_params(models, dtype=torch.float32)

    # Dataset
    train_dataset = PairedCaptionDataset(
        root_folder=args.root_folders,
        null_text_ratio=args.null_text_ratio,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and training steps
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare with accelerator
    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )

    # Recalculate training steps after dataloader changes
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Initialize trackers (WandB)
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        
        # Generate run name if not specified
        if args.wandb_run_name is None:
            args.wandb_run_name = f"dit4sr_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        accelerator.init_trackers(
            args.tracker_project_name, 
            config=tracker_config,
            init_kwargs={"wandb": {"name": args.wandb_run_name}}
        )

    # Training info
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    
    global_step = 0
    first_epoch = 0

    # Resume from checkpoint
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' not found. Starting fresh."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
            
            # Log resume event to wandb
            if accelerator.is_main_process and args.report_to == "wandb":
                accelerator.log({"resumed_from_step": global_step}, step=global_step)
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    free_memory()
    
    # EMA loss tracker
    ema_loss = EMALoss(decay=0.99)
    
    # Training loop
    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(transformer):
                # Get latents
                model_input = batch["pixel_values"].to(dtype=weight_dtype)
                controlnet_image = batch["conditioning_pixel_values"].to(dtype=weight_dtype)

                # Sample noise
                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0]
                
                # Sample timesteps
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)

                # Add noise (flow matching)
                sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
                noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

                # Get embeddings
                prompt_embeds = batch["prompt_embeds"].to(dtype=model_input.dtype)
                pooled_prompt_embeds = batch["pooled_prompt_embeds"].to(dtype=model_input.dtype)

                # Forward pass
                model_pred = transformer(
                    hidden_states=noisy_model_input,
                    controlnet_image=controlnet_image,
                    timestep=timesteps,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    return_dict=False,
                )[0]

                # Preconditioning
                if args.precondition_outputs:
                    model_pred = model_pred * (-sigmas) + noisy_model_input

                # Loss weighting
                weighting = compute_loss_weighting_for_sd3(
                    weighting_scheme=args.weighting_scheme, 
                    sigmas=sigmas
                )

                # Target
                if args.precondition_outputs:
                    target = model_input
                else:
                    target = noise - model_input

                # Compute loss
                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()

                accelerator.backward(loss)
                
                # Gradient clipping
                if accelerator.sync_gradients:
                    grad_norm = None
                    if args.log_grad_norm:
                        grad_norm = compute_gradient_norm(transformer)
                    accelerator.clip_grad_norm_(transformer.parameters(), args.max_grad_norm)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # After gradient sync
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # Checkpointing
                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # Limit checkpoints
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints exist, removing {len(removing_checkpoints)}"
                                )
                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            # Logging
            if global_step % args.log_every_n_steps == 0 or global_step == 1:
                ema_loss_val = ema_loss.update(loss.detach().item())
                
                logs = {
                    "loss": loss.detach().item(),
                    "loss_ema": ema_loss_val,
                    "lr": lr_scheduler.get_last_lr()[0],
                    "epoch": epoch,
                    "step": global_step,
                }
                
                if args.log_grad_norm and accelerator.sync_gradients and grad_norm is not None:
                    logs["grad_norm"] = grad_norm
                
                # GPU memory
                if torch.cuda.is_available():
                    logs["gpu_memory_gb"] = torch.cuda.max_memory_allocated() / 1e9
                
                progress_bar.set_postfix(**{"loss": logs["loss"], "loss_ema": logs["loss_ema"]})
                accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # End training
    accelerator.wait_for_everyone()
    
    # Save final model
    if accelerator.is_main_process:
        final_save_path = os.path.join(args.output_dir, "final_model")
        unwrap_model(transformer).save_pretrained(os.path.join(final_save_path, "transformer"))
        logger.info(f"Saved final model to {final_save_path}")
    
    accelerator.end_training()
    logger.info("Training complete!")


if __name__ == "__main__":
    args = parse_args()
    main(args)
