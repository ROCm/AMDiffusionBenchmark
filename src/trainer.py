import argparse
import contextlib
import functools
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Union

import accelerate
import datasets
import diffusers
import torch
import torch.distributed
import torch.optim.optimizer
import transformers
from accelerate import Accelerator, ProfileKwargs
# from accelerate.utils import DummyOptim, DummyScheduler, DynamoBackend, set_seed
from accelerate.utils.deepspeed import DummyOptim, DummyScheduler
from accelerate.utils import DynamoBackend, set_seed
from diffusers.optimization import get_scheduler
from diffusers.utils import export_to_video
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.utils.flop_counter import FlopCounterMode
from torchvision import transforms
from tqdm import tqdm
from transformers import logging as hf_logging

from src.data.cache_dataset import CacheDataset, create_or_append_to_xr_dataset
from src.data.datasets_utils import collate_fn, preprocess_train
from src.models import ModelManager
from src.models.base import BaseModel
from src.utils import configure_logging

# Configure the logger for outputs to terminal
logger = configure_logging()


class Trainer:
    """Trainer class for model training with multi-GPU support and mixed precision.

    This class encapsulates the training logic for a diffusion model using the Hugging Face
    Accelerate library. It handles model initialization, data loading, preprocessing,
    training loop execution, and validation image generation.

    Attributes:
        args (Namespace): Training configurations and command-line arguments.
        accelerator (Accelerator): Manages multi-GPU training and mixed precision.
        noise_scheduler: Schedules noise levels for the diffusion process.
        vae (AutoencoderKL): Variational Autoencoder for encoding and decoding images.
        transformer (FluxTransformer2DModel): Transformer model for the diffusion process.
        text_encoder (CLIPTextModel): First text encoder for processing captions.
        text_encoder_2 (T5EncoderModel): Second text encoder for processing captions.
        tokenizer (CLIPTokenizer): Tokenizer corresponding to the first text encoder.
        tokenizer_2 (T5TokenizerFast): Tokenizer corresponding to the second text encoder.
        vae_scale_factor (int): Scale factor derived from the VAE configuration.
        height (int): Height of the latent images after scaling.
        width (int): Width of the latent images after scaling.
        num_channels_latents (int): Number of channels in the latent representations.
        weight_dtype (torch.dtype): Data type used for mixed precision training.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        """Initialize the Trainer with configurations and models.

        Args:
            args (Namespace): Training configurations and arguments.
        """
        # Arguments
        self.args = args
        self.is_cached = False
        self.model_output_type = None

        # Multi-gpu with accelerate
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            log_with=self.args.report_to,
            project_dir=self.args.logging_dir,
        )
        self.accelerator.init_trackers(
            str(Path(self.args.report_to, datetime.now().strftime("%Y%m%d_%H%M%S")))
        )

        # Keep warning messages to the main process only. No progress bar spam.
        if self.accelerator.is_local_main_process:
            hf_logging.set_verbosity_warning()
            datasets.utils.logging.set_verbosity_warning()
            transformers.utils.logging.set_verbosity_warning()
        else:
            diffusers.utils.logging.disable_progress_bar()
            hf_logging.set_verbosity_error()
            datasets.disable_progress_bars()
            datasets.utils.logging.set_verbosity_error()
            transformers.utils.logging.set_verbosity_error()
            transformers.utils.logging.disable_progress_bar()

        if args.seed is not None:
            set_seed(args.seed, device_specific=True)

        self._prepare_model()

    def _prepare_model(self) -> None:
        """Initialize and prepare the model for training.

        1. Initializes the model using the appropriate model class
        2. Configures sigmas for the scheduler based on timesteps
        3. Sets up mixed precision training (fp16 or bf16) if specified
        4. Enables gradient checkpointing if requested (necessary for hunyuan-video)
        5. Initializes LoRA if enabled
        6. Creates the image transformation pipeline
        """
        # Initialize model and set type
        model_class, self.model_input_type, self.model_output_type = (
            ModelManager().get_model(self.args.model)
        )
        self.model: BaseModel = model_class(self.args)

        # For mixed precision training
        self.weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16

        # Modules to apply the lora matrix to
        if self.args.use_lora:
            self.model.init_lora()

        # Filter out parameters that do not require gradients
        self.trainable_params = list(
            filter(lambda p: p.requires_grad, self.model.denoiser.parameters())
        )

        # Image transformation pipeline
        width, height = self.args.resolution
        jitter_factor = 1.5
        jittered_width = int(random.uniform(width, width * jitter_factor))
        jittered_height = int(random.uniform(height, height * jitter_factor))
        self.transform_pipeline = transforms.Compose(
            [
                transforms.Resize(
                    (jittered_height, jittered_width),  # Resize to jittered size
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                (
                    transforms.CenterCrop(
                        (
                            height,
                            width,
                        )
                    )  # Crop to desired size
                    if self.args.center_crop
                    else transforms.RandomCrop(
                        (
                            height,
                            width,
                        )
                    )
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def generate(self, iter: int = 0) -> None:
        """Generate and save using the Hugging Face pipeline.

        Args:
            iter (int, optional): The current iteration number, used in the output filename.
                Defaults to 0.
        """
        Path(self.args.output_dir, "validation").mkdir(parents=True, exist_ok=True)

        # Load all necessary components into VRAM
        self.model.submodules_to(self.accelerator.device, self.weight_dtype)

        # Insert the initial config (required for deepspeed support)
        if hasattr(self.model.denoiser, "config"):
            temp_copy = self.model.denoiser.config
            self.model.denoiser.config = self.model.denoiser_config

        self.model.pipe.set_progress_bar_config(
            disable=not self.accelerator.is_local_main_process
        )
        prompt = self.args.validation_prompts
        generator = None
        if self.args.seed is not None:
            generator = torch.Generator(device=self.accelerator.device).manual_seed(
                self.args.seed
            )

        width, height = self.args.resolution
        # Known issue where latents are cast to torch.float32 before being fed into the VAE.
        # This forces the latents to be cast to float16/bfloat16 to avoid precision issues.
        if self.model_output_type == "image":

            def convertfp16(flux_pipeline, i, t, callback_kwargs):
                latents = callback_kwargs["latents"].to(self.weight_dtype)
                return {"latents": latents}

            image = self.model.pipe(
                prompt,
                guidance_scale=self.args.validation_guidance_scale,
                num_inference_steps=self.args.validation_inference_steps,
                max_sequence_length=256,
                height=height,
                width=width,
                callback_on_step_end=(
                    convertfp16 if self.weight_dtype != torch.float32 else None
                ),
                callback_on_step_end_tensor_inputs=(
                    ["latents"] if self.weight_dtype != torch.float32 else None
                ),
                generator=generator,
            ).images[0]
        else:
            # This is needed for some unknown reason.
            with self.model.denoiser.summon_full_params(
                self.model.denoiser, writeback=False
            ):
                output = self.model.pipe(
                    prompt=prompt,
                    height=height,
                    width=width,
                    num_frames=61,
                    num_inference_steps=self.args.validation_inference_steps,
                ).frames[0]

        # Return the deepspeed configuration
        if hasattr(self.model.denoiser, "config"):
            self.model.denoiser.config = temp_copy

        # Unload to CPU
        if self.args.use_cache:
            self.model.unload_submodules()
        if self.accelerator.is_local_main_process:
            if self.model_output_type == "image":
                image.save(
                    str(
                        Path(
                            self.args.output_dir,
                            "validation",
                            f"val_image_iter{iter}.png",
                        )
                    )
                )
            else:
                export_to_video(
                    output,
                    str(
                        Path(
                            self.args.output_dir,
                            "validation",
                            f"val_video_iter{iter}.mp4",
                        )
                    ),
                    fps=15,
                )

    # TODO: Generalize the data loading process to support different dataset and preproc.
    def _get_dataloader(self) -> torch.utils.data.DataLoader:
        """Create and return the training DataLoader.

        Returns:
            DataLoader: A PyTorch DataLoader instance for the training dataset.
        """
        # Create cache dir
        if self.args.use_cache and self.accelerator.is_local_main_process:
            Path(self.args.cache_dir).mkdir(parents=True, exist_ok=True)

        # Check whether all necessary files for the latents and text encodings have been cached.
        # Otherwise they need to be computed.
        cache_file = Path(self.args.cache_dir, "cached_data.zarr")
        if cache_file.exists() and self.args.use_cache:
            self.is_cached = True

        # Load cached dataset
        if self.is_cached:
            train_dataset = CacheDataset(str(cache_file), self.weight_dtype)

        else:
            # Load the dataset
            dataset = datasets.load_dataset(
                self.args.train_data_path,
                revision=(
                    "refs/pr/4"
                    if self.args.train_data_path
                    == "Wild-Heart/Disney-VideoGeneration-Dataset"
                    else "main"
                ),
                split="train",
            )

            # Limit dataset size if specified
            if (
                hasattr(self.args, "max_train_samples")
                and self.args.max_train_samples > 0
            ):
                max_samples = min(len(dataset), self.args.max_train_samples)
                dataset = dataset.select(range(max_samples))
                logger.info(f"Limited dataset to {max_samples} samples")

            # Process the data with the image transform
            with self.accelerator.main_process_first():
                train_dataset = dataset.with_transform(
                    lambda examples: preprocess_train(
                        examples,
                        self.args,
                        tokenizer=self.model.submodules["tokenizer"],
                        tokenizer_2=self.model.submodules["tokenizer_2"],
                        image_processor=self.model.submodules["image_processor"],
                        train_transforms=self.transform_pipeline,
                    ),
                )

        # Create the dataloader
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            shuffle=True,
            collate_fn=collate_fn,
            batch_size=self.args.train_batch_size,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=True,
            persistent_workers=self.args.dataloader_num_workers > 0,
        )
        return train_dataloader

    def _cache_latents_and_text(self, dataloader: torch.utils.data.DataLoader) -> None:
        """Calculate and cache the latents

        Args:
            DataLoader: A PyTorch DataLoader instance for the training dataset to be cached.
        """
        progress_bar = tqdm(
            range(
                0,
                len(dataloader),
            ),
            desc="Caching latents",
            # Only show the progress bar once on each machine.
            disable=not self.accelerator.is_local_main_process,
            leave=False,
        )

        # Cache all training data
        if self.accelerator.is_main_process:
            gathered_data = []

        for _, batch in enumerate(dataloader):
            with torch.inference_mode():
                encoded_batch = self.model.encode_batch(batch)

            gathered_batch = self.accelerator.gather(encoded_batch)
            if self.accelerator.is_main_process:
                for key, value in gathered_batch.items():
                    gathered_batch[key] = value.to("cpu")

                cache_file = str(Path(self.args.cache_dir, "cached_data.zarr"))
                create_or_append_to_xr_dataset(gathered_batch, cache_file=cache_file)

                progress_bar.update(1)

        self.accelerator.wait_for_everyone()
        self.is_cached = True

    def _prepare_training(
        self,
        optimizer: Union[DummyOptim, torch.optim.Optimizer],
        train_dataloader: torch.utils.data.DataLoader,
        lr_scheduler: Union[DummyScheduler, torch.optim.lr_scheduler.LRScheduler],
    ) -> tuple[
        Union[DummyOptim, torch.optim.Optimizer],
        torch.utils.data.DataLoader,
        Union[DummyScheduler, torch.optim.lr_scheduler.LRScheduler],
    ]:
        """Prepare the model and optimizer for training, especially for multi-GPU setups.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer initialized with model parameters.
            train_dataloader (torch.utils.data.DataLoader): The DataLoader for training data.
            lr_scheduler (_LRScheduler): The learning rate scheduler.

        Returns:
            Tuple[torch.optim.Optimizer, DataLoader, _LRScheduler]:
                - optimizer: The prepared optimizer.
                - train_dataloader: The prepared DataLoader.
                - lr_scheduler: The prepared learning rate scheduler.
        """
        # Only train the diffusion transformer
        for module in self.model.submodules.values():
            if hasattr(module, "requires_grad_"):
                module.requires_grad_(False)

        # Prepare multi-gpu training
        self.model.denoiser, optimizer, train_dataloader, lr_scheduler = (
            self.accelerator.prepare(
                self.model.denoiser, optimizer, train_dataloader, lr_scheduler
            )
        )
        # Apply the (partial) gradient checkpointing
        # We follow https://github.com/huggingface/accelerate/blob/33967d4733ec5bf402d85462ec2bbbcd8e872ea9/src/accelerate/accelerator.py#L1658-L1665
        # which matches with `fsdp_config: fsdp_activation_checkpointing=True`
        # NB! We don't use Diffuser's `enable_gradient_checkpointing()` because this results in performance regressions
        if self.args.gradient_checkpointing > 0:
            if self.accelerator.distributed_type != accelerate.DistributedType.FSDP:
                raise NotImplementedError(
                    "Activation/Gradient checkpointing is only available for FSDP"
                )

            if self.accelerator.state.fsdp_plugin.activation_checkpointing:
                logger.warning(
                    "Setting activation checkpointing via both `fsdp_config: fsdp_activation_checkpointing=True` \
                        and `train_args.gradient_checkpointing` can lead into unwanted behaviour: use either or, not both!"
                )

            def _check_fn(module):
                """
                Helper function to activate the activation/gradient checkpointing on the non-FSDP modules
                that are tagged with `use_in_partial_checkpointing=True`
                NB!
                * We use (base) model to annotate the modules with attribute `use_in_partial_checkpointing=True`
                * TRANSFORMER_BASED_WRAP iterates the modules as both "normal" module and "FSDP module".
                    We don't apply grad.ckpt on "FSDP modules", because this can lead into performance regressions
                """
                return getattr(
                    module, "use_in_partial_checkpointing", False
                ) and not module.__class__.__name__.startswith(
                    "FullyShardedDataParallel"
                )

            kwargs = dict(
                checkpoint_wrapper_fn=functools.partial(
                    checkpoint_wrapper,
                    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
                ),
                check_fn=_check_fn,
            )
            if self.args.gradient_checkpointing == 1.0:
                # NB! We use `auto_wrap_policy` in case of full (100%) checkpointing
                # This will automatically overwrite `check_fn` and apply gradient checkpointing on all modules
                kwargs["auto_wrap_policy"] = (
                    self.accelerator.state.fsdp_plugin.auto_wrap_policy
                )

            apply_activation_checkpointing(self.model.denoiser, **kwargs)

        # Move to correct gpu/dtype, offload vae to cpu if latents are cached
        if not self.args.use_cache:
            target_dtype = self.weight_dtype
            target_device = self.accelerator.device
        elif not self.is_cached:
            target_dtype = torch.float32
            target_device = self.accelerator.device
        else:
            target_dtype = self.weight_dtype
            target_device = "cpu"

        self.model.submodules_to(target_device, target_dtype)
        self.model.denoiser.to(self.accelerator.device, dtype=self.weight_dtype)

        # Prep for training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        self.model.denoiser.train()

        return optimizer, train_dataloader, lr_scheduler

    def _profiling_ctx(self, step: int) -> contextlib.contextmanager:
        """
        Context manager for profiling the training step.
        """
        if self.args.profiling_step == step:
            Path(
                self.args.profiling_logging_dir,
            ).mkdir(parents=True, exist_ok=True)
            profile_kwargs = ProfileKwargs(
                activities=["cpu", "cuda"],
                # schedule=torch.profiler.schedule(wait=1, warmup=1, active=2),
                record_shapes=True,
                with_stack=True,
                profile_memory=True,
                on_trace_ready=lambda prof: prof.export_chrome_trace(
                    str(
                        Path(
                            self.args.profiling_logging_dir,
                            f"pytorch_profile_gpu{self.accelerator.process_index}.json",
                        )
                    ),
                ),
            )
            return self.accelerator.profile(profile_kwargs)

        return contextlib.nullcontext()

    def _is_deepspeed_optimizer_configured(self) -> bool:
        """
        Checks if the DeepSpeed plugin is configured with an optimizer.
        """
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        return (
            deepspeed_plugin is not None
            and "optimizer" in deepspeed_plugin.deepspeed_config
        )

    def _get_optimizer(self):
        """Initialize the optimizer and learning rate scheduler."""
        # Set epsilon to 1e-07 for fp16
        epsilon = self.args.adam_epsilon
        if self.args.adam_epsilon < 1e-03 and self.weight_dtype == torch.float16:
            epsilon = 1e-03
            logger.warning(
                f"WARNING: Epsilon value too low for fp16. Increasing it to {epsilon}"
            )

        # Only optimize parameters that require grad, e.g. lora layers
        optimizer_params = dict(
            params=self.trainable_params,
            lr=self.args.learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            weight_decay=self.args.adam_weight_decay,
            eps=epsilon,
            fused=True,
        )

        # Initialize the optimizer and scheduler
        if self._is_deepspeed_optimizer_configured():
            optimizer = DummyOptim(**optimizer_params)
            lr_scheduler = DummyScheduler(
                optimizer,
                total_num_steps=self.args.num_iterations
                * self.accelerator.num_processes,
                num_warmup_steps=self.args.lr_warmup_steps
                * self.accelerator.num_processes,
            )
        else:
            # TODO: make optimizer params accessible from the model
            optimizer = torch.optim.AdamW(**optimizer_params)
            # Calculate the amount of steps for the scheduling
            lr_scheduler = get_scheduler(
                self.args.lr_scheduler,
                optimizer=optimizer,
                num_warmup_steps=self.args.lr_warmup_steps
                * self.accelerator.num_processes,
                num_training_steps=self.args.num_iterations
                * self.accelerator.num_processes,
            )
        return optimizer, lr_scheduler

    # stay as is, to modularise and generalise in loss and optimiser
    def run_training(self) -> None:
        """Executes the main training loop for the model.

        This method handles the complete training process including:
        - Setting up optimizer and scheduler
        - Preparing model and data
        - Executing training iterations
        - Managing latent caching
        - Tracking progress and metrics
        """
        # Prepare model before optimizer is supposed to be more efficient for fsdp
        if self.accelerator.distributed_type == accelerate.DistributedType.FSDP:
            self.model.denoiser = self.accelerator.prepare(self.model.denoiser)

        optimizer, lr_scheduler = self._get_optimizer()
        train_dataloader = self._get_dataloader()

        # Prepare model for multi-gpu training
        optimizer, train_dataloader, lr_scheduler = self._prepare_training(
            optimizer, train_dataloader, lr_scheduler
        )

        # Cache the latents and reload the dataloader
        if self.args.use_cache and not self.is_cached:
            self._cache_latents_and_text(train_dataloader)
            # Reload the dataloader
            train_dataloader = self._get_dataloader()
            optimizer, train_dataloader, lr_scheduler = self._prepare_training(
                optimizer, train_dataloader, lr_scheduler
            )

        # Calculate the flops for the training loop
        flop_counter = FlopCounterMode(display=False, depth=None)
        total_flops = 0.0

        # Disable flop counting when using torch.compile
        if self.accelerator.state.dynamo_plugin.backend != DynamoBackend.NO:
            flop_counter = contextlib.nullcontext()
            logger.warning(
                "WARNING: Disabling TFlops/s calculations due to incompatability with torch.compile"
            )

        # Training loop
        global_step = 0
        train_loss = 0.0

        # Initialize CUDA events
        start_event_gpu = torch.cuda.Event(enable_timing=True)
        end_event_gpu = torch.cuda.Event(enable_timing=True)

        # Training loop
        while global_step < self.args.num_iterations:
            # Record start time including data loading
            start_time_step = time.time()

            for step, batch in enumerate(train_dataloader):
                # Generate validation image
                if (
                    global_step % self.args.validation_iteration == 0
                    and self.args.validation_iteration != -1
                ):
                    with torch.no_grad():
                        self.generate(iter=global_step)

                # Train step
                if self.accelerator.use_distributed:
                    # We wait for every process to reach this,
                    # so that below timing would only contain current training step,
                    # not wait times of previous steps
                    # NB! This is strictly for benchmarking purposes
                    # and should be removed in real use
                    self.accelerator.wait_for_everyone()

                with self.accelerator.accumulate(
                    self.model.denoiser
                ), self._profiling_ctx(global_step), flop_counter:
                    start_event_gpu.record()

                    # If un-cached, encode the batch
                    if not self.is_cached:
                        batch = self.model.encode_batch(batch)

                    noised_latents, noise, timesteps = (
                        self.model.sample_timesteps_and_noise(batch["latents"])
                    )
                    batch["noised_latents"] = noised_latents
                    batch["timestep"] = timesteps

                    # Get the model specific inputs:
                    # Each model must implement `get_model_inputs` which posprocesses
                    # the batch into final model inputs fed to `model.denoiser`
                    model_inputs, timesteps = self.model.get_model_inputs(batch)

                    # Guidance conditioning
                    if self.model.guidance is not None:
                        guidance = torch.full(
                            [1],
                            self.model.guidance,
                            dtype=self.weight_dtype,
                            device=self.accelerator.device,
                        ).expand(batch["latents"].shape[0])
                        model_inputs["guidance"] = guidance

                    # Inference
                    pred = self.model.denoiser(
                        timestep=timesteps,
                        **model_inputs,
                        return_dict=False,
                    )[0]

                    loss = self.model.compute_loss(
                        pred, noise, batch["latents"], timesteps
                    )
                    self.accelerator.backward(loss)

                    # Clip gradients
                    if (
                        self.accelerator.sync_gradients
                        and self.args.max_grad_norm > 0.0
                    ):
                        self.accelerator.clip_grad_norm_(
                            self.trainable_params,
                            self.args.max_grad_norm,
                        )

                    # Optimizer step
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                # Wait for all processes to finish before recording the time
                if self.accelerator.use_distributed:
                    self.accelerator.wait_for_everyone()

                end_event_gpu.record()
                torch.cuda.synchronize()

                # Calculate GPU timing
                step_time_gpu = (
                    start_event_gpu.elapsed_time(end_event_gpu) / 1000
                )  # GPU step time in seconds. NB! event.elapsed_time is in ms
                fps_gpu = self.args.train_batch_size / step_time_gpu

                # Calculate loss across all processes
                # This is only for logging purposes; loss.backward() has already been done above
                avg_loss = self.accelerator.gather(
                    loss.repeat(self.args.train_batch_size)
                ).mean()
                train_loss += (
                    avg_loss.item() / self.accelerator.gradient_accumulation_steps
                )

                # Remove flop counter since we only need to compute the flops once.
                if isinstance(flop_counter, FlopCounterMode):
                    total_flops = flop_counter.get_total_flops()
                    flop_counter = contextlib.nullcontext()

                # Calculate entire step timing (in seconds)
                step_time_total = time.time() - start_time_step

                # Logging
                logs = {
                    "step_time_total": step_time_total,
                    "step_loss": loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                    "step_time_gpu": step_time_gpu,
                    "fps_gpu": fps_gpu,
                    "tflops/s": total_flops * 1e-12 / step_time_gpu,
                }
                if self.accelerator.sync_gradients:
                    global_step += 1
                    logs["train_loss"] = train_loss
                    train_loss = 0.0

                postpr_logs = {k: float(f"{v:.4E}") for k, v in logs.items()}
                
                # if self.accelerator.is_local_main_process:
                #     logger.info(f"Step {global_step}: {postpr_logs}")
                #     self.accelerator.log(logs, step=global_step)

                def _fmt_num(k: str, v):
                    if not isinstance(v, float):
                        return v
                    if k == "lr":
                        return f"{v:.10f}"   # e.g., 0.0000100000
                    return f"{v:.6f}"

                if self.accelerator.is_local_main_process:
                    formatted_logs = ", ".join([f"'{k}': {_fmt_num(k, v)}" for k, v in logs.items()])
                    log_line = f"INFO - Step {global_step} - {{{formatted_logs}}}"
                    print(log_line, flush=True)
                    logger.info(log_line)
                self.accelerator.log(logs, step=global_step)

                # Break if the max number of iterations are exceeded
                if global_step >= self.args.num_iterations:
                    break

                start_time_step = time.time()

            # Generate last validation image
            if global_step >= self.args.num_iterations:
                if (
                    global_step % self.args.validation_iteration == 0
                    and self.args.validation_iteration != -1
                ):
                    with torch.no_grad():
                        self.generate(iter=global_step)

        # Save the trained model's
        checkpoint_dir = Path(self.args.output_dir, "checkpoint", self.args.model)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        with (
            self.model.denoiser.summon_full_params(self.model.denoiser, writeback=False)
            if self.accelerator.distributed_type == accelerate.DistributedType.FSDP
            else contextlib.nullcontext()
        ):
            if self.accelerator.is_local_main_process:
                self.accelerator.unwrap_model(self.model.denoiser).save_pretrained(
                    str(checkpoint_dir)
                )

        # Empty the cache
        self.accelerator.clear()
        self.accelerator.end_training()
