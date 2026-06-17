import argparse
import os

from src.data.hf_torchvision_video_shim import apply_torchvision_io_video_reader_shim

apply_torchvision_io_video_reader_shim()

import torch
import torch.distributed
from dotenv import load_dotenv

from src.models import ModelManager
from src.trainer import Trainer
from src.utils import check_gpu_vendor, parse_resolution, safely_eval_as_bool

# Load environment variables from a .env file
load_dotenv()

# Adjust torch.dynamo.cache_limit if needed
torch._dynamo.config.cache_size_limit = int(
    os.getenv("TORCH_DYNAMO_CACHE_SIZE_LIMIT", torch._dynamo.config.cache_size_limit)
)


def parse_args():
    """Parse the arguments necessary for training:

    Returns:
        args: The parsed arguments containing the parameters necessary for training.
    """
    parser = argparse.ArgumentParser(
        description="Simple training script.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("MODEL", "flux-dev"),
        help="Model to train, either flux-dev, hunyuan-video or stable-diffusion-xl.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=os.getenv("MODEL_PATH", "black-forest-labs/FLUX.1-dev"),
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.getenv("OUTPUT_DIR", "./outputs"),
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--train_data_path",
        type=str,
        default=os.getenv("TRAIN_DATA_PATH", "bghira/pseudo-camera-10k"),
        help="The directory where the training data is stored or dataset identifier from huggingface.co/datasets.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=int(os.getenv("MAX_TRAIN_SAMPLES", -1)),
        help="Maximum number of samples to use from the training dataset.",
    )
    parser.add_argument(
        "--validation_prompts",
        type=str,
        default=os.getenv("VALIDATION_PROMPTS", "high-res picture of a cat"),
        help="A set of prompts evaluated every `--validation_iteration` and logged to `--report_to`.",
    )
    parser.add_argument(
        "--validation_iteration",
        type=int,
        default=int(os.getenv("VALIDATION_ITERATION", -1)),
        help="Number of iterations between validations.",
    )
    parser.add_argument(
        "--validation_guidance_scale",
        type=float,
        default=float(os.getenv("VALIDATION_GUIDANCE_SCALE", 0.0)),
        help="Number of iterations between validations.",
    )
    parser.add_argument(
        "--validation_inference_steps",
        type=int,
        default=int(os.getenv("VALIDATION_INFERENCE_STEPS", 20)),
        help="Number of iterations between validations.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=os.getenv("CACHE_DIR", "./cache"),
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--use_cache",
        default=os.getenv("USE_CACHE", "1"),
        type=lambda x: safely_eval_as_bool(x),
        help="Whether to use the cached latents or generate new ones. Accepts 'true', 'false', '1', or '0'.",
    )
    parser.add_argument(
        "--substitute_sdpa_with_flash_attn",
        default=os.getenv(
            "SUBSTITUTE_SDPA_WITH_FLASH_ATTN", check_gpu_vendor() == "rocm"
        ),
        type=lambda x: safely_eval_as_bool(x),
        help=(
            "Whether to use Flash-Attention as opposed to PyTorch's native SDPA as the attention backend. "
            "Accepts 'true', 'false', '1', or '0'. Note that Flash-Attention is not available for fp32; "
            "a warning will be issued if a datatype other than bf16 or fp16 is used, but execution will continue using SDPA. "
            "Consider using bf16 or fp16 for compatibility and potential speed-ups."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(os.getenv("SEED", 42)),
        help="A seed for reproducible training.",
    )
    parser.add_argument(
        "--resolution",
        type=parse_resolution,
        default=parse_resolution(os.getenv("RESOLUTION", "512")),
        help=(
            "The resolution (width, height) for input images."
            " All the images in the train/validation dataset will be resized to this resolution."
        ),
    )
    parser.add_argument(
        "--center_crop",
        type=lambda x: safely_eval_as_bool(x),
        default=safely_eval_as_bool(os.getenv("CENTER_CROP", "false")),
        help="Whether to center crop the input images to the resolution. If not set, the images will be randomly cropped.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=int(os.getenv("TRAIN_BATCH_SIZE", 1)),
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=int(os.getenv("GRADIENT_ACCUMULATION_STEPS", 1)),
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=int(os.getenv("NUM_ITERATIONS", 100)),
        help="Total number of training iterations.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=float(os.getenv("LEARNING_RATE", 1e-5)),
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--shift",
        type=float,
        default=float(os.getenv("SHIFT", 3.0)),
        help="Flow matching shift parameter. Common values lie between 0.0-4.0.",
    )
    parser.add_argument(
        "--scale_lr",
        type=lambda x: safely_eval_as_bool(x),
        default=safely_eval_as_bool(os.getenv("SCALE_LR", "false")),
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default=os.getenv("LR_SCHEDULER", "constant"),
        help="The scheduler type to use. Choose between"
        '["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"].',
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=int(os.getenv("LR_WARMUP_STEPS", 500)),
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=int(os.getenv("DATALOADER_NUM_WORKERS", 0)),
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=float(os.getenv("ADAM_BETA1", 0.9)),
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=float(os.getenv("ADAM_BETA2", 0.999)),
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay",
        type=float,
        default=float(os.getenv("ADAM_WEIGHT_DECAY", 0.0)),
        help="Weight decay to use.",
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=float(os.getenv("ADAM_EPSILON", 1e-08)),
        help="Epsilon value for the Adam optimizer.",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=float(os.getenv("MAX_GRAD_NORM", 0.0)),
        help="Max gradient norm.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=int(os.getenv("NUM_FRAMES", 49)),
        help="Maximum number of frames to train on at once. Only relevant for video models.",
    )
    parser.add_argument(
        "--use_lora",
        default=os.getenv("USE_LORA", "false"),
        type=lambda x: safely_eval_as_bool(x),
        help="Whether to use lora for fine-tuning purposes. Accepts 'true', 'false', '1', or '0'.",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=int(os.getenv("LORA_RANK", 8)),
        help="The LoRA rank to use while training.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        default=os.getenv("GRADIENT_CHECKPOINTING", 0),
        type=float,
        help=(
            "Percentage of gradient checkpointing as a float between 0 (disabled) and 1 (full checkp.). "
            "NB! If provided value is strictly between 0 and 1, "
            "only the corresponding proportion of the transformer blocks will be checkpointed. "
            "However, setting `gradient_checkpointing=1` checkpoints the whole model, not only the transf. blocks."
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=int(os.getenv("LOCAL_RANK", -1)),
        help="For distributed training: local_rank.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default=os.getenv("REPORT_TO", "tensorboard"),
        help="The integration to report the results and logs to."
        'Supported platforms are "tensorboard" (default), "wandb" and "comet_ml". Use "all" to report to all integrations.',
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default=os.getenv("LOGGING_DIR", "outputs/runs"),
        help="Experiment tracker (see args.report_to) log directory. Defaults to `logs`.",
    )
    parser.add_argument(
        "--profiling_step",
        type=int,
        default=int(os.getenv("PROFILING_STEP", -1)),
        help="Determines on which training iter to perform the profiling. Negative numbers disable the profiling (default).",
    )
    parser.add_argument(
        "--profiling_logging_dir",
        type=str,
        default=os.getenv("PROFILING_LOGGING_DIR", "outputs/runs/profile"),
        help="The logging directory where the profiling trace-files (JSON) will be written.",
    )
    args = parser.parse_args()
    args.cache_dir = os.path.join(
        args.cache_dir,
        f"{args.model}-{'x'.join(map(str, args.resolution))}",
    )
    _, _, model_output_type = ModelManager().get_model(args.model)
    if model_output_type == "video":
        args.cache_dir += f"x{args.num_frames}"

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.train_data_path is None:
        raise ValueError("Need to specify a training folder.")

    return args


def main():
    """
    Main function to execute the training process.

    This function parses the command-line arguments, initializes the Trainer
    with the provided arguments, and starts the training loop.

    """
    # Get arguments
    args = parse_args()

    # Run training loop
    trainer = Trainer(args)
    trainer.run_training()


if __name__ == "__main__":
    main()
