import argparse
import os
from typing import Dict, Iterable, List, Optional, Union

import numpy as np
import torch
import torchvision

from src.utils import safely_eval_as_bool


def tokenize_captions(
    examples: dict, tokenizer, tokenizer_2
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tokenize captions extracted from image filenames using two tokenizers.

    Args:
        examples (dict): A dictionary containing the input data.
        tokenizer (Tokenizer): The first tokenizer to process the captions.
        tokenizer_2 (Tokenizer): The second tokenizer to process the captions.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - inputs (torch.Tensor): Tokenized captions from the first tokenizer.
            - inputs_2 (torch.Tensor): Tokenized captions from the second tokenizer.
    """
    captions = []
    if "prompt" in examples.keys():
        captions = examples["prompt"]
    # else:
    #     for example in examples["image"]:
    #         path = example.filename
    #         filename = os.path.splitext(os.path.basename(path))[0]
    #         caption = filename.replace("_", " ")
    #         captions.append(caption)
    else:
        for i, img in enumerate(examples["image"]):
            # try several likely places for a path
            path = getattr(img, "filename", None)  # PIL Image opened from disk
            if path is None and isinstance(img, dict):
                path = img.get("path")             # HF datasets when decode=False
            if path is None and "image_path" in examples:
                path = examples["image_path"][i]   # custom parallel column if you have one
    
            if path:
                filename = os.path.splitext(os.path.basename(path))[0]
                caption = filename.replace("_", " ")
            else:
                caption = f"image_{i}"             
    
            captions.append(caption)

    inputs = tokenizer(captions)
    input_ids = inputs.input_ids
    if isinstance(input_ids, list):
        input_ids = torch.tensor(input_ids)  # Because of Wan2.1 I2V

    if hasattr(inputs, "attention_mask"):
        input_mask = inputs.attention_mask
        if isinstance(input_mask, list):
            input_mask = torch.tensor(input_mask)  # Because of Wan2.1 I2V
    else:
        input_mask = None

    input_ids_2 = None
    input_mask_2 = None
    if tokenizer_2 is not None:
        inputs_2 = tokenizer_2(captions)
        input_ids_2 = inputs_2.input_ids
        if hasattr(inputs_2, "attention_mask"):
            input_mask_2 = inputs_2.attention_mask

    return input_ids, input_mask, input_ids_2, input_mask_2


def _preprocess_images(
    images: Iterable,
    train_transforms: callable = None,
) -> List[Dict[str, torch.Tensor]]:

    dataset = [image.convert("RGB") for image in images]
    if train_transforms:
        dataset = [train_transforms(data) for data in dataset]

    outputs: Dict[str, List[torch.Tensor]] = {"pixel_values": dataset}
    return outputs


# def _preprocess_videos(
#     videos: Iterable,
#     num_frames: int,
#     train_transforms: callable = None,
#     image_processor: callable = None,
# ) -> List[Dict[str, torch.Tensor]]:
#     outputs: Dict[str, List[torch.Tensor]] = {"pixel_values": []}
#     if image_processor is not None:
#         outputs["processed_image"] = []

#     for video in videos:
#         current_length = len(video)

#         if num_frames > current_length:
#             if safely_eval_as_bool(os.getenv("PAD_VIDEOS_TO_NUM_FRAMES", "false")):
#                 # If num_frames is greater than the current length, pad with last frame
#                 pad_length = num_frames - current_length
#                 video = video.get_batch(list(range(current_length))).asnumpy()
#                 video = np.concatenate(
#                     [
#                         video,
#                         np.tile(video[-1], (pad_length, 1, 1, 1)),
#                     ],
#                     axis=0,
#                 )
#             else:
#                 raise ValueError(
#                     f"num_frames={num_frames} is longer than input video length {current_length}"
#                 )
#         else:
#             video = video.get_batch(list(range(num_frames))).asnumpy()

#         if image_processor is not None:
#             image = video[0]  # first frame of the video
#             image = image_processor(images=image, return_tensors="pt")["pixel_values"][
#                 0
#             ]
#             outputs["processed_image"].append(image)

#         if train_transforms:
#             video = torch.stack(
#                 [
#                     train_transforms(torchvision.transforms.ToPILImage()(frame))
#                     for frame in video
#                 ]
#             )

#         outputs["pixel_values"].append(video)

#     return outputs

from decord import VideoReader, cpu
import torch

def _preprocess_videos(
    videos: Iterable,
    num_frames: int,
    train_transforms: callable = None,
    image_processor: callable = None,
) -> List[Dict[str, torch.Tensor]]:
    outputs: Dict[str, List[torch.Tensor]] = {"pixel_values": []}
    if image_processor is not None:
        outputs["processed_image"] = []

    for video in videos:
        if not hasattr(video, "__len__"):
            if isinstance(video, str):
                video = VideoReader(video, ctx=cpu(0))
            else:
                frames = [f["data"] for f in video]  # TorchVision returns dicts
                video = torch.stack(frames).permute(0, 2, 3, 1).numpy()

        if hasattr(video, "__len__"):
            current_length = len(video)
        else:
            current_length = video.shape[0]

        if num_frames > current_length:
            pad_length = num_frames - current_length
            video = video.get_batch(list(range(current_length))).asnumpy() if isinstance(video, VideoReader) else video
            video = np.concatenate([video, np.tile(video[-1], (pad_length, 1, 1, 1))], axis=0)
        else:
            if isinstance(video, VideoReader):
                video = video.get_batch(list(range(num_frames))).asnumpy()
            else:
                video = video[:num_frames]

        if image_processor is not None:
            image = video[0]
            image = image_processor(images=image, return_tensors="pt")["pixel_values"][0]
            outputs["processed_image"].append(image)
        if train_transforms:
            video = torch.stack([
                train_transforms(torchvision.transforms.ToPILImage()(frame))
                for frame in video
            ])

        outputs["pixel_values"].append(video)

    return outputs



def preprocess_train(
    examples: dict[str, torch.Tensor],
    args: argparse.Namespace,
    tokenizer: callable,
    tokenizer_2: callable,
    image_processor: callable = None,
    train_transforms: callable = None,
    ignore_keys: List[str] = ["image", "video", "prompt"],
) -> dict[str, torch.Tensor]:
    """Preprocess training examples by transforming images and tokenizing captions.

    Args:
        examples (dict): A dictionary containing the input data.
        args (argparse.Namespace): Training args
        tokenizer (Tokenizer): The first tokenizer to process the captions.
        tokenizer_2 (Tokenizer): The second tokenizer to process the captions.
        image_processor: (ImageProcessor): Optional image preprocessor
        train_transforms (callable): A function or transform to apply to each image.
        ignore_keys (list): A list of keys to be excluded from the data as a postprocessing step.

    Returns:
        dict: The updated examples dictionary with the following keys added:
            - "pixel_values": List of transformed image/video tensors.
            - "input_ids": Tensor of tokenized captions from the first tokenizer.
            - "input_mask": First tokenizer attention mask.
            - "input_ids_2" (optional): Tensor of tokenized captions from the optional second tokenizer.
            - "input_mask_2" (optional): Optional second tokenizer attention mask.
    """
    if "image" in examples.keys():
        images = examples["image"]
        outputs = _preprocess_images(
            images=images,
            train_transforms=train_transforms,
        )

    elif "video" in examples.keys():
        num_frames = args.num_frames
        videos = examples["video"]
        outputs = _preprocess_videos(
            videos=videos,
            num_frames=num_frames,
            train_transforms=train_transforms,
            image_processor=image_processor,
        )
    else:
        raise NotImplementedError("Only 'image' and 'video' are supported currently.")

    # Tokenize captions: these may potentially contain null values
    keys = ["input_ids", "input_mask", "input_ids_2", "input_mask_2"]
    values = tokenize_captions(examples, tokenizer, tokenizer_2)
    outputs.update(dict(zip(keys, values)))

    # Only keep the non-null fields, that are not ignored by `ignore_keys`
    keys_to_remove = [
        key for key, value in outputs.items() if value is None or key in ignore_keys
    ]
    # Remove the keys marked for removal
    for key in keys_to_remove:
        outputs.pop(key)

    return outputs


def collate_fn(
    sample_dicts: List[Dict[str, Union[np.ndarray, torch.Tensor]]],
    output_as_numpy: bool = False,
    combine_with: callable = torch.stack,
    included_keys: Optional[List[str]] = None,
) -> Dict[str, torch.Tensor]:
    """Collate a list of examples into a batch for DataLoader.

    Args:
        sample_dicts (list): A list of sample dictionaries,
            each containing data to be collated, where data can be torch.Tensor or np.ndarray.
        output_as_numpy (bool): Whether to return the collated data as numpy.ndarray or torch.Tensor.
        combine_with (callable): Function to combine the data with (e.g., torch.stack or torch.cat).
        included_keys (optional list): If specified, only these keys will be kept for the collation.
            Otherwise all available keys will be kept.
    Returns:
        dict: A dictionary containing batched tensors as
            np.ndarray (output_as_numpy=True) or torch.Tensor (output_as_numpy=False).
    """

    def _convert_to_torch_tensor(
        x: Union[torch.Tensor, np.ndarray],
    ) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x)

        return x

    def _convert_to_target_instance(
        x: Union[torch.Tensor, np.ndarray],
    ) -> Union[torch.Tensor, np.ndarray]:
        if output_as_numpy:
            if isinstance(x, torch.Tensor):
                return x.numpy()

        elif isinstance(x, np.ndarray):
            return torch.from_numpy(x)

        return x

    batch = {}

    # List of keys to check for and collate
    included_keys = included_keys or set(sample_dicts[0].keys())

    # Check for each key in possible_keys, collating if present in examples
    for key in included_keys:
        value = combine_with(
            [_convert_to_torch_tensor(example[key]) for example in sample_dicts]
        )

        # Optionally, apply specific tensor formatting (e.g., contiguous) for certain keys
        if key == "pixel_values":
            value = value.to(memory_format=torch.contiguous_format).float()

        batch[key] = _convert_to_target_instance(value)

    return batch
