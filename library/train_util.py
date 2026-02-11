# common functions for training

import argparse
import ast
import asyncio
from concurrent.futures import Future, ThreadPoolExecutor
import datetime
import importlib
import json
import logging
import pathlib
import re
import shutil
import time
import typing
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union
from accelerate import Accelerator, InitProcessGroupKwargs, DistributedDataParallelKwargs, PartialState
import glob
import math
import os
import random
import hashlib
import subprocess
from io import BytesIO
import toml

# from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
from packaging.version import Version

import torch
from library.device_utils import init_ipex, clean_memory_on_device
from library.strategy_base import LatentsCachingStrategy, TokenizeStrategy, TextEncoderOutputsCachingStrategy, TextEncodingStrategy

init_ipex()

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torchvision import transforms
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection
import transformers
from diffusers.optimization import (
    SchedulerType as DiffusersSchedulerType,
    TYPE_TO_SCHEDULER_FUNCTION as DIFFUSERS_TYPE_TO_SCHEDULER_FUNCTION,
)
from transformers.optimization import SchedulerType, TYPE_TO_SCHEDULER_FUNCTION
from diffusers import (
    StableDiffusionPipeline,
    DDPMScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    KDPM2DiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    AutoencoderKL,
)
from library import custom_train_functions, sd3_utils
from library.original_unet import UNet2DConditionModel
from huggingface_hub import hf_hub_download
import numpy as np
from PIL import Image
import imagesize
import cv2
import safetensors.torch
from library.lpw_stable_diffusion import StableDiffusionLongPromptWeightingPipeline
from library.sdxl_lpw_stable_diffusion import SdxlStableDiffusionLongPromptWeightingPipeline
import library.model_util as model_util
import library.huggingface_util as huggingface_util
import library.sai_model_spec as sai_model_spec
import library.deepspeed_utils as deepspeed_utils
from library.utils import setup_logging, resize_image, validate_interpolation_fn

setup_logging()
import logging

logger = logging.getLogger(__name__)
# from library.attention_processors import FlashAttnProcessor
# from library.hypernetwork import replace_attentions_for_hypernetwork
from library.original_unet import UNet2DConditionModel

HIGH_VRAM = False

# checkpoint?????
EPOCH_STATE_NAME = "{}-{:06d}-state"
EPOCH_FILE_NAME = "{}-{:06d}"
EPOCH_DIFFUSERS_DIR_NAME = "{}-{:06d}"
LAST_STATE_NAME = "{}-state"
DEFAULT_EPOCH_NAME = "epoch"
DEFAULT_LAST_OUTPUT_NAME = "last"

DEFAULT_STEP_NAME = "at"
STEP_STATE_NAME = "{}-step{:08d}-state"
STEP_FILE_NAME = "{}-step{:08d}"
STEP_DIFFUSERS_DIR_NAME = "{}-step{:08d}"

# region dataset

IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".PNG", ".JPG", ".JPEG", ".WEBP", ".BMP"]

try:
    import pillow_avif

    IMAGE_EXTENSIONS.extend([".avif", ".AVIF"])
except:
    pass

# JPEG-XL on Linux
try:
    from jxlpy import JXLImagePlugin
    from library.jpeg_xl_util import get_jxl_size

    IMAGE_EXTENSIONS.extend([".jxl", ".JXL"])
except:
    pass

# JPEG-XL on Linux and Windows
try:
    import pillow_jxl
    from library.jpeg_xl_util import get_jxl_size

    IMAGE_EXTENSIONS.extend([".jxl", ".JXL"])
except:
    pass

IMAGE_TRANSFORMS = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

TEXT_ENCODER_OUTPUTS_CACHE_SUFFIX = "_te_outputs.npz"
TEXT_ENCODER_OUTPUTS_CACHE_SUFFIX_SD3 = "_sd3_te.npz"


def split_train_val(
    paths: List[str],
    sizes: List[Optional[Tuple[int, int]]],
    is_training_dataset: bool,
    validation_split: float,
    validation_seed: int | None,
) -> Tuple[List[str], List[Optional[Tuple[int, int]]]]:
    """
    Split the dataset into train and validation

    Shuffle the dataset based on the validation_seed or the current random seed.
    For example if the split of 0.2 of 100 images.
    [0:80] = 80 training images
    [80:] = 20 validation images
    """
    dataset = list(zip(paths, sizes))
    if validation_seed is not None:
        logging.info(f"Using validation seed: {validation_seed}")
        prevstate = random.getstate()
        random.seed(validation_seed)
        random.shuffle(dataset)
        random.setstate(prevstate)
    else:
        random.shuffle(dataset)

    paths, sizes = zip(*dataset)
    paths = list(paths)
    sizes = list(sizes)
    # Split the dataset between training and validation
    if is_training_dataset:
        # Training dataset we split to the first part
        split = math.ceil(len(paths) * (1 - validation_split))
        return paths[0:split], sizes[0:split]
    else:
        # Validation dataset we split to the second part
        split = len(paths) - round(len(paths) * validation_split)
        return paths[split:], sizes[split:]


class ImageInfo:
    def __init__(self, image_key: str, num_repeats: int, caption: str, is_reg: bool, absolute_path: str) -> None:
        self.image_key: str = image_key
        self.num_repeats: int = num_repeats
        self.caption: str = caption
        self.is_reg: bool = is_reg
        self.absolute_path: str = absolute_path
        self.image_size: Tuple[int, int] = None
        self.resized_size: Tuple[int, int] = None
        self.bucket_reso: Tuple[int, int] = None
        self.latents: Optional[torch.Tensor] = None
        self.latents_flipped: Optional[torch.Tensor] = None
        self.latents_npz: Optional[str] = None  # set in cache_latents
        self.latents_original_size: Optional[Tuple[int, int]] = None  # original image size, not latents size
        self.latents_crop_ltrb: Optional[Tuple[int, int]] = (
            None  # crop left top right bottom in original pixel size, not latents size
        )
        self.cond_img_path: Optional[str] = None
        self.image: Optional[Image.Image] = None  # optional, original PIL Image
        self.text_encoder_outputs_npz: Optional[str] = None  # set in cache_text_encoder_outputs

        # new
        self.text_encoder_outputs: Optional[List[torch.Tensor]] = None
        # old
        self.text_encoder_outputs1: Optional[torch.Tensor] = None
        self.text_encoder_outputs2: Optional[torch.Tensor] = None
        self.text_encoder_pool2: Optional[torch.Tensor] = None

        self.alpha_mask: Optional[torch.Tensor] = None  # alpha mask can be flipped in runtime
        self.resize_interpolation: Optional[str] = None


class BucketManager:
    def __init__(self, no_upscale, max_reso, min_size, max_size, reso_steps) -> None:
        if max_size is not None:
            if max_reso is not None:
                assert max_size >= max_reso[0], "the max_size should be larger than the width of max_reso"
                assert max_size >= max_reso[1], "the max_size should be larger than the height of max_reso"
            if min_size is not None:
                assert max_size >= min_size, "the max_size should be larger than the min_size"

        self.no_upscale = no_upscale
        if max_reso is None:
            self.max_reso = None
            self.max_area = None
        else:
            self.max_reso = max_reso
            self.max_area = max_reso[0] * max_reso[1]
        self.min_size = min_size
        self.max_size = max_size
        self.reso_steps = reso_steps

        self.resos = []
        self.reso_to_id = {}
        self.buckets = []  # ????? (image_key, image, original size, crop left/top)????? image_key

    def add_image(self, reso, image_or_info):
        bucket_id = self.reso_to_id[reso]
        self.buckets[bucket_id].append(image_or_info)

    def shuffle(self):
        for bucket in self.buckets:
            random.shuffle(bucket)

    def sort(self):
        # ??????????????????????????????????????buckets??????reso_to_id?????
        sorted_resos = self.resos.copy()
        sorted_resos.sort()

        sorted_buckets = []
        sorted_reso_to_id = {}
        for i, reso in enumerate(sorted_resos):
            bucket_id = self.reso_to_id[reso]
            sorted_buckets.append(self.buckets[bucket_id])
            sorted_reso_to_id[reso] = i

        self.resos = sorted_resos
        self.buckets = sorted_buckets
        self.reso_to_id = sorted_reso_to_id

    def make_buckets(self):
        resos = model_util.make_bucket_resolutions(self.max_reso, self.min_size, self.max_size, self.reso_steps)
        self.set_predefined_resos(resos)

    def set_predefined_resos(self, resos):
        # ????????????????aspect ratio??????????
        self.predefined_resos = resos.copy()
        self.predefined_resos_set = set(resos)
        self.predefined_aspect_ratios = np.array([w / h for w, h in resos])

    def add_if_new_reso(self, reso):
        if reso not in self.reso_to_id:
            bucket_id = len(self.resos)
            self.reso_to_id[reso] = bucket_id
            self.resos.append(reso)
            self.buckets.append([])
            # logger.info(reso, bucket_id, len(self.buckets))

    def round_to_steps(self, x):
        x = int(x + 0.5)
        return x - x % self.reso_steps

    def select_bucket(self, image_width, image_height):
        aspect_ratio = image_width / image_height
        if not self.no_upscale:
            # ??????????
            # ??aspect ratio????????????fine tuning??no_upscale=True???????????????????????
            reso = (image_width, image_height)
            if reso in self.predefined_resos_set:
                pass
            else:
                ar_errors = self.predefined_aspect_ratios - aspect_ratio
                predefined_bucket_id = np.abs(ar_errors).argmin()  # ????????aspect ratio error????????
                reso = self.predefined_resos[predefined_bucket_id]

            ar_reso = reso[0] / reso[1]
            if aspect_ratio > ar_reso:  # ???????????
                scale = reso[1] / image_height
            else:
                scale = reso[0] / image_width

            resized_size = (int(image_width * scale + 0.5), int(image_height * scale + 0.5))
            # logger.info(f"use predef, {image_width}, {image_height}, {reso}, {resized_size}")
        else:
            # ???????
            if image_width * image_height > self.max_area:
                # ????????????????????????????????bucket????
                resized_width = math.sqrt(self.max_area * aspect_ratio)
                resized_height = self.max_area / resized_width
                assert abs(resized_width / resized_height - aspect_ratio) < 1e-2, "aspect is illegal"

                # ??????????????reso_steps??????aspect ratio???????????
                # ??bucketing???????
                b_width_rounded = self.round_to_steps(resized_width)
                b_height_in_wr = self.round_to_steps(b_width_rounded / aspect_ratio)
                ar_width_rounded = b_width_rounded / b_height_in_wr

                b_height_rounded = self.round_to_steps(resized_height)
                b_width_in_hr = self.round_to_steps(b_height_rounded * aspect_ratio)
                ar_height_rounded = b_width_in_hr / b_height_rounded

                # logger.info(b_width_rounded, b_height_in_wr, ar_width_rounded)
                # logger.info(b_width_in_hr, b_height_rounded, ar_height_rounded)

                if abs(ar_width_rounded - aspect_ratio) < abs(ar_height_rounded - aspect_ratio):
                    resized_size = (b_width_rounded, int(b_width_rounded / aspect_ratio + 0.5))
                else:
                    resized_size = (int(b_height_rounded * aspect_ratio + 0.5), b_height_rounded)
                # logger.info(resized_size)
            else:
                resized_size = (image_width, image_height)  # ???????

            # ?????????bucket????????padding???cropping???
            bucket_width = resized_size[0] - resized_size[0] % self.reso_steps
            bucket_height = resized_size[1] - resized_size[1] % self.reso_steps
            # logger.info(f"use arbitrary {image_width}, {image_height}, {resized_size}, {bucket_width}, {bucket_height}")

            reso = (bucket_width, bucket_height)

        self.add_if_new_reso(reso)

        ar_error = (reso[0] / reso[1]) - aspect_ratio
        return reso, resized_size, ar_error

    @staticmethod
    def get_crop_ltrb(bucket_reso: Tuple[int, int], image_size: Tuple[int, int]):
        # Stability AI?????????crop left/top??????crop right?flip?augmentation???????
        # Calculate crop left/top according to the preprocessing of Stability AI. Crop right is calculated for flip augmentation.

        bucket_ar = bucket_reso[0] / bucket_reso[1]
        image_ar = image_size[0] / image_size[1]
        if bucket_ar > image_ar:
            # bucket?????????????
            resized_width = bucket_reso[1] * image_ar
            resized_height = bucket_reso[1]
        else:
            resized_width = bucket_reso[0]
            resized_height = bucket_reso[0] / image_ar
        crop_left = (bucket_reso[0] - resized_width) // 2
        crop_top = (bucket_reso[1] - resized_height) // 2
        crop_right = crop_left + resized_width
        crop_bottom = crop_top + resized_height
        return crop_left, crop_top, crop_right, crop_bottom


class BucketBatchIndex(NamedTuple):
    bucket_index: int
    bucket_batch_size: int
    batch_index: int


class AugHelper:
    # albumentations?????????????????interface?????

    def __init__(self):
        pass

    def color_aug(self, image: np.ndarray):
        # self.color_aug_method = albu.OneOf(
        #     [
        #         albu.HueSaturationValue(8, 0, 0, p=0.5),
        #         albu.RandomGamma((95, 105), p=0.5),
        #     ],
        #     p=0.33,
        # )
        hue_shift_limit = 8

        # remove dependency to albumentations
        if random.random() <= 0.33:
            if random.random() > 0.5:
                # hue shift
                hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                hue_shift = random.uniform(-hue_shift_limit, hue_shift_limit)
                if hue_shift < 0:
                    hue_shift = 180 + hue_shift
                hsv_img[:, :, 0] = (hsv_img[:, :, 0] + hue_shift) % 180
                image = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
            else:
                # random gamma
                gamma = random.uniform(0.95, 1.05)
                image = np.clip(image**gamma, 0, 255).astype(np.uint8)

        return {"image": image}

    def get_augmentor(self, use_color_aug: bool):  # -> Optional[Callable[[np.ndarray], Dict[str, np.ndarray]]]:
        return self.color_aug if use_color_aug else None


class BaseSubset:
    def __init__(
        self,
        image_dir: Optional[str],
        alpha_mask: Optional[bool],
        num_repeats: int,
        shuffle_caption: bool,
        caption_separator: str,
        keep_tokens: int,
        keep_tokens_separator: str,
        secondary_separator: Optional[str],
        enable_wildcard: bool,
        color_aug: bool,
        flip_aug: bool,
        face_crop_aug_range: Optional[Tuple[float, float]],
        random_crop: bool,
        caption_dropout_rate: float,
        caption_dropout_every_n_epochs: int,
        caption_tag_dropout_rate: float,
        caption_prefix: Optional[str],
        caption_suffix: Optional[str],
        token_warmup_min: int,
        token_warmup_step: Union[float, int],
        custom_attributes: Optional[Dict[str, Any]] = None,
        validation_seed: Optional[int] = None,
        validation_split: Optional[float] = 0.0,
        resize_interpolation: Optional[str] = None,
    ) -> None:
        self.image_dir = image_dir
        self.alpha_mask = alpha_mask if alpha_mask is not None else False
        self.num_repeats = num_repeats
        self.shuffle_caption = shuffle_caption
        self.caption_separator = caption_separator
        self.keep_tokens = keep_tokens
        self.keep_tokens_separator = keep_tokens_separator
        self.secondary_separator = secondary_separator
        self.enable_wildcard = enable_wildcard
        self.color_aug = color_aug
        self.flip_aug = flip_aug
        self.face_crop_aug_range = face_crop_aug_range
        self.random_crop = random_crop
        self.caption_dropout_rate = caption_dropout_rate
        self.caption_dropout_every_n_epochs = caption_dropout_every_n_epochs
        self.caption_tag_dropout_rate = caption_tag_dropout_rate
        self.caption_prefix = caption_prefix
        self.caption_suffix = caption_suffix

        self.token_warmup_min = token_warmup_min  # step=0????????
        self.token_warmup_step = token_warmup_step  # N?N<1??N*max_train_steps?????????????????

        self.custom_attributes = custom_attributes if custom_attributes is not None else {}

        self.img_count = 0

        self.validation_seed = validation_seed
        self.validation_split = validation_split

        self.resize_interpolation = resize_interpolation


class DreamBoothSubset(BaseSubset):
    def __init__(
        self,
        image_dir: str,
        is_reg: bool,
        class_tokens: Optional[str],
        caption_extension: str,
        cache_info: bool,
        alpha_mask: bool,
        num_repeats,
        shuffle_caption,
        caption_separator: str,
        keep_tokens,
        keep_tokens_separator,
        secondary_separator,
        enable_wildcard,
        color_aug,
        flip_aug,
        face_crop_aug_range,
        random_crop,
        caption_dropout_rate,
        caption_dropout_every_n_epochs,
        caption_tag_dropout_rate,
        caption_prefix,
        caption_suffix,
        token_warmup_min,
        token_warmup_step,
        custom_attributes: Optional[Dict[str, Any]] = None,
        validation_seed: Optional[int] = None,
        validation_split: Optional[float] = 0.0,
        resize_interpolation: Optional[str] = None,
    ) -> None:
        assert image_dir is not None, "image_dir must be specified / image_dir????????"

        super().__init__(
            image_dir,
            alpha_mask,
            num_repeats,
            shuffle_caption,
            caption_separator,
            keep_tokens,
            keep_tokens_separator,
            secondary_separator,
            enable_wildcard,
            color_aug,
            flip_aug,
            face_crop_aug_range,
            random_crop,
            caption_dropout_rate,
            caption_dropout_every_n_epochs,
            caption_tag_dropout_rate,
            caption_prefix,
            caption_suffix,
            token_warmup_min,
            token_warmup_step,
            custom_attributes=custom_attributes,
            validation_seed=validation_seed,
            validation_split=validation_split,
            resize_interpolation=resize_interpolation,
        )

        self.is_reg = is_reg
        self.class_tokens = class_tokens
        self.caption_extension = caption_extension
        if self.caption_extension and not self.caption_extension.startswith("."):
            self.caption_extension = "." + self.caption_extension
        self.cache_info = cache_info

    def __eq__(self, other) -> bool:
        if not isinstance(other, DreamBoothSubset):
            return NotImplemented
        return self.image_dir == other.image_dir


class FineTuningSubset(BaseSubset):
    def __init__(
        self,
        image_dir,
        metadata_file: str,
        alpha_mask: bool,
        num_repeats,
        shuffle_caption,
        caption_separator,
        keep_tokens,
        keep_tokens_separator,
        secondary_separator,
        enable_wildcard,
        color_aug,
        flip_aug,
        face_crop_aug_range,
        random_crop,
        caption_dropout_rate,
        caption_dropout_every_n_epochs,
        caption_tag_dropout_rate,
        caption_prefix,
        caption_suffix,
        token_warmup_min,
        token_warmup_step,
        custom_attributes: Optional[Dict[str, Any]] = None,
        validation_seed: Optional[int] = None,
        validation_split: Optional[float] = 0.0,
        resize_interpolation: Optional[str] = None,
    ) -> None:
        assert metadata_file is not None, "metadata_file must be specified / metadata_file????????"

        super().__init__(
            image_dir,
            alpha_mask,
            num_repeats,
            shuffle_caption,
            caption_separator,
            keep_tokens,
            keep_tokens_separator,
            secondary_separator,
            enable_wildcard,
            color_aug,
            flip_aug,
            face_crop_aug_range,
            random_crop,
            caption_dropout_rate,
            caption_dropout_every_n_epochs,
            caption_tag_dropout_rate,
            caption_prefix,
            caption_suffix,
            token_warmup_min,
            token_warmup_step,
            custom_attributes=custom_attributes,
            validation_seed=validation_seed,
            validation_split=validation_split,
            resize_interpolation=resize_interpolation,
        )

        self.metadata_file = metadata_file

    def __eq__(self, other) -> bool:
        if not isinstance(other, FineTuningSubset):
            return NotImplemented
        return self.metadata_file == other.metadata_file


class ControlNetSubset(BaseSubset):
    def __init__(
        self,
        image_dir: str,
        conditioning_data_dir: str,
        caption_extension: str,
        cache_info: bool,
        num_repeats,
        shuffle_caption,
        caption_separator,
        keep_tokens,
        keep_tokens_separator,
        secondary_separator,
        enable_wildcard,
        color_aug,
        flip_aug,
        face_crop_aug_range,
        random_crop,
        caption_dropout_rate,
        caption_dropout_every_n_epochs,
        caption_tag_dropout_rate,
        caption_prefix,
        caption_suffix,
        token_warmup_min,
        token_warmup_step,
        custom_attributes: Optional[Dict[str, Any]] = None,
        validation_seed: Optional[int] = None,
        validation_split: Optional[float] = 0.0,
        resize_interpolation: Optional[str] = None,
    ) -> None:
        assert image_dir is not None, "image_dir must be specified / image_dir????????"

        super().__init__(
            image_dir,
            False,  # alpha_mask
            num_repeats,
            shuffle_caption,
            caption_separator,
            keep_tokens,
            keep_tokens_separator,
            secondary_separator,
            enable_wildcard,
            color_aug,
            flip_aug,
            face_crop_aug_range,
            random_crop,
            caption_dropout_rate,
            caption_dropout_every_n_epochs,
            caption_tag_dropout_rate,
            caption_prefix,
            caption_suffix,
            token_warmup_min,
            token_warmup_step,
            custom_attributes=custom_attributes,
            validation_seed=validation_seed,
            validation_split=validation_split,
            resize_interpolation=resize_interpolation,
        )

        self.conditioning_data_dir = conditioning_data_dir
        self.caption_extension = caption_extension
        if self.caption_extension and not self.caption_extension.startswith("."):
            self.caption_extension = "." + self.caption_extension
        self.cache_info = cache_info

    def __eq__(self, other) -> bool:
        if not isinstance(other, ControlNetSubset):
            return NotImplemented
        return self.image_dir == other.image_dir and self.conditioning_data_dir == other.conditioning_data_dir


class BaseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        resolution: Optional[Tuple[int, int]],
        network_multiplier: float,
        debug_dataset: bool,
        resize_interpolation: Optional[str] = None,
    ) -> None:
        super().__init__()

        # width/height is used when enable_bucket==False
        self.width, self.height = (None, None) if resolution is None else resolution
        self.network_multiplier = network_multiplier
        self.debug_dataset = debug_dataset

        self.subsets: List[Union[DreamBoothSubset, FineTuningSubset]] = []

        self.token_padding_disabled = False
        self.tag_frequency = {}
        self.XTI_layers = None
        self.token_strings = None

        self.enable_bucket = False
        self.bucket_manager: BucketManager = None  # not initialized
        self.min_bucket_reso = None
        self.max_bucket_reso = None
        self.bucket_reso_steps = None
        self.bucket_no_upscale = None
        self.bucket_info = None  # for metadata

        self.current_epoch: int = 0  # ???????epoch??????????????????????????

        self.current_step: int = 0
        self.max_train_steps: int = 0
        self.seed: int = 0

        # augmentation
        self.aug_helper = AugHelper()

        self.image_transforms = IMAGE_TRANSFORMS

        if resize_interpolation is not None:
            assert validate_interpolation_fn(
                resize_interpolation
            ), f'Resize interpolation "{resize_interpolation}" is not a valid interpolation'
        self.resize_interpolation = resize_interpolation

        self.image_data: Dict[str, ImageInfo] = {}
        self.image_to_subset: Dict[str, Union[DreamBoothSubset, FineTuningSubset]] = {}

        self.replacements = {}

        # caching
        self.caching_mode = None  # None, 'latents', 'text'

        self.tokenize_strategy = None
        self.text_encoder_output_caching_strategy = None
        self.latents_caching_strategy = None

    def set_current_strategies(self):
        self.tokenize_strategy = TokenizeStrategy.get_strategy()
        self.text_encoder_output_caching_strategy = TextEncoderOutputsCachingStrategy.get_strategy()
        self.latents_caching_strategy = LatentsCachingStrategy.get_strategy()

    def adjust_min_max_bucket_reso_by_steps(
        self, resolution: Tuple[int, int], min_bucket_reso: int, max_bucket_reso: int, bucket_reso_steps: int
    ) -> Tuple[int, int]:
        # make min/max bucket reso to be multiple of bucket_reso_steps
        if min_bucket_reso % bucket_reso_steps != 0:
            adjusted_min_bucket_reso = min_bucket_reso - min_bucket_reso % bucket_reso_steps
            logger.warning(
                f"min_bucket_reso is adjusted to be multiple of bucket_reso_steps"
                f" / min_bucket_reso?bucket_reso_steps????????????????: {min_bucket_reso} -> {adjusted_min_bucket_reso}"
            )
            min_bucket_reso = adjusted_min_bucket_reso
        if max_bucket_reso % bucket_reso_steps != 0:
            adjusted_max_bucket_reso = max_bucket_reso + bucket_reso_steps - max_bucket_reso % bucket_reso_steps
            logger.warning(
                f"max_bucket_reso is adjusted to be multiple of bucket_reso_steps"
                f" / max_bucket_reso?bucket_reso_steps????????????????: {max_bucket_reso} -> {adjusted_max_bucket_reso}"
            )
            max_bucket_reso = adjusted_max_bucket_reso

        assert (
            min(resolution) >= min_bucket_reso
        ), f"min_bucket_reso must be equal or less than resolution / min_bucket_reso???????????????????????????min_bucket_reso??????????"
        assert (
            max(resolution) <= max_bucket_reso
        ), f"max_bucket_reso must be equal or greater than resolution / max_bucket_reso???????????????????????????min_bucket_reso??????????"

        return min_bucket_reso, max_bucket_reso

    def set_seed(self, seed):
        self.seed = seed

    def set_caching_mode(self, mode):
        self.caching_mode = mode

    def set_current_epoch(self, epoch):
        if not self.current_epoch == epoch:  # epoch???????????????????
            if epoch > self.current_epoch:
                logger.info("epoch is incremented. current_epoch: {}, epoch: {}".format(self.current_epoch, epoch))
                num_epochs = epoch - self.current_epoch
                for _ in range(num_epochs):
                    self.current_epoch += 1
                    self.shuffle_buckets()
                # self.current_epoch seem to be set to 0 again in the next epoch. it may be caused by skipped_dataloader?
            else:
                logger.warning("epoch is not incremented. current_epoch: {}, epoch: {}".format(self.current_epoch, epoch))
                self.current_epoch = epoch

    def set_current_step(self, step):
        self.current_step = step

    def set_max_train_steps(self, max_train_steps):
        self.max_train_steps = max_train_steps

    def set_tag_frequency(self, dir_name, captions):
        frequency_for_dir = self.tag_frequency.get(dir_name, {})
        self.tag_frequency[dir_name] = frequency_for_dir
        for caption in captions:
            for tag in caption.split(","):
                tag = tag.strip()
                if tag:
                    tag = tag.lower()
                    frequency = frequency_for_dir.get(tag, 0)
                    frequency_for_dir[tag] = frequency + 1

    def disable_token_padding(self):
        self.token_padding_disabled = True

    def enable_XTI(self, layers=None, token_strings=None):
        self.XTI_layers = layers
        self.token_strings = token_strings

    def add_replacement(self, str_from, str_to):
        self.replacements[str_from] = str_to

    def process_caption(self, subset: BaseSubset, caption):
        # caption ? prefix/suffix ????
        if subset.caption_prefix:
            caption = subset.caption_prefix + " " + caption
        if subset.caption_suffix:
            caption = caption + " " + subset.caption_suffix

        # dropout????tag drop??????????????????????
        is_drop_out = subset.caption_dropout_rate > 0 and random.random() < subset.caption_dropout_rate
        is_drop_out = (
            is_drop_out
            or subset.caption_dropout_every_n_epochs > 0
            and self.current_epoch % subset.caption_dropout_every_n_epochs == 0
        )

        if is_drop_out:
            caption = ""
        else:
            # process wildcards
            if subset.enable_wildcard:
                # if caption is multiline, random choice one line
                if "\n" in caption:
                    caption = random.choice(caption.split("\n"))

                # wildcard is like '{aaa|bbb|ccc...}'
                # escape the curly braces like {{ or }}
                replacer1 = "?"
                replacer2 = "?"
                while replacer1 in caption or replacer2 in caption:
                    replacer1 += "?"
                    replacer2 += "?"

                caption = caption.replace("{{", replacer1).replace("}}", replacer2)

                # replace the wildcard
                def replace_wildcard(match):
                    return random.choice(match.group(1).split("|"))

                caption = re.sub(r"\{([^}]+)\}", replace_wildcard, caption)

                # unescape the curly braces
                caption = caption.replace(replacer1, "{").replace(replacer2, "}")
            else:
                # if caption is multiline, use the first line
                caption = caption.split("\n")[0]

            if subset.shuffle_caption or subset.token_warmup_step > 0 or subset.caption_tag_dropout_rate > 0:
                fixed_tokens = []
                flex_tokens = []
                fixed_suffix_tokens = []
                if (
                    hasattr(subset, "keep_tokens_separator")
                    and subset.keep_tokens_separator
                    and subset.keep_tokens_separator in caption
                ):
                    fixed_part, flex_part = caption.split(subset.keep_tokens_separator, 1)
                    if subset.keep_tokens_separator in flex_part:
                        flex_part, fixed_suffix_part = flex_part.split(subset.keep_tokens_separator, 1)
                        fixed_suffix_tokens = [t.strip() for t in fixed_suffix_part.split(subset.caption_separator) if t.strip()]

                    fixed_tokens = [t.strip() for t in fixed_part.split(subset.caption_separator) if t.strip()]
                    flex_tokens = [t.strip() for t in flex_part.split(subset.caption_separator) if t.strip()]
                else:
                    tokens = [t.strip() for t in caption.strip().split(subset.caption_separator)]
                    flex_tokens = tokens[:]
                    if subset.keep_tokens > 0:
                        fixed_tokens = flex_tokens[: subset.keep_tokens]
                        flex_tokens = tokens[subset.keep_tokens :]

                if subset.token_warmup_step < 1:  # ????????
                    subset.token_warmup_step = math.floor(subset.token_warmup_step * self.max_train_steps)
                if subset.token_warmup_step and self.current_step < subset.token_warmup_step:
                    tokens_len = (
                        math.floor(
                            (self.current_step) * ((len(flex_tokens) - subset.token_warmup_min) / (subset.token_warmup_step))
                        )
                        + subset.token_warmup_min
                    )
                    flex_tokens = flex_tokens[:tokens_len]

                def dropout_tags(tokens):
                    if subset.caption_tag_dropout_rate <= 0:
                        return tokens
                    l = []
                    for token in tokens:
                        if random.random() >= subset.caption_tag_dropout_rate:
                            l.append(token)
                    return l

                if subset.shuffle_caption:
                    random.shuffle(flex_tokens)

                flex_tokens = dropout_tags(flex_tokens)

                caption = ", ".join(fixed_tokens + flex_tokens + fixed_suffix_tokens)

            # process secondary separator
            if subset.secondary_separator:
                caption = caption.replace(subset.secondary_separator, subset.caption_separator)

            # textual inversion??
            for str_from, str_to in self.replacements.items():
                if str_from == "":
                    # replace all
                    if type(str_to) == list:
                        caption = random.choice(str_to)
                    else:
                        caption = str_to
                else:
                    caption = caption.replace(str_from, str_to)

        return caption

    def get_input_ids(self, caption, tokenizer=None):
        if tokenizer is None:
            tokenizer = self.tokenizers[0]

        input_ids = tokenizer(
            caption, padding="max_length", truncation=True, max_length=self.tokenizer_max_length, return_tensors="pt"
        ).input_ids

        if self.tokenizer_max_length > tokenizer.model_max_length:
            input_ids = input_ids.squeeze(0)
            iids_list = []
            if tokenizer.pad_token_id == tokenizer.eos_token_id:
                # v1
                # 77????? "<BOS> .... <EOS> <EOS> <EOS>" ?????227???????????"<BOS>...<EOS>"????????
                # 1111????? , ????????????????????????
                for i in range(
                    1, self.tokenizer_max_length - tokenizer.model_max_length + 2, tokenizer.model_max_length - 2
                ):  # (1, 152, 75)
                    ids_chunk = (
                        input_ids[0].unsqueeze(0),
                        input_ids[i : i + tokenizer.model_max_length - 2],
                        input_ids[-1].unsqueeze(0),
                    )
                    ids_chunk = torch.cat(ids_chunk)
                    iids_list.append(ids_chunk)
            else:
                # v2 or SDXL
                # 77????? "<BOS> .... <EOS> <PAD> <PAD>..." ?????227???????????"<BOS>...<EOS> <PAD> <PAD> ..."????????
                for i in range(1, self.tokenizer_max_length - tokenizer.model_max_length + 2, tokenizer.model_max_length - 2):
                    ids_chunk = (
                        input_ids[0].unsqueeze(0),  # BOS
                        input_ids[i : i + tokenizer.model_max_length - 2],
                        input_ids[-1].unsqueeze(0),
                    )  # PAD or EOS
                    ids_chunk = torch.cat(ids_chunk)

                    # ??? <EOS> <PAD> ??? <PAD> <PAD> ?????????????
                    # ??? x <PAD/EOS> ??????? <EOS> ?????x <EOS> ???????????
                    if ids_chunk[-2] != tokenizer.eos_token_id and ids_chunk[-2] != tokenizer.pad_token_id:
                        ids_chunk[-1] = tokenizer.eos_token_id
                    # ??? <BOS> <PAD> ... ???? <BOS> <EOS> <PAD> ... ????
                    if ids_chunk[1] == tokenizer.pad_token_id:
                        ids_chunk[1] = tokenizer.eos_token_id

                    iids_list.append(ids_chunk)

            input_ids = torch.stack(iids_list)  # 3,77
        return input_ids

    def register_image(self, info: ImageInfo, subset: BaseSubset):
        self.image_data[info.image_key] = info
        self.image_to_subset[info.image_key] = subset

    def make_buckets(self):
        """
        bucketing????????????????????bucket????
        min_size and max_size are ignored when enable_bucket is False
        """
        logger.info("loading image sizes.")
        for info in tqdm(self.image_data.values()):
            if info.image_size is None:
                info.image_size = self.get_image_size(info.absolute_path)

        # # run in parallel
        # max_workers = min(os.cpu_count(), len(self.image_data))  # TODO consider multi-gpu (processes)
        # with ThreadPoolExecutor(max_workers) as executor:
        #     futures = []
        #     for info in tqdm(self.image_data.values(), desc="loading image sizes"):
        #         if info.image_size is None:
        #             def get_and_set_image_size(info):
        #                 info.image_size = self.get_image_size(info.absolute_path)
        #             futures.append(executor.submit(get_and_set_image_size, info))
        #             # consume futures to reduce memory usage and prevent Ctrl-C hang
        #             if len(futures) >= max_workers:
        #                 for future in futures:
        #                     future.result()
        #                 futures = []
        #     for future in futures:
        #         future.result()

        if self.enable_bucket:
            logger.info("make buckets")
        else:
            logger.info("prepare dataset")

        # bucket????????bucket??????
        if self.enable_bucket:
            if self.bucket_manager is None:  # fine tuning????metadata??????????????????
                self.bucket_manager = BucketManager(
                    self.bucket_no_upscale,
                    (self.width, self.height),
                    self.min_bucket_reso,
                    self.max_bucket_reso,
                    self.bucket_reso_steps,
                )
                if not self.bucket_no_upscale:
                    self.bucket_manager.make_buckets()
                else:
                    logger.warning(
                        "min_bucket_reso and max_bucket_reso are ignored if bucket_no_upscale is set, because bucket reso is defined by image size automatically / bucket_no_upscale??????????bucket??????????????????????min_bucket_reso?max_bucket_reso???????"
                    )

            img_ar_errors = []
            for image_info in self.image_data.values():
                image_width, image_height = image_info.image_size
                image_info.bucket_reso, image_info.resized_size, ar_error = self.bucket_manager.select_bucket(
                    image_width, image_height
                )

                # logger.info(image_info.image_key, image_info.bucket_reso)
                img_ar_errors.append(abs(ar_error))

            self.bucket_manager.sort()
        else:
            self.bucket_manager = BucketManager(False, (self.width, self.height), None, None, None)
            self.bucket_manager.set_predefined_resos([(self.width, self.height)])  # ?????????bucket??
            for image_info in self.image_data.values():
                image_width, image_height = image_info.image_size
                image_info.bucket_reso, image_info.resized_size, _ = self.bucket_manager.select_bucket(image_width, image_height)

        for image_info in self.image_data.values():
            for _ in range(image_info.num_repeats):
                self.bucket_manager.add_image(image_info.bucket_reso, image_info.image_key)

        # bucket??????????
        if self.enable_bucket:
            self.bucket_info = {"buckets": {}}
            logger.info("number of images (including repeats) / ?bucket????????????????")
            for i, (reso, bucket) in enumerate(zip(self.bucket_manager.resos, self.bucket_manager.buckets)):
                count = len(bucket)
                if count > 0:
                    self.bucket_info["buckets"][i] = {"resolution": reso, "count": len(bucket)}
                    logger.info(f"bucket {i}: resolution {reso}, count: {len(bucket)}")

            if len(img_ar_errors) == 0:
                mean_img_ar_error = 0  # avoid NaN
            else:
                img_ar_errors = np.array(img_ar_errors)
                mean_img_ar_error = np.mean(np.abs(img_ar_errors))
            self.bucket_info["mean_img_ar_error"] = mean_img_ar_error
            logger.info(f"mean ar error (without repeats): {mean_img_ar_error}")

        # ??????index??????index?dataset?shuffle??????
        self.buckets_indices: List[BucketBatchIndex] = []
        for bucket_index, bucket in enumerate(self.bucket_manager.buckets):
            batch_count = int(math.ceil(len(bucket) / self.batch_size))
            for batch_index in range(batch_count):
                self.buckets_indices.append(BucketBatchIndex(bucket_index, self.batch_size, batch_index))

        self.shuffle_buckets()
        self._length = len(self.buckets_indices)

    def shuffle_buckets(self):
        # set random seed for this epoch
        random.seed(self.seed + self.current_epoch)

        random.shuffle(self.buckets_indices)
        self.bucket_manager.shuffle()

    def verify_bucket_reso_steps(self, min_steps: int):
        assert self.bucket_reso_steps is None or self.bucket_reso_steps % min_steps == 0, (
            f"bucket_reso_steps is {self.bucket_reso_steps}. it must be divisible by {min_steps}.\n"
            + f"bucket_reso_steps?{self.bucket_reso_steps}???{min_steps}?????????????"
        )

    def is_latent_cacheable(self):
        return all([not subset.color_aug and not subset.random_crop for subset in self.subsets])

    def is_text_encoder_output_cacheable(self):
        return all(
            [
                not (
                    subset.caption_dropout_rate > 0
                    or subset.shuffle_caption
                    or subset.token_warmup_step > 0
                    or subset.caption_tag_dropout_rate > 0
                )
                for subset in self.subsets
            ]
        )

    def new_cache_latents(self, model: Any, accelerator: Accelerator):
        r"""
        a brand new method to cache latents. This method caches latents with caching strategy.
        normal cache_latents method is used by default, but this method is used when caching strategy is specified.
        """
        logger.info("caching latents with caching strategy.")
        caching_strategy = LatentsCachingStrategy.get_strategy()
        image_infos = list(self.image_data.values())

        # sort by resolution
        image_infos.sort(key=lambda info: info.bucket_reso[0] * info.bucket_reso[1])

        # split by resolution and some conditions
        class Condition:
            def __init__(self, reso, flip_aug, alpha_mask, random_crop):
                self.reso = reso
                self.flip_aug = flip_aug
                self.alpha_mask = alpha_mask
                self.random_crop = random_crop

            def __eq__(self, other):
                return (
                    other is not None
                    and self.reso == other.reso
                    and self.flip_aug == other.flip_aug
                    and self.alpha_mask == other.alpha_mask
                    and self.random_crop == other.random_crop
                )

        batch: List[ImageInfo] = []
        current_condition = None

        # support multiple-gpus
        num_processes = accelerator.num_processes
        process_index = accelerator.process_index

        # define a function to submit a batch to cache
        def submit_batch(batch, cond):
            for info in batch:
                if info.image is not None and isinstance(info.image, Future):
                    info.image = info.image.result()  # future to image
            caching_strategy.cache_batch_latents(model, batch, cond.flip_aug, cond.alpha_mask, cond.random_crop)

            # remove image from memory
            for info in batch:
                info.image = None

        # define ThreadPoolExecutor to load images in parallel
        max_workers = min(os.cpu_count(), len(image_infos))
        max_workers = max(1, max_workers // num_processes)  # consider multi-gpu
        max_workers = min(max_workers, caching_strategy.batch_size)  # max_workers should be less than batch_size
        executor = ThreadPoolExecutor(max_workers)

        try:
            # iterate images
            logger.info("caching latents...")
            for i, info in enumerate(tqdm(image_infos)):
                subset = self.image_to_subset[info.image_key]

                if info.latents_npz is not None:  # fine tuning dataset
                    continue

                # check disk cache exists and size of latents
                if caching_strategy.cache_to_disk:
                    # info.latents_npz = os.path.splitext(info.absolute_path)[0] + file_suffix
                    info.latents_npz = caching_strategy.get_latents_npz_path(info.absolute_path, info.image_size)

                    # if the modulo of num_processes is not equal to process_index, skip caching
                    # this makes each process cache different latents
                    if i % num_processes != process_index:
                        continue

                    # print(f"{process_index}/{num_processes} {i}/{len(image_infos)} {info.latents_npz}")

                    cache_available = caching_strategy.is_disk_cached_latents_expected(
                        info.bucket_reso, info.latents_npz, subset.flip_aug, subset.alpha_mask
                    )
                    if cache_available:  # do not add to batch
                        continue

                # if batch is not empty and condition is changed, flush the batch. Note that current_condition is not None if batch is not empty
                condition = Condition(info.bucket_reso, subset.flip_aug, subset.alpha_mask, subset.random_crop)
                if len(batch) > 0 and current_condition != condition:
                    submit_batch(batch, current_condition)
                    batch = []
                if condition != current_condition and HIGH_VRAM:  # even with high VRAM, if shape is changed
                    clean_memory_on_device(accelerator.device)

                if info.image is None:
                    # load image in parallel
                    info.image = executor.submit(load_image, info.absolute_path, condition.alpha_mask)

                batch.append(info)
                current_condition = condition

                # if number of data in batch is enough, flush the batch
                if len(batch) >= caching_strategy.batch_size:
                    submit_batch(batch, current_condition)
                    batch = []
                    # current_condition = None  # keep current_condition to avoid next `clean_memory_on_device` call

            if len(batch) > 0:
                submit_batch(batch, current_condition)

        finally:
            executor.shutdown()

    def cache_latents(self, vae, vae_batch_size=1, cache_to_disk=False, is_main_process=True, file_suffix=".npz"):
        # ???GPU????????????????tools/cache_latents.py?????
        logger.info("caching latents.")

        image_infos = list(self.image_data.values())

        # sort by resolution
        image_infos.sort(key=lambda info: info.bucket_reso[0] * info.bucket_reso[1])

        # split by resolution and some conditions
        class Condition:
            def __init__(self, reso, flip_aug, alpha_mask, random_crop):
                self.reso = reso
                self.flip_aug = flip_aug
                self.alpha_mask = alpha_mask
                self.random_crop = random_crop

            def __eq__(self, other):
                return (
                    self.reso == other.reso
                    and self.flip_aug == other.flip_aug
                    and self.alpha_mask == other.alpha_mask
                    and self.random_crop == other.random_crop
                )

        batches: List[Tuple[Condition, List[ImageInfo]]] = []
        batch: List[ImageInfo] = []
        current_condition = None

        logger.info("checking cache validity...")
        for info in tqdm(image_infos):
            subset = self.image_to_subset[info.image_key]

            if info.latents_npz is not None:  # fine tuning dataset
                continue

            # check disk cache exists and size of latents
            if cache_to_disk:
                info.latents_npz = os.path.splitext(info.absolute_path)[0] + file_suffix
                if not is_main_process:  # store to info only
                    continue

                cache_available = is_disk_cached_latents_is_expected(
                    info.bucket_reso, info.latents_npz, subset.flip_aug, subset.alpha_mask
                )

                if cache_available:  # do not add to batch
                    continue

            # if batch is not empty and condition is changed, flush the batch. Note that current_condition is not None if batch is not empty
            condition = Condition(info.bucket_reso, subset.flip_aug, subset.alpha_mask, subset.random_crop)
            if len(batch) > 0 and current_condition != condition:
                batches.append((current_condition, batch))
                batch = []

            batch.append(info)
            current_condition = condition

            # if number of data in batch is enough, flush the batch
            if len(batch) >= vae_batch_size:
                batches.append((current_condition, batch))
                batch = []
                current_condition = None

        if len(batch) > 0:
            batches.append((current_condition, batch))

        if cache_to_disk and not is_main_process:  # if cache to disk, don't cache latents in non-main process, set to info only
            return

        # iterate batches: batch doesn't have image, image will be loaded in cache_batch_latents and discarded
        logger.info("caching latents...")
        for condition, batch in tqdm(batches, smoothing=1, total=len(batches)):
            cache_batch_latents(vae, cache_to_disk, batch, condition.flip_aug, condition.alpha_mask, condition.random_crop)

    def new_cache_text_encoder_outputs(self, models: List[Any], accelerator: Accelerator):
        r"""
        a brand new method to cache text encoder outputs. This method caches text encoder outputs with caching strategy.
        """
        tokenize_strategy = TokenizeStrategy.get_strategy()
        text_encoding_strategy = TextEncodingStrategy.get_strategy()
        caching_strategy = TextEncoderOutputsCachingStrategy.get_strategy()
        batch_size = caching_strategy.batch_size or self.batch_size

        logger.info("caching Text Encoder outputs with caching strategy.")
        image_infos = list(self.image_data.values())

        # split by resolution
        batches = []
        batch = []

        # support multiple-gpus
        num_processes = accelerator.num_processes
        process_index = accelerator.process_index

        logger.info("checking cache validity...")
        for i, info in enumerate(tqdm(image_infos)):
            # check disk cache exists and size of text encoder outputs
            if caching_strategy.cache_to_disk:
                te_out_npz = caching_strategy.get_outputs_npz_path(info.absolute_path)
                info.text_encoder_outputs_npz = te_out_npz  # set npz filename regardless of cache availability

                # if the modulo of num_processes is not equal to process_index, skip caching
                # this makes each process cache different text encoder outputs
                if i % num_processes != process_index:
                    continue

                cache_available = caching_strategy.is_disk_cached_outputs_expected(te_out_npz)
                if cache_available:  # do not add to batch
                    continue

            batch.append(info)

            # if number of data in batch is enough, flush the batch
            if len(batch) >= batch_size:
                batches.append(batch)
                batch = []

        if len(batch) > 0:
            batches.append(batch)

        if len(batches) == 0:
            logger.info("no Text Encoder outputs to cache")
            return

        # iterate batches
        logger.info("caching Text Encoder outputs...")
        for batch in tqdm(batches, smoothing=1, total=len(batches)):
            # cache_batch_latents(vae, cache_to_disk, batch, subset.flip_aug, subset.alpha_mask, subset.random_crop)
            caching_strategy.cache_batch_outputs(tokenize_strategy, models, text_encoding_strategy, batch)

    # if weight_dtype is specified, Text Encoder itself and output will be converted to the dtype
    # this method is only for SDXL, but it should be implemented here because it needs to be a method of dataset
    # to support SD1/2, it needs a flag for v2, but it is postponed
    def cache_text_encoder_outputs(
        self, tokenizers, text_encoders, device, output_dtype, cache_to_disk=False, is_main_process=True
    ):
        assert len(tokenizers) == 2, "only support SDXL"
        return self.cache_text_encoder_outputs_common(
            tokenizers, text_encoders, [device, device], output_dtype, [output_dtype], cache_to_disk, is_main_process
        )

    # same as above, but for SD3
    def cache_text_encoder_outputs_sd3(
        self, tokenizer, text_encoders, devices, output_dtype, te_dtypes, cache_to_disk=False, is_main_process=True, batch_size=None
    ):
        return self.cache_text_encoder_outputs_common(
            [tokenizer],
            text_encoders,
            devices,
            output_dtype,
            te_dtypes,
            cache_to_disk,
            is_main_process,
            TEXT_ENCODER_OUTPUTS_CACHE_SUFFIX_SD3,
            batch_size,
        )

    def cache_text_encoder_outputs_common(
        self,
        tokenizers,
        text_encoders,
        devices,
        output_dtype,
        te_dtypes,
        cache_to_disk=False,
        is_main_process=True,
        file_suffix=TEXT_ENCODER_OUTPUTS_CACHE_SUFFIX,
        batch_size=None,
    ):
        # latents???????????????????????????
        # ?????GPU????????????????tools/cache_latents.py?????
        logger.info("caching text encoder outputs.")

        tokenize_strategy = TokenizeStrategy.get_strategy()

        if batch_size is None:
            batch_size = self.batch_size

        image_infos = list(self.image_data.values())

        logger.info("checking cache existence...")
        image_infos_to_cache = []
        for info in tqdm(image_infos):
            # subset = self.image_to_subset[info.image_key]
            if cache_to_disk:
                te_out_npz = os.path.splitext(info.absolute_path)[0] + file_suffix
                info.text_encoder_outputs_npz = te_out_npz

                if not is_main_process:  # store to info only
                    continue

                if os.path.exists(te_out_npz):
                    # TODO check varidity of cache here
                    continue

            image_infos_to_cache.append(info)

        if cache_to_disk and not is_main_process:  # if cache to disk, don't cache latents in non-main process, set to info only
            return

        # prepare tokenizers and text encoders
        for text_encoder, device, te_dtype in zip(text_encoders, devices, te_dtypes):
            text_encoder.to(device)
            if te_dtype is not None:
                text_encoder.to(dtype=te_dtype)

        # create batch
        is_sd3 = len(tokenizers) == 1
        batch = []
        batches = []
        for info in image_infos_to_cache:
            if not is_sd3:
                input_ids1 = self.get_input_ids(info.caption, tokenizers[0])
                input_ids2 = self.get_input_ids(info.caption, tokenizers[1])
                batch.append((info, input_ids1, input_ids2))
            else:
                l_tokens, g_tokens, t5_tokens = tokenize_strategy.tokenize(info.caption)
                batch.append((info, l_tokens, g_tokens, t5_tokens))

            if len(batch) >= batch_size:
                batches.append(batch)
                batch = []

        if len(batch) > 0:
            batches.append(batch)

        # iterate batches: call text encoder and cache outputs for memory or disk
        logger.info("caching text encoder outputs...")
        if not is_sd3:
            for batch in tqdm(batches):
                infos, input_ids1, input_ids2 = zip(*batch)
                input_ids1 = torch.stack(input_ids1, dim=0)
                input_ids2 = torch.stack(input_ids2, dim=0)
                cache_batch_text_encoder_outputs(
                    infos, tokenizers, text_encoders, self.max_token_length, cache_to_disk, input_ids1, input_ids2, output_dtype
                )
        else:
            for batch in tqdm(batches):
                infos, l_tokens, g_tokens, t5_tokens = zip(*batch)

                # stack tokens
                # l_tokens = [tokens[0] for tokens in l_tokens]
                # g_tokens = [tokens[0] for tokens in g_tokens]
                # t5_tokens = [tokens[0] for tokens in t5_tokens]

                cache_batch_text_encoder_outputs_sd3(
                    infos,
                    tokenizers[0],
                    text_encoders,
                    self.max_token_length,
                    cache_to_disk,
                    (l_tokens, g_tokens, t5_tokens),
                    output_dtype,
                )

    def get_image_size(self, image_path):
        if image_path.endswith(".jxl") or image_path.endswith(".JXL"):
            return get_jxl_size(image_path)
        # return imagesize.get(image_path)
        image_size = imagesize.get(image_path)
        if image_size[0] <= 0:
            # imagesize doesn't work for some images, so use PIL as a fallback
            try:
                with Image.open(image_path) as img:
                    image_size = img.size
            except Exception as e:
                logger.warning(f"failed to get image size: {image_path}, error: {e}")
                image_size = (0, 0)
        return image_size

    def load_image_with_face_info(self, subset: BaseSubset, image_path: str, alpha_mask=False):
        img = load_image(image_path, alpha_mask)

        face_cx = face_cy = face_w = face_h = 0
        if subset.face_crop_aug_range is not None:
            tokens = os.path.splitext(os.path.basename(image_path))[0].split("_")
            if len(tokens) >= 5:
                face_cx = int(tokens[-4])
                face_cy = int(tokens[-3])
                face_w = int(tokens[-2])
                face_h = int(tokens[-1])

        return img, face_cx, face_cy, face_w, face_h

    # ?????????
    def crop_target(self, subset: BaseSubset, image, face_cx, face_cy, face_w, face_h):
        height, width = image.shape[0:2]
        if height == self.height and width == self.width:
            return image

        # ??????size?????????????
        face_size = max(face_w, face_h)
        size = min(self.height, self.width)  # ????
        min_scale = max(self.height / height, self.width / width)  # ???????????????????????????
        min_scale = min(1.0, max(min_scale, size / (face_size * subset.face_crop_aug_range[1])))  # ??????????
        max_scale = min(1.0, max(min_scale, size / (face_size * subset.face_crop_aug_range[0])))  # ??????????
        if min_scale >= max_scale:  # range???min==max
            scale = min_scale
        else:
            scale = random.uniform(min_scale, max_scale)

        nh = int(height * scale + 0.5)
        nw = int(width * scale + 0.5)
        assert nh >= self.height and nw >= self.width, f"internal error. small scale {scale}, {width}*{height}"
        image = resize_image(image, width, height, nw, nh, subset.resize_interpolation)
        face_cx = int(face_cx * scale + 0.5)
        face_cy = int(face_cy * scale + 0.5)
        height, width = nh, nw

        # ???????448*640???????
        for axis, (target_size, length, face_p) in enumerate(zip((self.height, self.width), (height, width), (face_cy, face_cx))):
            p1 = face_p - target_size // 2  # ???????????????????

            if subset.random_crop:
                # ??????????????????????????
                range = max(length - face_p, face_p)  # ???????????????????
                p1 = p1 + (random.randint(0, range) + random.randint(0, range)) - range  # -range ~ +range ??????????
            else:
                # range???????????????????????????
                if subset.face_crop_aug_range[0] != subset.face_crop_aug_range[1]:
                    if face_size > size // 10 and face_size >= 40:
                        p1 = p1 + random.randint(-face_size // 20, +face_size // 20)

            p1 = max(0, min(p1, length - target_size))

            if axis == 0:
                image = image[p1 : p1 + target_size, :]
            else:
                image = image[:, p1 : p1 + target_size]

        return image

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        bucket = self.bucket_manager.buckets[self.buckets_indices[index].bucket_index]
        bucket_batch_size = self.buckets_indices[index].bucket_batch_size
        image_index = self.buckets_indices[index].batch_index * bucket_batch_size

        if self.caching_mode is not None:  # return batch for latents/text encoder outputs caching
            return self.get_item_for_caching(bucket, bucket_batch_size, image_index)

        loss_weights = []
        captions = []
        input_ids_list = []
        latents_list = []
        alpha_mask_list = []
        images = []
        original_sizes_hw = []
        crop_top_lefts = []
        target_sizes_hw = []
        flippeds = []  # ??????
        text_encoder_outputs_list = []
        custom_attributes = []

        for image_key in bucket[image_index : image_index + bucket_batch_size]:
            image_info = self.image_data[image_key]
            subset = self.image_to_subset[image_key]

            custom_attributes.append(subset.custom_attributes)

            # in case of fine tuning, is_reg is always False
            loss_weights.append(self.prior_loss_weight if image_info.is_reg else 1.0)

            flipped = subset.flip_aug and random.random() < 0.5  # not flipped or flipped with 50% chance

            # image/latents?????
            if image_info.latents is not None:  # cache_latents=True???
                original_size = image_info.latents_original_size
                crop_ltrb = image_info.latents_crop_ltrb  # calc values later if flipped
                if not flipped:
                    latents = image_info.latents
                    alpha_mask = image_info.alpha_mask
                else:
                    latents = image_info.latents_flipped
                    alpha_mask = None if image_info.alpha_mask is None else torch.flip(image_info.alpha_mask, [1])

                image = None
            elif image_info.latents_npz is not None:  # FineTuningDataset???cache_latents_to_disk=True???
                latents, original_size, crop_ltrb, flipped_latents, alpha_mask = (
                    self.latents_caching_strategy.load_latents_from_disk(image_info.latents_npz, image_info.bucket_reso)
                )
                if flipped:
                    latents = flipped_latents
                    alpha_mask = None if alpha_mask is None else alpha_mask[:, ::-1].copy()  # copy to avoid negative stride problem
                    del flipped_latents
                latents = torch.FloatTensor(latents)
                if alpha_mask is not None:
                    alpha_mask = torch.FloatTensor(alpha_mask)

                image = None
            else:
                # ????????????crop??
                img, face_cx, face_cy, face_w, face_h = self.load_image_with_face_info(
                    subset, image_info.absolute_path, subset.alpha_mask
                )
                im_h, im_w = img.shape[0:2]

                if self.enable_bucket:
                    img, original_size, crop_ltrb = trim_and_resize_if_required(
                        subset.random_crop,
                        img,
                        image_info.bucket_reso,
                        image_info.resized_size,
                        resize_interpolation=image_info.resize_interpolation,
                    )
                else:
                    if face_cx > 0:  # ???????
                        img = self.crop_target(subset, img, face_cx, face_cy, face_w, face_h)
                    elif im_h > self.height or im_w > self.width:
                        assert (
                            subset.random_crop
                        ), f"image too large, but cropping and bucketing are disabled / ???????????face_crop_aug_range?random_crop????bucket??????????: {image_info.absolute_path}"
                        if im_h > self.height:
                            p = random.randint(0, im_h - self.height)
                            img = img[p : p + self.height]
                        if im_w > self.width:
                            p = random.randint(0, im_w - self.width)
                            img = img[:, p : p + self.width]

                    im_h, im_w = img.shape[0:2]
                    assert (
                        im_h == self.height and im_w == self.width
                    ), f"image size is small / ?????????????: {image_info.absolute_path}"

                    original_size = [im_w, im_h]
                    crop_ltrb = (0, 0, 0, 0)

                # augmentation
                aug = self.aug_helper.get_augmentor(subset.color_aug)
                if aug is not None:
                    # augment RGB channels only
                    img_rgb = img[:, :, :3]
                    img_rgb = aug(image=img_rgb)["image"]
                    img[:, :, :3] = img_rgb

                if flipped:
                    img = img[:, ::-1, :].copy()  # copy to avoid negative stride problem

                if subset.alpha_mask:
                    if img.shape[2] == 4:
                        alpha_mask = img[:, :, 3]  # [H,W]
                        alpha_mask = alpha_mask.astype(np.float32) / 255.0  # 0.0~1.0
                        alpha_mask = torch.FloatTensor(alpha_mask)
                    else:
                        alpha_mask = torch.ones((img.shape[0], img.shape[1]), dtype=torch.float32)
                else:
                    alpha_mask = None

                img = img[:, :, :3]  # remove alpha channel

                latents = None
                image = self.image_transforms(img)  # -1.0~1.0?torch.Tensor???
                del img

            images.append(image)
            latents_list.append(latents)
            alpha_mask_list.append(alpha_mask)

            target_size = (image.shape[2], image.shape[1]) if image is not None else (latents.shape[2] * 8, latents.shape[1] * 8)

            if not flipped:
                crop_left_top = (crop_ltrb[0], crop_ltrb[1])
            else:
                # crop_ltrb[2] is right, so target_size[0] - crop_ltrb[2] is left in flipped image
                crop_left_top = (target_size[0] - crop_ltrb[2], crop_ltrb[1])

            original_sizes_hw.append((int(original_size[1]), int(original_size[0])))
            crop_top_lefts.append((int(crop_left_top[1]), int(crop_left_top[0])))
            target_sizes_hw.append((int(target_size[1]), int(target_size[0])))
            flippeds.append(flipped)

            # caption?text encoder output?????
            caption = image_info.caption  # default

            tokenization_required = (
                self.text_encoder_output_caching_strategy is None or self.text_encoder_output_caching_strategy.is_partial
            )
            text_encoder_outputs = None
            input_ids = None

            if image_info.text_encoder_outputs is not None:
                # cached
                text_encoder_outputs = image_info.text_encoder_outputs
            elif image_info.text_encoder_outputs_npz is not None:
                # on disk
                text_encoder_outputs = self.text_encoder_output_caching_strategy.load_outputs_npz(
                    image_info.text_encoder_outputs_npz
                )
            else:
                tokenization_required = True
            text_encoder_outputs_list.append(text_encoder_outputs)

            if tokenization_required:
                caption = self.process_caption(subset, image_info.caption)
                input_ids = [ids[0] for ids in self.tokenize_strategy.tokenize(caption)]  # remove batch dimension
                # if self.XTI_layers:
                #     caption_layer = []
                #     for layer in self.XTI_layers:
                #         token_strings_from = " ".join(self.token_strings)
                #         token_strings_to = " ".join([f"{x}_{layer}" for x in self.token_strings])
                #         caption_ = caption.replace(token_strings_from, token_strings_to)
                #         caption_layer.append(caption_)
                #     captions.append(caption_layer)
                # else:
                #     captions.append(caption)

                # if not self.token_padding_disabled:  # this option might be omitted in future
                #     # TODO get_input_ids must support SD3
                #     if self.XTI_layers:
                #         token_caption = self.get_input_ids(caption_layer, self.tokenizers[0])
                #     else:
                #         token_caption = self.get_input_ids(caption, self.tokenizers[0])
                #     input_ids_list.append(token_caption)

                #     if len(self.tokenizers) > 1:
                #         if self.XTI_layers:
                #             token_caption2 = self.get_input_ids(caption_layer, self.tokenizers[1])
                #         else:
                #             token_caption2 = self.get_input_ids(caption, self.tokenizers[1])
                #         input_ids2_list.append(token_caption2)

            input_ids_list.append(input_ids)
            captions.append(caption)

        def none_or_stack_elements(tensors_list, converter):
            # [[clip_l, clip_g, t5xxl], [clip_l, clip_g, t5xxl], ...] -> [torch.stack(clip_l), torch.stack(clip_g), torch.stack(t5xxl)]
            if len(tensors_list) == 0 or tensors_list[0] == None or len(tensors_list[0]) == 0 or tensors_list[0][0] is None:
                return None

            # old implementation without padding: all elements must have same length
            # return [torch.stack([converter(x[i]) for x in tensors_list]) for i in range(len(tensors_list[0]))]

            # new implementation with padding support
            result = []
            for i in range(len(tensors_list[0])):
                tensors = [x[i] for x in tensors_list]
                if tensors[0].ndim == 0:
                    # scalar value: e.g. ocr mask
                    result.append(torch.stack([converter(x[i]) for x in tensors_list]))
                    continue

                min_len = min([len(x) for x in tensors])
                max_len = max([len(x) for x in tensors])

                if min_len == max_len:
                    # no padding
                    result.append(torch.stack([converter(x) for x in tensors]))
                else:
                    # padding
                    tensors = [converter(x) for x in tensors]
                    if tensors[0].ndim == 1:
                        # input_ids or mask
                        result.append(torch.stack([(torch.nn.functional.pad(x, (0, max_len - x.shape[0]))) for x in tensors]))
                    else:
                        # text encoder outputs
                        result.append(torch.stack([(torch.nn.functional.pad(x, (0, 0, 0, max_len - x.shape[0]))) for x in tensors]))
            return result

        # set example
        example = {}
        example["custom_attributes"] = custom_attributes  # may be list of empty dict
        example["loss_weights"] = torch.FloatTensor(loss_weights)
        example["text_encoder_outputs_list"] = none_or_stack_elements(text_encoder_outputs_list, torch.FloatTensor)
        example["input_ids_list"] = none_or_stack_elements(input_ids_list, lambda x: x)

        # if one of alpha_masks is not None, we need to replace None with ones
        none_or_not = [x is None for x in alpha_mask_list]
        if all(none_or_not):
            example["alpha_masks"] = None
        elif any(none_or_not):
            for i in range(len(alpha_mask_list)):
                if alpha_mask_list[i] is None:
                    if images[i] is not None:
                        alpha_mask_list[i] = torch.ones((images[i].shape[1], images[i].shape[2]), dtype=torch.float32)
                    else:
                        alpha_mask_list[i] = torch.ones(
                            (latents_list[i].shape[1] * 8, latents_list[i].shape[2] * 8), dtype=torch.float32
                        )
            example["alpha_masks"] = torch.stack(alpha_mask_list)
        else:
            example["alpha_masks"] = torch.stack(alpha_mask_list)

        if images[0] is not None:
            images = torch.stack(images)
            images = images.to(memory_format=torch.contiguous_format).float()
        else:
            images = None
        example["images"] = images

        example["latents"] = torch.stack(latents_list) if latents_list[0] is not None else None
        example["captions"] = captions

        example["original_sizes_hw"] = torch.stack([torch.LongTensor(x) for x in original_sizes_hw])
        example["crop_top_lefts"] = torch.stack([torch.LongTensor(x) for x in crop_top_lefts])
        example["target_sizes_hw"] = torch.stack([torch.LongTensor(x) for x in target_sizes_hw])
        example["flippeds"] = flippeds

        example["network_multipliers"] = torch.FloatTensor([self.network_multiplier] * len(captions))

        if self.debug_dataset:
            example["image_keys"] = bucket[image_index : image_index + self.batch_size]
        return example

    def get_item_for_caching(self, bucket, bucket_batch_size, image_index):
        captions = []
        images = []
        input_ids1_list = []
        input_ids2_list = []
        absolute_paths = []
        resized_sizes = []
        bucket_reso = None
        flip_aug = None
        alpha_mask = None
        random_crop = None

        for image_key in bucket[image_index : image_index + bucket_batch_size]:
            image_info = self.image_data[image_key]
            subset = self.image_to_subset[image_key]

            if flip_aug is None:
                flip_aug = subset.flip_aug
                alpha_mask = subset.alpha_mask
                random_crop = subset.random_crop
                bucket_reso = image_info.bucket_reso
            else:
                # TODO ??????????????????????
                assert flip_aug == subset.flip_aug, "flip_aug must be same in a batch"
                assert alpha_mask == subset.alpha_mask, "alpha_mask must be same in a batch"
                assert random_crop == subset.random_crop, "random_crop must be same in a batch"
                assert bucket_reso == image_info.bucket_reso, "bucket_reso must be same in a batch"

            caption = image_info.caption  # TODO cache some patterns of dropping, shuffling, etc.

            if self.caching_mode == "latents":
                image = load_image(image_info.absolute_path)
            else:
                image = None

            if self.caching_mode == "text":
                input_ids1 = self.get_input_ids(caption, self.tokenizers[0])
                input_ids2 = self.get_input_ids(caption, self.tokenizers[1])
            else:
                input_ids1 = None
                input_ids2 = None

            captions.append(caption)
            images.append(image)
            input_ids1_list.append(input_ids1)
            input_ids2_list.append(input_ids2)
            absolute_paths.append(image_info.absolute_path)
            resized_sizes.append(image_info.resized_size)

        example = {}

        if images[0] is None:
            images = None
        example["images"] = images

        example["captions"] = captions
        example["input_ids1_list"] = input_ids1_list
        example["input_ids2_list"] = input_ids2_list
        example["absolute_paths"] = absolute_paths
        example["resized_sizes"] = resized_sizes
        example["flip_aug"] = flip_aug
        example["alpha_mask"] = alpha_mask
        example["random_crop"] = random_crop
        example["bucket_reso"] = bucket_reso
        return example


class DreamBoothDataset(BaseDataset):
    IMAGE_INFO_CACHE_FILE = "metadata_cache.json"

    # The is_training_dataset defines the type of dataset, training or validation
    # if is_training_dataset is True -> training dataset
    # if is_training_dataset is False -> validation dataset
    def __init__(
        self,
        subsets: Sequence[DreamBoothSubset],
        is_training_dataset: bool,
        batch_size: int,
        resolution,
        network_multiplier: float,
        enable_bucket: bool,
        min_bucket_reso: int,
        max_bucket_reso: int,
        bucket_reso_steps: int,
        bucket_no_upscale: bool,
        prior_loss_weight: float,
        debug_dataset: bool,
        validation_split: float,
        validation_seed: Optional[int],
        resize_interpolation: Optional[str],
    ) -> None:
        super().__init__(resolution, network_multiplier, debug_dataset, resize_interpolation)

        assert resolution is not None, f"resolution is required / resolution????????????"

        self.batch_size = batch_size
        self.size = min(self.width, self.height)  # ????
        self.prior_loss_weight = prior_loss_weight
        self.latents_cache = None
        self.is_training_dataset = is_training_dataset
        self.validation_seed = validation_seed
        self.validation_split = validation_split

        self.enable_bucket = enable_bucket
        if self.enable_bucket:
            min_bucket_reso, max_bucket_reso = self.adjust_min_max_bucket_reso_by_steps(
                resolution, min_bucket_reso, max_bucket_reso, bucket_reso_steps
            )
            self.min_bucket_reso = min_bucket_reso
            self.max_bucket_reso = max_bucket_reso
            self.bucket_reso_steps = bucket_reso_steps
            self.bucket_no_upscale = bucket_no_upscale
        else:
            self.min_bucket_reso = None
            self.max_bucket_reso = None
            self.bucket_reso_steps = None  # ??????????
            self.bucket_no_upscale = False

        def read_caption(img_path, caption_extension, enable_wildcard):
            # caption???????????
            base_name = os.path.splitext(img_path)[0]
            base_name_face_det = base_name
            tokens = base_name.split("_")
            if len(tokens) >= 5:
                base_name_face_det = "_".join(tokens[:-4])
            cap_paths = [base_name + caption_extension, base_name_face_det + caption_extension]

            caption = None
            for cap_path in cap_paths:
                if os.path.isfile(cap_path):
                    with open(cap_path, "rt", encoding="utf-8") as f:
                        try:
                            lines = f.readlines()
                        except UnicodeDecodeError as e:
                            logger.error(f"illegal char in file (not UTF-8) / ?????UTF-8??????????: {cap_path}")
                            raise e
                        assert len(lines) > 0, f"caption file is empty / ??????????????: {cap_path}"
                        if enable_wildcard:
                            caption = "\n".join([line.strip() for line in lines if line.strip() != ""])  # ???????????
                        else:
                            caption = lines[0].strip()
                    break
            return caption

        def load_dreambooth_dir(subset: DreamBoothSubset):
            if not os.path.isdir(subset.image_dir):
                logger.warning(f"not directory: {subset.image_dir}")
                return [], [], []

            info_cache_file = os.path.join(subset.image_dir, self.IMAGE_INFO_CACHE_FILE)
            use_cached_info_for_subset = subset.cache_info
            if use_cached_info_for_subset:
                logger.info(
                    f"using cached image info for this subset / ??????????????????????????: {info_cache_file}"
                )
                if not os.path.isfile(info_cache_file):
                    logger.warning(
                        f"image info file not found. You can ignore this warning if this is the first time to use this subset"
                        + " / ????????????????????????????????????????: {metadata_file}"
                    )
                    use_cached_info_for_subset = False

            if use_cached_info_for_subset:
                # json: {`img_path`:{"caption": "caption...", "resolution": [width, height]}, ...}
                with open(info_cache_file, "r", encoding="utf-8") as f:
                    metas = json.load(f)
                img_paths = list(metas.keys())
                sizes: List[Optional[Tuple[int, int]]] = [meta["resolution"] for meta in metas.values()]

                # we may need to check image size and existence of image files, but it takes time, so user should check it before training
            else:
                img_paths = glob_images(subset.image_dir, "*")
                sizes: List[Optional[Tuple[int, int]]] = [None] * len(img_paths)

                # new caching: get image size from cache files
                strategy = LatentsCachingStrategy.get_strategy()
                if strategy is not None:
                    logger.info("get image size from name of cache files")

                    # make image path to npz path mapping
                    npz_paths = glob.glob(os.path.join(subset.image_dir, "*" + strategy.cache_suffix))
                    npz_paths.sort(
                        key=lambda item: item.rsplit("_", maxsplit=2)[0]
                    )  # sort by name excluding resolution and cache_suffix
                    npz_path_index = 0

                    size_set_count = 0
                    for i, img_path in enumerate(tqdm(img_paths)):
                        l = len(os.path.splitext(img_path)[0])  # remove extension
                        found = False
                        while npz_path_index < len(npz_paths):  # until found or end of npz_paths
                            # npz_paths are sorted, so if npz_path > img_path, img_path is not found
                            if npz_paths[npz_path_index][:l] > img_path[:l]:
                                break
                            if npz_paths[npz_path_index][:l] == img_path[:l]:  # found
                                found = True
                                break
                            npz_path_index += 1  # next npz_path

                        if found:
                            w, h = strategy.get_image_size_from_disk_cache_path(img_path, npz_paths[npz_path_index])
                        else:
                            w, h = None, None

                        if w is not None and h is not None:
                            sizes[i] = (w, h)
                            size_set_count += 1
                    logger.info(f"set image size from cache files: {size_set_count}/{len(img_paths)}")

            # We want to create a training and validation split. This should be improved in the future
            # to allow a clearer distinction between training and validation. This can be seen as a
            # short-term solution to limit what is necessary to implement validation datasets
            #
            # We split the dataset for the subset based on if we are doing a validation split
            # The self.is_training_dataset defines the type of dataset, training or validation
            # if self.is_training_dataset is True -> training dataset
            # if self.is_training_dataset is False -> validation dataset
            if self.validation_split > 0.0:
                # For regularization images we do not want to split this dataset.
                if subset.is_reg is True:
                    # Skip any validation dataset for regularization images
                    if self.is_training_dataset is False:
                        img_paths = []
                        sizes = []
                    # Otherwise the img_paths remain as original img_paths and no split
                    # required for training images dataset of regularization images
                else:
                    img_paths, sizes = split_train_val(
                        img_paths, sizes, self.is_training_dataset, self.validation_split, self.validation_seed
                    )

            logger.info(f"found directory {subset.image_dir} contains {len(img_paths)} image files")

            if use_cached_info_for_subset:
                captions = [meta["caption"] for meta in metas.values()]
                missing_captions = [img_path for img_path, caption in zip(img_paths, captions) if caption is None or caption == ""]
            else:
                # ???????????????????????????????
                captions = []
                missing_captions = []
                for img_path in tqdm(img_paths, desc="read caption"):
                    cap_for_img = read_caption(img_path, subset.caption_extension, subset.enable_wildcard)
                    if cap_for_img is None and subset.class_tokens is None:
                        logger.warning(
                            f"neither caption file nor class tokens are found. use empty caption for {img_path} / ???????????class token??????????????????????????: {img_path}"
                        )
                        captions.append("")
                        missing_captions.append(img_path)
                    else:
                        if cap_for_img is None:
                            captions.append(subset.class_tokens)
                            missing_captions.append(img_path)
                        else:
                            captions.append(cap_for_img)

            self.set_tag_frequency(os.path.basename(subset.image_dir), captions)  # ???????

            if missing_captions:
                number_of_missing_captions = len(missing_captions)
                number_of_missing_captions_to_show = 5
                remaining_missing_captions = number_of_missing_captions - number_of_missing_captions_to_show

                logger.warning(
                    f"No caption file found for {number_of_missing_captions} images. Training will continue without captions for these images. If class token exists, it will be used. / {number_of_missing_captions}????????????????????????????????????????????????????????class token????????????????"
                )
                for i, missing_caption in enumerate(missing_captions):
                    if i >= number_of_missing_captions_to_show:
                        logger.warning(missing_caption + f"... and {remaining_missing_captions} more")
                        break
                    logger.warning(missing_caption)

            if not use_cached_info_for_subset and subset.cache_info:
                logger.info(f"cache image info for / ????????????? : {info_cache_file}")
                sizes = [self.get_image_size(img_path) for img_path in tqdm(img_paths, desc="get image size")]
                matas = {}
                for img_path, caption, size in zip(img_paths, captions, sizes):
                    matas[img_path] = {"caption": caption, "resolution": list(size)}
                with open(info_cache_file, "w", encoding="utf-8") as f:
                    json.dump(matas, f, ensure_ascii=False, indent=2)
                logger.info(f"cache image info done for / ??????????? : {info_cache_file}")

            # if sizes are not set, image size will be read in make_buckets
            return img_paths, captions, sizes

        logger.info("prepare images.")
        num_train_images = 0
        num_reg_images = 0
        reg_infos: List[Tuple[ImageInfo, DreamBoothSubset]] = []
        for subset in subsets:
            num_repeats = subset.num_repeats if self.is_training_dataset else 1
            if num_repeats < 1:
                logger.warning(
                    f"ignore subset with image_dir='{subset.image_dir}': num_repeats is less than 1 / num_repeats?1????????????????????: {num_repeats}"
                )
                continue

            if subset in self.subsets:
                logger.warning(
                    f"ignore duplicated subset with image_dir='{subset.image_dir}': use the first one / ????????????????????????????????????"
                )
                continue

            img_paths, captions, sizes = load_dreambooth_dir(subset)
            if len(img_paths) < 1:
                logger.warning(
                    f"ignore subset with image_dir='{subset.image_dir}': no images found / ??????????????????????"
                )
                continue

            if subset.is_reg:
                num_reg_images += num_repeats * len(img_paths)
            else:
                num_train_images += num_repeats * len(img_paths)

            for img_path, caption, size in zip(img_paths, captions, sizes):
                info = ImageInfo(img_path, num_repeats, caption, subset.is_reg, img_path)
                info.resize_interpolation = (
                    subset.resize_interpolation if subset.resize_interpolation is not None else self.resize_interpolation
                )
                if size is not None:
                    info.image_size = size
                if subset.is_reg:
                    reg_infos.append((info, subset))
                else:
                    self.register_image(info, subset)

            subset.img_count = len(img_paths)
            self.subsets.append(subset)

        images_split_name = "train" if self.is_training_dataset else "validation"
        logger.info(f"{num_train_images} {images_split_name} images with repeats.")

        self.num_train_images = num_train_images

        logger.info(f"{num_reg_images} reg images with repeats.")
        if num_train_images < num_reg_images:
            logger.warning("some of reg images are not used / ???????????????????????????????")

        if num_reg_images == 0:
            logger.warning("no regularization images / ????????????????")
        else:
            # num_repeats???????????????????????????
            n = 0
            first_loop = True
            while n < num_train_images:
                for info, subset in reg_infos:
                    if first_loop:
                        self.register_image(info, subset)
                        n += info.num_repeats
                    else:
                        info.num_repeats += 1  # rewrite registered info
                        n += 1
                    if n >= num_train_images:
                        break
                first_loop = False

        self.num_reg_images = num_reg_images


class FineTuningDataset(BaseDataset):
    def __init__(
        self,
        subsets: Sequence[FineTuningSubset],
        batch_size: int,
        resolution,
        network_multiplier: float,
        enable_bucket: bool,
        min_bucket_reso: int,
        max_bucket_reso: int,
        bucket_reso_steps: int,
        bucket_no_upscale: bool,
        debug_dataset: bool,
        validation_seed: int,
        validation_split: float,
        resize_interpolation: Optional[str],
    ) -> None:
        super().__init__(resolution, network_multiplier, debug_dataset, resize_interpolation)

        self.batch_size = batch_size
        self.size = min(self.width, self.height)  # ????
        self.latents_cache = None

        self.enable_bucket = enable_bucket
        if self.enable_bucket:
            min_bucket_reso, max_bucket_reso = self.adjust_min_max_bucket_reso_by_steps(
                resolution, min_bucket_reso, max_bucket_reso, bucket_reso_steps
            )
            self.min_bucket_reso = min_bucket_reso
            self.max_bucket_reso = max_bucket_reso
            self.bucket_reso_steps = bucket_reso_steps
            self.bucket_no_upscale = bucket_no_upscale
        else:
            self.min_bucket_reso = None
            self.max_bucket_reso = None
            self.bucket_reso_steps = None  # ??????????
            self.bucket_no_upscale = False

        self.num_train_images = 0
        self.num_reg_images = 0

        for subset in subsets:
            if subset.num_repeats < 1:
                logger.warning(
                    f"ignore subset with metadata_file='{subset.metadata_file}': num_repeats is less than 1 / num_repeats?1????????????????????: {subset.num_repeats}"
                )
                continue

            if subset in self.subsets:
                logger.warning(
                    f"ignore duplicated subset with metadata_file='{subset.metadata_file}': use the first one / ????????????????????????????????????"
                )
                continue

            # ??????????
            if os.path.exists(subset.metadata_file):
                if subset.metadata_file.endswith(".jsonl"):
                    logger.info(f"loading existing JSOL metadata: {subset.metadata_file}")
                    # optional JSONL format
                    # {"image_path": "/path/to/image1.jpg", "caption": "A caption for image1", "image_size": [width, height]}
                    metadata = {}
                    with open(subset.metadata_file, "rt", encoding="utf-8") as f:
                        for line in f:
                            line_md = json.loads(line)
                            image_md = {"caption": line_md.get("caption", "")}
                            if "image_size" in line_md:
                                image_md["image_size"] = line_md["image_size"]
                            if "tags" in line_md:
                                image_md["tags"] = line_md["tags"]
                            metadata[line_md["image_path"]] = image_md
                else:
                    # standard JSON format
                    logger.info(f"loading existing metadata: {subset.metadata_file}")
                    with open(subset.metadata_file, "rt", encoding="utf-8") as f:
                        metadata = json.load(f)
            else:
                raise ValueError(f"no metadata / ???????????????: {subset.metadata_file}")

            if len(metadata) < 1:
                logger.warning(
                    f"ignore subset with '{subset.metadata_file}': no image entries found / ?????????????????????????????"
                )
                continue

            # Add full path for image
            image_dirs = set()
            if subset.image_dir is not None:
                image_dirs.add(subset.image_dir)
            for image_key in metadata.keys():
                if not os.path.isabs(image_key):
                    assert (
                        subset.image_dir is not None
                    ), f"image_dir is required when image paths are relative / ?????????????image_dir????????: {image_key}"
                    abs_path = os.path.join(subset.image_dir, image_key)
                else:
                    abs_path = image_key
                    image_dirs.add(os.path.dirname(abs_path))
                metadata[image_key]["abs_path"] = abs_path

            # Enumerate existing npz files
            strategy = LatentsCachingStrategy.get_strategy()
            npz_paths = []
            for image_dir in image_dirs:
                npz_paths.extend(glob.glob(os.path.join(image_dir, "*" + strategy.cache_suffix)))
            npz_paths = sorted(npz_paths, key=lambda item: len(os.path.basename(item)), reverse=True)  # longer paths first

            # Match image filename longer to shorter because some images share same prefix
            image_keys_sorted_by_length_desc = sorted(metadata.keys(), key=len, reverse=True)

            # Collect tags and sizes
            tags_list = []
            size_set_from_metadata = 0
            size_set_from_cache_filename = 0
            for image_key in image_keys_sorted_by_length_desc:
                img_md = metadata[image_key]
                caption = img_md.get("caption")
                tags = img_md.get("tags")
                image_size = img_md.get("image_size")
                abs_path = img_md.get("abs_path")

                # search npz if image_size is not given
                npz_path = None
                if image_size is None:
                    image_without_ext = os.path.splitext(image_key)[0]
                    for candidate in npz_paths:
                        if candidate.startswith(image_without_ext):
                            npz_path = candidate
                            break
                    if npz_path is not None:
                        npz_paths.remove(npz_path)  # remove to avoid matching same file (share prefix)
                        abs_path = npz_path

                if caption is None:
                    caption = ""

                if subset.enable_wildcard:
                    # tags must be single line (split by caption separator)
                    if tags is not None:
                        tags = tags.replace("\n", subset.caption_separator)

                    # add tags to each line of caption
                    if tags is not None:
                        caption = "\n".join(
                            [f"{line}{subset.caption_separator}{tags}" for line in caption.split("\n") if line.strip() != ""]
                        )
                        tags_list.append(tags)
                else:
                    # use as is
                    if tags is not None and len(tags) > 0:
                        if len(caption) > 0:
                            caption = caption + subset.caption_separator
                        caption = caption + tags
                        tags_list.append(tags)

                if caption is None:
                    caption = ""

                image_info = ImageInfo(image_key, subset.num_repeats, caption, False, abs_path)
                image_info.resize_interpolation = (
                    subset.resize_interpolation if subset.resize_interpolation is not None else self.resize_interpolation
                )

                if image_size is not None:
                    image_info.image_size = tuple(image_size)  # width, height
                    size_set_from_metadata += 1
                elif npz_path is not None:
                    # get image size from npz filename
                    w, h = strategy.get_image_size_from_disk_cache_path(abs_path, npz_path)
                    image_info.image_size = (w, h)
                    size_set_from_cache_filename += 1

                self.register_image(image_info, subset)

            if size_set_from_cache_filename > 0:
                logger.info(
                    f"set image size from cache files: {size_set_from_cache_filename}/{len(image_keys_sorted_by_length_desc)}"
                )
            if size_set_from_metadata > 0:
                logger.info(f"set image size from metadata: {size_set_from_metadata}/{len(image_keys_sorted_by_length_desc)}")
            self.num_train_images += len(metadata) * subset.num_repeats

            # TODO do not record tag freq when no tag
            self.set_tag_frequency(os.path.basename(subset.metadata_file), tags_list)
            subset.img_count = len(metadata)
            self.subsets.append(subset)


class ControlNetDataset(BaseDataset):
    def __init__(
        self,
        subsets: Sequence[ControlNetSubset],
        batch_size: int,
        resolution,
        network_multiplier: float,
        enable_bucket: bool,
        min_bucket_reso: int,
        max_bucket_reso: int,
        bucket_reso_steps: int,
        bucket_no_upscale: bool,
        debug_dataset: bool,
        validation_split: float,
        validation_seed: Optional[int],
        resize_interpolation: Optional[str] = None,
    ) -> None:
        super().__init__(resolution, network_multiplier, debug_dataset, resize_interpolation)

        db_subsets = []
        for subset in subsets:
            assert (
                not subset.random_crop
            ), "random_crop is not supported in ControlNetDataset / random_crop?ControlNetDataset?????????????"
            db_subset = DreamBoothSubset(
                subset.image_dir,
                False,
                None,
                subset.caption_extension,
                subset.cache_info,
                False,
                subset.num_repeats,
                subset.shuffle_caption,
                subset.caption_separator,
                subset.keep_tokens,
                subset.keep_tokens_separator,
                subset.secondary_separator,
                subset.enable_wildcard,
                subset.color_aug,
                subset.flip_aug,
                subset.face_crop_aug_range,
                subset.random_crop,
                subset.caption_dropout_rate,
                subset.caption_dropout_every_n_epochs,
                subset.caption_tag_dropout_rate,
                subset.caption_prefix,
                subset.caption_suffix,
                subset.token_warmup_min,
                subset.token_warmup_step,
                resize_interpolation=subset.resize_interpolation,
            )
            db_subsets.append(db_subset)

        self.dreambooth_dataset_delegate = DreamBoothDataset(
            db_subsets,
            True,
            batch_size,
            resolution,
            network_multiplier,
            enable_bucket,
            min_bucket_reso,
            max_bucket_reso,
            bucket_reso_steps,
            bucket_no_upscale,
            1.0,
            debug_dataset,
            validation_split,
            validation_seed,
            resize_interpolation,
        )

        # config_util???????????????????????????????
        self.image_data = self.dreambooth_dataset_delegate.image_data
        self.batch_size = batch_size
        self.num_train_images = self.dreambooth_dataset_delegate.num_train_images
        self.num_reg_images = self.dreambooth_dataset_delegate.num_reg_images
        self.validation_split = validation_split
        self.validation_seed = validation_seed
        self.resize_interpolation = resize_interpolation

        # assert all conditioning data exists
        missing_imgs = []
        cond_imgs_with_pair = set()
        for image_key, info in self.dreambooth_dataset_delegate.image_data.items():
            db_subset = self.dreambooth_dataset_delegate.image_to_subset[image_key]
            subset = None
            for s in subsets:
                if s.image_dir == db_subset.image_dir:
                    subset = s
                    break
            assert subset is not None, "internal error: subset not found"

            if not os.path.isdir(subset.conditioning_data_dir):
                logger.warning(f"not directory: {subset.conditioning_data_dir}")
                continue

            img_basename = os.path.splitext(os.path.basename(info.absolute_path))[0]
            ctrl_img_path = glob_images(subset.conditioning_data_dir, img_basename)
            if len(ctrl_img_path) < 1:
                missing_imgs.append(img_basename)
                continue
            ctrl_img_path = ctrl_img_path[0]
            ctrl_img_path = os.path.abspath(ctrl_img_path)  # normalize path

            info.cond_img_path = ctrl_img_path
            cond_imgs_with_pair.add(os.path.splitext(ctrl_img_path)[0])  # remove extension because Windows is case insensitive

        extra_imgs = []
        for subset in subsets:
            conditioning_img_paths = glob_images(subset.conditioning_data_dir, "*")
            conditioning_img_paths = [os.path.abspath(p) for p in conditioning_img_paths]  # normalize path
            extra_imgs.extend([p for p in conditioning_img_paths if os.path.splitext(p)[0] not in cond_imgs_with_pair])

        assert (
            len(missing_imgs) == 0
        ), f"missing conditioning data for {len(missing_imgs)} images / ????????????????: {missing_imgs}"
        assert (
            len(extra_imgs) == 0
        ), f"extra conditioning data for {len(extra_imgs)} images / ?????????????: {extra_imgs}"

        self.conditioning_image_transforms = IMAGE_TRANSFORMS

    def set_current_strategies(self):
        return self.dreambooth_dataset_delegate.set_current_strategies()

    def make_buckets(self):
        self.dreambooth_dataset_delegate.make_buckets()
        self.bucket_manager = self.dreambooth_dataset_delegate.bucket_manager
        self.buckets_indices = self.dreambooth_dataset_delegate.buckets_indices

    def cache_latents(self, vae, vae_batch_size=1, cache_to_disk=False, is_main_process=True):
        return self.dreambooth_dataset_delegate.cache_latents(vae, vae_batch_size, cache_to_disk, is_main_process)

    def new_cache_latents(self, model: Any, accelerator: Accelerator):
        return self.dreambooth_dataset_delegate.new_cache_latents(model, accelerator)

    def new_cache_text_encoder_outputs(self, models: List[Any], is_main_process: bool):
        return self.dreambooth_dataset_delegate.new_cache_text_encoder_outputs(models, is_main_process)

    def __len__(self):
        return self.dreambooth_dataset_delegate.__len__()

    def __getitem__(self, index):
        example = self.dreambooth_dataset_delegate[index]

        bucket = self.dreambooth_dataset_delegate.bucket_manager.buckets[
            self.dreambooth_dataset_delegate.buckets_indices[index].bucket_index
        ]
        bucket_batch_size = self.dreambooth_dataset_delegate.buckets_indices[index].bucket_batch_size
        image_index = self.dreambooth_dataset_delegate.buckets_indices[index].batch_index * bucket_batch_size

        conditioning_images = []

        for i, image_key in enumerate(bucket[image_index : image_index + bucket_batch_size]):
            image_info = self.dreambooth_dataset_delegate.image_data[image_key]

            target_size_hw = example["target_sizes_hw"][i]
            original_size_hw = example["original_sizes_hw"][i]
            crop_top_left = example["crop_top_lefts"][i]
            flipped = example["flippeds"][i]
            cond_img = load_image(image_info.cond_img_path)

            if self.dreambooth_dataset_delegate.enable_bucket:
                assert (
                    cond_img.shape[0] == original_size_hw[0] and cond_img.shape[1] == original_size_hw[1]
                ), f"size of conditioning image is not match / ???????????: {image_info.absolute_path}"

                cond_img = resize_image(
                    cond_img,
                    original_size_hw[1],
                    original_size_hw[0],
                    target_size_hw[1],
                    target_size_hw[0],
                    self.resize_interpolation,
                )

                # TODO support random crop
                # ??????????crop?random????????
                h, w = target_size_hw
                ct = (cond_img.shape[0] - h) // 2
                cl = (cond_img.shape[1] - w) // 2
                cond_img = cond_img[ct : ct + h, cl : cl + w]
            else:
                # assert (
                #     cond_img.shape[0] == self.height and cond_img.shape[1] == self.width
                # ), f"image size is small / ?????????????: {image_info.absolute_path}"
                # resize to target
                if cond_img.shape[0] != target_size_hw[0] or cond_img.shape[1] != target_size_hw[1]:
                    cond_img = resize_image(
                        cond_img,
                        cond_img.shape[0],
                        cond_img.shape[1],
                        target_size_hw[1],
                        target_size_hw[0],
                        self.resize_interpolation,
                    )

            if flipped:
                cond_img = cond_img[:, ::-1, :].copy()  # copy to avoid negative stride

            cond_img = self.conditioning_image_transforms(cond_img)
            conditioning_images.append(cond_img)

        example["conditioning_images"] = torch.stack(conditioning_images).to(memory_format=torch.contiguous_format).float()

        return example


# behave as Dataset mock
class DatasetGroup(torch.utils.data.ConcatDataset):
    def __init__(self, datasets: Sequence[Union[DreamBoothDataset, FineTuningDataset]]):
        self.datasets: List[Union[DreamBoothDataset, FineTuningDataset]]

        super().__init__(datasets)

        self.image_data = {}
        self.num_train_images = 0
        self.num_reg_images = 0

        # simply concat together
        # TODO: handling image_data key duplication among dataset
        #   In practical, this is not the big issue because image_data is accessed from outside of dataset only for debug_dataset.
        for dataset in datasets:
            self.image_data.update(dataset.image_data)
            self.num_train_images += dataset.num_train_images
            self.num_reg_images += dataset.num_reg_images

    def add_replacement(self, str_from, str_to):
        for dataset in self.datasets:
            dataset.add_replacement(str_from, str_to)

    # def make_buckets(self):
    #   for dataset in self.datasets:
    #     dataset.make_buckets()

    def set_text_encoder_output_caching_strategy(self, strategy: TextEncoderOutputsCachingStrategy):
        """
        DataLoader is run in multiple processes, so we need to set the strategy manually.
        """
        for dataset in self.datasets:
            dataset.set_text_encoder_output_caching_strategy(strategy)

    def enable_XTI(self, *args, **kwargs):
        for dataset in self.datasets:
            dataset.enable_XTI(*args, **kwargs)

    def cache_latents(self, vae, vae_batch_size=1, cache_to_disk=False, is_main_process=True, file_suffix=".npz"):
        for i, dataset in enumerate(self.datasets):
            logger.info(f"[Dataset {i}]")
            dataset.cache_latents(vae, vae_batch_size, cache_to_disk, is_main_process, file_suffix)

    def new_cache_latents(self, model: Any, accelerator: Accelerator):
        for i, dataset in enumerate(self.datasets):
            logger.info(f"[Dataset {i}]")
            dataset.new_cache_latents(model, accelerator)
        accelerator.wait_for_everyone()

    def cache_text_encoder_outputs(
        self, tokenizers, text_encoders, device, weight_dtype, cache_to_disk=False, is_main_process=True
    ):
        for i, dataset in enumerate(self.datasets):
            logger.info(f"[Dataset {i}]")
            dataset.cache_text_encoder_outputs(tokenizers, text_encoders, device, weight_dtype, cache_to_disk, is_main_process)

    def cache_text_encoder_outputs_sd3(
        self, tokenizer, text_encoders, device, output_dtype, te_dtypes, cache_to_disk=False, is_main_process=True, batch_size=None
    ):
        for i, dataset in enumerate(self.datasets):
            logger.info(f"[Dataset {i}]")
            dataset.cache_text_encoder_outputs_sd3(
                tokenizer, text_encoders, device, output_dtype, te_dtypes, cache_to_disk, is_main_process, batch_size
            )

    def new_cache_text_encoder_outputs(self, models: List[Any], accelerator: Accelerator):
        for i, dataset in enumerate(self.datasets):
            logger.info(f"[Dataset {i}]")
            dataset.new_cache_text_encoder_outputs(models, accelerator)
        accelerator.wait_for_everyone()

    def set_caching_mode(self, caching_mode):
        for dataset in self.datasets:
            dataset.set_caching_mode(caching_mode)

    def verify_bucket_reso_steps(self, min_steps: int):
        for dataset in self.datasets:
            dataset.verify_bucket_reso_steps(min_steps)

    def get_resolutions(self) -> List[Tuple[int, int]]:
        return [(dataset.width, dataset.height) for dataset in self.datasets]

    def is_latent_cacheable(self) -> bool:
        return all([dataset.is_latent_cacheable() for dataset in self.datasets])

    def is_text_encoder_output_cacheable(self) -> bool:
        return all([dataset.is_text_encoder_output_cacheable() for dataset in self.datasets])

    def set_current_strategies(self):
        for dataset in self.datasets:
            dataset.set_current_strategies()

    def set_current_epoch(self, epoch):
        for dataset in self.datasets:
            dataset.set_current_epoch(epoch)

    def set_current_step(self, step):
        for dataset in self.datasets:
            dataset.set_current_step(step)

    def set_max_train_steps(self, max_train_steps):
        for dataset in self.datasets:
            dataset.set_max_train_steps(max_train_steps)

    def disable_token_padding(self):
        for dataset in self.datasets:
            dataset.disable_token_padding()


def is_disk_cached_latents_is_expected(reso, npz_path: str, flip_aug: bool, alpha_mask: bool):
    expected_latents_size = (reso[1] // 8, reso[0] // 8)  # bucket_reso?WxH?????

    if not os.path.exists(npz_path):
        return False

    try:
        npz = np.load(npz_path)
        if "latents" not in npz or "original_size" not in npz or "crop_ltrb" not in npz:  # old ver?
            return False
        if npz["latents"].shape[1:3] != expected_latents_size:
            return False

        if flip_aug:
            if "latents_flipped" not in npz:
                return False
            if npz["latents_flipped"].shape[1:3] != expected_latents_size:
                return False

        if alpha_mask:
            if "alpha_mask" not in npz:
                return False
            if (npz["alpha_mask"].shape[1], npz["alpha_mask"].shape[0]) != reso:  # HxW => WxH != reso
                return False
        else:
            if "alpha_mask" in npz:
                return False
    except Exception as e:
        logger.error(f"Error loading file: {npz_path}")
        raise e

    return True


# ?????latents_tensor, (original_size width, original_size height), (crop left, crop top)
# TODO update to use CachingStrategy
# def load_latents_from_disk(
#     npz_path,
# ) -> Tuple[Optional[np.ndarray], Optional[List[int]], Optional[List[int]], Optional[np.ndarray], Optional[np.ndarray]]:
#     npz = np.load(npz_path)
#     if "latents" not in npz:
#         raise ValueError(f"error: npz is old format. please re-generate {npz_path}")

#     latents = npz["latents"]
#     original_size = npz["original_size"].tolist()
#     crop_ltrb = npz["crop_ltrb"].tolist()
#     flipped_latents = npz["latents_flipped"] if "latents_flipped" in npz else None
#     alpha_mask = npz["alpha_mask"] if "alpha_mask" in npz else None
#     return latents, original_size, crop_ltrb, flipped_latents, alpha_mask


# def save_latents_to_disk(npz_path, latents_tensor, original_size, crop_ltrb, flipped_latents_tensor=None, alpha_mask=None):
#     kwargs = {}
#     if flipped_latents_tensor is not None:
#         kwargs["latents_flipped"] = flipped_latents_tensor.float().cpu().numpy()
#     if alpha_mask is not None:
#         kwargs["alpha_mask"] = alpha_mask.float().cpu().numpy()
#     np.savez(
#         npz_path,
#         latents=latents_tensor.float().cpu().numpy(),
#         original_size=np.array(original_size),
#         crop_ltrb=np.array(crop_ltrb),
#         **kwargs,
#     )


def debug_dataset(train_dataset, show_input_ids=False):
    logger.info(f"Total dataset length (steps) / ????????????????: {len(train_dataset)}")
    logger.info(
        "`S` for next step, `E` for next epoch no. , Escape for exit. / S??????????E??????????Esc???????????"
    )

    epoch = 1
    while True:
        logger.info(f"")
        logger.info(f"epoch: {epoch}")

        steps = (epoch - 1) * len(train_dataset) + 1
        indices = list(range(len(train_dataset)))
        random.shuffle(indices)

        k = 0
        for i, idx in enumerate(indices):
            train_dataset.set_current_epoch(epoch)
            train_dataset.set_current_step(steps)
            logger.info(f"steps: {steps} ({i + 1}/{len(train_dataset)})")

            example = train_dataset[idx]
            if example["latents"] is not None:
                logger.info(f"sample has latents from npz file: {example['latents'].size()}")
            for j, (ik, cap, lw, orgsz, crptl, trgsz, flpdz) in enumerate(
                zip(
                    example["image_keys"],
                    example["captions"],
                    example["loss_weights"],
                    # example["input_ids"],
                    example["original_sizes_hw"],
                    example["crop_top_lefts"],
                    example["target_sizes_hw"],
                    example["flippeds"],
                )
            ):
                logger.info(
                    f'{ik}, size: {train_dataset.image_data[ik].image_size}, loss weight: {lw}, caption: "{cap}", original size: {orgsz}, crop top left: {crptl}, target size: {trgsz}, flipped: {flpdz}'
                )
                if "network_multipliers" in example:
                    logger.info(f"network multiplier: {example['network_multipliers'][j]}")
                if "custom_attributes" in example:
                    logger.info(f"custom attributes: {example['custom_attributes'][j]}")

                # if show_input_ids:
                #     logger.info(f"input ids: {iid}")
                #     if "input_ids2" in example:
                #         logger.info(f"input ids2: {example['input_ids2'][j]}")
                if example["images"] is not None:
                    im = example["images"][j]
                    logger.info(f"image size: {im.size()}")
                    im = ((im.numpy() + 1.0) * 127.5).astype(np.uint8)
                    im = np.transpose(im, (1, 2, 0))  # c,H,W -> H,W,c
                    im = im[:, :, ::-1]  # RGB -> BGR (OpenCV)

                    if "conditioning_images" in example:
                        cond_img = example["conditioning_images"][j]
                        logger.info(f"conditioning image size: {cond_img.size()}")
                        cond_img = ((cond_img.numpy() + 1.0) * 127.5).astype(np.uint8)
                        cond_img = np.transpose(cond_img, (1, 2, 0))
                        cond_img = cond_img[:, :, ::-1]
                        if os.name == "nt":
                            cv2.imshow("cond_img", cond_img)

                    if "alpha_masks" in example and example["alpha_masks"] is not None:
                        alpha_mask = example["alpha_masks"][j]
                        logger.info(f"alpha mask size: {alpha_mask.size()}")
                        alpha_mask = (alpha_mask.numpy() * 255.0).astype(np.uint8)
                        if os.name == "nt":
                            cv2.imshow("alpha_mask", alpha_mask)

                    if os.name == "nt":  # only windows
                        cv2.imshow("img", im)
                        k = cv2.waitKey()
                        cv2.destroyAllWindows()
                    if k == 27 or k == ord("s") or k == ord("e"):
                        break
            steps += 1

            if k == ord("e"):
                break
            if k == 27 or (example["images"] is None and i >= 8):
                k = 27
                break
        if k == 27:
            break

        epoch += 1


def glob_images(directory, base="*"):
    img_paths = []
    for ext in IMAGE_EXTENSIONS:
        if base == "*":
            img_paths.extend(glob.glob(os.path.join(glob.escape(directory), base + ext)))
        else:
            img_paths.extend(glob.glob(glob.escape(os.path.join(directory, base + ext))))
    img_paths = list(set(img_paths))  # ?????
    img_paths.sort()
    return img_paths


def glob_images_pathlib(dir_path, recursive):
    image_paths = []
    if recursive:
        for ext in IMAGE_EXTENSIONS:
            image_paths += list(dir_path.rglob("*" + ext))
    else:
        for ext in IMAGE_EXTENSIONS:
            image_paths += list(dir_path.glob("*" + ext))
    image_paths = list(set(image_paths))  # ?????
    image_paths.sort()
    return image_paths


class MinimalDataset(BaseDataset):
    def __init__(self, resolution, network_multiplier, debug_dataset=False):
        super().__init__(resolution, network_multiplier, debug_dataset)

        self.num_train_images = 0  # update in subclass
        self.num_reg_images = 0  # update in subclass
        self.datasets = [self]
        self.batch_size = 1  # update in subclass

        self.subsets = [self]
        self.num_repeats = 1  # update in subclass if needed
        self.img_count = 1  # update in subclass if needed
        self.bucket_info = {}
        self.is_reg = False
        self.image_dir = "dummy"  # for metadata

    def verify_bucket_reso_steps(self, min_steps: int):
        pass

    def is_latent_cacheable(self) -> bool:
        return False

    def __len__(self):
        raise NotImplementedError

    # override to avoid shuffling buckets
    def set_current_epoch(self, epoch):
        self.current_epoch = epoch

    def __getitem__(self, idx):
        r"""
        The subclass may have image_data for debug_dataset, which is a dict of ImageInfo objects.

        Returns: example like this:

            for i in range(batch_size):
                image_key = ...  # whatever hashable
                image_keys.append(image_key)

                image = ...  # PIL Image
                img_tensor = self.image_transforms(img)
                images.append(img_tensor)

                caption = ...  # str
                input_ids = self.get_input_ids(caption)
                input_ids_list.append(input_ids)

                captions.append(caption)

            images = torch.stack(images, dim=0)
            input_ids_list = torch.stack(input_ids_list, dim=0)
            example = {
                "images": images,
                "input_ids": input_ids_list,
                "captions": captions,   # for debug_dataset
                "latents": None,
                "image_keys": image_keys,   # for debug_dataset
                "loss_weights": torch.ones(batch_size, dtype=torch.float32),
            }
            return example
        """
        raise NotImplementedError

    def get_resolutions(self) -> List[Tuple[int, int]]:
        return []


def load_arbitrary_dataset(args, tokenizer=None) -> MinimalDataset:
    module = ".".join(args.dataset_class.split(".")[:-1])
    dataset_class = args.dataset_class.split(".")[-1]
    module = importlib.import_module(module)
    dataset_class = getattr(module, dataset_class)
    train_dataset_group: MinimalDataset = dataset_class(tokenizer, args.max_token_length, args.resolution, args.debug_dataset)
    return train_dataset_group


def load_image(image_path, alpha=False):
    try:
        with Image.open(image_path) as image:
            if alpha:
                if not image.mode == "RGBA":
                    image = image.convert("RGBA")
            else:
                if not image.mode == "RGB":
                    image = image.convert("RGB")
            img = np.array(image, np.uint8)
            return img
    except (IOError, OSError) as e:
        logger.error(f"Error loading file: {image_path}")
        raise e


# ????????????numpy.ndarray,(original width, original height),(crop left, crop top, crop right, crop bottom)
def trim_and_resize_if_required(
    random_crop: bool, image: np.ndarray, reso, resized_size: Tuple[int, int], resize_interpolation: Optional[str] = None
) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int, int, int]]:
    image_height, image_width = image.shape[0:2]
    original_size = (image_width, image_height)  # size before resize

    if image_width != resized_size[0] or image_height != resized_size[1]:
        image = resize_image(image, image_width, image_height, resized_size[0], resized_size[1], resize_interpolation)

    image_height, image_width = image.shape[0:2]

    if image_width > reso[0]:
        trim_size = image_width - reso[0]
        p = trim_size // 2 if not random_crop else random.randint(0, trim_size)
        # logger.info(f"w {trim_size} {p}")
        image = image[:, p : p + reso[0]]
    if image_height > reso[1]:
        trim_size = image_height - reso[1]
        p = trim_size // 2 if not random_crop else random.randint(0, trim_size)
        # logger.info(f"h {trim_size} {p})
        image = image[p : p + reso[1]]

    # random crop????crop???????crop left/top?????????????????
    # I have no idea how to reflect the cropped value in crop left/top in the case of random crop

    crop_ltrb = BucketManager.get_crop_ltrb(reso, original_size)

    assert image.shape[0] == reso[1] and image.shape[1] == reso[0], f"internal error, illegal trimmed size: {image.shape}, {reso}"
    return image, original_size, crop_ltrb


# for new_cache_latents
def load_images_and_masks_for_caching(
    image_infos: List[ImageInfo], use_alpha_mask: bool, random_crop: bool
) -> Tuple[torch.Tensor, List[np.ndarray], List[Tuple[int, int]], List[Tuple[int, int, int, int]]]:
    r"""
    requires image_infos to have: [absolute_path or image], bucket_reso, resized_size

    returns: image_tensor, alpha_masks, original_sizes, crop_ltrbs

    image_tensor: torch.Tensor = torch.Size([B, 3, H, W]), ...], normalized to [-1, 1]
    alpha_masks: List[np.ndarray] = [np.ndarray([H, W]), ...], normalized to [0, 1]
    original_sizes: List[Tuple[int, int]] = [(W, H), ...]
    crop_ltrbs: List[Tuple[int, int, int, int]] = [(L, T, R, B), ...]
    """
    images: List[torch.Tensor] = []
    alpha_masks: List[np.ndarray] = []
    original_sizes: List[Tuple[int, int]] = []
    crop_ltrbs: List[Tuple[int, int, int, int]] = []
    for info in image_infos:
        image = load_image(info.absolute_path, use_alpha_mask) if info.image is None else np.array(info.image, np.uint8)
        # TODO ???????????????????????????bucket?????????????????????????????
        image, original_size, crop_ltrb = trim_and_resize_if_required(
            random_crop, image, info.bucket_reso, info.resized_size, resize_interpolation=info.resize_interpolation
        )

        original_sizes.append(original_size)
        crop_ltrbs.append(crop_ltrb)

        if use_alpha_mask:
            if image.shape[2] == 4:
                alpha_mask = image[:, :, 3]  # [H,W]
                alpha_mask = alpha_mask.astype(np.float32) / 255.0
                alpha_mask = torch.FloatTensor(alpha_mask)  # [H,W]
            else:
                alpha_mask = torch.ones_like(image[:, :, 0], dtype=torch.float32)  # [H,W]
        else:
            alpha_mask = None
        alpha_masks.append(alpha_mask)

        image = image[:, :, :3]  # remove alpha channel if exists
        image = IMAGE_TRANSFORMS(image)
        images.append(image)

    img_tensor = torch.stack(images, dim=0)
    return img_tensor, alpha_masks, original_sizes, crop_ltrbs


def cache_batch_latents(
    vae: AutoencoderKL, cache_to_disk: bool, image_infos: List[ImageInfo], flip_aug: bool, use_alpha_mask: bool, random_crop: bool
) -> None:
    r"""
    requires image_infos to have: absolute_path, bucket_reso, resized_size, latents_npz
    optionally requires image_infos to have: image
    if cache_to_disk is True, set info.latents_npz
        flipped latents is also saved if flip_aug is True
    if cache_to_disk is False, set info.latents
        latents_flipped is also set if flip_aug is True
    latents_original_size and latents_crop_ltrb are also set
    """
    images = []
    alpha_masks: List[np.ndarray] = []
    for info in image_infos:
        image = load_image(info.absolute_path, use_alpha_mask) if info.image is None else np.array(info.image, np.uint8)
        # TODO ???????????????????????????bucket?????????????????????????????
        image, original_size, crop_ltrb = trim_and_resize_if_required(
            random_crop, image, info.bucket_reso, info.resized_size, resize_interpolation=info.resize_interpolation
        )

        info.latents_original_size = original_size
        info.latents_crop_ltrb = crop_ltrb

        if use_alpha_mask:
            if image.shape[2] == 4:
                alpha_mask = image[:, :, 3]  # [H,W]
                alpha_mask = alpha_mask.astype(np.float32) / 255.0
                alpha_mask = torch.FloatTensor(alpha_mask)  # [H,W]
            else:
                alpha_mask = torch.ones_like(image[:, :, 0], dtype=torch.float32)  # [H,W]
        else:
            alpha_mask = None
        alpha_masks.append(alpha_mask)

        image = image[:, :, :3]  # remove alpha channel if exists
        image = IMAGE_TRANSFORMS(image)
        images.append(image)

    img_tensors = torch.stack(images, dim=0)
    img_tensors = img_tensors.to(device=vae.device, dtype=vae.dtype)

    with torch.no_grad():
        latents = vae.encode(img_tensors).latent_dist.sample().to("cpu")

    if flip_aug:
        img_tensors = torch.flip(img_tensors, dims=[3])
        with torch.no_grad():
            flipped_latents = vae.encode(img_tensors).latent_dist.sample().to("cpu")
    else:
        flipped_latents = [None] * len(latents)

    for info, latent, flipped_latent, alpha_mask in zip(image_infos, latents, flipped_latents,op_ltrb = trim_and_resize_if_required(
                        subset.random_crop,
                        img,
                        image_info.bucket_reso,
                        image_info.resized_size,
                        resize_interpolation=image_info.resize_interpolation,
                    )
                else:
                    if face_cx > 0:  # ???????
                        img = self.crop_target(subset, img, face_cx, face_cy, face_w, face_h)
                    elif im_h > self.height or im_w > self.width:
                        assert (
                            subset.random_crop
                        ), f"image too large, but cropping and bucketing are disabled / ???????????face_crop_aug_range?random_crop????bucket??????????: {image_info.absolute_path}"
                        if im_h > self.height:
                            p = random.randint(0, im_h - self.height)
                            img = img[p : p + self.height]
                        if im_w > self.width:
                            p = random.randint(0, im_w - self.width)
                            img = img[:, p : p + self.width]

                    im_h, im_w = img.shape[0:2]
                    assert (
                        im_h == self.height and im_w == self.width
                    ), f"image size is small / ?????????????: {image_info.absolute_path}"

                    original_size = [im_w, im_h]
                    crop_ltrb = (0, 0, 0, 0)

                # augmentation
                aug = self.aug_helper.get_augmentor(subset.color_aug)
                if aug is not None:
                    # augment RGB channels only
                    img_rgb = img[:, :, :3]
                    img_rgb = aug(image=img_rgb)["image"]
                    img[:, :, :3] = img_rgb

                if flipped:
                    img = img[:, ::-1, :].copy()  # copy to avoid negative stride problem

                if subset.alpha_mask:
                    if img.shape[2] == 4:
                        alpha_mask = img[:, :, 3]  # [H,W]
                        alpha_mask = alpha_mask.astype(np.float32) / 255.0  # 0.0~1.0
                        alpha_mask = torch.FloatTensor(alpha_mask)
                    else:
                        alpha_mask = torch.ones((img.shape[0], img.shape[1]), dtype=torch.float32)
                else:
                    alpha_mask = None

                img = img[:, :, :3]  # remove alpha channel

                latents = None
                image = self.image_transforms(img)  # -1.0~1.0?torch.Tensor???
                del img

            images.append(image)
            latents_list.append(latents)
            alpha_mask_list.append(alpha_mask)

            target_size = (image.shape[2], image.shape[1]) if image is not None else (latents.shape[2] * 8, latents.shape[1] * 8)

            if not flipped:
                crop_left_top = (crop_ltrb[0], crop_ltrb[1])
            else:
                # crop_ltrb[2] is right, so target_size[0] - crop_ltrb[2] is left in flipped image
                crop_left_top = (target_size[0] - crop_ltrb[2], crop_ltrb[1])

            original_sizes_hw.append((int(original_size[1]), int(original_size[0])))
            crop_top_lefts.append((int(crop_left_top[1]), int(crop_left_top[0])))
            target_sizes_hw.append((int(target_size[1]), int(target_size[0])))
            flippeds.append(flipped)

            # caption?text encoder output?????
            caption = image_info.caption  # default

            tokenization_required = (
                self.text_encoder_output_caching_strategy is None or self.text_encoder_output_caching_strategy.is_partial
            )
            text_encoder_outputs = None
            input_ids = None

            if image_info.text_encoder_outputs is not None:
                # cached
                text_encoder_outputs = image_info.text_encoder_outputs
            elif image_info.text_encoder_outputs_npz is not None:
                # on disk
                text_encoder_outputs = self.text_encoder_output_caching_strategy.load_outputs_npz(
                    image_info.text_encoder_outputs_npz
                )
            else:
                tokenization_required = True
            text_encoder_outputs_list.append(text_encoder_outputs)

            if tokenization_required:
                caption = self.process_caption(subset, image_info.caption)
                input_ids = [ids[0] for ids in self.tokenize_strategy.tokenize(caption)]  # remove batch dimension
                # if self.XTI_layers:
                #     caption_layer = []
                #     for layer in self.XTI_layers:
                #         token_strings_from = " ".join(self.token_strings)
                #         token_strings_to = " ".join([f"{x}_{layer}" for x in self.token_strings])
                #         caption_ = caption.replace(token_strings_from, token_strings_to)
                #         caption_layer.append(caption_)
                #     captions.append(caption_layer)
                # else:
                #     captions.append(caption)

                # if not self.token_padding_disabled:  # this option might be omitted in future
                #     # TODO get_input_ids must support SD3
                #     if self.XTI_layers:
                #         token_caption = self.get_input_ids(caption_layer, self.tokenizers[0])
                #     else:
                #         token_caption = self.get_input_ids(caption, self.tokenizers[0])
                #     input_ids_list.append(token_caption)

                #     if len(self.tokenizers) > 1:
                #         if self.XTI_layers:
                #             token_caption2 = self.get_input_ids(caption_layer, self.tokenizers[1])
                #         else:
                #             token_caption2 = self.get_input_ids(caption, self.tokenizers[1])
                #         input_ids2_list.append(token_caption2)

            input_ids_list.append(input_ids)
            captions.append(caption)

        def none_or_stack_elements(tensors_list, converter):
            # [[clip_l, clip_g, t5xxl], [clip_l, clip_g, t5xxl], ...] -> [torch.stack(clip_l), torch.stack(clip_g), torch.stack(t5xxl)]
            if len(tensors_list) == 0 or tensors_list[0] == None or len(tensors_list[0]) == 0 or tensors_list[0][0] is None:
                return None

            # old implementation without padding: all elements must have same length
            # return [torch.stack([converter(x[i]) for x in tensors_list]) for i in range(len(tensors_list[0]))]

            # new implementation with padding support
            result = []
            for i in range(len(tensors_list[0])):
                tensors = [x[i] for x in tensors_list]
                if tensors[0].ndim == 0:
                    # scalar value: e.g. ocr mask
                    result.append(torch.stack([converter(x[i]) for x in tensors_list]))
                    continue

                min_len = min([len(x) for x in tensors])
                max_len = max([len(x) for x in tensors])

                if min_len == max_len:
                    # no padding
                    result.append(torch.stack([converter(x) for x in tensors]))
                else:
                    # padding
                    tensors = [converter(x) for x in tensors]
                    if tensors[0].ndim == 1:
                        # input_ids or mask
                        result.append(torch.stack([(torch.nn.functional.pad(x, (0, max_len - x.shape[0]))) for x in tensors]))
                    else:
                        # text encoder outputs
                        result.append(torch.stack([(torch.nn.functional.pad(x, (0, 0, 0, max_len - x.shape[0]))) for x in tensors]))
            return result

        # set example
        example = {}
        example["custom_attributes"] = custom_attributes  # may be list of empty dict
        example["loss_weights"] = torch.FloatTensor(loss_weights)
        example["text_encoder_outputs_list"] = none_or_stack_elements(text_encoder_outputs_list, torch.FloatTensor)
        example["input_ids_list"] = none_or_stack_elements(input_ids_list, lambda x: x)

        # if one of alpha_masks is not None, we need to replace None with ones
        none_or_not = [x is None for x in alpha_mask_list]
        if all(none_or_not):
            example["alpha_masks"] = None
        elif any(none_or_not):
            for i in range(len(alpha_mask_list)):
                if alpha_mask_list[i] is None:
                    if images[i] is not None:
                        alpha_mask_list[i] = torch.ones((images[i].shape[1], images[i].shape[2]), dtype=torch.float32)
                    else:
                        alpha_mask_list[i] = torch.ones(
                            (latents_list[i].shape[1] * 8, latents_list[i].shape[2] * 8), dtype=torch.float32
                        )
            example["alpha_masks"] = torch.stack(alpha_mask_list)
        else:
            example["alpha_masks"] = torch.stack(alpha_mask_list)

        if images[0] is not None:
            images = torch.stack(images)
            images = images.to(memory_format=torch.contiguous_format).float()
        else:
            images = None
        example["images"] = images

        example["latents"] = torch.stack(latents_list) if latents_list[0] is not None else None
        example["captions"] = captions

        example["original_sizes_hw"] = torch.stack([torch.LongTensor(x) for x in original_sizes_hw])
        example["crop_top_lefts"] = torch.stack([torch.LongTensor(x) for x in crop_top_lefts])
        example["target_sizes_hw"] = torch.stack([torch.LongTensor(x) for x in target_sizes_hw])
        example["flippeds"] = flippeds

        example["network_multipliers"] = torch.FloatTensor([self.network_multiplier] * len(captions))

        if self.debug_dataset:
            example["image_keys"] = bucket[image_index : image_index + self.batch_size]
        return example

    def get_item_for_caching(self, bucket, bucket_batch_size, image_index):
        captions = []
        images = []
        input_ids1_list = []
        input_ids2_list = []
        absolute_paths = []
        resized_sizes = []
        bucket_reso = None
        flip_aug = None
        alpha_mask = None
        random_crop = None

        for image_key in bucket[image_index : image_index + bucket_batch_size]:
            image_info = self.image_data[image_key]
            subset = self.image_to_subset[image_key]

            if flip_aug is None:
                flip_aug = subset.flip_aug
                alpha_mask = subset.alpha_mask
                random_crop = subset.random_crop
                bucket_reso = image_info.bucket_reso
            else:
                # TODO ??????????????????????
                assert flip_aug == subset.flip_aug, "flip_aug must be same in a batch"
                assert alpha_mask == subset.alpha_mask, "alpha_mask must be same in a batch"
                assert random_crop == subset.random_crop, "random_crop must be same in a batch"
                assert bucket_reso == image_info.bucket_reso, "bucket_reso must be same in a batch"

            caption = image_info.caption  # TODO cache some patterns of dropping, shuffling, etc.

            if self.caching_mode == "latents":
                image = load_image(image_info.absolute_path)
            else:
                image = None

            if self.caching_mode == "text":
                input_ids1 = self.get_input_ids(caption, self.tokenizers[0])
                input_ids2 = self.get_input_ids(caption, self.tokenizers[1])
            else:
                input_ids1 = None
                input_ids2 = None

            captions.append(caption)
            images.append(image)
            input_ids1_list.append(input_ids1)
            input_ids2_list.append(input_ids2)
            absolute_paths.append(image_info.absolute_path)
            resized_sizes.append(image_info.resized_size)

        example = {}

        if images[0] is None:
            images = None
        example["images"] = images

        example["captions"] = captions
        example["input_ids1_list"] = input_ids1_list
        example["input_ids2_list"] = input_ids2_list
        example["absolute_paths"] = absolute_paths
        example["resized_sizes"] = resized_sizes
        example["flip_aug"] = flip_aug
        example["alpha_mask"] = alpha_mask
        example["random_crop"] = random_crop
        example["bucket_reso"] = bucket_reso
        return example


class DreamBoothDataset(BaseDataset):
    IMAGE_INFO_CACHE_FILE = "metadata_cache.json"

    # The is_training_dataset defines the type of dataset, training or validation
    # if is_training_dataset is True -> training dataset
    # if is_training_dataset is False -> validation dataset
    def __init__(
        self,
        subsets: Sequence[DreamBoothSubset],
        is_training_dataset: bool,
        batch_size: int,
        resolution,
        network_multiplier: float,
        enable_bucket: bool,
        min_bucket_reso: int,
        max_bucket_reso: int,
        bucket_reso_steps: int,
        bucket_no_upscale: bool,
        prior_loss_weight: float,
        debug_dataset: bool,
        validation_split: float,
        validation_seed: Optional[int],
        resize_interpolation: Optional[str],
    ) -> None:
        super().__init__(resolution, network_multiplier, debug_dataset, resize_interpolation)

        assert resolution is not None, f"resolution is required / resolution????????????"

        self.batch_size = batch_size
        self.size = min(self.width, self.height)  # ????
        self.prior_loss_weight = prior_loss_weight
        self.latents_cache = None
        self.is_training_dataset = is_training_dataset
        self.validation_seed = validation_seed
        self.validation_split = validation_split

        self.enable_bucket = enable_bucket
        if self.enable_bucket:
            min_bucket_reso, max_bucket_reso = self.adjust_min_max_bucket_reso_by_steps(
                resolution, min_bucket_reso, max_bucket_reso, bucket_reso_steps
            )
            self.min_bucket_reso = min_bucket_reso
            self.max_bucket_reso = max_bucket_reso
            self.bucket_reso_steps = bucket_reso_steps
            self.bucket_no_upscale = bucket_no_upscale
        else:
            self.min_bucket_reso = None
            self.max_bucket_reso = None
            self.bucket_reso_steps = None  # ??????????
            self.bucket_no_upscale = False

        def read_caption(img_path, caption_extension, enable_wildcard):
            # caption???????????
            base_name = os.path.splitext(img_path)[0]
            base_name_face_det = base_name
            tokens = base_name.split("_")
            if len(tokens) >= 5:
                base_name_face_det = "_".join(tokens[:-4])
            cap_paths = [base_name + caption_extension, base_name_face_det + caption_extension]

            caption = None
            for cap_path in cap_paths:
                if os.path.isfile(cap_path):
                    with open(cap_path, "rt", encoding="utf-8") as f:
                        try:
                            lines = f.readlines()
                        except UnicodeDecodeError as e:
                            logger.error(f"illegal char in file (not UTF-8) / ?????UTF-8??????????: {cap_path}")
                            raise e
                        assert len(lines) > 0, f"caption file is empty / ??????????????: {cap_path}"
                        if enable_wildcard:
                            caption = "\n".join([line.strip() for line in lines if line.strip() != ""])  # ???????????
                        else:
                            caption = lines[0].strip()
                    break
            return caption

        def load_dreambooth_dir(subset: DreamBoothSubset):
            if not os.path.isdir(subset.image_dir):
                logger.warning(f"not directory: {subset.image_dir}")
                return [], [], []

            info_cache_file = os.path.join(subset.image_dir, self.IMAGE_INFO_CACHE_FILE)
            use_cached_info_for_subset = subset.cache_info
            if use_cached_info_for_subset:
                logger.info(
                    f"using cached image info for this subset / ??????????????????????????: {info_cache_file}"
                )
                if not os.path.isfile(info_cache_file):
                    logger.warning(
                        f"image info file not found. You can ignore this warning if this is the first time to use this subset"
                        + " / ????????????????????????????????????????: {metadata_file}"
                    )
                    use_cached_info_for_subset = False

            if use_cached_info_for_subset:
                # json: {`img_path`:{"caption": "caption...", "resolution": [width, height]}, ...}
                with open(info_cache_file, "r", encoding="utf-8") as f:
                    metas = json.load(f)
                img_paths = list(metas.keys())
                sizes: List[Optional[Tuple[int, int]]] = [meta["resolution"] for meta in metas.values()]

                # we may need to check image size and existence of image files, but it takes time, so user should check it before training
            else:
                img_paths = glob_images(subset.image_dir, "*")
                sizes: List[Optional[Tuple[int, int]]] = [None] * len(img_paths)

                # new caching: get image size from cache files
                strategy = LatentsCachingStrategy.get_strategy()
                if strategy is not None:
                    logger.info("get image size from name of cache files")

                    # make image path to npz path mapping
                    npz_paths = glob.glob(os.path.join(subset.image_dir, "*" + strategy.cache_suffix))
                    npz_paths.sort(
                        key=lambda item: item.rsplit("_", maxsplit=2)[0]
                    )  # sort by name excluding resolution and cache_suffix
                    npz_path_index = 0

                    size_set_count = 0
                    for i, img_path in enumerate(tqdm(img_paths)):
                        l = len(os.path.splitext(img_path)[0])  # remove extension
                        found = False
                        while npz_path_index < len(npz_paths):  # until found or end of npz_paths
                            # npz_paths are sorted, so if npz_path > img_path, img_path is not found
                            if npz_paths[npz_path_index][:l] > img_path[:l]:
                                break
                            if npz_paths[npz_path_index][:l] == img_path[:l]:  # found
                                found = True
                                break
                            npz_path_index += 1  # next npz_path

                        if found:
                            w, h = strategy.get_image_size_from_disk_cache_path(img_path, npz_paths[npz_path_index])
                        else:
                            w, h = None, None

                        if w is not None and h is not None:
                            sizes[i] = (w, h)
                            size_set_count += 1
                    logger.info(f"set image size from cache files: {size_set_count}/{len(img_paths)}")

            # We want to create a training and validation split. This should be improved in the future
            # to allow a clearer distinction between training and validation. This can be seen as a
            # short-term solution to limit what is necessary to implement validation datasets
            #
            # We split the dataset for the subset based on if we are doing a validation split
            # The self.is_training_dataset defines the type of dataset, training or validation
            # if self.is_training_dataset is True -> training dataset
            # if self.is_training_dataset is False -> validation dataset
            if self.validation_split > 0.0:
                # For regularization images we do not want to split this dataset.
                if subset.is_reg is True:
                    # Skip any validation dataset for regularization images
                    if self.is_training_dataset is False:
                        img_paths = []
                        sizes = []
                    # Otherwise the img_paths remain as original img_paths and no split
                    # required for training images dataset of regularization images
                else:
                    img_paths, sizes = split_train_val(
                        img_paths, sizes, self.is_training_dataset, self.validation_split, self.validation_seed
                    )

            logger.info(f"found directory {subset.image_dir} contains {len(img_paths)} image files")

            if use_cached_info_for_subset:
                captions = [meta["caption"] for meta in metas.values()]
                missing_captions = [img_path for img_path, caption in zip(img_paths, captions) if caption is None or caption == ""]
            else:
                # ???????????????????????????????
                captions = []
                missing_captions = []
                for img_path in tqdm(img_paths, desc="read caption"):
                    cap_for_img = read_caption(img_path, subset.caption_extension, subset.enable_wildcard)
                    if cap_for_img is None and subset.class_tokens is None:
                        logger.warning(
                            f"neither caption file nor class tokens are found. use empty caption for {img_path} / ???????????class token??????????????????????????: {img_path}"
                        )
                        captions.append("")
                        missing_captions.append(img_path)
                    else:
                        if cap_for_img is None:
                            captions.append(subset.class_tokens)
                            missing_captions.append(img_path)
                        else:
                            captions.append(cap_for_img)

            self.set_tag_frequency(os.path.basename(subset.image_dir), captions)  # ???????

            if missing_captions:
                number_of_missing_captions = len(missing_captions)
                number_of_missing_captions_to_show = 5
                remaining_missing_captions = number_of_missing_captions - number_of_missing_captions_to_show

                logger.warning(
                    f"No caption file found for {number_of_missing_captions} images. Training will continue without captions for these images. If class token exists, it will be used. / {number_of_missing_captions}????????????????????????????????????????????????????????class token????????????????"
                )
                for i, missing_caption in enumerate(missing_captions):
                    if i >= number_of_missing_captions_to_show:
                        logger.warning(missing_caption + f"... and {remaining_missing_captions} more")
                        break
                    logger.warning(missing_caption)

            if not use_cached_info_for_subset and subset.cache_info:
                logger.info(f"cache image info for / ????????????? : {info_cache_file}")
                sizes = [self.get_image_size(img_path) for img_path in tqdm(img_paths, desc="get image size")]
                matas = {}
                for img_path, caption, size in zip(img_paths, captions, sizes):
                    matas[img_path] = {"caption": caption, "resolution": list(size)}
                with open(info_cache_file, "w", encoding="utf-8") as f:
                    json.dump(matas, f, ensure_ascii=False, indent=2)
                logger.info(f"cache image info done for / ??????????? : {info_cache_file}")

            # if sizes are not set, image size will be read in make_buckets
            return img_paths, captions, sizes

        logger.info("prepare images.")
        num_train_images = 0
        num_reg_images = 0
        reg_infos: List[Tuple[ImageInfo, DreamBoothSubset]] = []
        for subset in subsets:
            num_repeats = subset.num_repeats if self.is_training_dataset else 1
            if num_repeats < 1:
                logger.warning(
                    f"ignore subset with image_dir='{subset.image_dir}': num_repeats is less than 1 / num_repeats?1????????????????????: {num_repeats}"
                )
                continue

            if subset in self.subsets:
                logger.warning(
                    f"ignore duplicated subset with image_dir='{subset.image_dir}': use the first one / ????????????????????????????????????"
                )
                continue

            img_paths, captions, sizes = load_dreambooth_dir(subset)
            if len(img_paths) < 1:
                logger.warning(
                    f"ignore subset with image_dir='{subset.image_dir}': no images found / ??????????????????????"
                )
                continue

            if subset.is_reg:
                num_reg_images += num_repeats * len(img_paths)
            else:
                num_train_images += num_repeats * len(img_paths)

            for img_path, caption, size in zip(img_paths, captions, sizes):
                info = ImageInfo(img_path, num_repeats, caption, subset.is_reg, img_path)
                info.resize_interpolation = (
                    subset.resize_interpolation if subset.resize_interpolation is not None else self.resize_interpolation
                )
                if size is not None:
                    info.image_size = size
                if subset.is_reg:
                    reg_infos.append((info, subset))
                else:
                    self.register_image(info, subset)

            subset.img_count = len(img_paths)
            self.subsets.append(subset)

        images_split_name = "train" if self.is_training_dataset else "validation"
        logger.info(f"{num_train_images} {images_split_name} images with repeats.")

        self.num_train_images = num_train_images

        logger.info(f"{num_reg_images} reg images with repeats.")
        if num_train_images < num_reg_images:
            logger.warning("some of reg images are not used / ???????????????????????????????")

        if num_reg_images == 0:
            logger.warning("no regularization images / ????????????????")
        else:
            # num_repeats???????????????????????????
            n = 0
            first_loop = True
            while n < num_train_images:
                for info, subset in reg_infos:
                    if first_loop:
                        self.register_image(info, subset)
                        n += info.num_repeats
                    else:
                        info.num_repeats += 1  # rewrite registered info
                        n += 1
                    if n >= num_train_images:
                        break
                first_loop = False

        self.num_reg_images = num_reg_images


class FineTuningDataset(BaseDataset):
    def __init__(
        self,
        subsets: Sequence[FineTuningSubset],
        batch_size: int,
        resolution,
        network_multiplier: float,
        enable_bucket: bool,
        min_bucket_reso: int,
        max_bucket_reso: int,
        bucket_reso_steps: int,
        bucket_no_upscale: bool,
        debug_dataset: bool,
        validation_seed: int,
        validation_split: float,
        resize_interpolation: Optional[str],
    ) -> None:
        super().__init__(resolution, network_multiplier, debug_dataset, resize_interpolation)

        self.batch_size = batch_size
        self.size = min(self.width, self.height)  # ????
        self.latents_cache = None

        self.enable_bucket = enable_bucket
        if self.enable_bucket:
            min_bucket_reso, max_bucket_reso = self.adjust_min_max_bucket_reso_by_steps(
                resolution, min_bucket_reso, max_bucket_reso, bucket_reso_steps
            )
            self.min_bucket_reso = min_bucket_reso
            self.max_bucket_reso = max_bucket_reso
            self.bucket_reso_steps = bucket_reso_steps
            self.bucket_no_upscale = bucket_no_upscale
        else:
            self.min_bucket_reso = None
            self.max_bucket_reso = None
            self.bucket_reso_steps = None  # ??????????
            self.bucket_no_upscale = False

        self.num_train_images = 0
        self.num_reg_images = 0

        for subset in subsets:
            if subset.num_repeats < 1:
                logger.warning(
                    f"ignore subset with metadata_file='{subset.metadata_file}': num_repeats is less than 1 / num_repeats?1????????????????????: {subset.num_repeats}"
                )
                continue

            if subset in self.subsets:
                logger.warning(
                    f"ignore duplicated subset with metadata_file='{subset.metadata_file}': use the first one / ????????????????????????????????????"
                )
                continue

            # ??????????
            if os.path.exists(subset.metadata_file):
                if subset.metadata_file.endswith(".jsonl"):
                    logger.info(f"loading existing JSOL metadata: {subset.metadata_file}")
                    # optional JSONL format
                    # {"image_path": "/path/to/image1.jpg", "caption": "A caption for image1", "image_size": [width, height]}
                    metadata = {}
                    with open(subset.metadata_file, "rt", encoding="utf-8") as f:
                        for line in f:
                            line_md = json.loads(line)
                            image_md = {"caption": line_md.get("caption", "")}
                            if "image_size" in line_md:
                                image_md["image_size"] = line_md["image_size"]
                            if "tags" in line_md:
                                image_md["tags"] = line_md["tags"]
                            metadata[line_md["image_path"]] = image_md
                else:
                    # standard JSON format
                    logger.info(f"loading existing metadata: {subset.metadata_file}")
                    with open(subset.metadata_file, "rt", encoding="utf-8") as f:
                        metadata = json.load(f)
            else:
                raise ValueError(f"no metadata / ???????????????: {subset.metadata_file}")

            if len(metadata) < 1:
                logger.warning(
                    f"ignore subset with '{subset.metadata_file}': no image entries found / ?????????????????????????????"
                )
                continue

            # Add full path for image
            image_dirs = set()
            if subset.image_dir is not None:
                image_dirs.add(subset.image_dir)
            for image_key in metadata.keys():
                if not os.path.isabs(image_key):
                    assert (
                        subset.image_dir is not None
                    ), f"image_dir is required when image paths are relative / ?????????????image_dir????????: {image_key}"
                    abs_path = os.path.join(subset.image_dir, image_key)
                else:
                    abs_path = image_key
                    image_dirs.add(os.path.dirname(abs_path))
                metadata[image_key]["abs_path"] = abs_path

            # Enumerate existing npz files
            strategy = LatentsCachingStrategy.get_strategy()
            npz_paths = []
            for image_dir in image_dirs:
                npz_paths.extend(glob.glob(os.path.join(image_dir, "*" + strategy.cache_suffix)))
            npz_paths = sorted(npz_paths, key=lambda item: len(os.path.basename(item)), reverse=True)  # longer paths first

            # Match image filename longer to shorter because some images share same prefix
            image_keys_sorted_by_length_desc = sorted(metadata.keys(), key=len, reverse=True)

            # Collect tags and sizes
            tags_list = []
            size_set_from_metadata = 0
            size_set_from_cache_filename = 0
            for image_key in image_keys_sorted_by_length_desc:
                img_md = metadata[image_key]
                caption = img_md.get("caption")
                tags = img_md.get("tags")
                image_size = img_md.get("image_size")
                abs_path = img_md.get("abs_path")

                # search npz if image_size is not given
                npz_path = None
                if image_size is None:
                    image_without_ext = os.path.splitext(image_key)[0]
                    for candidate in npz_paths:
                        if candidate.startswith(image_without_ext):
                            npz_path = candidate
                            break
                    if npz_path is not None:
                        npz_paths.remove(npz_path)  # remove to avoid matching same file (share prefix)
                        abs_path = npz_path

                if caption is None:
                    caption = ""

                if subset.enable_wildcard:
                    # tags must be single line (split by caption separator)
                    if tags is not None:
                        tags = tags.replace("\n", subset.caption_separator)

                    # add tags to each line of caption
                    if tags is not None:
                        caption = "\n".join(
                            [f"{line}{subset.caption_separator}{tags}" for line in caption.split("\n") if line.strip() != ""]
                        )
                        tags_list.append(tags)
                else:
                    # use as is
                    if tags is not None and len(tags) > 0:
                        if len(caption) > 0:
                            caption = caption + subset.caption_separator
                        caption = caption + tags
                        tags_list.append(tags)

                if caption is None:
                    caption = ""

                image_info = ImageInfo(image_key, subset.num_repeats, caption, False, abs_path)
                image_info.resize_interpolation = (
                    subset.resize_interpolation if subset.resize_interpolation is not None else self.resize_interpolation
                )

                if image_size is not None:
                    image_info.image_size = tuple(image_size)  # width, height
                    size_set_from_metadata += 1
                elif npz_path is not None:
                    # get image size from npz filename
                    w, h = strategy.get_image_size_from_disk_cache_path(abs_path, npz_path)
                    image_info.image_size = (w, h)
                    size_set_from_cache_filename += 1

                self.register_image(image_info, subset)

            if size_set_from_cache_filename > 0:
                logger.info(
                    f"set image size from cache files: {size_set_from_cache_filename}/{len(image_keys_sorted_by_length_desc)}"
                )
            if size_set_from_metadata > 0:
                logger.info(f"set image size from metadata: {size_set_from_metadata}/{len(image_keys_sorted_by_length_desc)}")
            self.num_train_images += len(metadata) * subset.num_repeats

            # TODO do not record tag freq when no tag
            self.set_tag_frequency(os.path.basename(subset.metadata_file), tags_list)
            subset.img_count = len(metadata)
            self.subsets.append(subset)


class ControlNetDataset(BaseDataset):
    def __init__(
        self,
        subsets: Sequence[ControlNetSubset],
        batch_size: int,
        resolution,
        network_multiplier: float,
        enable_bucket: bool,
        min_bucket_reso: int,
        max_bucket_reso: int,
        bucket_reso_steps: int,
        bucket_no_upscale: bool,
        debug_dataset: bool,
        validation_split: float,
        validation_seed: Optional[int],
        resize_interpolation: Optional[str] = None,
    ) -> None:
        super().__init__(resolution, network_multiplier, debug_dataset, resize_interpolation)

        db_subsets = []
        for subset in subsets:
            assert (
                not subset.random_crop
            ), "random_crop is not supported in ControlNetDataset / random_crop?ControlNetDataset?????????????"
            db_subset = DreamBoothSubset(
                subset.image_dir,
                False,
                None,
                subset.caption_extension,
                subset.cache_info,
                False,
                subset.num_repeats,
                subset.shuffle_caption,
                subset.caption_separator,
                subset.keep_tokens,
                subset.keep_tokens_separator,
                subset.secondary_separator,
                subset.enable_wildcard,
                subset.color_aug,
                subset.flip_aug,
                subset.face_crop_aug_range,
                subset.random_crop,
                subset.caption_dropout_rate,
                subset.caption_dropout_every_n_epochs,
                subset.caption_tag_dropout_rate,
                subset.caption_prefix,
                subset.caption_suffix,
                subset.token_warmup_min,
                subset.token_warmup_step,
                resize_interpolation=subset.resize_interpolation,
            )
            db_subsets.append(db_subset)

        self.dreambooth_dataset_delegate = DreamBoothDataset(
            db_subsets,
            True,
            batch_size,
            resolution,
            network_multiplier,
            enable_bucket,
            min_bucket_reso,
            max_bucket_reso,
            bucket_reso_steps,
            bucket_no_upscale,
            1.0,
            debug_dataset,
            validation_split,
            validation_seed,
            resize_interpolation,
        )

        # config_util???????????????????????????????
        self.image_data = self.dreambooth_dataset_delegate.image_data
        self.batch_size = batch_size
        self.num_train_images = self.dreambooth_dataset_delegate.num_train_images
        self.num_reg_images = self.dreambooth_dataset_delegate.num_reg_images
        self.validation_split = validation_split
        self.validation_seed = validation_seed
        self.resize_interpolation = resize_interpolation

        # assert all conditioning data exists
        missing_imgs = []
        cond_imgs_with_pair = set()
        for image_key, info in self.dreambooth_dataset_delegate.image_data.items():
            db_subset = self.dreambooth_dataset_delegate.image_to_subset[image_key]
            subset = None
            for s in subsets:
                if s.image_dir == db_subset.image_dir:
                    subset = s
                    break
            assert subset is not None, "internal error: subset not found"

            if not os.path.isdir(subset.conditioning_data_dir):
                logger.warning(f"not directory: {subset.conditioning_data_dir}")
                continue

            img_basename = os.path.splitext(os.path.basename(info.absolute_path))[0]
            ctrl_img_path = glob_images(subset.conditioning_data_dir, img_basename)
            if len(ctrl_img_path) < 1:
                missing_imgs.append(img_basename)
                continue
            ctrl_img_path = ctrl_img_path[0]
            ctrl_img_path = os.path.abspath(ctrl_img_path)  # normalize path

            info.cond_img_path = ctrl_img_path
            cond_imgs_with_pair.add(os.path.splitext(ctrl_img_path)[0])  # remove extension because Windows is case insensitive

        extra_imgs = []
        for subset in subsets:
            conditioning_img_paths = glob_images(subset.conditioning_data_dir, "*")
            conditioning_img_paths = [os.path.abspath(p) for p in conditioning_img_paths]  # normalize path
            extra_imgs.extend([p for p in conditioning_img_paths if os.path.splitext(p)[0] not in cond_imgs_with_pair])

        assert (
            len(missing_imgs) == 0
        ), f"missing conditioning data for {len(missing_imgs)} images / ????????????????: {missing_imgs}"
        assert (
            len(extra_imgs) == 0
        ), f"extra conditioning data for {len(extra_imgs)} images / ?????????????: {extra_imgs}"

        self.conditioning_image_transforms = IMAGE_TRANSFORMS

    def set_current_strategies(self):
        return self.dreambooth_dataset_delegate.set_current_strategies()

    def make_buckets(self):
        self.dreambooth_dataset_delegate.make_buckets()
        self.bucket_manager = self.dreambooth_dataset_delegate.bucket_manager
        self.buckets_indices = self.dreambooth_dataset_delegate.buckets_indices

    def cache_latents(self, vae, vae_batch_size=1, cache_to_disk=False, is_main_process=True):
        return self.dreambooth_dataset_delegate.cache_latents(vae, vae_batch_size, cache_to_disk, is_main_process)

    def new_cache_latents(self, model: Any, accelerator: Accelerator):
        return self.dreambooth_dataset_delegate.new_cache_latents(model, accelerator)

    def new_cache_text_encoder_outputs(self, models: List[Any], is_main_process: bool):
        return self.dreambooth_dataset_delegate.new_cache_text_encoder_outputs(models, is_main_process)

    def __len__(self):
        return self.dreambooth_dataset_delegate.__len__()

    def __getitem__(self, index):
        example = self.dreambooth_dataset_delegate[index]

        bucket = self.dreambooth_dataset_delegate.bucket_manager.buckets[
            self.dreambooth_dataset_delegate.buckets_indices[index].bucket_index
        ]
        bucket_batch_size = self.dreambooth_dataset_delegate.buckets_indices[index].bucket_batch_size
        image_index = self.dreambooth_dataset_delegate.buckets_indices[index].batch_index * bucket_batch_size

        conditioning_images = []

        for i, image_key in enumerate(bucket[image_index : image_index + bucket_batch_size]):
            image_info = self.dreambooth_dataset_delegate.image_data[image_key]

            target_size_hw = example["target_sizes_hw"][i]
            original_size_hw = example["original_sizes_hw"][i]
            crop_top_left = example["crop_top_lefts"][i]
            flipped = example["flippeds"][i]
            cond_img = load_image(image_info.cond_img_path)

            if self.dreambooth_dataset_delegate.enable_bucket:
                assert (
                    cond_img.shape[0] == original_size_hw[0] and cond_img.shape[1] == original_size_hw[1]
                ), f"size of conditioning image is not match / ???????????: {image_info.absolute_path}"

                cond_img = resize_image(
                    cond_img,
                    original_size_hw[1],
                    original_size_hw[0],
                    target_size_hw[1],
                    target_size_hw[0],
                    self.resize_interpolation,
                )

                # TODO support random crop
                # ??????????crop?random????????
                h, w = target_size_hw
                ct = (cond_img.shape[0] - h) // 2
                cl = (cond_img.shape[1] - w) // 2
                cond_img = cond_img[ct : ct + h, cl : cl + w]
            else:
                # assert (
                #     cond_img.shape[0] == self.height and cond_img.shape[1] == self.width
                # ), f"image size is small / ?????????????: {image_info.absolute_path}"
                # resize to target
                if cond_img.shape[0] != target_size_hw[0] or cond_img.shape[1] != target_size_hw[1]:
                    cond_img = resize_image(
                        cond_img,
                        cond_img.shape[0],
                        cond_img.shape[1],
                        target_size_hw[1],
                        target_size_hw[0],
                        self.resize_interpolation,
                    )

            if flipped:
                cond_img = cond_img[:, ::-1, :].copy()  # copy to avoid negative stride

            cond_img = self.conditioning_image_transforms(cond_img)
            conditioning_images.append(cond_img)

        example["conditioning_images"] = torch.stack(conditioning_images).to(memory_format=torch.contiguous_format).float()

        return example


# behave as Dataset mock
class DatasetGroup(torch.utils.data.ConcatDataset):
    def __init__(self, datasets: Sequence[Union[DreamBoothDataset, FineTuningDataset]]):
        self.datasets: List[Union[DreamBoothDataset, FineTuningDataset]]

        super().__init__(datasets)

        self.image_data = {}
        self.num_train_images = 0
        self.num_reg_images = 0

        # simply concat together
        # TODO: handling image_data key duplication among dataset
        #   In practical, this is not the big issue because image_data is accessed from outside of dataset only for debug_dataset.
        for dataset in datasets:
            self.image_data.update(dataset.image_data)
            self.num_train_images += dataset.num_train_images
            self.num_reg_images += dataset.num_reg_images

    def add_replacement(self, str_from, str_to):
        for dataset in self.datasets:
            dataset.add_replacement(str_from, str_to)

    # def make_buckets(self):
    #   for dataset in self.datasets:
    #     dataset.make_buckets()

    def set_text_encoder_output_caching_strategy(self, strategy: TextEncoderOutputsCachingStrategy):
        """
        DataLoader is run in multiple processes, so we need to set the strategy manually.
        """
        for dataset in self.datasets:
            dataset.set_text_encoder_output_caching_strategy(strategy)

    def enable_XTI(self, *args, **kwargs):
        for dataset in self.datasets:
            dataset.enable_XTI(*args, **kwargs)

    def cache_latents(self, vae, vae_batch_size=1, cache_to_disk=False, is_main_process=True, file_suffix=".npz"):
        for i, dataset in enumerate(self.datasets):
            logger.info(f"[Dataset {i}]")
            dataset.cache_latents(vae, vae_batch_size, cache_to_disk, is_main_process, file_suffix)

    def new_cache_latents(self, model: Any, accelerator: Accelerator):
        for i, dataset in enumerate(self.datasets):
            logger.info(f"[Dataset {i}]")
            dataset.new_cache_latents(model, accelerator)
        accelerator.wait_for_everyone()

    def cache_text_encoder_outputs(
        self, tokenizers, text_encoders, device, weight_dtype, cache_to_disk=False, is_main_process=True
    ):
        for i, dataset in enumerate(self.datasets):
            logger.info(f"[Dataset {i}]")
            dataset.cache_text_encoder_outputs(tokenizers, text_encoders, device, weight_dtype, cache_to_disk, is_main_process)

    def cache_text_encoder_outputs_sd3(
        self, tokenizer, text_encoders, device, output_dtype, te_dtypes, cache_to_disk=False, is_main_process=True, batch_size=None
    ):
        for i, dataset in enumerate(self.datasets):
            logger.info(f"[Dataset {i}]")
            dataset.cache_text_encoder_outputs_sd3(
                tokenizer, text_encoders, device, output_dtype, te_dtypes, cache_to_disk, is_main_process, batch_size
            )

    def new_cache_text_encoder_outputs(self, models: List[Any], accelerator: Accelerator):
        for i, dataset in enumerate(self.datasets):
            logger.info(f"[Dataset {i}]")
            dataset.new_cache_text_encoder_outputs(models, accelerator)
        accelerator.wait_for_everyone()

    def set_caching_mode(self, caching_mode):
        for dataset in self.datasets:
            dataset.set_caching_mode(caching_mode)

    def verify_bucket_reso_steps(self, min_steps: int):
        for dataset in self.datasets:
            dataset.verify_bucket_reso_steps(min_steps)

    def get_resolutions(self) -> List[Tuple[int, int]]:
        return [(dataset.width, dataset.height) for dataset in self.datasets]

    def is_latent_cacheable(self) -> bool:
        return all([dataset.is_latent_cacheable() for dataset in self.datasets])

    def is_text_encoder_output_cacheable(self) -> bool:
        return all([dataset.is_text_encoder_output_cacheable() for dataset in self.datasets])

    def set_current_strategies(self):
        for dataset in self.datasets:
            dataset.set_current_strategies()

    def set_current_epoch(self, epoch):
        for dataset in self.datasets:
            dataset.set_current_epoch(epoch)

    def set_current_step(self, step):
        for dataset in self.datasets:
            dataset.set_current_step(step)

    def set_max_train_steps(self, max_train_steps):
        for dataset in self.datasets:
            dataset.set_max_train_steps(max_train_steps)

    def disable_token_padding(self):
        for dataset in self.datasets:
            dataset.disable_token_padding()


def is_disk_cached_latents_is_expected(reso, npz_path: str, flip_aug: bool, alpha_mask: bool):
    expected_latents_size = (reso[1] // 8, reso[0] // 8)  # bucket_reso?WxH?????

    if not os.path.exists(npz_path):
        return False

    try:
        npz = np.load(npz_path)
        if "latents" not in npz or "original_size" not in npz or "crop_ltrb" not in npz:  # old ver?
            return False
        if npz["latents"].shape[1:3] != expected_latents_size:
            return False

        if flip_aug:
            if "latents_flipped" not in npz:
                return False
            if npz["latents_flipped"].shape[1:3] != expected_latents_size:
                return False

        if alpha_mask:
            if "alpha_mask" not in npz:
                return False
            if (npz["alpha_mask"].shape[1], npz["alpha_mask"].shape[0]) != reso:  # HxW => WxH != reso
                return False
        else:
            if "alpha_mask" in npz:
                return False
    except Exception as e:
        logger.error(f"Error loading file: {npz_path}")
        raise e

    return True


# ?????latents_tensor, (original_size width, original_size height), (crop left, crop top)
# TODO update to use CachingStrategy
# def load_latents_from_disk(
#     npz_path,
# ) -> Tuple[Optional[np.ndarray], Optional[List[int]], Optional[List[int]], Optional[np.ndarray], Optional[np.ndarray]]:
#     npz = np.load(npz_path)
#     if "latents" not in npz:
#         raise ValueError(f"error: npz is old format. please re-generate {npz_path}")

#     latents = npz["latents"]
#     original_size = npz["original_size"].tolist()
#     crop_ltrb = npz["crop_ltrb"].tolist()
#     flipped_latents = npz["latents_flipped"] if "latents_flipped" in npz else None
#     alpha_mask = npz["alpha_mask"] if "alpha_mask" in npz else None
#     return latents, original_size, crop_ltrb, flipped_latents, alpha_mask


# def save_latents_to_disk(npz_path, latents_tensor, original_size, crop_ltrb, flipped_latents_tensor=None, alpha_mask=None):
#     kwargs = {}
#     if flipped_latents_tensor is not None:
#         kwargs["latents_flipped"] = flipped_latents_tensor.float().cpu().numpy()
#     if alpha_mask is not None:
#         kwargs["alpha_mask"] = alpha_mask.float().cpu().numpy()
#     np.savez(
#         npz_path,
#         latents=latents_tensor.float().cpu().numpy(),
#         original_size=np.array(original_size),
#         crop_ltrb=np.array(crop_ltrb),
#         **kwargs,
#     )


def debug_dataset(train_dataset, show_input_ids=False):
    logger.info(f"Total dataset length (steps) / ????????????????: {len(train_dataset)}")
    logger.info(
        "`S` for next step, `E` for next epoch no. , Escape for exit. / S??????????E??????????Esc???????????"
    )

    epoch = 1
    while True:
        logger.info(f"")
        logger.info(f"epoch: {epoch}")

        steps = (epoch - 1) * len(train_dataset) + 1
        indices = list(range(len(train_dataset)))
        random.shuffle(indices)

        k = 0
        for i, idx in enumerate(indices):
            train_dataset.set_current_epoch(epoch)
            train_dataset.set_current_step(steps)
            logger.info(f"steps: {steps} ({i + 1}/{len(train_dataset)})")

            example = train_dataset[idx]
            if example["latents"] is not None:
                logger.info(f"sample has latents from npz file: {example['latents'].size()}")
            for j, (ik, cap, lw, orgsz, crptl, trgsz, flpdz) in enumerate(
                zip(
                    example["image_keys"],
                    example["captions"],
                    example["loss_weights"],
                    # example["input_ids"],
                    example["original_sizes_hw"],
                    example["crop_top_lefts"],
                    example["target_sizes_hw"],
                    example["flippeds"],
                )
            ):
                logger.info(
                    f'{ik}, size: {train_dataset.image_data[ik].image_size}, loss weight: {lw}, caption: "{cap}", original size: {orgsz}, crop top left: {crptl}, target size: {trgsz}, flipped: {flpdz}'
                )
                if "network_multipliers" in example:
                    logger.info(f"network multiplier: {example['network_multipliers'][j]}")
                if "custom_attributes" in example:
                    logger.info(f"custom attributes: {example['custom_attributes'][j]}")

                # if show_input_ids:
                #     logger.info(f"input ids: {iid}")
                #     if "input_ids2" in example:
                #         logger.info(f"input ids2: {example['input_ids2'][j]}")
                if example["images"] is not None:
                    im = example["images"][j]
                    logger.info(f"image size: {im.size()}")
                    im = ((im.numpy() + 1.0) * 127.5).astype(np.uint8)
                    im = np.transpose(im, (1, 2, 0))  # c,H,W -> H,W,c
                    im = im[:, :, ::-1]  # RGB -> BGR (OpenCV)

                    if "conditioning_images" in example:
                        cond_img = example["conditioning_images"][j]
                        logger.info(f"conditioning image size: {cond_img.size()}")
                        cond_img = ((cond_img.numpy() + 1.0) * 127.5).astype(np.uint8)
                        cond_img = np.transpose(cond_img, (1, 2, 0))
                        cond_img = cond_img[:, :, ::-1]
                        if os.name == "nt":
                            cv2.imshow("cond_img", cond_img)

                    if "alpha_masks" in example and example["alpha_masks"] is not None:
                        alpha_mask = example["alpha_masks"][j]
                        logger.info(f"alpha mask size: {alpha_mask.size()}")
                        alpha_mask = (alpha_mask.numpy() * 255.0).astype(np.uint8)
                        if os.name == "nt":
                            cv2.imshow("alpha_mask", alpha_mask)

                    if os.name == "nt":  # only windows
                        cv2.imshow("img", im)
                        k = cv2.waitKey()
                        cv2.destroyAllWindows()
                    if k == 27 or k == ord("s") or k == ord("e"):
                        break
            steps += 1

            if k == ord("e"):
                break
            if k == 27 or (example["images"] is None and i >= 8):
                k = 27
                break
        if k == 27:
            break

        epoch += 1


def glob_images(directory, base="*"):
    img_paths = []
    for ext in IMAGE_EXTENSIONS:
        if base == "*":
            img_paths.extend(glob.glob(os.path.join(glob.escape(directory), base + ext)))
        else:
            img_paths.extend(glob.glob(glob.escape(os.path.join(directory, base + ext))))
    img_paths = list(set(img_paths))  # ?????
    img_paths.sort()
    return img_paths


def glob_images_pathlib(dir_path, recursive):
    image_paths = []
    if recursive:
        for ext in IMAGE_EXTENSIONS:
            image_paths += list(dir_path.rglob("*" + ext))
    else:
        for ext in IMAGE_EXTENSIONS:
            image_paths += list(dir_path.glob("*" + ext))
    image_paths = list(set(image_paths))  # ?????
    image_paths.sort()
    return image_paths


class MinimalDataset(BaseDataset):
    def __init__(self, resolution, network_multiplier, debug_dataset=False):
        super().__init__(resolution, network_multiplier, debug_dataset)

        self.num_train_images = 0  # update in subclass
        self.num_reg_images = 0  # update in subclass
        self.datasets = [self]
        self.batch_size = 1  # update in subclass

        self.subsets = [self]
        self.num_repeats = 1  # update in subclass if needed
        self.img_count = 1  # update in subclass if needed
        self.bucket_info = {}
        self.is_reg = False
        self.image_dir = "dummy"  # for metadata

    def verify_bucket_reso_steps(self, min_steps: int):
        pass

    def is_latent_cacheable(self) -> bool:
        return False

    def __len__(self):
        raise NotImplementedError

    # override to avoid shuffling buckets
    def set_current_epoch(self, epoch):
        self.current_epoch = epoch

    def __getitem__(self, idx):
        r"""
        The subclass may have image_data for debug_dataset, which is a dict of ImageInfo objects.

        Returns: example like this:

            for i in range(batch_size):
                image_key = ...  # whatever hashable
                image_keys.append(image_key)

                image = ...  # PIL Image
                img_tensor = self.image_transforms(img)
                images.append(img_tensor)

                caption = ...  # str
                input_ids = self.get_input_ids(caption)
                input_ids_list.append(input_ids)

                captions.append(caption)

            images = torch.stack(images, dim=0)
            input_ids_list = torch.stack(input_ids_list, dim=0)
            example = {
                "images": images,
                "input_ids": input_ids_list,
                "captions": captions,   # for debug_dataset
                "latents": None,
                "image_keys": image_keys,   # for debug_dataset
                "loss_weights": torch.ones(batch_size, dtype=torch.float32),
            }
            return example
        """
        raise NotImplementedError

    def get_resolutions(self) -> List[Tuple[int, int]]:
        return []


def load_arbitrary_dataset(args, tokenizer=None) -> MinimalDataset:
    module = ".".join(args.dataset_class.split(".")[:-1])
    dataset_class = args.dataset_class.split(".")[-1]
    module = importlib.import_module(module)
    dataset_class = getattr(module, dataset_class)
    train_dataset_group: MinimalDataset = dataset_class(tokenizer, args.max_token_length, args.resolution, args.debug_dataset)
    return train_dataset_group


def load_image(image_path, alpha=False):
    try:
        with Image.open(image_path) as image:
            if alpha:
                if not image.mode == "RGBA":
                    image = image.convert("RGBA")
            else:
                if not image.mode == "RGB":
                    image = image.convert("RGB")
            img = np.array(image, np.uint8)
            return img
    except (IOError, OSError) as e:
        logger.error(f"Error loading file: {image_path}")
        raise e


# ????????????numpy.ndarray,(original width, original height),(crop left, crop top, crop right, crop bottom)
def trim_and_resize_if_required(
    random_crop: bool, image: np.ndarray, reso, resized_size: Tuple[int, int], resize_interpolation: Optional[str] = None
) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int, int, int]]:
    image_height, image_width = image.shape[0:2]
    original_size = (image_width, image_height)  # size before resize

    if image_width != resized_size[0] or image_height != resized_size[1]:
        image = resize_image(image, image_width, image_height, resized_size[0], resized_size[1], resize_interpolation)

    image_height, image_width = image.shape[0:2]

    if image_width > reso[0]:
        trim_size = image_width - reso[0]
        p = trim_size // 2 if not random_crop else random.randint(0, trim_size)
        # logger.info(f"w {trim_size} {p}")
        image = image[:, p : p + reso[0]]
    if image_height > reso[1]:
        trim_size = image_height - reso[1]
        p = trim_size // 2 if not random_crop else random.randint(0, trim_size)
        # logger.info(f"h {trim_size} {p})
        image = image[p : p + reso[1]]

    # random crop????crop???????crop left/top?????????????????
    # I have no idea how to reflect the cropped value in crop left/top in the case of random crop

    crop_ltrb = BucketManager.get_crop_ltrb(reso, original_size)

    assert image.shape[0] == reso[1] and image.shape[1] == reso[0], f"internal error, illegal trimmed size: {image.shape}, {reso}"
    return image, original_size, crop_ltrb


# for new_cache_latents
def load_images_and_masks_for_caching(
    image_infos: List[ImageInfo], use_alpha_mask: bool, random_crop: bool
) -> Tuple[torch.Tensor, List[np.ndarray], List[Tuple[int, int]], List[Tuple[int, int, int, int]]]:
    r"""
    requires image_infos to have: [absolute_path or image], bucket_reso, resized_size

    returns: image_tensor, alpha_masks, original_sizes, crop_ltrbs

    image_tensor: torch.Tensor = torch.Size([B, 3, H, W]), ...], normalized to [-1, 1]
    alpha_masks: List[np.ndarray] = [np.ndarray([H, W]), ...], normalized to [0, 1]
    original_sizes: List[Tuple[int, int]] = [(W, H), ...]
    crop_ltrbs: List[Tuple[int, int, int, int]] = [(L, T, R, B), ...]
    """
    images: List[torch.Tensor] = []
    alpha_masks: List[np.ndarray] = []
    original_sizes: List[Tuple[int, int]] = []
    crop_ltrbs: List[Tuple[int, int, int, int]] = []
    for info in image_infos:
        image = load_image(info.absolute_path, use_alpha_mask) if info.image is None else np.array(info.image, np.uint8)
        # TODO ???????????????????????????bucket?????????????????????????????
        image, original_size, crop_ltrb = trim_and_resize_if_required(
            random_crop, image, info.bucket_reso, info.resized_size, resize_interpolation=info.resize_interpolation
        )

        original_sizes.append(original_size)
        crop_ltrbs.append(crop_ltrb)

        if use_alpha_mask:
            if image.shape[2] == 4:
                alpha_mask = image[:, :, 3]  # [H,W]
                alpha_mask = alpha_mask.astype(np.float32) / 255.0
                alpha_mask = torch.FloatTensor(alpha_mask)  # [H,W]
            else:
                alpha_mask = torch.ones_like(image[:, :, 0], dtype=torch.float32)  # [H,W]
        else:
            alpha_mask = None
        alpha_masks.append(alpha_mask)

        image = image[:, :, :3]  # remove alpha channel if exists
        image = IMAGE_TRANSFORMS(image)
        images.append(image)

    img_tensor = torch.stack(images, dim=0)
    return img_tensor, alpha_masks, original_sizes, crop_ltrbs


def cache_batch_latents(
    vae: AutoencoderKL, cache_to_disk: bool, image_infos: List[ImageInfo], flip_aug: bool, use_alpha_mask: bool, random_crop: bool
) -> None:
    r"""
    requires image_infos to have: absolute_path, bucket_reso, resized_size, latents_npz
    optionally requires image_infos to have: image
    if cache_to_disk is True, set info.latents_npz
        flipped latents is also saved if flip_aug is True
    if cache_to_disk is False, set info.latents
        latents_flipped is also set if flip_aug is True
    latents_original_size and latents_crop_ltrb are also set
    """
    images = []
    alpha_masks: List[np.ndarray] = []
    for info in image_infos:
        image = load_image(info.absolute_path, use_alpha_mask) if info.image is None else np.array(info.image, np.uint8)
        # TODO ???????????????????????????bucket?????????????????????????????
        image, original_size, crop_ltrb = trim_and_resize_if_required(
            random_crop, image, info.bucket_reso, info.resized_size, resize_interpolation=info.resize_interpolation
        )

        info.latents_original_size = original_size
        info.latents_crop_ltrb = crop_ltrb

        if use_alpha_mask:
            if image.shape[2] == 4:
                alpha_mask = image[:, :, 3]  # [H,W]
                alpha_mask = alpha_mask.astype(np.float32) / 255.0
                alpha_mask = torch.FloatTensor(alpha_mask)  # [H,W]
            else:
                alpha_mask = torch.ones_like(image[:, :, 0], dtype=torch.float32)  # [H,W]
        else:
            alpha_mask = None
        alpha_masks.append(alpha_mask)

        image = image[:, :, :3]  # remove alpha channel if exists
        image = IMAGE_TRANSFORMS(image)
        images.append(image)

    img_tensors = torch.stack(images, dim=0)
    img_tensors = img_tensors.to(device=vae.device, dtype=vae.dtype)

    with torch.no_grad():
        latents = vae.encode(img_tensors).latent_dist.sample().to("cpu")

    if flip_aug:
        img_tensors = torch.flip(img_tensors, dims=[3])
        with torch.no_grad():
            flipped_latents = vae.encode(img_tensors).latent_dist.sample().to("cpu")
    else:
        flipped_latents = [None] * len(latents)

    for info, latent, flipped_latent, alpha_mask in zip(image_infos, latents, flipped_latents,op_ltrb = trim_and_resize_if_required(
                        subset.random_crop,
                        img,
                        image_info.bucket_reso,
                        image_info.resized_size,
                        resize_interpolation=image_info.resize_interpolation,
                    )
                else:
                    if face_cx > 0:  # ???????
                        img = self.crop_target(subset, img, face_cx, face_cy, face_w, face_h)
                    elif im_h > self.height or im_w > self.width:
                        assert (
                            subset.random_crop
                        ), f"image too large, but cropping and bucketing are disabled / ???????????face_crop_aug_range?random_crop????bucket??????????: {image_info.absolute_path}"
                        if im_h > self.height:
                            p = random.randint(0, im_h - self.height)
                            img = img[p : p + self.height]
                        if im_w > self.width:
                            p = random.randint(0, im_w - self.width)
                            img = img[:, p : p + self.width]

                    im_h, im_w = img.shape[0:2]
                    assert (
                        im_h == self.height and im_w == self.width
                    ), f"image size is small / ?????????????: {image_info.absolute_path}"

                    original_size = [im_w, im_h]
                    crop_ltrb = (0, 0, 0, 0)

                # augmentation
                aug = self.aug_helper.get_augmentor(subset.color_aug)
                if aug is not None:
                    # augment RGB channels only
                    img_rgb = img[:, :, :3]
                    img_rgb = aug(image=img_rgb)["image"]
                    img[:, :, :3] = img_rgb

                if flipped:
                    img = img[:, ::-1, :].copy()  # copy to avoid negative stride problem

                if subset.alpha_mask:
                    if img.shape[2] == 4:
                        alpha_mask = img[:, :, 3]  # [H,W]
                        alpha_mask = alpha_mask.astype(np.float32) / 255.0  # 0.0~1.0
                        alpha_mask = torch.FloatTensor(alpha_mask)
                    else:
                        alpha_mask = torch.ones((img.shape[0], img.shape[1]), dtype=torch.float32)
                else:
                    alpha_mask = None

                img = img[:, :, :3]  # remove alpha channel

                latents = None
                image = self.image_transforms(img)  # -1.0~1.0?torch.Tensor???
                del img

            images.append(image)
            latents_list.append(latents)
            alpha_mask_list.append(alpha_mask)

            target_size = (image.shape[2], image.shape[1]) if image is not None else (latents.shape[2] * 8, latents.shape[1] * 8)

            if not flipped:
                crop_left_top = (crop_ltrb[0], crop_ltrb[1])
            else:
                # crop_ltrb[2] is right, so target_size[0] - crop_ltrb[2] is left in flipped image
                crop_left_top = (target_size[0] - crop_ltrb[2], crop_ltrb[1])

            original_sizes_hw.append((int(original_size[1]), int(original_size[0])))
            crop_top_lefts.append((int(crop_left_top[1]), int(crop_left_top[0])))
            target_sizes_hw.append((int(target_size[1]), int(target_size[0])))
            flippeds.append(flipped)

            # caption?text encoder output?????
            caption = image_info.caption  # default

            tokenization_required = (
                self.text_encoder_output_caching_strategy is None or self.text_encoder_output_caching_strategy.is_partial
            )
            text_encoder_outputs = None
            input_ids = None

            if image_info.text_encoder_outputs is not None:
                # cached
                text_encoder_outputs = image_info.text_encoder_outputs
            elif image_info.text_encoder_outputs_npz is not None:
                # on disk
                text_encoder_outputs = self.text_encoder_output_caching_strategy.load_outputs_npz(
                    image_info.text_encoder_outputs_npz
                )
            else:
                tokenization_required = True
            text_encoder_outputs_list.append(text_encoder_outputs)

            if tokenization_required:
                caption = self.process_caption(subset, image_info.caption)
                input_ids = [ids[0] for ids in self.tokenize_strategy.tokenize(caption)]  # remove batch dimension
                # if self.XTI_layers:
                #     caption_layer = []
                #     for layer in self.XTI_layers:
                #         token_strings_from = " ".join(self.token_strings)
                #         token_strings_to = " ".join([f"{x}_{layer}" for x in self.token_strings])
                #         caption_ = caption.replace(token_strings_from, token_strings_to)
                #         caption_layer.append(caption_)
                #     captions.append(caption_layer)
                # else:
                #     captions.append(caption)

                # if not self.token_padding_disabled:  # this option might be omitted in future
                #     # TODO get_input_ids must support SD3
                #     if self.XTI_layers:
                #         token_caption = self.get_input_ids(caption_layer, self.tokenizers[0])
                #     else:
                #         token_caption = self.get_input_ids(caption, self.tokenizers[0])
                #     input_ids_list.append(token_caption)

                #     if len(self.tokenizers) > 1:
                #         if self.XTI_layers:
                #             token_caption2 = self.get_input_ids(caption_layer, self.tokenizers[1])
                #         else:
                #             token_caption2 = self.get_input_ids(caption, self.tokenizers[1])
                #         input_ids2_list.append(token_caption2)

            input_ids_list.append(input_ids)
            captions.append(caption)

        def none_or_stack_elements(tensors_list, converter):
            # [[clip_l, clip_g, t5xxl], [clip_l, clip_g, t5xxl], ...] -> [torch.stack(clip_l), torch.stack(clip_g), torch.stack(t5xxl)]
            if len(tensors_list) == 0 or tensors_list[0] == None or len(tensors_list[0]) == 0 or tensors_list[0][0] is None:
                return None

            # old implementation without padding: all elements must have same length
            # return [torch.stack([converter(x[i]) for x in tensors_list]) for i in range(len(tensors_list[0]))]

            # new implementation with padding support
            result = []
            for i in range(len(tensors_list[0])):
                tensors = [x[i] for x in tensors_list]
                if tensors[0].ndim == 0:
                    # scalar value: e.g. ocr mask
                    result.append(torch.stack([converter(x[i]) for x in tensors_list]))
                    continue

                min_len = min([len(x) for x in tensors])
                max_len = max([len(x) for x in tensors])

                if min_len == max_len:
                    # no padding
                    result.append(torch.stack([converter(x) for x in tensors]))
                else:
                    # padding
                    tensors = [converter(x) for x in tensors]
                    if tensors[0].ndim == 1:
                        # input_ids or mask
                        result.append(torch.stack([(torch.nn.functional.pad(x, (0, max_len - x.shape[0]))) for x in tensors]))
                    else:
                        # text encoder outputs
                        result.append(torch.stack([(torch.nn.functional.pad(x, (0, 0, 0, max_len - x.shape[0]))) for x in tensors]))
            return result

        # set example
        example = {}
        example["custom_attributes"] = custom_attributes  # may be list of empty dict
        example["loss_weights"] = torch.FloatTensor(loss_weights)
        example["text_encoder_outputs_list"] = none_or_stack_elements(text_encoder_outputs_list, torch.FloatTensor)
        example["input_ids_list"] = none_or_stack_elements(input_ids_list, lambda x: x)

        # if one of alpha_masks is not None, we need to replace None with ones
        none_or_not = [x is None for x in alpha_mask_list]
        if all(none_or_not):
            example["alpha_masks"] = None
        elif any(none_or_not):
            for i in range(len(alpha_mask_list)):
                if alpha_mask_list[i] is None:
                    if images[i] is not None:
                        alpha_mask_list[i] = torch.ones((images[i].shape[1], images[i].shape[2]), dtype=torch.float32)
                    else:
                        alpha_mask_list[i] = torch.ones(
                            (latents_list[i].shape[1] * 8, latents_list[i].shape[2] * 8), dtype=torch.float32
                        )
            example["alpha_masks"] = torch.stack(alpha_mask_list)
        else:
            example["alpha_masks"] = torch.stack(alpha_mask_list)

        if images[0] is not None:
            images = torch.stack(images)
            images = images.to(memory_format=torch.contiguous_format).float()
        else:
            images = None
        example["images"] = images

        example["latents"] = torch.stack(latents_list) if latents_list[0] is not None else None
        example["captions"] = captions

        example["original_sizes_hw"] = torch.stack([torch.LongTensor(x) for x in original_sizes_hw])
        example["crop_top_lefts"] = torch.stack([torch.LongTensor(x) for x in crop_top_lefts])
        example["target_sizes_hw"] = torch.stack([torch.LongTensor(x) for x in target_sizes_hw])
        example["flippeds"] = flippeds

        example["network_multipliers"] = torch.FloatTensor([self.network_multiplier] * len(captions))

        if self.debug_dataset:
            example["image_keys"] = bucket[image_index : image_index + self.batch_size]
        return example

    def get_item_for_caching(self, bucket, bucket_batch_size, image_index):
        captions = []
        images = []
        input_ids1_list = []
        input_ids2_list = []
        absolute_paths = []
        resized_sizes = []
        bucket_reso = None
        flip_aug = None
        alpha_mask = None
        random_crop = None

        for image_key in bucket[image_index : image_index + bucket_batch_size]:
            image_info = self.image_data[image_key]
            subset = self.image_to_subset[image_key]

            if flip_aug is None:
                flip_aug = subset.flip_aug
                alpha_mask = subset.alpha_mask
                random_crop = subset.random_crop
                bucket_reso = image_info.bucket_reso
            else:
                # TODO ??????????????????????
                assert flip_aug == subset.flip_aug, "flip_aug must be same in a batch"
                assert alpha_mask == subset.alpha_mask, "alpha_mask must be same in a batch"
                assert random_crop == subset.random_crop, "random_crop must be same in a batch"
                assert bucket_reso == image_info.bucket_reso, "bucket_reso must be same in a batch"

            caption = image_info.caption  # TODO cache some patterns of dropping, shuffling, etc.

            if self.caching_mode == "latents":
                image = load_image(image_info.absolute_path)
            else:
                image = None

            if self.caching_mode == "text":
                input_ids1 = self.get_input_ids(caption, self.tokenizers[0])
                input_ids2 = self.get_input_ids(caption, self.tokenizers[1])
            else:
                input_ids1 = None
                input_ids2 = None

            captions.append(caption)
            images.append(image)
            input_ids1_list.append(input_ids1)
            input_ids2_list.append(input_ids2)
            absolute_paths.append(image_info.absolute_path)
            resized_sizes.append(image_info.resized_size)

        example = {}

        if images[0] is None:
            images = None
        example["images"] = images

        example["captions"] = captions
        example["input_ids1_list"] = input_ids1_list
        example["input_ids2_list"] = input_ids2_list
        example["absolute_paths"] = absolute_paths
        example["resized_sizes"] = resized_sizes
        example["flip_aug"] = flip_aug
        example["alpha_mask"] = alpha_mask
        example["random_crop"] = random_crop
        example["bucket_reso"] = bucket_reso
        return example


class DreamBoothDataset(BaseDataset):
    IMAGE_INFO_CACHE_FILE = "metadata_cache.json"

    # The is_training_dataset defines the type of dataset, training or validation
    # if is_training_dataset is True -> training dataset
    # if is_training_dataset is False -> validation dataset
    def __init__(
        self,
        subsets: Sequence[DreamBoothSubset],
        is_training_dataset: bool,
        batch_size: int,
        resolution,
        network_multiplier: float,
        enable_bucket: bool,
        min_bucket_reso: int,
        max_bucket_reso: int,
        bucket_reso_steps: int,
        bucket_no_upscale: bool,
        prior_loss_weight: float,
        debug_dataset: bool,
        validation_split: float,
        validation_seed: Optional[int],
        resize_interpolation: Optional[str],
    ) -> None:
        super().__init__(resolution, network_multiplier, debug_dataset, resize_interpolation)

        assert resolution is not None, f"resolution is required / resolution????????????"

        self.batch_size = batch_size
        self.size = min(self.width, self.height)  # ????
        self.prior_loss_weight = prior_loss_weight
        self.latents_cache = None
        self.is_training_dataset = is_training_dataset
        self.validation_seed = validation_seed
        self.validation_split = validation_split

        self.enable_bucket = enable_bucket
        if self.enable_bucket:
            min_bucket_reso, max_bucket_reso = self.adjust_min_max_bucket_reso_by_steps(
                resolution, min_bucket_reso, max_bucket_reso, bucket_reso_steps
            )
            self.min_bucket_reso = min_bucket_reso
            self.max_bucket_reso = max_bucket_reso
            self.bucket_reso_steps = bucket_reso_steps
            self.bucket_no_upscale = bucket_no_upscale
        else:
            self.min_bucket_reso = None
            self.max_bucket_reso = None
            self.bucket_reso_steps = None  # ??????????
            self.bucket_no_upscale = False

        def read_caption(img_path, caption_extension, enable_wildcard):
            # caption???????????
            base_name = os.path.splitext(img_path)[0]
            base_name_face_det = base_name
            tokens = base_name.split("_")
            if len(tokens) >= 5:
                base_name_face_det = "_".join(tokens[:-4])
            cap_paths = [base_name + caption_extension, base_name_face_det + caption_extension]

            caption = None
            for cap_path in cap_paths:
                if os.path.isfile(cap_path):
                    with open(cap_path, "rt", encoding="utf-8") as f:
                        try:
                            lines = f.readlines()
                        except UnicodeDecodeError as e:
                            logger.error(f"illegal char in file (not UTF-8) / ?????UTF-8??????????: {cap_path}")
                            raise e
                        assert len(lines) > 0, f"caption file is empty / ??????????????: {cap_path}"
                        if enable_wildcard:
                            caption = "\n".join([line.strip() for line in lines if line.strip() != ""])  # ???????????
                        else:
                            caption = lines[0].strip()
                    break
            return caption

        def load_dreambooth_dir(subset: DreamBoothSubset):
            if not os.path.isdir(subset.image_dir):
                logger.warning(f"not directory: {subset.image_dir}")
                return [], [], []

            info_cache_file = os.path.join(subset.image_dir, self.IMAGE_INFO_CACHE_FILE)
            use_cached_info_for_subset = subset.cache_info
            if use_cached_info_for_subset:
                logger.info(
                    f"using cached image info for this subset / ??????????????????????????: {info_cache_file}"
                )
                if not os.path.isfile(info_cache_file):
                    logger.warning(
                        f"image info file not found. You can ignore this warning if this is the first time to use this subset"
                        + " / ????????????????????????????????????????: {metadata_file}"
                    )
                    use_cached_info_for_subset = False

            if use_cached_info_for_subset:
                # json: {`img_path`:{"caption": "caption...", "resolution": [width, height]}, ...}
                with open(info_cache_file, "r", encoding="utf-8") as f:
                    metas = json.load(f)
                img_paths = list(metas.keys())
                sizes: List[Optional[Tuple[int, int]]] = [meta["resolution"] for meta in metas.values()]

                # we may need to check image size and existence of image files, but it takes time, so user should check it before training
            else:
                img_paths = glob_images(subset.image_dir, "*")
                sizes: List[Optional[Tuple[int, int]]] = [None] * len(img_paths)

                # new caching: get image size from cache files
                strategy = LatentsCachingStrategy.get_strategy()
                if strategy is not None:
                    logger.info("get image size from name of cache files")

                    # make image path to npz path mapping
                    npz_paths = glob.glob(os.path.join(subset.image_dir, "*" + strategy.cache_suffix))
                    npz_paths.sort(
                        key=lambda item: item.rsplit("_", maxsplit=2)[0]
                    )  # sort by name excluding resolution and cache_suffix
                    npz_path_index = 0

                    size_set_count = 0
                    for i, img_path in enumerate(tqdm(img_paths)):
                        l = len(os.path.splitext(img_path)[0])  # remove extension
                        found = False
                        while npz_path_index < len(npz_paths):  # until found or end of npz_paths
                            # npz_paths are sorted, so if npz_path > img_path, img_path is not found
                            if npz_paths[npz_path_index][:l] > img_path[:l]:
                                break
                            if npz_paths[npz_path_index][:l] == img_path[:l]:  # found
                                found = True
                                break
                            npz_path_index += 1  # next npz_path

                        if found:
                            w, h = strategy.get_image_size_from_disk_cache_path(img_path, npz_paths[npz_path_index])
                        else:
                            w, h = None, None

                        if w is not None and h is not None:
                            sizes[i] = (w, h)
                            size_set_count += 1
                    logger.info(f"set image size from cache files: {size_set_count}/{len(img_paths)}")

            # We want to create a training and validation split. This should be improved in the future
            # to allow a clearer distinction between training and validation. This can be seen as a
            # short-term solution to limit what is necessary to implement validation datasets
            #
            # We split the dataset for the subset based on if we are doing a validation split
            # The self.is_training_dataset defines the type of dataset, training or validation
            # if self.is_training_dataset is True -> training dataset
            # if self.is_training_dataset is False -> validation dataset
            if self.validation_split > 0.0:
                # For regularization images we do not want to split this dataset.
                if subset.is_reg is True:
                    # Skip any validation dataset for regularization images
                    if self.is_training_dataset is False:
                        img_paths = []
                        sizes = []
                    # Otherwise the img_paths remain as original img_paths and no split
                    # required for training images dataset of regularization images
                else:
                    img_paths, sizes = split_train_val(
                        img_paths, sizes, self.is_training_dataset, self.validation_split, self.validation_seed
                    )

            logger.info(f"found directory {subset.image_dir} contains {len(img_paths)} image files")

            if use_cached_info_for_subset:
                captions = [meta["caption"] for meta in metas.values()]
                missing_captions = [img_path for img_path, caption in zip(img_paths, captions) if caption is None or caption == ""]
            else:
                # ???????????????????????????????
                captions = []
                missing_captions = []
                for img_path in tqdm(img_paths, desc="read caption"):
                    cap_for_img = read_caption(img_path, subset.caption_extension, subset.enable_wildcard)
                    if cap_for_img is None and subset.class_tokens is None:
                        logger.warning(
                            f"neither caption file nor class tokens are found. use empty caption for {img_path} / ???????????class token??????????????????????????: {img_path}"
                        )
                        captions.append("")
                        missing_captions.append(img_path)
                    else:
                        if cap_for_img is None:
                            captions.append(subset.class_tokens)
                            missing_captions.append(img_path)
                        else:
                            captions.append(cap_for_img)

            self.set_tag_frequency(os.path.basename(subset.image_dir), captions)  # ???????

            if missing_captions:
                number_of_missing_captions = len(missing_captions)
                number_of_missing_captions_to_show = 5
                remaining_missing_captions = number_of_missing_captions - number_of_missing_captions_to_show

                logger.warning(
                    f"No caption file found for {number_of_missing_captions} images. Training will continue without captions for these images. If class token exists, it will be used. / {number_of_missing_captions}????????????????????????????????????????????????????????class token????????????????"
                )
                for i, missing_caption in enumerate(missing_captions):
                    if i >= number_of_missing_captions_to_show:
                        logger.warning(missing_caption + f"... and {remaining_missing_captions} more")
                        break
                    logger.warning(missing_caption)

            if not use_cached_info_for_subset and subset.cache_info:
                logger.info(f"cache image info for / ????????????? : {info_cache_file}")
                sizes = [self.get_image_size(img_path) for img_path in tqdm(img_paths, desc="get image size")]
                matas = {}
                for img_path, caption, size in zip(img_paths, captions, sizes):
                    matas[img_path] = {"caption": caption, "resolution": list(size)}
                with open(info_cache_file, "w", encoding="utf-8") as f:
                    json.dump(matas, f, ensure_ascii=False, indent=2)
                logger.info(f"cache image info done for / ??????????? : {info_cache_file}")

            # if sizes are not set, image size will be read in make_buckets
            return img_paths, captions, sizes

        logger.info("prepare images.")
        num_train_images = 0
        num_reg_images = 0
        reg_infos: List[Tuple[ImageInfo, DreamBoothSubset]] = []
        for subset in subsets:
            num_repeats = subset.num_repeats if self.is_training_dataset else 1
            if num_repeats < 1:
                logger.warning(
                    f"ignore subset with image_dir='{subset.image_dir}': num_repeats is less than 1 / num_repeats?1????????????????????: {num_repeats}"
                )
                continue

            if subset in self.subsets:
                logger.warning(
                    f"ignore duplicated subset with image_dir='{subset.image_dir}': use the first one / ????????????????????????????????????"
                )
                continue

            img_paths, captions, sizes = load_dreambooth_dir(subset)
            if len(img_paths) < 1:
                logger.warning(
                    f"ignore subset with image_dir='{subset.image_dir}': no images found / ??????????????????????"
                )
                continue

            if subset.is_reg:
                num_reg_images += num_repeats * len(img_paths)
            else:
                num_train_images += num_repeats * len(img_paths)

            for img_path, caption, size in zip(img_paths, captions, sizes):
                info = ImageInfo(img_path, num_repeats, caption, subset.is_reg, img_path)
                info.resize_interpolation = (
                    subset.resize_interpolation if subset.resize_interpolation is not None else self.resize_interpolation
                )
                if size is not None:
                    info.image_size = size
                if subset.is_reg:
                    reg_infos.append((info, subset))
                else:
                    self.register_image(info, subset)

            subset.img_count = len(img_paths)
            self.subsets.append(subset)

        images_split_name = "train" if self.is_training_dataset else "validation"
        logger.info(f"{num_train_images} {images_split_name} images with repeats.")

        self.num_train_images = num_train_images

        logger.info(f"{num_reg_images} reg images with repeats.")
        if num_train_images < num_reg_images:
            logger.warning("some of reg images are not used / ???????????????????????????????")

        if num_reg_images == 0:
            logger.warning("no regularization images / ????????????????")
        else:
            # num_repeats???????????????????????????
            n = 0
            first_loop = True
            while n < num_train_images:
                for info, subset in reg_infos:
                    if first_loop:
                        self.register_image(info, subset)
                        n += info.num_repeats
                    else:
                        info.num_repeats += 1  # rewrite registered info
                        n += 1
                    if n >= num_train_images:
                        break
                first_loop = False

        self.num_reg_images = num_reg_images


class FineTuningDataset(BaseDataset):
    def __init__(
        self,
        subsets: Sequence[FineTuningSubset],
        batch_size: int,
        resolution,
        network_multiplier: float,
        enable_bucket: bool,
        min_bucket_reso: int,
        max_bucket_reso: int,
        bucket_reso_steps: int,
        bucket_no_upscale: bool,
        debug_dataset: bool,
        validation_seed: int,
        validation_split: float,
        resize_interpolation: Optional[str],
    ) -> None:
        super().__init__(resolution, network_multiplier, debug_dataset, resize_interpolation)

        self.batch_size = batch_size
        self.size = min(self.width, self.height)  # ????
        self.latents_cache = None

        self.enable_bucket = enable_bucket
        if self.enable_bucket:
            min_bucket_reso, max_bucket_reso = self.adjust_min_max_bucket_reso_by_steps(
                resolution, min_bucket_reso, max_bucket_reso, bucket_reso_steps
            )
            self.min_bucket_reso = min_bucket_reso
            self.max_bucket_reso = max_bucket_reso
            self.bucket_reso_steps = bucket_reso_steps
            self.bucket_no_upscale = bucket_no_upscale
        else:
            self.min_bucket_reso = None
            self.max_bucket_reso = None
            self.bucket_reso_steps = None  # ??????????
            self.bucket_no_upscale = False

        self.num_train_images = 0
        self.num_reg_images = 0

        for subset in subsets:
            if subset.num_repeats < 1:
                logger.warning(
                    f"ignore subset with metadata_file='{subset.metadata_file}': num_repeats is less than 1 / num_repeats?1????????????????????: {subset.num_repeats}"
                )
                continue

            if subset in self.subsets:
                logger.warning(
                    f"ignore duplicated subset with metadata_file='{subset.metadata_file}': use the first one / ????????????????????????????????????"
                )
                continue

            # ??????????
            if os.path.exists(subset.metadata_file):
                if subset.metadata_file.endswith(".jsonl"):
                    logger.info(f"loading existing JSOL metadata: {subset.metadata_file}")
                    # optional JSONL format
                    # {"image_path": "/path/to/image1.jpg", "caption": "A caption for image1", "image_size": [width, height]}
                    metadata = {}
                    with open(subset.metadata_file, "rt", encoding="utf-8") as f:
                        for line in f:
                            line_md = json.loads(line)
                            image_md = {"caption": line_md.get("caption", "")}
                            if "image_size" in line_md:
                                image_md["image_size"] = line_md["image_size"]
                            if "tags" in line_md:
                                image_md["tags"] = line_md["tags"]
                            metadata[line_md["image_path"]] = image_md
                else:
                    # standard JSON format
                    logger.info(f"loading existing metadata: {subset.metadata_file}")
                    with open(subset.metadata_file, "rt", encoding="utf-8") as f:
                        metadata = json.load(f)
            else:
                raise ValueError(f"no metadata / ???????????????: {subset.metadata_file}")

            if len(metadata) < 1:
                logger.warning(
                    f"ignore subset with '{subset.metadata_file}': no image entries found / ?????????????????????????????"
                )
                continue

            # Add full path for image
            image_dirs = set()
            if subset.image_dir is not None:
                image_dirs.add(subset.image_dir)
            for image_key in metadata.keys():
                if not os.path.isabs(image_key):
                    assert (
                        subset.image_dir is not None
                    ), f"image_dir is required when image paths are relative / ?????????????image_dir????????: {image_key}"
                    abs_path = os.path.join(subset.image_dir, image_key)
                else:
                    abs_path = image_key
                    image_dirs.add(os.path.dirname(abs_path))
                metadata[image_key]["abs_path"] = abs_path

            # Enumerate existing npz files
            strategy = LatentsCachingStrategy.get_strategy()
            npz_paths = []
            for image_dir in image_dirs:
                npz_paths.extend(glob.glob(os.path.join(image_dir, "*" + strategy.cache_suffix)))
            npz_paths = sorted(npz_paths, key=lambda item: len(os.path.basename(item)), reverse=True)  # longer paths first

            # Match image filename longer to shorter because some images share same prefix
            image_keys_sorted_by_length_desc = sorted(metadata.keys(), key=len, reverse=True)

            # Collect tags and sizes
            tags_list = []
            size_set_from_metadata = 0
            size_set_from_cache_filename = 0
            for image_key in image_keys_sorted_by_length_desc:
                img_md = metadata[image_key]
                caption = img_md.get("caption")
                tags = img_md.get("tags")
                image_size = img_md.get("image_size")
                abs_path = img_md.get("abs_path")

                # search npz if image_size is not given
                npz_path = None
                if image_size is None:
                    image_without_ext = os.path.splitext(image_key)[0]
                    for candidate in npz_paths:
                        if candidate.startswith(image_without_ext):
                            npz_path = candidate
                            break
                    if npz_path is not None:
                        npz_paths.remove(npz_path)  # remove to avoid matching same file (share prefix)
                        abs_path = npz_path

                if caption is None:
                    caption = ""

                if subset.enable_wildcard:
                    # tags must be single line (split by caption separator)
                    if tags is not None:
                        tags = tags.replace("\n", subset.caption_separator)

                    # add tags to each line of caption
                    if tags is not None:
                        caption = "\n".join(
                            [f"{line}{subset.caption_separator}{tags}" for line in caption.split("\n") if line.strip() != ""]
                        )
                        tags_list.append(tags)
                else:
                    # use as is
                    if tags is not None and len(tags) > 0:
                        if len(caption) > 0:
                            caption = caption + subset.caption_separator
                        caption = caption + tags
                        tags_list.append(tags)

                if caption is None:
                    caption = ""

                image_info = ImageInfo(image_key, subset.num_repeats, caption, False, abs_path)
                image_info.resize_interpolation = (
                    subset.resize_interpolation if subset.resize_interpolation is not None else self.resize_interpolation
                )

                if image_size is not None:
                    image_info.image_size = tuple(image_size)  # width, height
                    size_set_from_metadata += 1
                elif npz_path is not None:
                    # get image size from npz filename
                    w, h = strategy.get_image_size_from_disk_cache_path(abs_path, npz_path)
                    image_info.image_size = (w, h)
                    size_set_from_cache_filename += 1

                self.register_image(image_info, subset)

            if size_set_from_cache_filename > 0:
                logger.info(
                    f"set image size from cache files: {size_set_from_cache_filename}/{len(image_keys_sorted_by_length_desc)}"
                )
            if size_set_from_metadata > 0:
                logger.info(f"set image size from metadata: {size_set_from_metadata}/{len(image_keys_sorted_by_length_desc)}")
            self.num_train_images += len(metadata) * subset.num_repeats

            # TODO do not record tag freq when no tag
            self.set_tag_frequency(os.path.basename(subset.metadata_file), tags_list)
            subset.img_count = len(metadata)
            self.subsets.append(subset)


class ControlNetDataset(BaseDataset):
    def __init__(
        self,
        subsets: Sequence[ControlNetSubset],
        batch_size: int,
        resolution,
        network_multiplier: float,
        enable_bucket: bool,
        min_bucket_reso: int,
        max_bucket_reso: int,
        bucket_reso_steps: int,
        bucket_no_upscale: bool,
        debug_dataset: bool,
        validation_split: float,
        validation_seed: Optional[int],
        resize_interpolation: Optional[str] = None,
    ) -> None:
        super().__init__(resolution, network_multiplier, debug_dataset, resize_interpolation)

        db_subsets = []
        for subset in subsets:
            assert (
                not subset.random_crop
            ), "random_crop is not supported in ControlNetDataset / random_crop?ControlNetDataset?????????????"
            db_subset = DreamBoothSubset(
                subset.image_dir,
                False,
                None,
                subset.caption_extension,
                subset.cache_info,
                False,
                subset.num_repeats,
                subset.shuffle_caption,
                subset.caption_separator,
                subset.keep_tokens,
                subset.keep_tokens_separator,
                subset.secondary_separator,
                subset.enable_wildcard,
                subset.color_aug,
                subset.flip_aug,
                subset.face_crop_aug_range,
                subset.random_crop,
                subset.caption_dropout_rate,
                subset.caption_dropout_every_n_epochs,
                subset.caption_tag_dropout_rate,
                subset.caption_prefix,
                subset.caption_suffix,
                subset.token_warmup_min,
                subset.token_warmup_step,
                resize_interpolation=subset.resize_interpolation,
            )
            db_subsets.append(db_subset)

        self.dreambooth_dataset_delegate = DreamBoothDataset(
            db_subsets,
            True,
            batch_size,
            resolution,
            network_multiplier,
            enable_bucket,
            min_bucket_reso,
            max_bucket_reso,
            bucket_reso_steps,
            bucket_no_upscale,
            1.0,
            debug_dataset,
            validation_split,
            validation_seed,
            resize_interpolation,
        )

        # config_util???????????????????????????????
        self.image_data = self.dreambooth_dataset_delegate.image_data
        self.batch_size = batch_size
        self.num_train_images = self.dreambooth_dataset_delegate.num_train_images
        self.num_reg_images = self.dreambooth_dataset_delegate.num_reg_images
        self.validation_split = validation_split
        self.validation_seed = validation_seed
        self.resize_interpolation = resize_interpolation

        # assert all conditioning data exists
        missing_imgs = []
        cond_imgs_with_pair = set()
        for image_key, info in self.dreambooth_dataset_delegate.image_data.items():
            db_subset = self.dreambooth_dataset_delegate.image_to_subset[image_key]
            subset = None
            for s in subsets:
                if s.image_dir == db_subset.image_dir:
                    subset = s
                    break
            assert subset is not None, "internal error: subset not found"

            if not os.path.isdir(subset.conditioning_data_dir):
                logger.warning(f"not directory: {subset.conditioning_data_dir}")
                continue

            img_basename = os.path.splitext(os.path.basename(info.absolute_path))[0]
            ctrl_img_path = glob_images(subset.conditioning_data_dir, img_basename)
            if len(ctrl_img_path) < 1:
                missing_imgs.append(img_basename)
                continue
            ctrl_img_path = ctrl_img_path[0]
            ctrl_img_path = os.path.abspath(ctrl_img_path)  # normalize path

            info.cond_img_path = ctrl_img_path
            cond_imgs_with_pair.add(os.path.splitext(ctrl_img_path)[0])  # remove extension because Windows is case insensitive

        extra_imgs = []
        for subset in subsets:
            conditioning_img_paths = glob_images(subset.conditioning_data_dir, "*")
            conditioning_img_paths = [os.path.abspath(p) for p in conditioning_img_paths]  # normalize path
            extra_imgs.extend([p for p in conditioning_img_paths if os.path.splitext(p)[0] not in cond_imgs_with_pair])

        assert (
            len(missing_imgs) == 0
        ), f"missing conditioning data for {len(missing_imgs)} images / ????????????????: {missing_imgs}"
        assert (
            len(extra_imgs) == 0
        ), f"extra conditioning data for {len(extra_imgs)} images / ?????????????: {extra_imgs}"

        self.conditioning_image_transforms = IMAGE_TRANSFORMS

    def set_current_strategies(self):
        return self.dreambooth_dataset_delegate.set_current_strategies()

    def make_buckets(self):
        self.dreambooth_dataset_delegate.make_buckets()
        self.bucket_manager = self.dreambooth_dataset_delegate.bucket_manager
        self.buckets_indices = self.dreambooth_dataset_delegate.buckets_indices

    def cache_latents(self, vae, vae_batch_size=1, cache_to_disk=False, is_main_process=True):
        return self.dreambooth_dataset_delegate.cache_latents(vae, vae_batch_size, cache_to_disk, is_main_process)

    def new_cache_latents(self, model: Any, accelerator: Accelerator):
        return self.dreambooth_dataset_delegate.new_cache_latents(model, accelerator)

    def new_cache_text_encoder_outputs(self, models: List[Any], is_main_process: bool):
        return self.dreambooth_dataset_delegate.new_cache_text_encoder_outputs(models, is_main_process)

    def __len__(self):
        return self.dreambooth_dataset_delegate.__len__()

    def __getitem__(self, index):
        example = self.dreambooth_dataset_delegate[index]

        bucket = self.dreambooth_dataset_delegate.bucket_manager.buckets[
            self.dreambooth_dataset_delegate.buckets_indices[index].bucket_index
        ]
        bucket_batch_size = self.dreambooth_dataset_delegate.buckets_indices[index].bucket_batch_size
        image_index = self.dreambooth_dataset_delegate.buckets_indices[index].batch_index * bucket_batch_size

        conditioning_images = []

        for i, image_key in enumerate(bucket[image_index : image_index + bucket_batch_size]):
            image_info = self.dreambooth_dataset_delegate.image_data[image_key]

            target_size_hw = example["target_sizes_hw"][i]
            original_size_hw = example["original_sizes_hw"][i]
            crop_top_left = example["crop_top_lefts"][i]
            flipped = example["flippeds"][i]
            cond_img = load_image(image_info.cond_img_path)

            if self.dreambooth_dataset_delegate.enable_bucket:
                assert (
                    cond_img.shape[0] == original_size_hw[0] and cond_img.shape[1] == original_size_hw[1]
                ), f"size of conditioning image is not match / ???????????: {image_info.absolute_path}"

                cond_img = resize_image(
                    cond_img,
                    original_size_hw[1],
                    original_size_hw[0],
                    target_size_hw[1],
                    target_size_hw[0],
                    self.resize_interpolation,
                )

                # TODO support random crop
                # ??????????crop?random????????
                h, w = target_size_hw
                ct = (cond_img.shape[0] - h) // 2
                cl = (cond_img.shape[1] - w) // 2
                cond_img = cond_img[ct : ct + h, cl : cl + w]
            else:
                # assert (
                #     cond_img.shape[0] == self.height and cond_img.shape[1] == self.width
                # ), f"image size is small / ?????????????: {image_info.absolute_path}"
                # resize to target
                if cond_img.shape[0] != target_size_hw[0] or cond_img.shape[1] != target_size_hw[1]:
                    cond_img = resize_image(
                        cond_img,
                        cond_img.shape[0],
                        cond_img.shape[1],
                        target_size_hw[1],
                        target_size_hw[0],
                        self.resize_interpolation,
                    )

            if flipped:
                cond_img = cond_img[:, ::-1, :].copy()  # copy to avoid negative stride

            cond_img = self.conditioning_image_transforms(cond_img)
            conditioning_images.append(cond_img)

        example["conditioning_images"] = torch.stack(conditioning_images).to(memory_format=torch.contiguous_format).float()

        return example


# behave as Dataset mock
class DatasetGroup(torch.utils.data.ConcatDataset):
    def __init__(self, datasets: Sequence[Union[DreamBoothDataset, FineTuningDataset]]):
        self.datasets: List[Union[DreamBoothDataset, FineTuningDataset]]

        super().__init__(datasets)

        self.image_data = {}
        self.num_train_images = 0
        self.num_reg_images = 0

        # simply concat together
        # TODO: handling image_data key duplication among dataset
        #   In practical, this is not the big issue because image_data is accessed from outside of dataset only for debug_dataset.
        for dataset in datasets:
            self.image_data.update(dataset.image_data)
            self.num_train_images += dataset.num_train_images
            self.num_reg_images += dataset.num_reg_images

    def add_replacement(self, str_from, str_to):
        for dataset in self.datasets:
            dataset.add_replacement(str_from, str_to)

    # def make_buckets(self):
    #   for dataset in self.datasets:
    #     dataset.make_buckets()

    def set_text_encoder_output_caching_strategy(self, strategy: TextEncoderOutputsCachingStrategy):
        """
        DataLoader is run in multiple processes, so we need to set the strategy manually.
        """
        for dataset in self.datasets:
            dataset.set_text_encoder_output_caching_strategy(strategy)

    def enable_XTI(self, *args, **kwargs):
        for dataset in self.datasets:
            dataset.enable_XTI(*args, **kwargs)

    def cache_latents(self, vae, vae_batch_size=1, cache_to_disk=False, is_main_process=True, file_suffix=".npz"):
        for i, dataset in enumerate(self.datasets):
            logger.info(f"[Dataset {i}]")
            dataset.cache_latents(vae, vae_batch_size, cache_to_disk, is_main_process, file_suffix)

    def new_cache_latents(self, model: Any, accelerator: Accelerator):
        for i, dataset in enumerate(self.datasets):
            logger.info(f"[Dataset {i}]")
            dataset.new_cache_latents(model, accelerator)
        accelerator.wait_for_everyone()

    def cache_text_encoder_outputs(
        self, tokenizers, text_encoders, device, weight_dtype, cache_to_disk=False, is_main_process=True
    ):
        for i, dataset in enumerate(self.datasets):
            logger.info(f"[Dataset {i}]")
            dataset.cache_text_encoder_outputs(tokenizers, text_encoders, device, weight_dtype, cache_to_disk, is_main_process)

    def cache_text_encoder_outputs_sd3(
        self, tokenizer, text_encoders, device, output_dtype, te_dtypes, cache_to_disk=False, is_main_process=True, batch_size=None
    ):
        for i, dataset in enumerate(self.datasets):
            logger.info(f"[Dataset {i}]")
            dataset.cache_text_encoder_outputs_sd3(
                tokenizer, text_encoders, device, output_dtype, te_dtypes, cache_to_disk, is_main_process, batch_size
            )

    def new_cache_text_encoder_outputs(self, models: List[Any], accelerator: Accelerator):
        for i, dataset in enumerate(self.datasets):
            logger.info(f"[Dataset {i}]")
            dataset.new_cache_text_encoder_outputs(models, accelerator)
        accelerator.wait_for_everyone()

    def set_caching_mode(self, caching_mode):
        for dataset in self.datasets:
            dataset.set_caching_mode(caching_mode)

    def verify_bucket_reso_steps(self, min_steps: int):
        for dataset in self.datasets:
            dataset.verify_bucket_reso_steps(min_steps)

    def get_resolutions(self) -> List[Tuple[int, int]]:
        return [(dataset.width, dataset.height) for dataset in self.datasets]

    def is_latent_cacheable(self) -> bool:
        return all([dataset.is_latent_cacheable() for dataset in self.datasets])

    def is_text_encoder_output_cacheable(self) -> bool:
        return all([dataset.is_text_encoder_output_cacheable() for dataset in self.datasets])

    def set_current_strategies(self):
        for dataset in self.datasets:
            dataset.set_current_strategies()

    def set_current_epoch(self, epoch):
        for dataset in self.datasets:
            dataset.set_current_epoch(epoch)

    def set_current_step(self, step):
        for dataset in self.datasets:
            dataset.set_current_step(step)

    def set_max_train_steps(self, max_train_steps):
        for dataset in self.datasets:
            dataset.set_max_train_steps(max_train_steps)

    def disable_token_padding(self):
        for dataset in self.datasets:
            dataset.disable_token_padding()


def is_disk_cached_latents_is_expected(reso, npz_path: str, flip_aug: bool, alpha_mask: bool):
    expected_latents_size = (reso[1] // 8, reso[0] // 8)  # bucket_reso?WxH?????

    if not os.path.exists(npz_path):
        return False

    try:
        npz = np.load(npz_path)
        if "latents" not in npz or "original_size" not in npz or "crop_ltrb" not in npz:  # old ver?
            return False
        if npz["latents"].shape[1:3] != expected_latents_size:
            return False

        if flip_aug:
            if "latents_flipped" not in npz:
                return False
            if npz["latents_flipped"].shape[1:3] != expected_latents_size:
                return False

        if alpha_mask:
            if "alpha_mask" not in npz:
                return False
            if (npz["alpha_mask"].shape[1], npz["alpha_mask"].shape[0]) != reso:  # HxW => WxH != reso
                return False
        else:
            if "alpha_mask" in npz:
                return False
    except Exception as e:
        logger.error(f"Error loading file: {npz_path}")
        raise e

    return True


# ?????latents_tensor, (original_size width, original_size height), (crop left, crop top)
# TODO update to use CachingStrategy
# def load_latents_from_disk(
#     npz_path,
# ) -> Tuple[Optional[np.ndarray], Optional[List[int]], Optional[List[int]], Optional[np.ndarray], Optional[np.ndarray]]:
#     npz = np.load(npz_path)
#     if "latents" not in npz:
#         raise ValueError(f"error: npz is old format. please re-generate {npz_path}")

#     latents = npz["latents"]
#     original_size = npz["original_size"].tolist()
#     crop_ltrb = npz["crop_ltrb"].tolist()
#     flipped_latents = npz["latents_flipped"] if "latents_flipped" in npz else None
#     alpha_mask = npz["alpha_mask"] if "alpha_mask" in npz else None
#     return latents, original_size, crop_ltrb, flipped_latents, alpha_mask


# def save_latents_to_disk(npz_path, latents_tensor, original_size, crop_ltrb, flipped_latents_tensor=None, alpha_mask=None):
#     kwargs = {}
#     if flipped_latents_tensor is not None:
#         kwargs["latents_flipped"] = flipped_latents_tensor.float().cpu().numpy()
#     if alpha_mask is not None:
#         kwargs["alpha_mask"] = alpha_mask.float().cpu().numpy()
#     np.savez(
#         npz_path,
#         latents=latents_tensor.float().cpu().numpy(),
#         original_size=np.array(original_size),
#         crop_ltrb=np.array(crop_ltrb),
#         **kwargs,
#     )


def debug_dataset(train_dataset, show_input_ids=False):
    logger.info(f"Total dataset length (steps) / ????????????????: {len(train_dataset)}")
    logger.info(
        "`S` for next step, `E` for next epoch no. , Escape for exit. / S??????????E??????????Esc???????????"
    )

    epoch = 1
    while True:
        logger.info(f"")
        logger.info(f"epoch: {epoch}")

        steps = (epoch - 1) * len(train_dataset) + 1
        indices = list(range(len(train_dataset)))
        random.shuffle(indices)

        k = 0
        for i, idx in enumerate(indices):
            train_dataset.set_current_epoch(epoch)
            train_dataset.set_current_step(steps)
            logger.info(f"steps: {steps} ({i + 1}/{len(train_dataset)})")

            example = train_dataset[idx]
            if example["latents"] is not None:
                logger.info(f"sample has latents from npz file: {example['latents'].size()}")
            for j, (ik, cap, lw, orgsz, crptl, trgsz, flpdz) in enumerate(
                zip(
                    example["image_keys"],
                    example["captions"],
                    example["loss_weights"],
                    # example["input_ids"],
                    example["original_sizes_hw"],
                    example["crop_top_lefts"],
                    example["target_sizes_hw"],
                    example["flippeds"],
                )
            ):
                logger.info(
                    f'{ik}, size: {train_dataset.image_data[ik].image_size}, loss weight: {lw}, caption: "{cap}", original size: {orgsz}, crop top left: {crptl}, target size: {trgsz}, flipped: {flpdz}'
                )
                if "network_multipliers" in example:
                    logger.info(f"network multiplier: {example['network_multipliers'][j]}")
                if "custom_attributes" in example:
                    logger.info(f"custom attributes: {example['custom_attributes'][j]}")

                # if show_input_ids:
                #     logger.info(f"input ids: {iid}")
                #     if "input_ids2" in example:
                #         logger.info(f"input ids2: {example['input_ids2'][j]}")
                if example["images"] is not None:
                    im = example["images"][j]
                    logger.info(f"image size: {im.size()}")
                    im = ((im.numpy() + 1.0) * 127.5).astype(np.uint8)
                    im = np.transpose(im, (1, 2, 0))  # c,H,W -> H,W,c
                    im = im[:, :, ::-1]  # RGB -> BGR (OpenCV)

                    if "conditioning_images" in example:
                        cond_img = example["conditioning_images"][j]
                        logger.info(f"conditioning image size: {cond_img.size()}")
                        cond_img = ((cond_img.numpy() + 1.0) * 127.5).astype(np.uint8)
                        cond_img = np.transpose(cond_img, (1, 2, 0))
                        cond_img = cond_img[:, :, ::-1]
                        if os.name == "nt":
                            cv2.imshow("cond_img", cond_img)

                    if "alpha_masks" in example and example["alpha_masks"] is not None:
                        alpha_mask = example["alpha_masks"][j]
                        logger.info(f"alpha mask size: {alpha_mask.size()}")
                        alpha_mask = (alpha_mask.numpy() * 255.0).astype(np.uint8)
                        if os.name == "nt":
                            cv2.imshow("alpha_mask", alpha_mask)

                    if os.name == "nt":  # only windows
                        cv2.imshow("img", im)
                        k = cv2.waitKey()
                        cv2.destroyAllWindows()
                    if k == 27 or k == ord("s") or k == ord("e"):
                        break
            steps += 1

            if k == ord("e"):
                break
            if k == 27 or (example["images"] is None and i >= 8):
                k = 27
                break
        if k == 27:
            break

        epoch += 1


def glob_images(directory, base="*"):
    img_paths = []
    for ext in IMAGE_EXTENSIONS:
        if base == "*":
            img_paths.extend(glob.glob(os.path.join(glob.escape(directory), base + ext)))
        else:
            img_paths.extend(glob.glob(glob.escape(os.path.join(directory, base + ext))))
    img_paths = list(set(img_paths))  # ?????
    img_paths.sort()
    return img_paths


def glob_images_pathlib(dir_path, recursive):
    image_paths = []
    if recursive:
        for ext in IMAGE_EXTENSIONS:
            image_paths += list(dir_path.rglob("*" + ext))
    else:
        for ext in IMAGE_EXTENSIONS:
            image_paths += list(dir_path.glob("*" + ext))
    image_paths = list(set(image_paths))  # ?????
    image_paths.sort()
    return image_paths


class MinimalDataset(BaseDataset):
    def __init__(self, resolution, network_multiplier, debug_dataset=False):
        super().__init__(resolution, network_multiplier, debug_dataset)

        self.num_train_images = 0  # update in subclass
        self.num_reg_images = 0  # update in subclass
        self.datasets = [self]
        self.batch_size = 1  # update in subclass

        self.subsets = [self]
        self.num_repeats = 1  # update in subclass if needed
        self.img_count = 1  # update in subclass if needed
        self.bucket_info = {}
        self.is_reg = False
        self.image_dir = "dummy"  # for metadata

    def verify_bucket_reso_steps(self, min_steps: int):
        pass

    def is_latent_cacheable(self) -> bool:
        return False

    def __len__(self):
        raise NotImplementedError

    # override to avoid shuffling buckets
    def set_current_epoch(self, epoch):
        self.current_epoch = epoch

    def __getitem__(self, idx):
        r"""
        The subclass may have image_data for debug_dataset, which is a dict of ImageInfo objects.

        Returns: example like this:

            for i in range(batch_size):
                image_key = ...  # whatever hashable
                image_keys.append(image_key)

                image = ...  # PIL Image
                img_tensor = self.image_transforms(img)
                images.append(img_tensor)

                caption = ...  # str
                input_ids = self.get_input_ids(caption)
                input_ids_list.append(input_ids)

                captions.append(caption)

            images = torch.stack(images, dim=0)
            input_ids_list = torch.stack(input_ids_list, dim=0)
            example = {
                "images": images,
                "input_ids": input_ids_list,
                "captions": captions,   # for debug_dataset
                "latents": None,
                "image_keys": image_keys,   # for debug_dataset
                "loss_weights": torch.ones(batch_size, dtype=torch.float32),
            }
            return example
        """
        raise NotImplementedError

    def get_resolutions(self) -> List[Tuple[int, int]]:
        return []


def load_arbitrary_dataset(args, tokenizer=None) -> MinimalDataset:
    module = ".".join(args.dataset_class.split(".")[:-1])
    dataset_class = args.dataset_class.split(".")[-1]
    module = importlib.import_module(module)
    dataset_class = getattr(module, dataset_class)
    train_dataset_group: MinimalDataset = dataset_class(tokenizer, args.max_token_length, args.resolution, args.debug_dataset)
    return train_dataset_group


def load_image(image_path, alpha=False):
    try:
        with Image.open(image_path) as image:
            if alpha:
                if not image.mode == "RGBA":
                    image = image.convert("RGBA")
            else:
                if not image.mode == "RGB":
                    image = image.convert("RGB")
            img = np.array(image, np.uint8)
            return img
    except (IOError, OSError) as e:
        logger.error(f"Error loading file: {image_path}")
        raise e


# ????????????numpy.ndarray,(original width, original height),(crop left, crop top, crop right, crop bottom)
def trim_and_resize_if_required(
    random_crop: bool, image: np.ndarray, reso, resized_size: Tuple[int, int], resize_interpolation: Optional[str] = None
) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int, int, int]]:
    image_height, image_width = image.shape[0:2]
    original_size = (image_width, image_height)  # size before resize

    if image_width != resized_size[0] or image_height != resized_size[1]:
        image = resize_image(image, image_width, image_height, resized_size[0], resized_size[1], resize_interpolation)

    image_height, image_width = image.shape[0:2]

    if image_width > reso[0]:
        trim_size = image_width - reso[0]
        p = trim_size // 2 if not random_crop else random.randint(0, trim_size)
        # logger.info(f"w {trim_size} {p}")
        image = image[:, p : p + reso[0]]
    if image_height > reso[1]:
        trim_size = image_height - reso[1]
        p = trim_size // 2 if not random_crop else random.randint(0, trim_size)
        # logger.info(f"h {trim_size} {p})
        image = image[p : p + reso[1]]

    # random crop????crop???????crop left/top?????????????????
    # I have no idea how to reflect the cropped value in crop left/top in the case of random crop

    crop_ltrb = BucketManager.get_crop_ltrb(reso, original_size)

    assert image.shape[0] == reso[1] and image.shape[1] == reso[0], f"internal error, illegal trimmed size: {image.shape}, {reso}"
    return image, original_size, crop_ltrb


# for new_cache_latents
def load_images_and_masks_for_caching(
    image_infos: List[ImageInfo], use_alpha_mask: bool, random_crop: bool
) -> Tuple[torch.Tensor, List[np.ndarray], List[Tuple[int, int]], List[Tuple[int, int, int, int]]]:
    r"""
    requires image_infos to have: [absolute_path or image], bucket_reso, resized_size

    returns: image_tensor, alpha_masks, original_sizes, crop_ltrbs

    image_tensor: torch.Tensor = torch.Size([B, 3, H, W]), ...], normalized to [-1, 1]
    alpha_masks: List[np.ndarray] = [np.ndarray([H, W]), ...], normalized to [0, 1]
    original_sizes: List[Tuple[int, int]] = [(W, H), ...]
    crop_ltrbs: List[Tuple[int, int, int, int]] = [(L, T, R, B), ...]
    """
    images: List[torch.Tensor] = []
    alpha_masks: List[np.ndarray] = []
    original_sizes: List[Tuple[int, int]] = []
    crop_ltrbs: List[Tuple[int, int, int, int]] = []
    for info in image_infos:
        image = load_image(info.absolute_path, use_alpha_mask) if info.image is None else np.array(info.image, np.uint8)
        # TODO ???????????????????????????bucket?????????????????????????????
        image, original_size, crop_ltrb = trim_and_resize_if_required(
            random_crop, image, info.bucket_reso, info.resized_size, resize_interpolation=info.resize_interpolation
        )

        original_sizes.append(original_size)
        crop_ltrbs.append(crop_ltrb)

        if use_alpha_mask:
            if image.shape[2] == 4:
                alpha_mask = image[:, :, 3]  # [H,W]
                alpha_mask = alpha_mask.astype(np.float32) / 255.0
                alpha_mask = torch.FloatTensor(alpha_mask)  # [H,W]
            else:
                alpha_mask = torch.ones_like(image[:, :, 0], dtype=torch.float32)  # [H,W]
        else:
            alpha_mask = None
        alpha_masks.append(alpha_mask)

        image = image[:, :, :3]  # remove alpha channel if exists
        image = IMAGE_TRANSFORMS(image)
        images.append(image)

    img_tensor = torch.stack(images, dim=0)
    return img_tensor, alpha_masks, original_sizes, crop_ltrbs


def cache_batch_latents(
    vae: AutoencoderKL, cache_to_disk: bool, image_infos: List[ImageInfo], flip_aug: bool, use_alpha_mask: bool, random_crop: bool
) -> None:
    r"""
    requires image_infos to have: absolute_path, bucket_reso, resized_size, latents_npz
    optionally requires image_infos to have: image
    if cache_to_disk is True, set info.latents_npz
        flipped latents is also saved if flip_aug is True
    if cache_to_disk is False, set info.latents
        latents_flipped is also set if flip_aug is True
    latents_original_size and latents_crop_ltrb are also set
    """
    images = []
    alpha_masks: List[np.ndarray] = []
    for info in image_infos:
        image = load_image(info.absolute_path, use_alpha_mask) if info.image is None else np.array(info.image, np.uint8)
        # TODO ???????????????????????????bucket?????????????????????????????
        image, original_size, crop_ltrb = trim_and_resize_if_required(
            random_crop, image, info.bucket_reso, info.resized_size, resize_interpolation=info.resize_interpolation
        )

        info.latents_original_size = original_size
        info.latents_crop_ltrb = crop_ltrb

        if use_alpha_mask:
            if image.shape[2] == 4:
                alpha_mask = image[:, :, 3]  # [H,W]
                alpha_mask = alpha_mask.astype(np.float32) / 255.0
                alpha_mask = torch.FloatTensor(alpha_mask)  # [H,W]
            else:
                alpha_mask = torch.ones_like(image[:, :, 0], dtype=torch.float32)  # [H,W]
        else:
            alpha_mask = None
        alpha_masks.append(alpha_mask)

        image = image[:, :, :3]  # remove alpha channel if exists
        image = IMAGE_TRANSFORMS(image)
        images.append(image)

    img_tensors = torch.stack(images, dim=0)
    img_tensors = img_tensors.to(device=vae.device, dtype=vae.dtype)

    with torch.no_grad():
        latents = vae.encode(img_tensors).latent_dist.sample().to("cpu")

    if flip_aug:
        img_tensors = torch.flip(img_tensors, dims=[3])
        with torch.no_grad():
            flipped_latents = vae.encode(img_tensors).latent_dist.sample().to("cpu")
    else:
        flipped_latents = [None] * len(latents)

    for info, latent, flipped_latent, alpha_mask in zip(image_infos, latents, flipped_latents,
