from __future__ import annotations
import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List
from array import array
from bisect import bisect_right

import torch
import random


import glob
import transformers

from torch.utils.data import Dataset
from fgclip2.train.local_trainer import CLIPTrainer
from fgclip2.train.projector_utils import configure_projector_only_training, freeze_projector, load_projector_from_path


import torch.distributed as dist

import copy
import os
import json
import torch
import re
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
from torchvision import transforms
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.transforms.functional import InterpolationMode
from einops import rearrange
# import cv2
from random import choice
from PIL import Image

import gzip
from io import BytesIO
import base64
from torch.utils.data import  IterableDataset
import random
import numpy as np

from fgclip2.model.strcs.configuration_fgclip2 import Fgclip2Config
from fgclip2.model.strcs.fgclip2 import FG_CLIP2_Model
from transformers import AutoProcessor,Siglip2ImageProcessor


from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)


import gc



local_rank = None
IMAGE_TOKEN_PATTERN = re.compile(r"<image>")


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="qihoo360/fg-clip2-base")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    base_model: Optional[str] = field(default=None)
    download_root: Optional[str] = field(default=None)
    log_scale: float = 4.6052
    loss_type: Optional[str] = field(default=None)

@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    image_grid_pinpoints: Optional[str] = field(default=None)
    max_seq_length: int = 64*4-60
    base_seq_length: int = 64
    use_short_caption: bool = field(
        default=True,
        metadata={"help": "Whether to train with the short caption image-text loss."},
    )
    box_image_size: int = 224
    add_box_loss: bool = field(default=False)
    use_hard_neg: bool = field(default=False)
    cn_pair_root: Optional[str] = field(default=None)
    cn_image_root: Optional[str] = field(default=None)
    max_num_patches: int = 0
    long_loss_weight: float = field(default=1.0)
    caption_loss_weight: float = field(default=0.0)
    llm_model_path: Optional[str] = field(default=None)
    llm_gradient_checkpointing: bool = field(default=False)
    caption_pool_2x2_tokens: bool = field(default=False)
    train_projector_only: bool = field(default=False)
    freeze_projector: bool = field(default=False)
    load_projector_from: Optional[str] = field(default=None)
    missing_image_log_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a jsonl log file for missing or unreadable training images."},
    )
    large_image_log_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a jsonl log file for skipped over-large training images."},
    )
    max_image_pixels: int = field(
        default=50000000,
        metadata={"help": "Skip images whose width * height exceeds this value. Set 0 to disable."},
    )
    max_num_patches: int = 0


    

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    train_use_word_size: int = 8
    text_model_lr: Optional[float] = None
    projector_lr: Optional[float] = None
    from_siglip2: bool = field(default=False)
    cn_and_en_2_train: bool = field(default=False)
    naflex_train: bool = field(default=False)


from datetime import datetime
    
def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()

    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

import ast


class JsonListStore:
    def __init__(self, data_file: str, max_records: Optional[int] = None):
        self.data_file = data_file
        with open(data_file, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        if max_records is not None:
            self.data = self.data[:max_records]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class JsonlOffsetStore:
    def __init__(
        self,
        data_file: str,
        max_records: Optional[int] = None,
        sample_seed: Optional[int] = None,
        index_file: Optional[str] = None,
    ):
        self.data_file = data_file
        self.offsets = array("Q")
        self._fp = None
        self.index_file = index_file

        if index_file is not None and os.path.exists(index_file):
            with open(index_file, "rb") as f:
                self.offsets.fromfile(f, os.path.getsize(index_file) // self.offsets.itemsize)
            if max_records is not None and len(self.offsets) != max_records:
                raise ValueError(
                    f"Index file {index_file} has {len(self.offsets)} records, expected {max_records}."
                )
            return

        rng = random.Random(sample_seed) if sample_seed is not None else None
        record_count = 0

        with open(data_file, "rb") as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                if line.strip():
                    record_count += 1
                    if rng is not None and max_records is not None:
                        if len(self.offsets) < max_records:
                            self.offsets.append(offset)
                        else:
                            sample_idx = rng.randrange(record_count)
                            if sample_idx < max_records:
                                self.offsets[sample_idx] = offset
                    else:
                        self.offsets.append(offset)
                    if rng is None and max_records is not None and len(self.offsets) >= max_records:
                        break

        if sample_seed is not None and max_records is not None:
            if max_records > record_count:
                raise ValueError(
                    f"Cannot sample {max_records} records from {self.data_file}; only {record_count} records found."
                )

        if index_file is not None:
            index_dir = os.path.dirname(index_file)
            if index_dir:
                os.makedirs(index_dir, exist_ok=True)
            tmp_index_file = f"{index_file}.tmp.{os.getpid()}"
            with open(tmp_index_file, "wb") as f:
                self.offsets.tofile(f)
            os.replace(tmp_index_file, index_file)

    def __len__(self):
        return len(self.offsets)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_fp"] = None
        return state

    def _file(self):
        if self._fp is None:
            self._fp = open(self.data_file, "rb")
        return self._fp

    def __getitem__(self, index):
        if index < 0:
            index += len(self.offsets)
        if index < 0 or index >= len(self.offsets):
            raise IndexError(index)

        f = self._file()
        f.seek(self.offsets[index])
        line = f.readline()
        return json.loads(line.decode("utf-8"))


class ConcatStore:
    def __init__(self, stores):
        self.stores = [store for store in stores if len(store) > 0]
        self.cumulative_sizes = []
        total = 0
        for store in self.stores:
            total += len(store)
            self.cumulative_sizes.append(total)

    def __len__(self):
        if not self.cumulative_sizes:
            return 0
        return self.cumulative_sizes[-1]

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)

        store_idx = bisect_right(self.cumulative_sizes, index)
        previous_size = 0 if store_idx == 0 else self.cumulative_sizes[store_idx - 1]
        return self.stores[store_idx][index - previous_size]


def parse_manifest_line(line: str, manifest_dir: str):
    parts = line.strip().split()
    if not parts:
        return None, None, None, None

    data_path = parts[0]
    max_records = None
    sample_seed = None
    index_file = None
    if len(parts) > 1:
        max_records = int(parts[1])
    if len(parts) > 2:
        sample_seed = int(parts[2])
    if len(parts) > 3:
        index_file = parts[3]

    if not os.path.isabs(data_path):
        data_path = os.path.join(manifest_dir, data_path)
    if index_file is not None and not os.path.isabs(index_file):
        index_file = os.path.join(manifest_dir, index_file)

    return data_path, max_records, sample_seed, index_file


def build_data_store(
    data_path: str,
    max_records: Optional[int] = None,
    sample_seed: Optional[int] = None,
    index_file: Optional[str] = None,
):
    data_path = os.path.abspath(data_path)

    if data_path.endswith(".jsonl"):
        return JsonlOffsetStore(
            data_path,
            max_records=max_records,
            sample_seed=sample_seed,
            index_file=index_file,
        )

    if data_path.endswith(".json"):
        return JsonListStore(data_path, max_records=max_records)

    if data_path.endswith(".txt"):
        stores = []
        manifest_dir = os.path.dirname(data_path)
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                json_file, json_limit, json_sample_seed, json_index_file = parse_manifest_line(line, manifest_dir)
                if not json_file:
                    continue
                stores.append(
                    build_data_store(
                        json_file,
                        max_records=json_limit,
                        sample_seed=json_sample_seed,
                        index_file=json_index_file,
                    )
                )
        return ConcatStore(stores)

    stores = []
    for json_file in sorted(glob.glob(os.path.join(data_path, "*.json"))):
        stores.append(JsonListStore(json_file))
    for jsonl_file in sorted(glob.glob(os.path.join(data_path, "*.jsonl"))):
        stores.append(JsonlOffsetStore(jsonl_file))
    return ConcatStore(stores)



class LazySupervisedBboxDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 data_args: DataArguments,
                 img_preprocess=None, tokenizer=None, llm_tokenizer=None):
        # super(LazySupervisedBboxDataset, self).__init__()

        # if data_path.endswith('.json') or data_path.endswith('.jsonl'):
        #     list_data_dict = json.load(open(data_path, "r", encoding="utf-8"))
        # elif data_path.endswith('.txt'):
        #     lines = open(data_path, "r", encoding="utf-8").readlines()
        #     list_data_dict = []
        #     for line in lines:
        #         json_file = line.rstrip()
        #         list_data_dict += json.load(open(json_file, "r",encoding="utf-8"))
        # else:
        #     json_files = glob.glob(os.path.join(data_path, '*.json'))
        #     list_data_dict = []
        #     for json_file in json_files:
        #         list_data_dict += json.load(open(json_file, "r",encoding="utf-8"))

        #     jsonl_files = glob.glob(os.path.join(data_path, '*.jsonl'))
        #     for jsonl_file in jsonl_files:
        #         list_data_dict += json.load(open(jsonl_file, "r",encoding="utf-8"))


        # self.en_data_length = len(list_data_dict)

        # if data_args.cn_pair_root is not None:
        #     cn_data_path = data_args.cn_pair_root
        #     json_files = glob.glob(os.path.join(cn_data_path, '*.json'))
        #     cn_list = []
        #     for json_file in json_files:
        #         cn_list += json.load(open(json_file, "r",encoding="utf-8"))
        #     list_data_dict += cn_list

        # self.all_data_length = len(list_data_dict)
        super(LazySupervisedBboxDataset, self).__init__()

        data_store = build_data_store(data_path)
        self.en_data_length = len(data_store)

        if data_args.cn_pair_root is not None:
            cn_data_store = build_data_store(data_args.cn_pair_root)
            data_store = ConcatStore([data_store, cn_data_store])

        self.all_data_length = len(data_store)


        rank0_print("Formatting inputs...Skip in lazy mode")

        self.total_len = 1000
        self.tokenizer = tokenizer
        self.data_store = data_store
        self.max_anns = 4

        self.data_args = data_args
        self.preprocess = img_preprocess
        self.image_root = data_args.image_folder
        self.max_length = data_args.max_seq_length
        self.base_length = data_args.base_seq_length
        self.use_short_caption = data_args.use_short_caption
        self.box_image_size = data_args.box_image_size
        self.add_box_loss = data_args.add_box_loss
        self.use_hard_neg = data_args.use_hard_neg
        self.cn_image_root = data_args.cn_image_root
        self.caption_loss_weight = data_args.caption_loss_weight
        self.llm_tokenizer = llm_tokenizer

        self.missing_image_log_path = data_args.missing_image_log_path
        self.large_image_log_path = data_args.large_image_log_path
        self.max_image_pixels = data_args.max_image_pixels
        self.logged_missing_images = set()
        self.logged_large_images = set()

    def __len__(self):
        return len(self.data_store)

    def get_caption(self, item):
        if "caption" in item:
            return item["caption"]

        messages = item.get("messages")
        if isinstance(messages, list):
            for message in messages:
                if isinstance(message, dict) and message.get("role") == "assistant" and message.get("content"):
                    return message["content"]
            for message in messages:
                if isinstance(message, dict) and message.get("content"):
                    return message["content"]

        raise KeyError("caption is required, or messages must contain a content field")

    def get_image_path(self, item):
        if "f_path" in item:
            return item["f_path"]

        images = item.get("images")
        if isinstance(images, list) and len(images) > 0:
            return images[0]

        raise KeyError("f_path is required, or images must contain at least one path")

    def resolve_image_name(self, image_path, is_cn):
        if os.path.isabs(image_path):
            return image_path
        if is_cn:
            return os.path.join(self.cn_image_root, image_path)
        return os.path.join(self.image_root, image_path)

    def log_missing_image(self, index, image_path, image_name, error):
        if self.missing_image_log_path is None:
            return

        if image_name in self.logged_missing_images:
            return

        self.logged_missing_images.add(image_name)
        log_dir = os.path.dirname(self.missing_image_log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        record = {
            "index": index,
            "image_path": image_path,
            "resolved_path": image_name,
            "error": error,
        }
        with open(self.missing_image_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def log_large_image(self, index, image_path, image_name, width, height):
        if self.large_image_log_path is None:
            return

        if image_name in self.logged_large_images:
            return

        self.logged_large_images.add(image_name)
        log_dir = os.path.dirname(self.large_image_log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        record = {
            "index": index,
            "image_path": image_path,
            "resolved_path": image_name,
            "width": width,
            "height": height,
            "pixels": width * height,
            "max_image_pixels": self.max_image_pixels,
        }
        with open(self.large_image_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def load_valid_item(self, i):
        dataset_len = len(self.data_store)
        for offset in range(dataset_len):
            cur_idx = (i + offset) % dataset_len
            item = self.data_store[cur_idx]
            caption = IMAGE_TOKEN_PATTERN.sub("", self.get_caption(item)).strip()
            image_path = self.get_image_path(item)
            caption_short = None

            if "is_cn" not in item.keys():
                is_cn = False
                if self.use_short_caption:
                    if "short_caption" not in item:
                        raise KeyError("short_caption is required when use_short_caption=True")
                    caption_short = "a photo of "+item["short_caption"]
            else:
                is_cn = True
                if self.use_short_caption:
                    if "short_caption" not in item:
                        raise KeyError("short_caption is required when use_short_caption=True")
                    caption_short = item["short_caption"]

            image_name = self.resolve_image_name(image_path, is_cn)

            try:
                image = Image.open(image_name)
                width, height = image.size
                if self.max_image_pixels > 0 and width * height > self.max_image_pixels:
                    self.log_large_image(cur_idx, image_path, image_name, width, height)
                    image.close()
                    continue
                image = image.convert("RGB")
            except (FileNotFoundError, OSError) as e:
                self.log_missing_image(cur_idx, image_path, image_name, repr(e))
                continue

            return item, caption, caption_short, is_cn, image, image_name

        raise RuntimeError("No readable image was found in the dataset.")


    @property
    def modality_lengths(self):
        length_list = []
        for cur_idx in range(self.all_data_length):
            if cur_idx < self.en_data_length:
                length_list.append(1)
            else:
                length_list.append(0)
        return length_list
 

    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:

        item, caption, caption_short, is_cn, image, image_name = self.load_valid_item(i)
        

        prewidth, preheight = image.size

        if self.data_args.max_num_patches !=0:
            # NOTE The low unilateral resolution may cause a bug, forced to resize.
            if prewidth < 128 or preheight < 128:
                image = image.resize((self.box_image_size, self.box_image_size))

            width, height = image.size
            max_img_token = (width//16)*(height//16)
            image_tensor = image
            pixel_attention_mask = None
            spatial_shapes = None
        else:
            image = image.resize((self.box_image_size, self.box_image_size))
            width, height = image.size
            max_img_token = (width//16)*(height//16)
            pixel_attention_mask = None
            spatial_shapes = None
            image_tensor = self.preprocess(images=image, return_tensors='pt')['pixel_values'][0]

        
        max_img_token = torch.tensor([max_img_token])
        
        text =  torch.tensor(self.tokenizer([caption.lower()], max_length=self.max_length, padding="max_length", truncation=True).input_ids, dtype=torch.long)
        short_text = None
        if self.use_short_caption:
            short_text = torch.tensor(self.tokenizer([caption_short.lower()], max_length=self.base_length, padding="max_length", truncation=True).input_ids, dtype=torch.long)
        tensor_device = text.device



        if self.add_box_loss:

            box_texts = []
            total_num = self.max_anns
            if "is_cn" not in item.keys():
                bbox_info = item["bbox_info"]
                valid_num = min(len(bbox_info), self.max_anns)
            else:
                valid_num = 0

            boxes_template = torch.zeros((total_num, 4), device=tensor_device)
            width, height = image.size

            for i in range(total_num):
                if i<valid_num:
                    bbox_data = bbox_info[i]
                    box = bbox_data["bbox"]
                    box_caption = random.choice([bbox_data["short_expr"], bbox_data["long_expr"]])
                else:
                    box = [0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.000000000]
                    box_caption = ""


                box_tensor = torch.tensor(box[:4])
                boxes_template[i] = box_tensor

                if box[0] > box[2] or box[1] > box[3]:
                    raise ValueError("Box coordinates are invalid.")

                left = int(box[0] * width)
                top = int(box[1] * height)
                right = int(box[2] * width)
                bottom = int(box[3] * height)
                box_text = torch.tensor(self.tokenizer([box_caption.lower()], max_length=self.base_length, padding="max_length", truncation=True).input_ids, dtype=torch.long, device=tensor_device)
                box_texts.append(box_text)

            box_texts = torch.cat(box_texts,dim=0)

            bbox_num = torch.tensor([valid_num], device=tensor_device)

        if self.use_hard_neg:
            hard_texts = []

            width, height = image.size
            total_num = self.max_anns
           
            if "is_cn" not in item.keys():
                bbox_info = item["bbox_info"]
                valid_num = min(len(bbox_info), self.max_anns)
            else:
                valid_num = 0

            hard_boxes = torch.zeros((total_num, 4), device=tensor_device)
            valid_hard = 0
            for i in range(total_num):
                if i<valid_num:
                    bbox_data = bbox_info[i]
                    box = bbox_data["bbox"]
                    box_caption = bbox_data["short_expr"]
                    
                    box_tensor = torch.tensor(box[:4])
                    if box[0] > box[2] or box[1] > box[3]:
                        raise ValueError("Box coordinates are invalid.")
    
                    if bbox_data["flag_short_neg"] == 1:
                        cur_texts = [box_caption]
                        hard_negs = bbox_data["short_expr_negs"]
                        for key in hard_negs.keys():
                            cur_texts.append(hard_negs[key].lower())
                        box_text = torch.tensor(self.tokenizer(cur_texts, max_length=self.base_length, padding="max_length", truncation=True).input_ids, dtype=torch.long, device=tensor_device)
                        hard_texts.append(box_text)

                        hard_boxes[valid_hard] = box_tensor
                        valid_hard = valid_hard+1
    
                        left = int(box[0] * width)
                        top = int(box[1] * height)
                        right = int(box[2] * width)
                        bottom = int(box[3] * height)
  

            valid_hard = torch.tensor([valid_hard], device=tensor_device)
   
            if len(hard_texts) > 0:
                hard_texts = torch.cat(hard_texts,dim=0)
            else:
                hard_texts = None

        data_dict = {}
        data_dict['image'] = image_tensor
        data_dict['pixel_attention_mask'] = pixel_attention_mask
        data_dict['spatial_shapes'] = spatial_shapes
        
        
        data_dict['text'] = text
        data_dict['short_text'] = short_text

        data_dict['add_box_loss'] = self.add_box_loss
        data_dict['use_hard_neg'] = self.use_hard_neg
        data_dict['max_img_token'] = max_img_token
        data_dict['is_cn'] = is_cn

        if self.caption_loss_weight > 0.0 and self.llm_tokenizer is not None:
            llm_enc = self.llm_tokenizer(
                caption.lower(),
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            llm_input_ids = llm_enc.input_ids          # [1, L]
            llm_attention_mask = llm_enc.attention_mask  # [1, L]
            # labels: shift left, mask pad positions with -100
            labels = llm_input_ids[0, 1:].clone()
            labels = torch.cat([labels, torch.tensor([-100], dtype=labels.dtype)])
            # mask pad positions: attention_mask[1:] aligns with labels[:-1], last position is always -100
            pad_mask = llm_attention_mask[0, 1:] == 0  # [L-1]
            labels[:-1][pad_mask] = -100
            data_dict['llm_input_ids'] = llm_input_ids
            data_dict['llm_attention_mask'] = llm_attention_mask
            data_dict['caption_labels'] = labels.unsqueeze(0)  # [1, L]

        if self.add_box_loss:
            # data_dict['box_images'] = box_images
            data_dict['box_texts'] = box_texts
            data_dict['box_infos'] = boxes_template
            data_dict['box_nums'] = bbox_num
        if self.use_hard_neg:

            data_dict['hard_texts'] = hard_texts
            data_dict['hard_infos'] = hard_boxes
            data_dict['hard_nums'] = valid_hard
            
        return data_dict



@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    preprocess: transformers.Siglip2ImageProcessor
    is_naflex: bool

    def determine_max_value(self,values):

        max_val = torch.max(values).item()

        if max_val > 784:
            return 1024
        elif max_val > 576:
            return 784
        elif max_val > 256:
            return 576
        elif max_val > 128:
            return 256
        else:
            return 128

    def __call__(self, instances: Sequence[Dict]):
        
        batch = {}

        if self.is_naflex:
            batch_max_img_token = self.determine_max_value(torch.stack([instance['max_img_token'] for instance in instances]))

            pixel_values = []
            pixel_attention_masks = []
            spatial_shapes = []

            for instance in instances:

                try:
                    image_input = self.preprocess(images=instance['image'].convert("RGB"), max_num_patches=batch_max_img_token, return_tensors='pt')
                except Exception as e:
                    print(e)
                    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! get fail image !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    width, height = 384, 384  # 
                    channels = 3  
                    black_image_array = np.zeros((height, width, channels), dtype=np.uint8)
                    black_image = Image.fromarray(black_image_array, mode="RGB")
                    image_input = self.preprocess(images=black_image, max_num_patches=batch_max_img_token, return_tensors='pt')

                pixel_values.append(image_input["pixel_values"])
                pixel_attention_masks.append(image_input["pixel_attention_mask"])
                spatial_shapes.append(image_input["spatial_shapes"])

            batch['pixel_values'] = torch.cat(pixel_values,dim=0)
            batch['pixel_attention_mask'] = torch.cat(pixel_attention_masks,dim=0)
            batch['spatial_shapes'] = torch.cat(spatial_shapes,dim=0)

        else:
            batch['pixel_attention_mask'] = None
            batch['spatial_shapes'] = None
            images = [instance['image'] for instance in instances]
            batch['pixel_values'] = torch.stack(images)

        texts = [instance['text'] for instance in instances]

        if None in texts:
            batch['text_long'] = None
            batch['text_long_flag'] = torch.tensor([0], device=batch['pixel_values'].device)
        else:
            batch['text_long_flag'] = torch.tensor([1], device=batch['pixel_values'].device)
            batch['text_long'] = torch.cat(texts,dim=0)

        short_texts = [instance['short_text'] for instance in instances]
        if any(short_text is None for short_text in short_texts):
            batch['text_short'] = None
        else:
            batch['text_short'] = torch.cat(short_texts,dim=0)
        
        batch["add_box_loss"] = instances[0]["add_box_loss"]
        batch["use_hard_neg"] = instances[0]["use_hard_neg"]
        
        if batch["add_box_loss"]:

            box_texts = [instance['box_texts'] for instance in instances]
            batch['box_texts'] = torch.cat(box_texts,dim=0)
            box_infos = [instance['box_infos'] for instance in instances]
            batch['box_infos'] = torch.cat(box_infos,dim=0)
            box_nums = [instance['box_nums'] for instance in instances]
            batch['box_nums'] = torch.cat(box_nums, dim=0)
            
        if batch["use_hard_neg"] :
            hard_texts = []
            for instance in instances:
                if instance['hard_texts'] != None:
                    hard_texts.append(instance['hard_texts'])
            if len(hard_texts)!=0:
                batch['hard_texts'] = torch.cat(hard_texts,dim=0)
            else:
                batch['hard_texts'] = None
            hard_infos = [instance['hard_infos'] for instance in instances]
            batch['hard_infos'] = torch.cat(hard_infos,dim=0)
            hard_nums = [instance['hard_nums'] for instance in instances]
            batch['hard_nums'] = torch.cat(hard_nums, dim=0)

        if 'caption_labels' in instances[0]:
            batch['caption_labels'] = torch.cat([instance['caption_labels'] for instance in instances], dim=0)
            batch['llm_input_ids'] = torch.cat([instance['llm_input_ids'] for instance in instances], dim=0)
            batch['llm_attention_mask'] = torch.cat([instance['llm_attention_mask'] for instance in instances], dim=0)

        return batch




def make_supervised_data_module(data_args, img_preprocess, tokenizer, llm_tokenizer=None, is_naflex=False) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""

    train_dataset = LazySupervisedBboxDataset(
                                data_path=data_args.data_path,
                                data_args=data_args,
                                img_preprocess=img_preprocess,
                                tokenizer=tokenizer,
                                llm_tokenizer=llm_tokenizer,)
            
    data_collator = DataCollatorForSupervisedDataset(preprocess=img_preprocess,is_naflex=is_naflex)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)



def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if data_args.freeze_projector and data_args.train_projector_only:
        raise ValueError("--freeze_projector True and --train_projector_only True are mutually exclusive.")
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    # compute_dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_args.base_model)

    assert training_args.naflex_train

    if training_args.naflex_train:
        assert data_args.max_num_patches in [128, 256, 576, 784, 1024, 4096]
        image_processor = Siglip2ImageProcessor.from_pretrained(model_args.base_model)
    else:
        pass

    config = Fgclip2Config.from_pretrained(model_args.model_name_or_path)
    config.enable_region_heads = data_args.add_box_loss or data_args.use_hard_neg
    model = FG_CLIP2_Model.from_pretrained(model_args.model_name_or_path, config=config)

    config = model.config
    import numpy as np

    model.logit_scale_finegraind = (
        torch.nn.Parameter(torch.ones([]) * model_args.log_scale) if data_args.add_box_loss else None
    )
    model.logit_scale_hardneg = (
        torch.nn.Parameter(torch.ones([]) * model_args.log_scale) if data_args.use_hard_neg else None
    )
    
    if training_args.from_siglip2:
        print("copy and resize")
        model.resize_postion_embeding()
        model.copy_weight()
        print("copy_weight")
        if model.enable_region_heads:
            model.copy_dense_feature_head()
            print("copy_dense_feature_head")
        print("fine")

    model.world_size = training_args.train_use_word_size
    model.loss_type = model_args.loss_type
    model.long_loss_weight = data_args.long_loss_weight
    model.caption_loss_weight = data_args.caption_loss_weight

    llm_tokenizer = None
    if data_args.llm_model_path is not None and data_args.caption_loss_weight > 0.0:
        from fgclip2.model.strcs.caption_decoder import LLMCaptionDecoder
        llm_caption_decoder = LLMCaptionDecoder(
            vis_hidden_dim=model.config.vision_config.hidden_size,
            llm_model_path=data_args.llm_model_path,
            caption_pool_2x2_tokens=data_args.caption_pool_2x2_tokens,
        )
        model.llm_caption_decoder = llm_caption_decoder
        if data_args.llm_gradient_checkpointing:
            model.llm_caption_decoder.llm.gradient_checkpointing_enable()
            print(f"[DEBUG] LLM gradient checkpointing enabled")
        llm_tokenizer = AutoTokenizer.from_pretrained(data_args.llm_model_path)
        if llm_tokenizer.pad_token is None:
            llm_tokenizer.pad_token = llm_tokenizer.eos_token
        print(f"[DEBUG] LLMCaptionDecoder loaded from {data_args.llm_model_path}")
        if data_args.load_projector_from is not None:
            load_projector_from_path(model, data_args.load_projector_from)
            print(f"[DEBUG] projector loaded from {data_args.load_projector_from}")
        if data_args.freeze_projector:
            freeze_projector(model)
            print("[DEBUG] freeze_projector enabled")
        if data_args.train_projector_only:
            configure_projector_only_training(model)
            model.train_projector_only = True
            print("[DEBUG] train_projector_only enabled")
    print(f"[DEBUG] caption_loss_weight={data_args.caption_loss_weight}, llm_model_path={data_args.llm_model_path}")

    data_module = make_supervised_data_module(data_args=data_args, img_preprocess=image_processor, tokenizer=tokenizer, llm_tokenizer=llm_tokenizer, is_naflex=training_args.naflex_train)
    
    model.to(dtype=compute_dtype, device=training_args.device)

    # old: --gradient_checkpointing_kwargs {"use_reentrant":True} \
    training_args.gradient_checkpointing_kwargs = {"use_reentrant":False}

    trainer = CLIPTrainer(model=model,
                        args=training_args,
                        **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer,output_dir=training_args.output_dir)



if __name__ == "__main__":
    train()
