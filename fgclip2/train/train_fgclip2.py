from __future__ import annotations
import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List

import torch
import random


import glob
import transformers

from torch.utils.data import Dataset
from fgclip2.train.local_trainer import CLIPTrainer


import torch.distributed as dist

import copy
import os
import json
import torch
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
    box_image_size: int = 224
    add_box_loss: bool = field(default=False)
    use_hard_neg: bool = field(default=False)
    cn_pair_root: Optional[str] = field(default=None)
    cn_image_root: Optional[str] = field(default=None)
    max_num_patches: int = 0
    caption_loss_weight: float = field(default=0.0)
    llm_model_path: Optional[str] = field(default=None)
    llm_gradient_checkpointing: bool = field(default=False)


    

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



class LazySupervisedBboxDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 data_args: DataArguments,
                 img_preprocess=None, tokenizer=None, llm_tokenizer=None):
        super(LazySupervisedBboxDataset, self).__init__()

        if data_path.endswith('.json') or data_path.endswith('.jsonl'):
            list_data_dict = json.load(open(data_path, "r", encoding="utf-8"))
        elif data_path.endswith('.txt'):
            lines = open(data_path, "r", encoding="utf-8").readlines()
            list_data_dict = []
            for line in lines:
                json_file = line.rstrip()
                list_data_dict += json.load(open(json_file, "r",encoding="utf-8"))
        else:
            json_files = glob.glob(os.path.join(data_path, '*.json'))
            list_data_dict = []
            for json_file in json_files:
                list_data_dict += json.load(open(json_file, "r",encoding="utf-8"))

            jsonl_files = glob.glob(os.path.join(data_path, '*.jsonl'))
            for jsonl_file in jsonl_files:
                list_data_dict += json.load(open(jsonl_file, "r",encoding="utf-8"))


        self.en_data_length = len(list_data_dict)

        if data_args.cn_pair_root is not None:
            cn_data_path = data_args.cn_pair_root
            json_files = glob.glob(os.path.join(cn_data_path, '*.json'))
            cn_list = []
            for json_file in json_files:
                cn_list += json.load(open(json_file, "r",encoding="utf-8"))
            list_data_dict += cn_list

        self.all_data_length = len(list_data_dict)


        rank0_print("Formatting inputs...Skip in lazy mode")

        self.total_len = 1000
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.max_anns = 4

        self.data_args = data_args
        self.preprocess = img_preprocess
        self.image_root = data_args.image_folder
        self.max_length = data_args.max_seq_length
        self.base_length = data_args.base_seq_length
        self.box_image_size = data_args.box_image_size
        self.add_box_loss = data_args.add_box_loss
        self.use_hard_neg = data_args.use_hard_neg
        self.cn_image_root = data_args.cn_image_root
        self.caption_loss_weight = data_args.caption_loss_weight
        self.llm_tokenizer = llm_tokenizer

    def __len__(self):
        return len(self.list_data_dict)


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

        item = self.list_data_dict[i]
        caption = item["caption"]
        image_path = item["f_path"]
        
        if "is_cn" not in item.keys():
            is_cn = False
            caption_short = "a photo of "+item["short_caption"]
        else:
            is_cn = True
            caption_short = item["short_caption"]


        if is_cn:
            image_name = os.path.join(self.cn_image_root,image_path)
        else:
            image_name = os.path.join(self.image_root,image_path)

        
        
        image = Image.open(image_name).convert("RGB")
        

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
        short_text = torch.tensor(self.tokenizer([caption_short.lower()], max_length=self.base_length, padding="max_length", truncation=True).input_ids, dtype=torch.long)        



        if self.add_box_loss:

            box_texts = []
            total_num = self.max_anns
            if "is_cn" not in item.keys():
                bbox_info = item["bbox_info"]
                valid_num = min(len(bbox_info), self.max_anns)
            else:
                valid_num = 0

            boxes_template = torch.zeros((total_num, 4), device=short_text.device)
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
                box_text = torch.tensor(self.tokenizer([box_caption.lower()], max_length=self.base_length, padding="max_length", truncation=True).input_ids, dtype=torch.long, device=short_text.device)        
                box_texts.append(box_text)

            box_texts = torch.cat(box_texts,dim=0)

            bbox_num = torch.tensor([valid_num], device=short_text.device)

        if self.use_hard_neg:
            hard_texts = []

            width, height = image.size
            total_num = self.max_anns
           
            if "is_cn" not in item.keys():
                bbox_info = item["bbox_info"]
                valid_num = min(len(bbox_info), self.max_anns)
            else:
                valid_num = 0

            hard_boxes = torch.zeros((total_num, 4), device=short_text.device)
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
                        box_text = torch.tensor(self.tokenizer(cur_texts, max_length=self.base_length, padding="max_length", truncation=True).input_ids, dtype=torch.long, device=short_text.device)        
                        hard_texts.append(box_text)

                        hard_boxes[valid_hard] = box_tensor
                        valid_hard = valid_hard+1
    
                        left = int(box[0] * width)
                        top = int(box[1] * height)
                        right = int(box[2] * width)
                        bottom = int(box[3] * height)
  

            valid_hard = torch.tensor([valid_hard], device=short_text.device)
   
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

    model = FG_CLIP2_Model.from_pretrained(model_args.model_name_or_path)

    config = model.config
    import numpy as np

    model.logit_scale_finegraind = torch.nn.Parameter(torch.ones([]) * model_args.log_scale)
    model.logit_scale_hardneg = torch.nn.Parameter(torch.ones([]) * model_args.log_scale)
    
    if training_args.from_siglip2:
        print("copy and resize")
        model.resize_postion_embeding()
        model.copy_weight()
        print("copy_weight")
        model.copy_dense_feature_head()
        print("copy_dense_feature_head")
        print("fine")

    model.world_size = training_args.train_use_word_size
    model.loss_type = model_args.loss_type
    model.caption_loss_weight = data_args.caption_loss_weight

    llm_tokenizer = None
    if data_args.llm_model_path is not None and data_args.caption_loss_weight > 0.0:
        from fgclip2.model.strcs.caption_decoder import LLMCaptionDecoder
        llm_caption_decoder = LLMCaptionDecoder(
            vis_hidden_dim=model.config.vision_config.hidden_size,
            llm_model_path=data_args.llm_model_path,
        )
        model.llm_caption_decoder = llm_caption_decoder
        if data_args.llm_gradient_checkpointing:
            model.llm_caption_decoder.llm.gradient_checkpointing_enable()
            print(f"[DEBUG] LLM gradient checkpointing enabled")
        llm_tokenizer = AutoTokenizer.from_pretrained(data_args.llm_model_path)
        if llm_tokenizer.pad_token is None:
            llm_tokenizer.pad_token = llm_tokenizer.eos_token
        print(f"[DEBUG] LLMCaptionDecoder loaded from {data_args.llm_model_path}")
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
