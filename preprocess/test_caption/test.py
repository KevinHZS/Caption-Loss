#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import re
from array import array
from bisect import bisect_right
from typing import Any, Dict, Iterable, List, Optional

import torch
import torch.nn.functional as F
from PIL import Image

DEFAULT_BASE_MODEL = "/gemini/space/gjx/FG-CLIP/siglip2-so400m-patch16-naflex"
DEFAULT_DATA_PATH = (
    "/gemini/space/gjx/FG-CLIP/data/TeleMM/stage1_longonly_2M_cleaned_manifest.txt"
)
DEFAULT_IMAGE_FOLDER = "/gemini/space/gjx/FG-CLIP/data"
DEFAULT_LLM_MODEL_PATH = "/gemini/space/zyf/models/Qwen/Qwen3-1.7B"
DEFAULT_PROJECTOR_PATH = "/gemini/space/zyf/FG-CLIP/output/stage1_siglip2_projector_joint_finehard_fixed/checkpoint-652/projector"
DEFAULT_VISION_ENCODER_PATH = "/gemini/space/zyf/FG-CLIP/output/stage1_siglip2_projector_joint_finehard_fixed/checkpoint-652"
DEFAULT_OUTPUT_LOG = "/gemini/space/zyf/FG-CLIP/output/stage1_siglip2_projector_joint_w_pretrained_projector/test_llm_caption_generation.jsonl"
DEFAULT_CAPTION_PROMPT = "Describe the image in one detailed sentence."
IMAGE_TOKEN_PATTERN = re.compile(r"<image>")


def progress(message: str, quiet: bool = False) -> None:
    if not quiet:
        print(f"[caption-test] {message}", flush=True)


class JsonListStore:
    def __init__(self, data_file: str, max_records: Optional[int] = None):
        with open(data_file, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        if max_records is not None:
            self.data = self.data[:max_records]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.data[index]


class JsonlOffsetStore:
    def __init__(self, data_file: str, max_records: Optional[int] = None):
        self.data_file = data_file
        self.offsets = array("Q")
        self._fp = None

        with open(data_file, "rb") as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                if line.strip():
                    self.offsets.append(offset)
                    if max_records is not None and len(self.offsets) >= max_records:
                        break

    def __len__(self) -> int:
        return len(self.offsets)

    def _file(self):
        if self._fp is None:
            self._fp = open(self.data_file, "rb")
        return self._fp

    def __getitem__(self, index: int) -> Dict[str, Any]:
        f = self._file()
        f.seek(self.offsets[index])
        return json.loads(f.readline().decode("utf-8"))


class ConcatStore:
    def __init__(self, stores: Iterable[Any]):
        self.stores = [store for store in stores if len(store) > 0]
        self.cumulative_sizes: List[int] = []
        total = 0
        for store in self.stores:
            total += len(store)
            self.cumulative_sizes.append(total)

    def __len__(self) -> int:
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0

    def __getitem__(self, index: int) -> Dict[str, Any]:
        store_idx = bisect_right(self.cumulative_sizes, index)
        previous_size = 0 if store_idx == 0 else self.cumulative_sizes[store_idx - 1]
        return self.stores[store_idx][index - previous_size]


def parse_manifest_line(line: str, manifest_dir: str):
    parts = line.strip().split()
    if not parts:
        return None, None
    data_path = parts[0]
    max_records = int(parts[1]) if len(parts) > 1 else None
    if not os.path.isabs(data_path):
        data_path = os.path.join(manifest_dir, data_path)
    return data_path, max_records


def build_data_store(data_path: str, max_records: Optional[int] = None):
    data_path = os.path.abspath(data_path)
    if data_path.endswith(".json"):
        return JsonListStore(data_path, max_records=max_records)
    if data_path.endswith(".jsonl"):
        return JsonlOffsetStore(data_path, max_records=max_records)
    if data_path.endswith(".txt"):
        stores = []
        manifest_dir = os.path.dirname(data_path)
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                record_path, record_limit = parse_manifest_line(line, manifest_dir)
                if record_path:
                    stores.append(
                        build_data_store(record_path, max_records=record_limit)
                    )
        return ConcatStore(stores)
    raise ValueError(f"Unsupported data path: {data_path}")


def get_caption(item: Dict[str, Any]) -> str:
    if item.get("caption"):
        return item["caption"]
    messages = item.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if (
                isinstance(message, dict)
                and message.get("role") == "assistant"
                and message.get("content")
            ):
                return message["content"]
        for message in messages:
            if isinstance(message, dict) and message.get("content"):
                return message["content"]
    raise KeyError("caption is required, or messages must contain a content field")


def get_image_path(item: Dict[str, Any]) -> str:
    if item.get("f_path"):
        return item["f_path"]
    images = item.get("images")
    if isinstance(images, list) and images:
        return images[0]
    raise KeyError("f_path is required, or images must contain at least one path")


def resolve_image_path(
    image_path: str, image_folder: str, cn_image_folder: Optional[str], is_cn: bool
) -> str:
    if os.path.isabs(image_path):
        return image_path
    if is_cn and cn_image_folder:
        return os.path.join(cn_image_folder, image_path)
    return os.path.join(image_folder, image_path)


def bucket_max_num_patches(width: int, height: int, patch_size: int = 16) -> int:
    max_img_token = (width // patch_size) * (height // patch_size)
    if max_img_token > 784:
        return 1024
    if max_img_token > 576:
        return 784
    if max_img_token > 256:
        return 576
    if max_img_token > 128:
        return 256
    return 128


def load_projector_into_decoder(
    decoder: LLMCaptionDecoder, projector_path: str
) -> None:
    from fgclip2.train.projector_utils import (
        extract_projector_state_dict,
        load_checkpoint_state_dict,
    )

    state_dict = load_checkpoint_state_dict(projector_path)
    decoder.projector.load_state_dict(
        extract_projector_state_dict(state_dict), strict=True
    )


def build_prompt(
    tokenizer, prompt: str, use_chat_template: bool, enable_thinking: bool
) -> str:
    from fgclip2.train.caption_label_utils import build_caption_prompt

    return build_caption_prompt(
        tokenizer=tokenizer,
        prompt=prompt,
        use_chat_template=use_chat_template,
        enable_thinking=enable_thinking,
    )


def build_caption_prefix(tokenizer, caption: str, prefix_token_count: int) -> str:
    token_ids = tokenizer(
        caption.lower(),
        add_special_tokens=True,
        truncation=False,
    ).input_ids
    if prefix_token_count > 0:
        token_ids = token_ids[:prefix_token_count]
    return tokenizer.decode(token_ids, skip_special_tokens=True).strip()


def build_generation_inputs(
    vision_model,
    caption_decoder: LLMCaptionDecoder,
    image_processor,
    tokenizer,
    image: Image.Image,
    prompt: str,
    device: torch.device,
    use_chat_template: bool,
    enable_thinking: bool,
):
    if image.width < 128 or image.height < 128:
        image = image.resize((512, 512))

    max_num_patches = bucket_max_num_patches(image.width, image.height)
    image_inputs = image_processor(
        images=image, max_num_patches=max_num_patches, return_tensors="pt"
    )
    pixel_values = image_inputs["pixel_values"].to(device)
    pixel_attention_mask = image_inputs["pixel_attention_mask"].to(device)
    spatial_shapes = image_inputs["spatial_shapes"].to(device)

    vision_outputs = vision_model(
        pixel_values=pixel_values,
        attention_mask=pixel_attention_mask,
        spatial_shapes=spatial_shapes,
        output_hidden_states=True,
        return_dict=True,
    )
    image_patch_tokens = vision_outputs.last_hidden_state

    if caption_decoder.caption_pool_2x2_tokens:
        from fgclip2.model.strcs.caption_decoder import pool_caption_image_patch_tokens

        image_patch_tokens, pixel_attention_mask = pool_caption_image_patch_tokens(
            image_patch_tokens=image_patch_tokens,
            pixel_attention_mask=pixel_attention_mask,
            spatial_shapes=spatial_shapes,
        )

    visual_embeds = caption_decoder.projector(image_patch_tokens)
    prompt_text = build_prompt(tokenizer, prompt, use_chat_template, enable_thinking)
    prompt_inputs = tokenizer(
        prompt_text, return_tensors="pt", add_special_tokens=True
    ).to(device)
    text_embeds = caption_decoder.llm.model.embed_tokens(prompt_inputs.input_ids)
    visual_embeds = visual_embeds.to(dtype=text_embeds.dtype)

    inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
    attention_mask = torch.cat(
        [pixel_attention_mask.long(), prompt_inputs.attention_mask.long()],
        dim=1,
    )
    return (
        inputs_embeds,
        attention_mask,
        prompt_inputs.input_ids.shape[1],
        prompt_text,
        max_num_patches,
    )


def compute_reference_caption_ppl(
    vision_model,
    caption_decoder: LLMCaptionDecoder,
    image_processor,
    tokenizer,
    image: Image.Image,
    reference_caption: str,
    prompt: str,
    device: torch.device,
    max_seq_length: int,
    use_chat_template: bool,
    enable_thinking: bool,
):
    from fgclip2.model.strcs.caption_alignment import get_caption_text_logits
    from fgclip2.train.caption_label_utils import build_caption_lm_inputs

    if image.width < 128 or image.height < 128:
        image = image.resize((512, 512))

    max_num_patches = bucket_max_num_patches(image.width, image.height)
    image_inputs = image_processor(
        images=image, max_num_patches=max_num_patches, return_tensors="pt"
    )
    pixel_values = image_inputs["pixel_values"].to(device)
    pixel_attention_mask = image_inputs["pixel_attention_mask"].to(device)
    spatial_shapes = image_inputs["spatial_shapes"].to(device)

    vision_outputs = vision_model(
        pixel_values=pixel_values,
        attention_mask=pixel_attention_mask,
        spatial_shapes=spatial_shapes,
        output_hidden_states=True,
        return_dict=True,
    )

    llm_enc = build_caption_lm_inputs(
        tokenizer=tokenizer,
        caption=reference_caption,
        max_length=max_seq_length,
        prompt=prompt,
        use_chat_template=use_chat_template,
        enable_thinking=enable_thinking,
    )
    llm_input_ids = llm_enc.input_ids.to(device)
    llm_attention_mask = llm_enc.attention_mask.to(device)
    labels = llm_enc.labels.to(device)

    caption_logits, visual_token_count = caption_decoder(
        image_patch_tokens=vision_outputs.last_hidden_state,
        pixel_attention_mask=pixel_attention_mask,
        spatial_shapes=spatial_shapes,
        llm_input_ids=llm_input_ids,
        llm_attention_mask=llm_attention_mask,
    )
    text_logits = get_caption_text_logits(caption_logits, visual_token_count)
    loss = F.cross_entropy(
        text_logits.reshape(-1, text_logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=-100,
    )
    return loss.float().item(), torch.exp(loss.float()).item()


def generate_caption(
    caption_decoder: LLMCaptionDecoder,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_token_count: int,
    tokenizer,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> str:
    generation_kwargs = {
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p
    output_ids = caption_decoder.llm.generate(**generation_kwargs)

    input_embed_count = inputs_embeds.shape[1]
    if output_ids.shape[1] > input_embed_count:
        generated_ids = output_ids[0, input_embed_count:]
    elif output_ids.shape[1] > prompt_token_count:
        generated_ids = output_ids[0, prompt_token_count:]
    else:
        generated_ids = output_ids[0]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def select_indices(
    dataset_size: int, count: int, start_index: int, seed: Optional[int]
) -> List[int]:
    if seed is None:
        return list(range(start_index, min(dataset_size, start_index + count)))
    rng = random.Random(seed)
    return rng.sample(range(dataset_size), k=min(count, dataset_size))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate captions from training images using FG-CLIP visual tokens."
    )
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--image-folder", default=DEFAULT_IMAGE_FOLDER)
    parser.add_argument("--cn-image-folder", default=None)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--vision-encoder-path", default=DEFAULT_VISION_ENCODER_PATH)
    parser.add_argument("--projector-path", default=DEFAULT_PROJECTOR_PATH)
    parser.add_argument("--llm-model-path", default=DEFAULT_LLM_MODEL_PATH)
    parser.add_argument("--output-log", default=DEFAULT_OUTPUT_LOG)
    parser.add_argument(
        "--prompt",
        default=DEFAULT_CAPTION_PROMPT,
        help="Generation prompt. The prompt is excluded from reference caption loss.",
    )
    parser.add_argument(
        "--prefix-token-count",
        type=int,
        default=8,
        help="Number of tokenizer tokens to take from the reference caption as the generation prefix.",
    )
    parser.add_argument("--max-seq-length", type=int, default=196)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--use-chat-template", action="store_true", default=True)
    parser.add_argument(
        "--no-chat-template",
        action="store_false",
        dest="use_chat_template",
        help="Disable tokenizer chat template.",
    )
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--caption-pool-2x2-tokens", action="store_true")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    parser.add_argument("--max-image-pixels", type=int, default=50_000_000)
    parser.add_argument("--quiet", action="store_true", help="Disable progress prints.")
    return parser.parse_args()


def resolve_torch_dtype(dtype_name: str):
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = resolve_torch_dtype(args.dtype)

    progress("Starting LLM caption generation test", args.quiet)
    progress(f"data_path={args.data_path}", args.quiet)
    progress(f"image_folder={args.image_folder}", args.quiet)
    progress(f"vision_encoder_path={args.vision_encoder_path}", args.quiet)
    progress(f"projector_path={args.projector_path}", args.quiet)
    progress(f"llm_model_path={args.llm_model_path}", args.quiet)
    progress(f"output_log={args.output_log}", args.quiet)
    progress(
        f"num_samples={args.num_samples}, start_index={args.start_index}, seed={args.seed}",
        args.quiet,
    )
    progress(
        f"prefix_token_count={args.prefix_token_count}, max_seq_length={args.max_seq_length}",
        args.quiet,
    )
    progress(f"device={device}, dtype={args.dtype}", args.quiet)

    from transformers import AutoTokenizer, Siglip2ImageProcessor

    from fgclip2.model.strcs.caption_decoder import LLMCaptionDecoder
    from fgclip2.model.strcs.configuration_fgclip2 import Fgclip2Config
    from fgclip2.model.strcs.fgclip2 import FG_CLIP2_Model

    progress("Loading data store", args.quiet)
    store = build_data_store(args.data_path)
    progress(f"Loaded data store with {len(store)} records", args.quiet)

    progress("Loading LLM tokenizer", args.quiet)
    tokenizer = AutoTokenizer.from_pretrained(args.llm_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    progress(f"Loaded LLM tokenizer; pad_token_id={tokenizer.pad_token_id}", args.quiet)

    progress("Loading SigLIP2 image processor", args.quiet)
    image_processor = Siglip2ImageProcessor.from_pretrained(args.base_model)

    progress("Loading FG-CLIP vision checkpoint", args.quiet)
    config = Fgclip2Config.from_pretrained(args.vision_encoder_path)
    config.enable_region_heads = False
    fgclip_model = FG_CLIP2_Model.from_pretrained(
        args.vision_encoder_path, config=config
    )
    fgclip_model.eval().to(device=device, dtype=dtype)
    progress("Loaded FG-CLIP vision checkpoint", args.quiet)

    progress("Loading LLM caption decoder", args.quiet)
    caption_decoder = LLMCaptionDecoder(
        vis_hidden_dim=fgclip_model.config.vision_config.hidden_size,
        llm_model_path=args.llm_model_path,
        caption_pool_2x2_tokens=args.caption_pool_2x2_tokens,
    )
    progress("Loading projector weights", args.quiet)
    load_projector_into_decoder(caption_decoder, args.projector_path)
    caption_decoder.eval().to(device=device, dtype=dtype)
    progress("Loaded LLM caption decoder and projector", args.quiet)

    output_dir = os.path.dirname(args.output_log)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    written = 0
    indices = select_indices(len(store), args.num_samples, args.start_index, args.seed)
    progress(f"Selected {len(indices)} sample indices: {indices}", args.quiet)
    with open(args.output_log, "w", encoding="utf-8") as log_file:
        for sample_number, index in enumerate(indices, start=1):
            item = store[index]
            is_cn = bool(item.get("is_cn", False))
            image_path = get_image_path(item)
            resolved_image_path = resolve_image_path(
                image_path, args.image_folder, args.cn_image_folder, is_cn
            )
            reference_caption = IMAGE_TOKEN_PATTERN.sub("", get_caption(item)).strip()
            short_caption = item.get("short_caption")
            caption_prefix = args.prompt
            progress(
                f"Processing sample {sample_number}/{len(indices)}: index={index}, image={resolved_image_path}",
                args.quiet,
            )
            progress(f"Caption prefix: {caption_prefix!r}", args.quiet)

            try:
                image = Image.open(resolved_image_path)
                width, height = image.size
                progress(f"Opened image: width={width}, height={height}", args.quiet)
                if args.max_image_pixels > 0 and width * height > args.max_image_pixels:
                    raise ValueError(
                        f"image has {width * height} pixels, max_image_pixels={args.max_image_pixels}"
                    )
                image = image.convert("RGB")
            except Exception as exc:
                record = {
                    "id": item.get("id"),
                    "index": index,
                    "image_path": image_path,
                    "resolved_image_path": resolved_image_path,
                    "reference_caption": reference_caption,
                    "short_caption": short_caption,
                    "caption_prefix": caption_prefix,
                    "prompt": caption_prefix,
                    "generated_caption": None,
                    "reference_caption_loss": None,
                    "reference_caption_ppl": None,
                    "error": repr(exc),
                }
                log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                progress(f"Skipped sample index={index}: {exc!r}", args.quiet)
                continue

            with torch.inference_mode():
                (
                    inputs_embeds,
                    attention_mask,
                    prompt_token_count,
                    prompt_text,
                    max_num_patches,
                ) = build_generation_inputs(
                    vision_model=fgclip_model.vision_model,
                    caption_decoder=caption_decoder,
                    image_processor=image_processor,
                    tokenizer=tokenizer,
                    image=image,
                    prompt=caption_prefix,
                    device=device,
                    use_chat_template=args.use_chat_template,
                    enable_thinking=args.enable_thinking,
                )
                reference_caption_loss, reference_caption_ppl = compute_reference_caption_ppl(
                    vision_model=fgclip_model.vision_model,
                    caption_decoder=caption_decoder,
                    image_processor=image_processor,
                    tokenizer=tokenizer,
                    image=image,
                    reference_caption=reference_caption,
                    prompt=caption_prefix,
                    device=device,
                    max_seq_length=args.max_seq_length,
                    use_chat_template=args.use_chat_template,
                    enable_thinking=args.enable_thinking,
                )
                generated_caption = generate_caption(
                    caption_decoder=caption_decoder,
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    prompt_token_count=prompt_token_count,
                    tokenizer=tokenizer,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
            progress(
                f"Finished sample {sample_number}/{len(indices)}: index={index}, "
                f"ppl={reference_caption_ppl:.4f}, generated={generated_caption[:120]!r}",
                args.quiet,
            )

            record = {
                "id": item.get("id"),
                "index": index,
                "image_path": image_path,
                "resolved_image_path": resolved_image_path,
                "width": image.width,
                "height": image.height,
                "max_num_patches": max_num_patches,
                "reference_caption": reference_caption,
                "short_caption": short_caption,
                "caption_prefix": caption_prefix,
                "prompt": caption_prefix,
                "prompt_text": prompt_text,
                "generated_caption": generated_caption,
                "reference_caption_loss": reference_caption_loss,
                "reference_caption_ppl": reference_caption_ppl,
            }
            log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    progress(f"Wrote {written} generated caption records to {args.output_log}", args.quiet)


if __name__ == "__main__":
    main()
