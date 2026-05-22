from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch


DEFAULT_LLM_CAPTION_PROMPT = "Describe the image in one detailed sentence."


@dataclass
class CaptionLmInputs:
    input_ids: torch.LongTensor
    attention_mask: torch.LongTensor
    labels: torch.LongTensor
    prompt_token_count: int


def _as_token_ids(encoded) -> List[int]:
    input_ids = encoded.input_ids
    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.tolist()
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    return [int(token_id) for token_id in input_ids]


def build_caption_prompt(
    tokenizer,
    prompt: str = DEFAULT_LLM_CAPTION_PROMPT,
    use_chat_template: bool = True,
    enable_thinking: bool = False,
) -> str:
    if use_chat_template and getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": prompt}]
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
    return prompt


def _tokenize_without_special_tokens(tokenizer, text: str) -> List[int]:
    return _as_token_ids(tokenizer(text, add_special_tokens=False))


def build_caption_lm_inputs(
    tokenizer,
    caption: str,
    max_length: int,
    prompt: str = DEFAULT_LLM_CAPTION_PROMPT,
    use_chat_template: bool = True,
    enable_thinking: bool = False,
    append_eos: bool = True,
) -> CaptionLmInputs:
    prompt_text = build_caption_prompt(
        tokenizer=tokenizer,
        prompt=prompt,
        use_chat_template=use_chat_template,
        enable_thinking=enable_thinking,
    )
    prompt_ids = _tokenize_without_special_tokens(tokenizer, prompt_text)

    target_ids = _tokenize_without_special_tokens(tokenizer, caption.strip())
    if append_eos and getattr(tokenizer, "eos_token", None):
        target_ids = target_ids + _tokenize_without_special_tokens(
            tokenizer, tokenizer.eos_token
        )

    input_ids = (prompt_ids + target_ids)[:max_length]
    prompt_token_count = min(len(prompt_ids), max_length)
    label_ids = [-100] * prompt_token_count + input_ids[prompt_token_count:]

    attention_mask = [1] * len(input_ids)
    if len(input_ids) < max_length:
        pad_count = max_length - len(input_ids)
        input_ids = input_ids + [tokenizer.pad_token_id] * pad_count
        attention_mask = attention_mask + [0] * pad_count
        label_ids = label_ids + [-100] * pad_count

    return CaptionLmInputs(
        input_ids=torch.tensor([input_ids], dtype=torch.long),
        attention_mask=torch.tensor([attention_mask], dtype=torch.long),
        labels=torch.tensor([label_ids], dtype=torch.long),
        prompt_token_count=prompt_token_count,
    )
