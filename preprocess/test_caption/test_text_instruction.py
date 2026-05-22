#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional

import torch

DEFAULT_LLM_MODEL_PATH = "/gemini/space/zyf/models/Qwen/Qwen3-1.7B"
# DEFAULT_LLM_MODEL_PATH = "/gemini/space/zyf/models/Qwen/Qwen3-4B"
DEFAULT_OUTPUT_LOG = "/gemini/space/zyf/FG-CLIP/output/test_text_instruction_generation.jsonl"

DEFAULT_TEST_CASES: List[Dict[str, Any]] = [
    {
        "id": "capital_france",
        "prompt": "What is the capital of France?",
    },
    {
        "id": "simple_math",
        "prompt": "What is 17 + 25?",
    },
    {
        "id": "short_definition",
        "prompt": "Explain what photosynthesis is in one sentence.",
    },
    {
        "id": "exact_word",
        "prompt": "Reply with exactly one word: OK",
    },
    {
        "id": "translation",
        "prompt": "Translate 'Good morning' into Spanish.",
    },
    {
        "id": "summarization",
        "prompt": "Summarize this sentence in five words or fewer: The laptop battery lasted all day during meetings and travel.",
    },
    {
        "id": "list_format",
        "prompt": "List three primary colors as a comma-separated list.",
    },
    {
        "id": "comparison",
        "prompt": "Which is larger, 9.11 or 9.9? Briefly explain.",
    },
    {
        "id": "classification",
        "prompt": "Classify the sentiment of this sentence as positive, neutral, or negative: I am satisfied with the result.",
    },
    {
        "id": "rewrite_tone",
        "prompt": "Rewrite this sentence in a polite tone: Send me the report now.",
    },
]


def progress(message: str, quiet: bool = False) -> None:
    if not quiet:
        print(f"[text-instruction-test] {message}", flush=True)


def load_test_cases(path: Optional[str]) -> List[Dict[str, Any]]:
    if path is None:
        return list(DEFAULT_TEST_CASES)

    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON list in {path}")
        return [normalize_case(item, index) for index, item in enumerate(data)]

    if path.endswith(".jsonl"):
        cases = []
        with open(path, "r", encoding="utf-8") as f:
            for index, line in enumerate(f):
                if line.strip():
                    cases.append(normalize_case(json.loads(line), index))
        return cases

    raise ValueError(f"Unsupported prompts path: {path}")


def normalize_case(item: Dict[str, Any], index: int) -> Dict[str, Any]:
    if not isinstance(item, dict):
        raise TypeError(f"Test case #{index} must be a JSON object")
    if not item.get("prompt"):
        raise ValueError(f"Test case #{index} must contain a non-empty prompt")

    case = dict(item)
    case.setdefault("id", f"case_{index}")
    return case


def build_prompt(
    tokenizer,
    prompt: str,
    use_chat_template: bool,
    enable_thinking: bool,
    system_prompt: Optional[str],
) -> str:
    if use_chat_template and getattr(tokenizer, "chat_template", None):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
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


def generate_text(
    model,
    tokenizer,
    prompt_text: str,
    device: torch.device,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> str:
    inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=True).to(device)
    eos_token_id = getattr(model.generation_config, "eos_token_id", None)
    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id

    generation_kwargs = {
        "input_ids": inputs.input_ids,
        "attention_mask": inputs.attention_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": eos_token_id,
        "use_cache": True,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p

    output_ids = model.generate(**generation_kwargs)
    generated_ids = output_ids[0, inputs.input_ids.shape[1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test text-only instruction following of the LLM without image or visual tokens."
    )
    parser.add_argument("--llm-model-path", default=DEFAULT_LLM_MODEL_PATH)
    parser.add_argument(
        "--prompts-path",
        default=None,
        help="Optional JSON/JSONL file. Each record needs prompt and may include id.",
    )
    parser.add_argument("--output-log", default=DEFAULT_OUTPUT_LOG)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Disable tokenizer chat template and feed the raw prompt directly.",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Pass enable_thinking=True to chat templates that support it.",
    )
    parser.add_argument(
        "--system-prompt",
        default="You are a helpful assistant. Answer the user directly.",
        help="Optional system prompt used when chat template is enabled. Use an empty string to disable it.",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
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

    progress("Starting text-only instruction following test", args.quiet)
    progress(f"llm_model_path={args.llm_model_path}", args.quiet)
    progress(f"prompts_path={args.prompts_path}", args.quiet)
    progress(f"output_log={args.output_log}", args.quiet)
    progress(f"use_chat_template={not args.no_chat_template}", args.quiet)
    progress(f"device={device}, dtype={args.dtype}", args.quiet)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    test_cases = load_test_cases(args.prompts_path)
    progress(f"Loaded {len(test_cases)} test cases", args.quiet)

    progress("Loading LLM tokenizer", args.quiet)
    tokenizer = AutoTokenizer.from_pretrained(args.llm_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    progress(f"Loaded tokenizer; pad_token_id={tokenizer.pad_token_id}", args.quiet)

    progress("Loading LLM", args.quiet)
    model = AutoModelForCausalLM.from_pretrained(args.llm_model_path, dtype=dtype)
    model.eval().to(device=device)
    progress("Loaded LLM", args.quiet)

    output_dir = os.path.dirname(args.output_log)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output_log, "w", encoding="utf-8") as log_file:
        for index, case in enumerate(test_cases):
            prompt = case["prompt"]
            prompt_text = build_prompt(
                tokenizer,
                prompt,
                not args.no_chat_template,
                args.enable_thinking,
                args.system_prompt,
            )
            progress(
                f"Processing {index + 1}/{len(test_cases)}: id={case.get('id')}, prompt={prompt!r}",
                args.quiet,
            )

            try:
                with torch.inference_mode():
                    generated_text = generate_text(
                        model=model,
                        tokenizer=tokenizer,
                        prompt_text=prompt_text,
                        device=device,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=args.do_sample,
                        temperature=args.temperature,
                        top_p=args.top_p,
                    )
                record = {
                    "id": case.get("id"),
                    "prompt": prompt,
                    "prompt_text": prompt_text,
                    "generated_text": generated_text,
                    "error": None,
                }
                progress(
                    f"Finished id={case.get('id')}: generated={generated_text[:120]!r}",
                    args.quiet,
                )
            except Exception as exc:
                record = {
                    "id": case.get("id"),
                    "prompt": prompt,
                    "prompt_text": prompt_text,
                    "generated_text": None,
                    "error": repr(exc),
                }
                progress(f"Failed id={case.get('id')}: {exc!r}", args.quiet)

            log_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    progress(
        f"Wrote {len(test_cases)} records to {args.output_log}",
        args.quiet,
    )


if __name__ == "__main__":
    main()
