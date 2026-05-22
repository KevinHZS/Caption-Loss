from __future__ import annotations

import unittest

from fgclip2.train.caption_label_utils import build_caption_lm_inputs


class Encoded:
    def __init__(self, ids):
        self.input_ids = [ids]
        self.attention_mask = [[1] * len(ids)]


class ToyTokenizer:
    chat_template = "toy"
    eos_token = "<eos>"
    pad_token_id = 0

    def __init__(self):
        self.vocab = {
            "<pad>": 0,
            "<eos>": 1,
            "<user>": 2,
            "<assistant>": 3,
            "Describe": 4,
            "the": 5,
            "image": 6,
            "A": 7,
            "cat": 8,
            "sleeps": 9,
        }

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    ):
        assert not tokenize
        assert add_generation_prompt
        assert enable_thinking is False
        return "<user> " + messages[-1]["content"] + " <assistant>"

    def __call__(
        self,
        text,
        add_special_tokens=False,
        max_length=None,
        padding=False,
        truncation=False,
        return_tensors=None,
    ):
        assert add_special_tokens is False
        ids = [self.vocab[token] for token in text.split()]
        if truncation and max_length is not None:
            ids = ids[:max_length]
        if padding == "max_length" and max_length is not None:
            ids = ids + [self.pad_token_id] * (max_length - len(ids))
        return Encoded(ids)


class CaptionLabelUtilsTest(unittest.TestCase):
    def test_build_caption_lm_inputs_masks_prompt_tokens_from_loss(self):
        tokenizer = ToyTokenizer()

        enc = build_caption_lm_inputs(
            tokenizer=tokenizer,
            caption="A cat sleeps",
            max_length=10,
            prompt="Describe the image",
            use_chat_template=True,
            enable_thinking=False,
        )

        self.assertEqual(enc.input_ids.tolist(), [[2, 4, 5, 6, 3, 7, 8, 9, 1, 0]])
        self.assertEqual(enc.attention_mask.tolist(), [[1, 1, 1, 1, 1, 1, 1, 1, 1, 0]])
        self.assertEqual(enc.labels.tolist(), [[-100, -100, -100, -100, -100, 7, 8, 9, 1, -100]])


if __name__ == "__main__":
    unittest.main()
