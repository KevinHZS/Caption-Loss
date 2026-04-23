import os
os.environ["HF_HOME"] = "/gemini/space/gjx/hf_cache"

import torch
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoTokenizer,
    AutoModelForCausalLM,
)

model_root = "qihoo360/fg-clip2-base"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(
    model_root,
    trust_remote_code=True
).to(device)

tokenizer = AutoTokenizer.from_pretrained(model_root)
image_processor = AutoImageProcessor.from_pretrained(model_root)

print("device:", device)
print("hf cache:", os.environ["HF_HOME"])