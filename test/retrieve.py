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

def determine_max_value(image):
    w,h = image.size
    max_val = (w//16)*(h//16)
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

img_root = "/gemini/space/gjx/FG-CLIP/use_imgs/cat_dfclor.jpg"
image = Image.open(img_root).convert("RGB")

image_input = image_processor(images=image, max_num_patches=determine_max_value(image), return_tensors="pt").to(device)

# NOTE Short captions: max_length=64 walk_type="short"(default)
# NOTE Long captions: max_length=196 walk_type="long"

captions = [
"一只黑猫",
"一只黑猫和一只白猫，白猫站在电脑上",
"一只黑猫和一只白猫，黑猫站在电脑上",
"一个繁忙的街头市场，摊位上摆满水果，背景是高楼大厦，人们在喧闹中购物。"
]
captions = [caption.lower() for caption in captions]

caption_input = tokenizer(captions, padding="max_length", max_length=64, truncation=True, return_tensors="pt").to(device)


with torch.no_grad():
  image_feature = model.get_image_features(**image_input)
#   text_feature = model.get_text_features(**caption_input,walk_type="long")
  print("input_ids shape:", caption_input["input_ids"].shape)
  print("tokenizer.model_max_length:", tokenizer.model_max_length)
  print("position_embedding size:", model.text_model.embeddings.position_embedding.num_embeddings)
  text_feature = model.get_text_features(**caption_input)
  image_feature = image_feature / image_feature.norm(p=2, dim=-1, keepdim=True)
  text_feature = text_feature / text_feature.norm(p=2, dim=-1, keepdim=True)

logits_per_image = image_feature @ text_feature.T
logit_scale, logit_bias = model.logit_scale.to(text_feature.device), model.logit_bias.to(text_feature.device)
logits_per_image = logits_per_image * logit_scale.exp() + logit_bias
print("logits_per_image:", logits_per_image)