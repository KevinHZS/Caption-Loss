# LLM Caption Decoder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用冻结的 LLM（如 Qwen3-1.7B）替换原有 CaptionDecoder，通过 MLP projector 将 vision tokens 投影到 LLM 空间，计算 caption cross-entropy loss 训练 vision encoder 和 projector。

**Architecture:**
- Vision encoder → `last_hidden_state` [B, N, D_vis]
- MLP projector: D_vis → D_llm（从 LLM config 动态读取）
- 冻结 LLM：`[vision_tokens | llm_caption_tokens]` 作为输入，对 caption 部分计算 cross-entropy loss

**Tech Stack:** PyTorch, HuggingFace Transformers, DeepSpeed ZeRO-2

---

## File Changes

| 文件 | 操作 |
|---|---|
| `fgclip2/model/strcs/caption_decoder.py` | 完全替换为 `LLMCaptionDecoder` |
| `fgclip2/model/strcs/fgclip2.py` | 替换 CaptionDecoder 实例化；删除 `detach_caption_decoder`；修改 forward caption loss 分支 |
| `fgclip2/train/train_fgclip2.py` | 新增 `llm_model_path` 参数；删除 `detach_caption_decoder`；dataset 加 LLM tokenizer caption labels |
| `scripts/train/stage2_fgclip2_pretrain_decoder.sh` | 删除 `detach_caption_decoder`；加 `llm_model_path` |
| `scripts/train/stage2_fgclip2_joint.sh` | 加 `llm_model_path` |

---

## Task 1: 替换 caption_decoder.py

**Files:**
- Modify: `fgclip2/model/strcs/caption_decoder.py`

- [ ] **Step 1: 完全替换文件内容**

```python
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig


class LLMCaptionDecoder(nn.Module):
    """
    LLM-based caption decoder.
    Vision tokens are projected via MLP to LLM hidden dim,
    then prepended to caption token embeddings as input to a frozen LLM.
    Only the MLP projector is trainable; LLM is always frozen.
    """

    def __init__(self, vis_hidden_dim: int, llm_model_path: str):
        super().__init__()
        llm_config = AutoConfig.from_pretrained(llm_model_path)
        llm_hidden_dim = llm_config.hidden_size

        self.projector = nn.Sequential(
            nn.Linear(vis_hidden_dim, llm_hidden_dim),
            nn.GELU(),
            nn.Linear(llm_hidden_dim, llm_hidden_dim),
        )

        self.llm = AutoModelForCausalLM.from_pretrained(llm_model_path)
        for param in self.llm.parameters():
            param.requires_grad_(False)

    def forward(
        self,
        image_patch_tokens: torch.Tensor,   # [B, N, D_vis]
        pixel_attention_mask: torch.Tensor, # [B, N], 1=valid 0=pad
        llm_input_ids: torch.Tensor,        # [B, L]
        llm_attention_mask: torch.Tensor,   # [B, L]
    ) -> torch.Tensor:                      # [B, N+L, vocab_size]
        B, N, _ = image_patch_tokens.shape
        device = image_patch_tokens.device

        # project vision tokens to LLM space
        vis_embeds = self.projector(image_patch_tokens)  # [B, N, D_llm]

        # get LLM text token embeddings
        txt_embeds = self.llm.model.embed_tokens(llm_input_ids)  # [B, L, D_llm]

        # concat: [vision_tokens | text_tokens]
        inputs_embeds = torch.cat([vis_embeds, txt_embeds], dim=1)  # [B, N+L, D_llm]

        # attention mask: vision mask + text mask
        attention_mask = torch.cat([pixel_attention_mask, llm_attention_mask], dim=1)  # [B, N+L]

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        return outputs.logits  # [B, N+L, vocab_size]
```

- [ ] **Step 2: 验证语法**

```bash
cd /Users/zyf/FG-CLIP && python -c "from fgclip2.model.strcs.caption_decoder import LLMCaptionDecoder; print('OK')"
```

---

## Task 2: 修改 fgclip2.py

**Files:**
- Modify: `fgclip2/model/strcs/fgclip2.py`

- [ ] **Step 1: 修改 import**

将：
```python
from .caption_decoder import CaptionDecoder
```
改为：
```python
from .caption_decoder import LLMCaptionDecoder
```

- [ ] **Step 2: 修改 `__init__` 中的实例化**

将：
```python
self.caption_decoder = CaptionDecoder(
    hidden_dim=text_config.hidden_size,
    num_layers=text_config.num_hidden_layers,
    num_heads=text_config.num_attention_heads,
    num_learnable_tokens=196,
    vocab_size=text_config.vocab_size,
)
```
改为：
```python
self.caption_decoder = None  # 由 train() 在 from_pretrained 后设置
```

并删除 `self.detach_caption_decoder = False`，保留 `self.caption_loss_weight = 0.0`。

- [ ] **Step 3: 修改 forward 中的 caption loss 分支**

将：
```python
if self.caption_loss_weight > 0.0 and caption_labels is not None:
    img_tokens = vision_outputs.last_hidden_state.detach() if self.detach_caption_decoder else vision_outputs.last_hidden_state
    txt_tokens = long_text_outputs.last_hidden_state.detach() if self.detach_caption_decoder else long_text_outputs.last_hidden_state
    print(f"[DEBUG] caption decoder detach={self.detach_caption_decoder}, img_tokens shape: {img_tokens.shape}, txt_tokens shape: {txt_tokens.shape}")
    caption_logits = self.caption_decoder(
        image_patch_tokens=img_tokens,
        text_token_embs=txt_tokens,
        pixel_attention_mask=pixel_attention_mask,
    )
    loss_caption = F.cross_entropy(
        caption_logits.reshape(-1, caption_logits.shape[-1]),
        caption_labels.reshape(-1),
        ignore_index=1,
    )
    print(f"[DEBUG] caption loss: {loss_caption.item():.4f}, weight: {self.caption_loss_weight}")
    loss = loss + self.caption_loss_weight * loss_caption
```

改为：
```python
if self.caption_loss_weight > 0.0 and self.caption_decoder is not None and caption_labels is not None:
    N = vision_outputs.last_hidden_state.shape[1]
    caption_logits = self.caption_decoder(
        image_patch_tokens=vision_outputs.last_hidden_state,
        pixel_attention_mask=pixel_attention_mask,
        llm_input_ids=llm_input_ids,
        llm_attention_mask=llm_attention_mask,
    )
    # 只对 text 部分（后 L 个位置）计算 loss，vision 位置 label=-100
    text_logits = caption_logits[:, N:, :]  # [B, L, vocab_size]
    loss_caption = F.cross_entropy(
        text_logits.reshape(-1, text_logits.shape[-1]),
        caption_labels.reshape(-1),
        ignore_index=-100,
    )
    print(f"[DEBUG] caption loss: {loss_caption.item():.4f}, weight: {self.caption_loss_weight}")
    loss = loss + self.caption_loss_weight * loss_caption
```

- [ ] **Step 4: 修改 forward 签名，加入 llm_input_ids 和 llm_attention_mask**

在 forward 参数列表中加入：
```python
llm_input_ids: Optional[torch.LongTensor] = None,
llm_attention_mask: Optional[torch.Tensor] = None,
```

---

## Task 3: 修改 train_fgclip2.py

**Files:**
- Modify: `fgclip2/train/train_fgclip2.py`

- [ ] **Step 1: DataArguments 加 llm_model_path，删除 detach_caption_decoder**

将：
```python
    detach_caption_decoder: bool = field(default=False)
```
改为：
```python
    llm_model_path: Optional[str] = field(default=None)
```

- [ ] **Step 2: TrainingArguments 删除 caption_decoder_lr（移到 DataArguments 不合适，保留在 TrainingArguments）**

保持不变。

- [ ] **Step 3: LazySupervisedBboxDataset.__init__ 加 LLM tokenizer**

在 `__init__` 参数中加 `llm_tokenizer=None`，并保存：
```python
self.llm_tokenizer = llm_tokenizer
```

- [ ] **Step 4: __getitem__ 加 llm caption labels**

将：
```python
if self.caption_loss_weight > 0.0:
    caption_labels = torch.cat([text[0, 1:], torch.tensor([1], dtype=text.dtype)])
    data_dict['caption_labels'] = caption_labels.unsqueeze(0)
```
改为：
```python
if self.caption_loss_weight > 0.0 and self.llm_tokenizer is not None:
    llm_enc = self.llm_tokenizer(
        caption.lower(),
        max_length=self.max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    llm_input_ids = llm_enc.input_ids  # [1, L]
    llm_attention_mask = llm_enc.attention_mask  # [1, L]
    # labels: shift left, pad=-100
    labels = llm_input_ids[0, 1:].clone()
    labels = torch.cat([labels, torch.tensor([-100], dtype=labels.dtype)])
    data_dict['llm_input_ids'] = llm_input_ids
    data_dict['llm_attention_mask'] = llm_attention_mask
    data_dict['caption_labels'] = labels.unsqueeze(0)
```

- [ ] **Step 5: DataCollatorForSupervisedDataset.__call__ 加 llm 字段 collate**

在 `caption_labels` collate 后加：
```python
if 'llm_input_ids' in instances[0]:
    batch['llm_input_ids'] = torch.cat([inst['llm_input_ids'] for inst in instances], dim=0)
    batch['llm_attention_mask'] = torch.cat([inst['llm_attention_mask'] for inst in instances], dim=0)
```

- [ ] **Step 6: train() 加载 LLM tokenizer，构建 LLMCaptionDecoder，删除 detach_caption_decoder**

在 `model.caption_loss_weight = ...` 后加：
```python
if data_args.llm_model_path is not None and data_args.caption_loss_weight > 0.0:
    from fgclip2.model.strcs.caption_decoder import LLMCaptionDecoder
    llm_caption_decoder = LLMCaptionDecoder(
        vis_hidden_dim=model.config.vision_config.hidden_size,
        llm_model_path=data_args.llm_model_path,
    )
    model.caption_decoder = llm_caption_decoder
    llm_tokenizer = AutoTokenizer.from_pretrained(data_args.llm_model_path)
    print(f"[DEBUG] LLMCaptionDecoder loaded from {data_args.llm_model_path}")
else:
    llm_tokenizer = None
```

删除：
```python
model.detach_caption_decoder = data_args.detach_caption_decoder
print(f"[DEBUG] caption_loss_weight=..., detach_caption_decoder=...")
```

改为：
```python
print(f"[DEBUG] caption_loss_weight={data_args.caption_loss_weight}, llm_model_path={data_args.llm_model_path}")
```

并将 `make_supervised_data_module` 调用改为传入 `llm_tokenizer`：
```python
data_module = make_supervised_data_module(
    data_args=data_args,
    img_preprocess=image_processor,
    tokenizer=tokenizer,
    llm_tokenizer=llm_tokenizer,
    is_naflex=training_args.naflex_train,
)
```

- [ ] **Step 7: make_supervised_data_module 加 llm_tokenizer 参数**

```python
def make_supervised_data_module(data_args, img_preprocess, tokenizer, llm_tokenizer=None, is_naflex=False):
    train_dataset = LazySupervisedBboxDataset(
        data_path=data_args.data_path,
        data_args=data_args,
        img_preprocess=img_preprocess,
        tokenizer=tokenizer,
        llm_tokenizer=llm_tokenizer,
    )
    data_collator = DataCollatorForSupervisedDataset(preprocess=img_preprocess, is_naflex=is_naflex)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)
```

---

## Task 4: 更新训练脚本

**Files:**
- Modify: `scripts/train/stage2_fgclip2_pretrain_decoder.sh`
- Modify: `scripts/train/stage2_fgclip2_joint.sh`

- [ ] **Step 1: stage2_fgclip2_pretrain_decoder.sh**

删除 `--detach_caption_decoder True \`，加入：
```bash
LLM_MODEL_PATH="/path/to/local/qwen3-1.7b"
```
并在 deepspeed 命令中加：
```bash
    --llm_model_path $LLM_MODEL_PATH \
```

- [ ] **Step 2: stage2_fgclip2_joint.sh**

同样加入 `--llm_model_path $LLM_MODEL_PATH \`。

---

## 验证点

| 验证点 | 预期 |
|---|---|
| LLM 参数全部冻结 | `requires_grad=False` for all `caption_decoder.llm.*` |
| projector 可训练 | `requires_grad=True` for `caption_decoder.projector.*` |
| caption loss 只在 text 位置计算 | logits[:, N:, :] 对应 caption_labels |
| vision encoder 梯度正常流入 | loss.backward() 后 vision_model 参数有 grad |
| caption loss 初始值合理 | ≈ log(151936) ≈ 11.9 |