# L_caption 损失实现文档

## 概述

为 FG-CLIP2 Stage 2 训练添加自回归 Caption 损失（L_caption），参考 OpenVision/docs/autoregressive_training.md 的 CoCa 范式，采用方案 A（learnable tokens 作为 query）。

---

## 设计决策

| 决策项 | 选择 | 原因 |
|---|---|---|
| Decoder query | learnable tokens（方案 A） | 不依赖文本输入，与 CoCa 论文一致 |
| K/V 来源 | concat(image_patch_tokens, text_token_embs) | 同时利用视觉和文本信息 |
| text_token_embs 来源 | 复用 `long_text_outputs.last_hidden_state`（方案 X） | 避免额外跑一次 text_model |
| 文本目标 | long caption（196 tokens） | 提供更丰富的监督信号 |
| Tokenizer | 沿用 SigLIP2 tokenizer | 无需引入新依赖 |
| caption_loss_weight | 可配置参数，默认 0.0 | 方便消融实验 |
| 方案 C | TODO，暂不实现 | text tokens 直接作为 query |

### Tokenizer 特殊 token

| token | id |
|---|---|
| pad_token_id | 1 |
| bos_token_id | 49406 |
| eos_token_id | 49407 |
| vocab_size | 32000 |

---

## 文件改动

### 1. 新建 `fgclip2/model/strcs/caption_decoder.py`

实现 CoCa 风格的文本解码器。

**架构**：

```
CaptionDecoder
├── image_proj: Linear(768, 768)        # image patch tokens 投影
├── text_proj:  Linear(768, 768)        # text token embs 投影
├── learnable_tokens: Parameter[196, 768]
├── layers: ModuleList[CaptionDecoderLayer × 6]
│   ├── self_attn: MultiheadAttention   # causal mask
│   ├── cross_attn: MultiheadAttention  # attend to image+text kv
│   ├── norm1, norm2, norm3: LayerNorm
│   └── ffn: Linear → GELU → Linear
└── lm_head: Linear(768, 32000)
```

**前向流程**：

```
image_patch_tokens [B, N, 768]  →  image_proj  →  img_kv [B, N, 768]
text_token_embs    [B, L, 768]  →  text_proj   →  txt_kv [B, L, 768]
                                                    kv = concat([img_kv, txt_kv])  [B, N+L, 768]

kv_key_padding_mask = ~concat([pixel_attention_mask, ones(B,L)])  # True=ignore padding

learnable_tokens.expand(B, 196, 768)  →  q

for each layer:
    q = causal_self_attn(q)
    q = cross_attn(q, kv, key_padding_mask=kv_key_padding_mask)
    q = ffn(q)

logits = lm_head(q)  →  [B, 196, 32000]
```

**NaFlex padding 处理**：`pixel_attention_mask`（1=valid, 0=pad）取反后作为 cross-attention 的 `key_padding_mask`，屏蔽 image patch 的 padding 位置。

---

### 2. 修改 `fgclip2/model/strcs/fgclip2.py`

**改动 1**：新增 import

```python
from .caption_decoder import CaptionDecoder
```

**改动 2**：`FG_CLIP2_Model.__init__` 新增

```python
self.caption_decoder = CaptionDecoder(
    hidden_dim=text_config.hidden_size,        # 768
    num_layers=text_config.num_hidden_layers,  # 12
    num_heads=text_config.num_attention_heads, # 12
    num_learnable_tokens=196,
    vocab_size=text_config.vocab_size,         # 32000
)
self.caption_loss_weight = 0.0  # 默认不启用，由训练脚本设置
```

**改动 3**：`forward` 新增参数

```python
caption_labels: Optional[torch.LongTensor] = None,  # [B, 196]，shifted left
```

**改动 4**：`forward` 中 caption loss 计算（在 `loss = loss_long + loss_short` 之后）

```python
if text_long is not None:
    loss = loss_long + loss_short
    if self.caption_loss_weight > 0.0 and caption_labels is not None:
        caption_logits = self.caption_decoder(
            image_patch_tokens=vision_outputs.last_hidden_state,
            text_token_embs=long_text_outputs.last_hidden_state,
            pixel_attention_mask=pixel_attention_mask,
        )
        loss_caption = F.cross_entropy(
            caption_logits.reshape(-1, caption_logits.shape[-1]),
            caption_labels.reshape(-1),
            ignore_index=1,  # pad_token_id=1
        )
        loss = loss + self.caption_loss_weight * loss_caption
```

**梯度流向**：`L_caption` → `lm_head` → `CaptionDecoderLayer.cross_attn`（K/V 路径）→ `image_proj(vision_outputs.last_hidden_state)` → vision encoder 参数更新。

---

### 3. 修改 `fgclip2/train/train_fgclip2.py`

**改动 1**：`DataArguments` 新增字段

```python
caption_loss_weight: float = field(default=0.0)
```

**改动 2**：`LazySupervisedBboxDataset.__init__` 新增

```python
self.caption_loss_weight = data_args.caption_loss_weight
```

**改动 3**：`LazySupervisedBboxDataset.__getitem__` 新增

```python
if self.caption_loss_weight > 0.0:
    # text 字段为 [BOS, t1, t2, ..., EOS, PAD]，长度 max_seq_length
    # caption_labels: shifted left → [t1, t2, ..., EOS, PAD, PAD]
    caption_labels = torch.cat([text[0, 1:], torch.tensor([1], dtype=text.dtype)])
    data_dict['caption_labels'] = caption_labels.unsqueeze(0)  # [1, max_seq_length]
```

注意：`caption_labels` 直接从已有的 `text`（long caption token ids）构造，不额外 tokenize。

**改动 4**：`DataCollatorForSupervisedDataset.__call__` 新增

```python
if 'caption_labels' in instances[0]:
    batch['caption_labels'] = torch.cat(
        [instance['caption_labels'] for instance in instances], dim=0
    )
```

**改动 5**：`train()` 函数新增

```python
model.caption_loss_weight = data_args.caption_loss_weight
```

---

### 4. 修改 `scripts/train/stage2_fgclip2.sh`

新增参数：

```bash
--caption_loss_weight 2.0 \
```

---

## 总损失公式

```
total_loss = loss_short + loss_long                          # SigLIP 全局对比损失
           + 0.2 * loss_bbox_itcl + 0.1 * loss_bbox_rcc    # box 级对比损失
           + 0.5 * loss_bbox_hitc                           # hard negative 损失
           + caption_loss_weight * loss_caption             # 自回归 caption 损失（新增）
```

---

## 验证点

| 验证点 | 说明 |
|---|---|
| `caption_loss_weight=0.0` 时零开销 | `self.caption_loss_weight > 0.0` 条件短路，decoder 不执行 |
| `long_text_outputs` 作用域安全 | caption loss 在 `text_long is not None` 分支内，变量一定存在 |
| `caption_labels` 与 `caption_logits` 长度对齐 | 均为 196（`max_seq_length = num_learnable_tokens`） |
| NaFlex padding 屏蔽 | `pixel_attention_mask` 传入 cross-attention `key_padding_mask` |
| `ignore_index=1` | 与 `pad_token_id=1` 一致，PAD 位置不计入 loss |
| DeepSpeed ZeRO-2 兼容 | `CaptionDecoder` 为标准 `nn.Module`，无特殊分布式依赖 |

---

## TODO

**方案 C**：用 long caption text token embeddings 直接作为 decoder query，去掉 learnable tokens，decoder 退化为标准 encoder-decoder cross-attention。适合后续对比实验。
