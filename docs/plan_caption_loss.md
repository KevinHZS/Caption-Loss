# 实现计划：为 FG-CLIP2 添加 L_caption 损失

## 背景与设计决策

参考 OpenVision/docs/autoregressive_training.md，采用 CoCa 范式的方案 A：

- **Query**：`num_learnable_tokens` 个可学习向量（不依赖文本输入）
- **K/V**：`concat(image_patch_tokens, text_token_embs)`，分别经过 `image_proj` 和 `text_proj` 投影到 decoder 空间
- **text_token_embs**：复用 `long_text_outputs.last_hidden_state`（方案 X，不额外跑 text_model）
- **Label**：long caption tokens shifted left（去掉 BOS，末尾补 PAD），mask 掉 `pad_token_id=1` 的位置
- **方案 C**（text tokens 直接作为 query）列为 TODO，暂不实现

### Tokenizer 特殊 token（SigLIP2）

| token | id |
|---|---|
| pad_token_id | 1 |
| bos_token_id | 49406 |
| eos_token_id | 49407 |
| vocab_size | 32000 |

### 关键约束

- NaFlex 模式下 `vision_outputs.last_hidden_state` shape 为 `[B, max_num_patches, 768]`，含 padding，需用 `pixel_attention_mask` 在 cross-attention 中屏蔽
- `text_model` 本身无 causal mask，decoder 的 causal self-attention mask 需在 `CaptionDecoder` 内部实现
- `long_text_outputs` 仅在 `text_long is not None` 时存在，caption loss 的启用条件与此一致

---

## 文件改动汇总

| 文件 | 操作 |
|---|---|
| `fgclip2/model/strcs/caption_decoder.py` | 新建：CaptionDecoder 模块 |
| `fgclip2/model/strcs/fgclip2.py` | 修改：集成 decoder，添加 caption loss |
| `fgclip2/train/train_fgclip2.py` | 修改：DataArguments、Dataset、Collator |
| `scripts/train/stage2_fgclip2.sh` | 修改：新增 caption_loss_weight 参数 |

---

## Step 1：新建 `caption_decoder.py`

**文件**：`fgclip2/model/strcs/caption_decoder.py`

### 模块结构

```
CaptionDecoder(
    hidden_dim=768,
    num_layers=6,
    num_heads=12,
    num_learnable_tokens=196,   # 对应 max_seq_length
    vocab_size=32000,
)
├── image_proj: nn.Linear(768, 768)   # image patch tokens → decoder space
├── text_proj:  nn.Linear(768, 768)   # text token embs → decoder space
├── learnable_tokens: nn.Parameter([num_learnable_tokens, 768])
├── layers: nn.ModuleList of CaptionDecoderLayer × num_layers
│   ├── self_attn: nn.MultiheadAttention (causal mask)
│   ├── cross_attn: nn.MultiheadAttention
│   ├── norm1, norm2, norm3: nn.LayerNorm
│   └── ffn: nn.Sequential(Linear, GELU, Linear)
└── lm_head: nn.Linear(768, vocab_size, bias=False)
```

### forward 签名

```python
def forward(
    self,
    image_patch_tokens: torch.Tensor,      # [B, N_patches, 768]
    text_token_embs: torch.Tensor,         # [B, L, 768]
    pixel_attention_mask: torch.Tensor,    # [B, N_patches]，NaFlex padding mask
) -> torch.Tensor:                         # logits [B, num_learnable_tokens, vocab_size]
```

### 前向流程

```
1. img_kv = image_proj(image_patch_tokens)          [B, N, 768]
2. txt_kv = text_proj(text_token_embs)              [B, L, 768]
3. kv = concat([img_kv, txt_kv], dim=1)             [B, N+L, 768]
4. kv_mask: concat([pixel_attention_mask, ones(B,L)], dim=1)  [B, N+L]
5. q = learnable_tokens.unsqueeze(0).expand(B,-1,-1) [B, T, 768]
6. for each layer:
     q = self_attn(q, q, q, causal_mask=True)
     q = cross_attn(q, kv, kv, key_padding_mask=~kv_mask)
7. logits = lm_head(q)                              [B, T, vocab_size]
```

---

## Step 2：修改 `train_fgclip2.py`

### 2.1 DataArguments 新增字段

```python
caption_loss_weight: float = field(default=0.0)
```

### 2.2 LazySupervisedBboxDataset.__getitem__ 新增返回

仅当 `caption_loss_weight > 0` 时构造：

```python
# caption_input_ids: [BOS, t1, t2, ..., EOS, PAD]，长度 max_seq_length
caption_input_ids = tokenizer(
    [caption.lower()],
    max_length=max_seq_length,
    padding="max_length",
    truncation=True
).input_ids  # 与 text_long 完全相同

# caption_labels: shifted left，[t1, t2, ..., EOS, PAD, PAD]
caption_labels = caption_input_ids[1:] + [pad_token_id]

# caption_loss_mask: 非 pad_token_id 位置为 1
caption_loss_mask = [1 if t != pad_token_id else 0 for t in caption_labels]
```

注意：`caption_input_ids` 与 `text_long`（即 `text` 字段）内容完全相同，
可直接复用 `data_dict['text']`，不需要额外 tokenize。

### 2.3 DataCollatorForSupervisedDataset.__call__ 新增 collate

```python
if caption_loss_weight > 0:
    batch['caption_labels'] = torch.cat([instance['caption_labels'] for instance in instances])
    batch['caption_loss_mask'] = torch.cat([instance['caption_loss_mask'] for instance in instances])
```

---

## Step 3：修改 `fgclip2.py`

### 3.1 __init__ 新增

```python
from .caption_decoder import CaptionDecoder

self.caption_decoder = CaptionDecoder(
    hidden_dim=config.text_config.hidden_size,   # 768
    num_layers=config.text_config.num_hidden_layers,  # 12
    num_heads=config.text_config.num_attention_heads,  # 12
    num_learnable_tokens=196,
    vocab_size=config.text_config.vocab_size,    # 32000
)
```

### 3.2 forward 新增参数

```python
caption_labels: Optional[torch.LongTensor] = None,    # [B, max_seq_length]
caption_loss_mask: Optional[torch.Tensor] = None,     # [B, max_seq_length]
caption_loss_weight: float = 0.0,
```

### 3.3 forward 新增 caption loss 计算

在现有 `loss = loss_short + loss_long` 之后追加：

```python
if caption_loss_weight > 0.0 and caption_labels is not None and text_long is not None:
    # 复用已计算的 long_text_outputs.last_hidden_state（方案 X）
    caption_logits = self.caption_decoder(
        image_patch_tokens=vision_outputs.last_hidden_state,
        text_token_embs=long_text_outputs.last_hidden_state,
        pixel_attention_mask=pixel_attention_mask,
    )
    # caption_logits: [B, num_learnable_tokens, vocab_size]
    # caption_labels: [B, max_seq_length]，两者长度均为 196，对齐
    loss_caption = F.cross_entropy(
        caption_logits.reshape(-1, caption_logits.shape[-1]),
        caption_labels.reshape(-1),
        ignore_index=1,   # pad_token_id=1
    )
    loss = loss + caption_loss_weight * loss_caption
```

---

## Step 4：修改训练脚本

**文件**：`scripts/train/stage2_fgclip2.sh`

新增一行：

```bash
--caption_loss_weight 2.0 \
```

---

## 执行正确性验证点

| 验证点 | 说明 |
|---|---|
| `long_text_outputs` 作用域 | 在 `text_long is not None` 分支内，caption loss 条件与此一致，不会出现未定义变量 |
| `caption_logits` 与 `caption_labels` 长度对齐 | 两者均为 `max_seq_length=196`，`num_learnable_tokens=196` |
| NaFlex padding 屏蔽 | `pixel_attention_mask` 传入 decoder cross-attention 的 `key_padding_mask` |
| `pad_token_id=1` | `F.cross_entropy` 的 `ignore_index=1`，与 tokenizer 配置一致 |
| DeepSpeed ZeRO-2 兼容 | `CaptionDecoder` 为标准 `nn.Module`，无特殊分布式依赖 |
| `caption_loss_weight=0.0` 时零开销 | 条件判断在 forward 最外层，decoder 不执行 |

---

## TODO（方案 C，暂不实现）

用 long caption text token embeddings 直接作为 decoder query，去掉 learnable tokens，
decoder 退化为标准 encoder-decoder cross-attention。适合后续对比实验。