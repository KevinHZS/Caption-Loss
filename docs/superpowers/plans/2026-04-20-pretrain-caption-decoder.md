# Plan: Pretrain Caption Decoder (Plan A)

**Goal:** 冻结 vision encoder 和 text encoder，只训练 CaptionDecoder，验证 decoder 架构正确性。

---

## File Changes

| 文件 | 操作 |
|---|---|
| `fgclip2/train/train_fgclip2.py` | 新增 `pretrain_caption_decoder` 参数；冻结 encoder |
| `fgclip2/model/strcs/fgclip2.py` | `forward` 支持 pretrain 模式：跳过对比损失，只算 caption loss |
| `scripts/train/stage2_fgclip2_pretrain_decoder.sh` | 新建训练脚本 |

---

## Step 1: `DataArguments` 新增 `pretrain_caption_decoder`

**文件**: `fgclip2/train/train_fgclip2.py`

在 `caption_loss_weight` 字段后添加：

```python
pretrain_caption_decoder: bool = field(default=False)
```

在 `train()` 的 `model.caption_loss_weight = ...` 之后添加：

```python
model.pretrain_caption_decoder = data_args.pretrain_caption_decoder
if data_args.pretrain_caption_decoder:
    for name, param in model.named_parameters():
        if 'caption_decoder' not in name:
            param.requires_grad_(False)
```

---

## Step 2: `FG_CLIP2_Model.__init__` 初始化属性

**文件**: `fgclip2/model/strcs/fgclip2.py`

在 `self.caption_loss_weight = 0.0` 之后添加：

```python
self.pretrain_caption_decoder = False
```

---

## Step 3: `forward` 支持 pretrain 模式

**文件**: `fgclip2/model/strcs/fgclip2.py`

在计算 `logit_scale` 之前插入分支（`long_text_outputs` 已在此前计算）：

```python
if self.pretrain_caption_decoder:
    assert text_long is not None, "pretrain_caption_decoder requires text_long"
    assert caption_labels is not None, "pretrain_caption_decoder requires caption_labels"
    caption_logits = self.caption_decoder(
        image_patch_tokens=vision_outputs.last_hidden_state.detach(),
        text_token_embs=long_text_outputs.last_hidden_state.detach(),
        pixel_attention_mask=pixel_attention_mask,
    )
    loss = F.cross_entropy(
        caption_logits.reshape(-1, caption_logits.shape[-1]),
        caption_labels.reshape(-1),
        ignore_index=1,
    )
    return Fgclip2Output(loss=loss)
```

注意：`.detach()` 是双重保险（参数已冻结），确保梯度不流回 encoder。

---

## Step 4: 新建训练脚本

**文件**: `scripts/train/stage2_fgclip2_pretrain_decoder.sh`

关键差异（相对于 `stage2_fgclip2.sh`）：
- `--model_name_or_path`：指向已完成对比训练的 checkpoint
- `--from_siglip2 False`：不重新初始化权重
- `--add_box_loss False` / `--use_hard_neg False`：不需要 box loss
- `--learning_rate 1e-4`：decoder 从随机初始化，需要更大学习率
- `--caption_loss_weight 1.0`
- `--pretrain_caption_decoder True`

---

## 验证点

| 验证点 | 预期 |
|---|---|
| 只有 decoder 参数可训练 | `requires_grad=True` 的参数只含 `caption_decoder.*` |
| 对比损失不计算 | `pretrain_caption_decoder=True` 时直接 return，不进入 SigLIP loss |
| caption loss 初始值合理 | ≈ log(32000) ≈ 10.4 |
| caption loss 正常下降 | 数千步内降至 3-5 |