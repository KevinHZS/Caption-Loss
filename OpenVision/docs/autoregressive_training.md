# OpenVision：使用自回归标签训练视觉编码器

## 目录

1. [设计动机：为什么要用自回归标签训练视觉编码器](#1-设计动机)
2. [整体训练范式：CoCa](#2-整体训练范式coca)
3. [数据预处理：自回归标签的生成](#3-数据预处理自回归标签的生成)
4. [模型架构](#4-模型架构)
   - 4.1 [视觉编码器（ViT）](#41-视觉编码器vit)
   - 4.2 [文本编码器](#42-文本编码器)
   - 4.3 [文本解码器（核心）](#43-文本解码器核心)
5. [前向传播：logits 的生成过程](#5-前向传播logits-的生成过程)
6. [损失函数](#6-损失函数)
7. [反向传播：梯度如何流回视觉编码器](#7-反向传播梯度如何流回视觉编码器)
8. [与标准 CLIP 的对比](#8-与标准-clip-的对比)
9. [关键超参数](#9-关键超参数)

---

## 1. 设计动机

标准 CLIP 只使用**对比损失**（Contrastive Loss）训练视觉编码器：让图像嵌入和对应文本嵌入在特征空间中靠近，与其他样本的嵌入远离。这种方式的局限在于：

- 监督信号来自**嵌入空间的相对距离**，信息密度有限
- 视觉编码器只需要输出一个"足够好"的全局嵌入，不需要理解图像的细节语义
- 文本的丰富语义信息（词序、语法结构）没有被充分利用

**自回归 Caption 损失**（Autoregressive Caption Loss）提供了额外的监督信号：

> 给定图像，模型必须逐 token 地生成对应的文字描述。

这要求视觉编码器输出的 patch token 序列必须包含足够丰富的局部语义信息，供文本解码器"读取"并生成准确的文字。这种设计来自 **CoCa**（Contrastive Captioners，Yu et al. 2022）。

---

## 2. 整体训练范式：CoCa

OpenVision 采用 CoCa 范式，同时优化两个损失：

```
总损失 = 1 × L_contrastive + 2 × L_caption
```

| 损失 | 作用 | 监督视觉编码器的方式 |
|------|------|---------------------|
| `L_contrastive` | 图文对齐（全局语义） | 通过全局嵌入 `zimg` 的梯度 |
| `L_caption` | 图像描述生成（局部语义） | 通过 patch token 序列的梯度 |

两个损失共同反向传播，视觉编码器同时接收来自两个方向的梯度更新。

配置文件（`src/configs/openvision.py`）：
```python
config.loss_type = 'coca'
config.clip_loss_weight = 1
config.coca_caption_loss_weight = 2
```

---

## 3. 数据预处理：自回归标签的生成

### 3.1 完整预处理流水线

预处理以字符串形式配置，在 `tf.data` 流水线中按顺序执行：

```
inception_crop(size=84)          # 随机裁剪并缩放到目标分辨率
| simclr_jitter_gray              # 颜色抖动 + 随机灰度化
| my_bert_tokenize(                # BERT tokenizer
    max_len=80,
    vocab_path='assets/bert_base_vocab_bos_eos.txt',
    inkey='text',
    outkey='labels_for_regress'
  )
| get_autoreg_label(pad_token=0)  # 生成自回归标签
| keep(image, labels, labels_for_regress, autoreg_labels, cap_loss_mask)
```

### 3.2 BERT Tokenize：生成 `labels_for_regress`

`my_bert_tokenize` 将原始文本转换为 token id 序列，并在首尾添加特殊 token：

```
原始文本:         "a dog running in the park"
tokenize 后:      [BOS, 1045, 3899, 2770, 1999, 1996, 2380, EOS, PAD, PAD, ...]
                   ↑                                          ↑
                  token=101                               token=102
```

- 序列长度固定为 `token_len=80`
- `BOS`（Beginning of Sentence）= token id 101
- `EOS`（End of Sentence）= token id 102
- `PAD`（Padding）= token id 0

这个序列存储在 `labels_for_regress` 字段中，同时也作为文本编码器的输入（存储在 `labels` 字段）。

### 3.3 `get_autoreg_label`：生成自回归训练目标

**文件**：`src/transforms/ops_text.py:40-51`

```python
@Registry.register("preprocess_ops.get_autoreg_label")
@InKeyOutKey(indefault="labels_for_regress", outdefault="autoreg_labels")
def get_pp_autoreg_label(pad_token):
    def _pp_autoreg_label(label):
        shift_label = label[1:]                              # 去掉 BOS token
        shift_label = tf.concat([shift_label,
                                  tf.constant([pad_token])], # 末尾补 PAD
                                 axis=0)
        return shift_label
    return _pp_autoreg_label
```

这是标准的 **teacher-forcing** 自回归标签构造方式——将输入序列向左移一位：

```
labels_for_regress: [BOS, t1,  t2,  t3,  EOS, PAD, PAD, ...]  ← 文本编码器输入
autoreg_labels:     [t1,  t2,  t3,  EOS, PAD, PAD, PAD, ...]  ← 解码器预测目标
                     ↑
                    位置 0 的预测目标是 t1（即给定 BOS，预测第一个词）
```

**语义**：`autoreg_labels[i]` 是模型在看到前 `i` 个 token 后，应该预测的下一个 token。

### 3.4 `cap_loss_mask`：损失掩码

`cap_loss_mask` 标记哪些位置需要计算损失（非 PAD 位置），避免 PAD token 影响梯度：

```
autoreg_labels: [t1,  t2,  t3,  EOS, PAD, PAD, ...]
cap_loss_mask:  [1,   1,   1,   1,   0,   0,   ...]
```

---

## 4. 模型架构

### 4.1 视觉编码器（ViT）

**文件**：`src/models/vit.py`

ViT 将图像分割为 patch，经过 Transformer 编码后输出**两种表示**：

```python
# output_tokens=True 时同时返回全局嵌入和 patch token 序列
zimg, image_embs = image_model(image, train=train)
#      ↑                ↑
#  全局嵌入          patch token 序列
# shape: [B, D]    shape: [B, N_patches, D]
```

- `zimg`：经过 GAP 池化 + L2 归一化的全局嵌入，用于对比损失
- `image_embs`：所有 patch 的 token 序列（含 CLS token），用于文本解码器的交叉注意力

对于 84px 分辨率、patch size=16 的 ViT-S/16：
- patch 数量 = (84/16)² ≈ 25 个 patch（加 CLS token 共 26 个）

### 4.2 文本编码器

**文件**：`src/models/text_transformer.py`

文本编码器接收 `labels`（含 BOS 的完整 token 序列），输出：

```python
ztxt, token_embs = text_model(text)
#      ↑               ↑
#  全局嵌入         token 序列
# shape: [B, D]   shape: [B, L, D]
```

**关键细节**：在 CoCa 训练中，batch 被分为两半：

```python
# two_towers.py:97
# 训练时只取前半个 batch 的 token_embs 送入解码器
token_embs = token_embs[:token_embs.shape[0] // 2]
```

这是因为 `labels` 在 `update_fn` 中被拼接了两份（`labels1` 和 `labels2`），文本编码器处理 `2B` 个样本，但解码器只需要 `B` 个（与图像 batch 大小对齐）。

### 4.3 文本解码器（核心）

**文件**：`src/models/text_decoder.py`，核心类 `_Model`

文本解码器是一个**交叉注意力 Transformer**，接收图像 patch tokens 和文本 token 序列，输出每个位置的词表概率分布。

#### 架构概览

```
输入：
  image_embs  [B, N_patches, D_img]   ← ViT 输出的 patch token 序列
  token_embs  [B, L_text, D_txt]      ← 文本编码器输出的 token 序列

Step 1: 线性投影到统一维度 width
  image_embs → Dense(width) → projected_image_embs  [B, N_patches, width]
  token_embs → Dense(width) → projected_text_embs   [B, L_text, width]

Step 2: 构造解码器的 query 和 key/value
  image_embeds = concat([projected_image_embs, projected_text_embs], axis=1)
                 shape: [B, N_patches + L_text, width]   ← 作为 cross-attn 的 K/V
  text_embeds  = learnable_tokens                        ← 作为 cross-attn 的 Q
                 shape: [B, num_learnable_tokens, width]

Step 3: CrossAttnEncoder（depth//2 层）
  每层 = 自注意力(text_embeds) + 交叉注意力(text_embeds → image_embeds)

Step 4: LayerNorm + Dense(vocab_size)
  logits  shape: [B, num_learnable_tokens, vocab_size]
```

#### Learnable Tokens

解码器使用 `num_learnable_tokens`（默认 80，对应 `output_token_len`）个可学习 token 作为 query，而不是直接用文本 token 作为 query。这些 token 通过交叉注意力从图像特征中"读取"信息，然后预测对应位置的文字。

```python
# text_decoder.py:461-466
learnable_tokens = self.param(
    'learnable_tokens',
    nn.initializers.normal(stddev=1.0),
    (self.num_learnable_tokens, self.width)
)
learnable_tokens = jnp.tile(learnable_tokens[None, :, :], (ni, 1, 1))
```

#### CrossAttnEncoder 结构

**文件**：`src/models/text_decoder.py:335-411`

```
for lyr in range(depth // 2):
    # 1. 自注意力：learnable tokens 之间互相关注（带因果掩码）
    x = Encoder1DBlock(casual_mask=True)(x)

    # 2. 交叉注意力：learnable tokens 关注图像+文本特征
    x = CrossAttnEncoder1DBlock()(
        x,           # Q: learnable tokens
        u,           # K/V: image_embeds (patch tokens + text tokens)
    )
```

`CrossAttnEncoder1DBlock`（`text_decoder.py:249-331`）的结构：

```
输入 x (learnable tokens), u (image+text tokens)
  ↓
LayerNorm(x) → Q
LayerNorm(u) → K, V
  ↓
MultiHeadDotProductAttention(Q, K, V)   ← 交叉注意力
  ↓
x = x + attn_output                     ← 残差连接
  ↓
LayerNorm(x) → MlpBlock → x = x + mlp  ← FFN + 残差
  ↓
输出 x
```

---

## 5. 前向传播：logits 的生成过程

以下是从原始输入到 `logits` 的完整数据流：

```
batch["image"]          [B, H, W, 3]
      ↓ ViT
image_embs              [B, N_patches, D_vit]    ← patch token 序列
zimg                    [B, D]                   ← 全局嵌入（L2归一化）

batch["labels"]         [2B, L]                  ← 含 BOS 的 token 序列（两份）
      ↓ Text Encoder
token_embs              [2B, L, D_txt]
ztxt                    [2B, D]                  ← 全局嵌入（L2归一化）

# CoCa 分割：取前半 batch 的 token_embs
token_embs              [B, L, D_txt]            ← two_towers.py:97

      ↓ Text Decoder
# 线性投影
projected_image_embs    [B, N_patches, width]
projected_text_embs     [B, L, width]

# 拼接为 K/V
image_embeds            [B, N_patches+L, width]

# Learnable tokens 作为 Q
text_embeds             [B, num_learnable_tokens, width]

      ↓ CrossAttnEncoder（depth//2 层）
x                       [B, num_learnable_tokens, width]

      ↓ LayerNorm + Dense(vocab_size)
logits                  [B, num_learnable_tokens, vocab_size]
```

**代码对应**（`two_towers.py:91-99`）：

```python
if (text is not None) and (image is not None) and (self.text_decoder != 'none'):
    text_decoder = importlib.import_module(f"src.models.{self.text_decoder}").Model(
        **(self.text_decoder_config or {}), name="txt_decoder", **kw)

    if train:
        token_embs = token_embs[:token_embs.shape[0] // 2]  # 取前半 batch

    logits, out_decoder_txt = text_decoder(image_embs, token_embs, train=train)
    out_dict["logits"] = logits
```

---

## 6. 损失函数

### 6.1 Caption 损失（自回归交叉熵）

**文件**：`src/losses/common.py:225-251`，`src/main_clip.py:457-465`

```python
# main_clip.py:457-465
autoreg_labels = batch["autoreg_labels"]   # [B, L]  下一个 token 的 ground truth
logits_txt = extras["logits"]              # [B, num_learnable_tokens, vocab_size]
cap_loss_mask = batch["cap_loss_mask"]     # [B, L]  非 PAD 位置为 1

caption_l = losses.softmax_xent(
    logits=logits_txt,
    labels=autoreg_labels,
    mask=cap_loss_mask,
    reduction=True,
    axis=-1
)
```

`softmax_xent` 的实现（`src/losses/common.py:225-251`）：

```python
def softmax_xent(*, logits, labels, mask=None, reduction=True, axis=-1, smoothing=0.1):
    vocab_size = logits.shape[axis]
    one_hot_labels = jax.nn.one_hot(labels, vocab_size)   # [B, L, vocab_size]
    log_p = jax.nn.log_softmax(logits, axis=axis)          # [B, L, vocab_size]
    nll = -jnp.sum(one_hot_labels * log_p, axis=axis)      # [B, L]

    # 用 mask 屏蔽 PAD 位置
    if mask is not None:
        def redux(x): return jnp.sum(x * mask) / (jnp.sum(mask) + 1e-8)
    return redux(nll)
```

### 6.2 对比损失（CLIP Loss）

**文件**：`src/losses/common.py:50-189`

CoCa 中文本编码器输出 `2B` 个嵌入，分为两半：

```python
# main_clip.py:449-454
half_batch_size = ztxt.shape[0] // 2
ztxt_1, ztxt_2 = ztxt[:half_batch_size], ztxt[half_batch_size:]

l, l_extras = losses.bidirectional_contrastive_loss(
    zimg, ztxt_1, ztxt_2, extras["t"], ...
)
```

`ztxt_1` 和 `ztxt_2` 分别是两份文本输入的嵌入，通过 `local_loss` 在每个设备本地计算对比损失，减少跨设备通信开销。

### 6.3 总损失

```python
# main_clip.py:461-465
clip_loss_weight = 1    # config.clip_loss_weight
cap_loss_weight = 2     # config.coca_caption_loss_weight

l = clip_loss_weight * l_contrastive + cap_loss_weight * caption_l
```

---

## 7. 反向传播：梯度如何流回视觉编码器

这是整个设计的核心：**自回归损失的梯度是如何更新视觉编码器的？**

```
L_caption
    ↓ ∂L/∂logits
Dense(vocab_size)
    ↓ ∂L/∂x
CrossAttnEncoder（多层）
    ↓ ∂L/∂image_embeds（通过交叉注意力的 K/V 路径）
projected_image_embs = Dense(width)(image_embs)
    ↓ ∂L/∂image_embs
ViT（视觉编码器）
    ↓ 参数更新
```

**关键路径**：梯度通过文本解码器中的**交叉注意力层**（`CrossAttnEncoder1DBlock`）反向传播到 `image_embs`（ViT 输出的 patch token 序列），再经过线性投影层传回 ViT 的参数。

具体来说，在 `CrossAttnEncoder1DBlock` 中：

```
Q = LayerNorm(learnable_tokens)
K = LayerNorm(image_embeds)   ← 梯度从这里流向 image_embs
V = LayerNorm(image_embeds)   ← 梯度从这里流向 image_embs

attn_output = softmax(QK^T / sqrt(d)) × V
```

`∂L/∂image_embs` 通过 K 和 V 两条路径传回，要求 `image_embs` 必须包含足够的语义信息来支持文字生成。

**与对比损失的梯度叠加**：

```python
# JAX 自动微分，两个损失的梯度在 ViT 参数上叠加
(l, measurements), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, ...)
# grads["img"] 包含来自 L_contrastive 和 L_caption 的梯度之和
```

---

## 8. 与标准 CLIP 的对比

| 特性 | 标准 CLIP | OpenVision (CoCa) |
|------|-----------|-------------------|
| 损失函数 | 对比损失 | 对比损失 + Caption 损失 |
| 视觉编码器输出 | 全局嵌入 | 全局嵌入 + patch token 序列 |
| 文本模型 | 文本编码器 | 文本编码器 + 文本解码器 |
| 梯度来源 | 嵌入空间对齐 | 嵌入对齐 + 逐 token 生成 |
| 局部语义理解 | 弱（全局嵌入） | 强（patch tokens 参与解码） |
| 计算开销 | 低 | 中（额外的解码器前向/反向） |
| 下游任务 | 零样本分类、检索 | 分类、检索、图像描述生成 |

---

## 9. 关键超参数

| 参数 | 配置键 | 默认值 | 说明 |
|------|--------|--------|------|
| Caption 损失权重 | `coca_caption_loss_weight` | 2 | 相对于对比损失的权重 |
| 对比损失权重 | `clip_loss_weight` | 1 | — |
| 文本 token 长度 | `token_len` | 80 | 文本编码器输入长度 |
| 解码器输出长度 | `output_token_len` | 128 | `num_learnable_tokens` |
| 解码器变体 | `txt_decoder_name` | `S` | S/B/L/H，决定 width/depth |
| 解码器融合方式 | `fusion_style` | `cross_attn` | 交叉注意力（vs concat） |
| 因果掩码 | `casual_mask` | `True` | 解码器自注意力使用因果掩码 |
