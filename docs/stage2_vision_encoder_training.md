# FG-CLIP2 Vision Encoder 训练文档

## 1. 概述

FG-CLIP2 是基于 SigLIP2 架构的细粒度视觉-语言对比学习模型。Stage 2 训练在预训练的 `siglip2-base-patch16-naflex` 基础上进行微调，核心目标是让 vision encoder 具备**细粒度区域理解**能力，即不仅能做图文全局匹配，还能对图像中的局部区域（bounding box）与文本描述做精确对齐。

训练入口：`scripts/train/stage2_fgclip2.sh` → `fgclip2/train/train.py` → `fgclip2/train/train_fgclip2.py`

---

## 2. 模型架构

### 2.1 Vision Encoder

基于 SigLIP2 ViT-Base 架构：

| 参数 | 值 |
|---|---|
| hidden_size | 768 |
| intermediate_size | 3072 |
| num_hidden_layers | 12 |
| num_attention_heads | 12 |
| patch_size | 16×16 |
| num_patches（默认） | 256 |

**NaFlex 动态分辨率**：启用 `naflex_train=True`，图像不再固定 resize 到某个尺寸，而是保持原始宽高比，按 patch 数量动态分配。本次训练 `max_num_patches=1024`，即最多允许 1024 个 patch（对应约 512×512 分辨率）。

### 2.2 新增模块

在标准 SigLIP2 基础上，FG-CLIP2 额外添加了三个 head：

| 模块 | 类型 | 用途 |
|---|---|---|
| `dense_feature_head` | `Fgclip2MultiheadAttentionPoolingHead` | 提取 dense 空间特征，用于 RoI Align |
| `longtext_head` | `nn.Linear(768, 768)` | 长文本特征投影 |
| `boxtext_head` | `nn.Linear(768, 768)` | bbox 文本特征投影 |

以及两个可学习的 logit 参数：
- `logit_scale_finegraind`：用于 box 级对比损失
- `logit_scale_hardneg`：用于 hard negative 损失

---

## 3. 权重初始化（from_siglip2=True）

训练开始时执行三步初始化（`fgclip2/model/strcs/fgclip2.py`）：

**Step 1 — `resize_postion_embeding()`**

将文本位置编码从原始长度（64）插值扩展到约 4× 长度，用于支持更长文本（`max_seq_length=196`）。前 20 个 token（`keep_len=20`）直接复制，后续 token 做线性插值，末尾 4 个 token 做外推。

**Step 2 — `copy_weight()`**

将 `text_model.head` 的权重同时复制给 `longtext_head` 和 `boxtext_head`，保证初始化时两个 head 与原始文本 head 一致。

**Step 3 — `copy_dense_feature_head()`**

将 `vision_model.head`（MultiheadAttentionPoolingHead）的权重复制给 `dense_feature_head`，用于后续提取 dense 特征。

---

## 4. 数据

### 4.1 数据格式

每条样本（JSON）包含：

| 字段 | 说明 |
|---|---|
| `f_path` | 图像相对路径 |
| `caption` | 长文本描述（最多 196 tokens） |
| `short_caption` | 短文本描述（最多 64 tokens），英文自动加前缀 `"a photo of "` |
| `bbox_info`（可选） | bounding box 列表 |
| `is_cn`（可选） | 标记为中文数据 |

每个 `bbox_info` 条目包含：
- `bbox`：归一化坐标 `[x1, y1, x2, y2]`
- `short_expr` / `long_expr`：区域描述文本
- `short_expr_negs`：hard negative 文本字典
- `flag_short_neg`：是否有 hard negative（1 表示有）

### 4.2 双语数据

同时使用英文（`en_pairs/`）和中文（`cn_pairs/`）数据。通过 `BilingualSampler` 保证每个 batch 内中英文样本不混合——中文 batch 和英文 batch 分别组成 megabatch，然后随机打乱顺序交替训练。中文数据不包含 bbox 信息（`valid_num=0`）。

### 4.3 动态 patch 数量

`DataCollatorForSupervisedDataset` 在 collate 阶段统一处理一个 batch 内所有图像的 patch 数量，按 batch 内最大 token 数向上取整到固定档位：

```
128 → 256 → 576 → 784 → 1024
```

---

## 5. 损失函数

Forward 过程计算三类损失，最终加权求和。

### 5.1 全局图文对比损失（SigLIP loss）

使用 SigLIP 的 sigmoid 对比损失（而非 softmax），支持跨 GPU 的负样本：

```
loss = -logsigmoid(labels * logits).sum() / batch_size
```

其中 `labels` 为 `{-1, +1}` 矩阵（对角线为 +1，其余为 -1）。

分两路计算：
- **短文本路径**：`short_text_embeds`（64 tokens）直接与 `image_embeds` 计算
- **长文本路径**：`long_text_embeds`（196 tokens）经过 `longtext_head` 投影后与 `image_embeds` 计算

```
loss_global = loss_short + loss_long
```

跨 GPU 负样本通过 `all_reduce_siglip_loss`（`loss_type=reduce`）实现：每个 rank 广播自己的 text features，其他 rank 用这些 features 计算 negative-only loss。

### 5.2 Box 级对比损失（add_box_loss=True）

1. **Vision 侧**：从 `dense_feature_head` 输出的 dense feature map 上，用 **RoI Align**（输出 1×1）提取每个 bbox 对应的区域特征 `bbox_image_embeds`
2. **Text 侧**：bbox 描述文本经 `text_model`（`walk_type="box"`）+ `boxtext_head` 得到 `bbox_text_embeds`
3. **损失**：
   - `pairwise_contrastive_loss`（标准 InfoNCE）× 0.2
   - `hard_category_contrastive_loss`（区域文本间排斥损失）× 0.1

`hard_category_contrastive_loss` 计算 box 文本之间的相似度，对 top-10 最相似（但相似度 < 0.95）的文本对施加排斥，防止不同区域的文本表示过于接近。

### 5.3 Hard Negative 损失（use_hard_neg=True）

针对每个 bbox，提供 1 个正样本 + 10 个 hard negative 文本（共 11 个），计算：

```
hard_contrastive_total_loss = 0.6 * contrastive_loss
                            + 3.0 * pair_loss
                            + 0.4 * cmr_loss
```

- **contrastive_loss**：图像 region vs 全部文本（正+负）的 InfoNCE
- **pair_loss**：每个图像 region 与其对应的 11 个文本做分类（label=0 为正样本）
- **cmr_loss**（Contrastive Margin Regularization）：动态 margin，要求正样本相似度比 hard negative 高出 `threshold`，threshold 通过 `all_reduce` 跨 GPU 同步并动态更新

### 5.4 总损失

```
total_loss = loss_short + loss_long
           + 0.2 * loss_bbox_itcl + 0.1 * loss_bbox_rcc
           + 0.5 * loss_bbox_hitc
```

---

## 6. 训练配置

### 6.1 优化器

AdamW，参数分组（非 LayerNorm / bias 参数施加 weight decay）：

| 超参 | 值 |
|---|---|
| learning_rate | 1e-6 |
| weight_decay | 0.001 |
| adam_beta1 | 0.9 |
| adam_beta2 | 0.98 |
| adam_epsilon | 1e-6 |
| lr_scheduler | cosine |
| warmup_ratio | 0.03 |

### 6.2 训练规模

| 参数 | 值 |
|---|---|
| per_device_train_batch_size | 512 |
| gradient_accumulation_steps | 1 |
| num_train_epochs | 6 |
| world_size | 8 |
| 精度 | BF16 + TF32 |
| gradient_checkpointing | True（use_reentrant=False） |
| save_steps | 2000 |
| save_total_limit | 36 |

### 6.3 分布式训练（DeepSpeed ZeRO Stage 2）

配置文件：`scripts/zero2.json`

| 配置项 | 值 |
|---|---|
| zero_optimization.stage | 2（梯度分片） |
| overlap_comm | true |
| contiguous_gradients | true |
| reduce_bucket_size | 5e8 |

---

## 7. 训练流程总结

```
加载 siglip2-base-patch16-naflex 预训练权重
    ↓
初始化 FG_CLIP2_Model（添加 dense_feature_head / longtext_head / boxtext_head）
    ↓
resize_postion_embeding() + copy_weight() + copy_dense_feature_head()
    ↓
构建双语数据集（英文 bbox 数据 + 中文图文对数据）
    ↓
BilingualSampler 保证 batch 内语言一致
    ↓
每个 step：
  1. Vision encoder 提取图像全局特征 + dense feature map
  2. Text encoder 分别编码 short text / long text / box text / hard neg text
  3. RoI Align 从 dense feature map 提取 bbox 区域特征
  4. 计算 SigLIP 全局损失 + box 对比损失 + hard negative 损失
  5. 反向传播，DeepSpeed ZeRO-2 更新参数
    ↓
每 2000 steps 保存 checkpoint，共训练 6 epochs
```