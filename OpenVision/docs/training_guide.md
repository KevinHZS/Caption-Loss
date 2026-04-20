# OpenVision 训练流程、模型架构与 GPU 适配分析

## 目录

1. [训练流程总览](#1-训练流程总览)
2. [三阶段训练详解](#2-三阶段训练详解)
3. [模型架构](#3-模型架构)
4. [数据流水线](#4-数据流水线)
5. [并行策略](#5-并行策略)
6. [TPU 专属代码分析](#6-tpu-专属代码分析)
7. [GPU 适配方案](#7-gpu-适配方案)

---

## 1. 训练流程总览

OpenVision 是一个基于 **JAX/Flax** 框架、在 **Google TPU** 上训练的多模态视觉-语言模型，采用 **CoCa**（Contrastive Captioners）范式，同时优化对比学习损失（CLIP loss）和自回归字幕生成损失（Caption loss）。

### 整体训练分三个阶段：

```
阶段一：小分辨率预训练 (84px)
    ↓  checkpoint.npz
阶段二：中分辨率微调 (224px)
    ↓  ft_224/checkpoint.npz
阶段三：高分辨率微调 (384px / 336px)
    ↓  ft_384/checkpoint.npz
```

入口文件：`src/main_clip.py`
配置文件：`src/configs/openvision.py`

---

## 2. 三阶段训练详解

### 阶段一：小分辨率预训练（84px）

| 参数 | 值 |
|------|-----|
| 分辨率 | 84×84 |
| 批大小 | `1024 × 16 × BATCH_FACTOR` = 32,768（BATCH_FACTOR=2） |
| 学习率 | 8e-6 |
| 训练轮数 | 10,000 epoch（≈ 12.8M 样本/epoch × 10,000 = 128B 样本） |
| Warmup | 40 epoch |
| 数据集 | DataComp（大规模图文对） |

**目的**：用低分辨率快速学习图文对齐的语义表示，降低计算成本。

### 阶段二：中分辨率微调（224px）

| 参数 | 值 |
|------|-----|
| 分辨率 | 224×224 |
| 批大小 | 16,384（FT_BATCH_FACTOR=1） |
| 学习率 | 4e-7 |
| 训练轮数 | 800 epoch（≈ 1.024B 样本） |
| 初始化 | 阶段一的 `checkpoint.npz` |

**目的**：将低分辨率预训练的表示迁移到标准 224px 分辨率，适配下游任务。

### 阶段三：高分辨率微调（384px / 336px）

| 参数 | 值 |
|------|-----|
| 分辨率 | 384×384（ViT-L/14、ViT-H/14 用 336px） |
| 批大小 | 8,192（FT_BATCH_FACTOR=0.5） |
| 学习率 | 1e-7 |
| 训练轮数 | 200 epoch（≈ 256M 样本） |
| 初始化 | 阶段二的 `ft_224/checkpoint.npz` |

**目的**：进一步提升细粒度视觉理解能力。

### 训练循环核心逻辑（`src/main_clip.py`）

```python
# 初始化分布式 TPU
jax.distributed.initialize()

# 创建 3D 设备网格
mesh = create_mesh(config)  # axes: ['data', 'fsdp', 'tensor']

# 构建数据迭代器
train_iter = input_pipeline.start_input_pipeline(...)

# 编译 JIT 更新函数（参数原地更新，donate_argnums 避免内存拷贝）
@functools.partial(jax.jit, donate_argnums=(0,), ...)
def update_fn(train_state, batch, rng):
    (loss, measurements), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, ...)
    updates, opt = tx.update(grads, opt, params)
    params = optax.apply_updates(params, updates)
    return {"params": params, "opt": opt}, measurements

# 主训练循环
for step, batch in zip(range(first_step + 1, total_steps + 1), train_iter):
    with mesh, nn.logical_axis_rules(config.sharding.logical_axis_rules):
        train_state, measurements = update_fn(train_state, batch, rng_loop)
```

### 损失函数（`src/losses/common.py`）

CoCa 范式同时计算两种损失：

```
总损失 = clip_loss_weight × CLIP对比损失 + coca_caption_loss_weight × Caption自回归损失
       = 1 × L_contrastive + 2 × L_caption
```

- **CLIP 对比损失**：图像嵌入与文本嵌入的双向 InfoNCE 损失，使用 `local_loss=True` 在每个设备本地计算，减少跨设备通信
- **Caption 自回归损失**：文本解码器的交叉熵损失，预测下一个 token

### 评估（训练中周期性执行）

- **零样本分类**：ImageNet-1K（`disclf` evaluator）
- **图文检索**：COCO、Flickr30K（`retrieval` evaluator）

---

## 3. 模型架构

### 3.1 整体架构：Two-Towers + Decoder（`src/models/two_towers.py`）

```
输入图像 ──→ [图像编码器 ViT] ──→ zimg (L2归一化)
                                        ↘
                                    对比损失 (CLIP)
                                        ↗
输入文本 ──→ [文本编码器 Transformer] ──→ ztxt (L2归一化)
              ↓ (patch tokens)
         [文本解码器 Transformer] ──→ Caption 损失
```

- 可学习温度参数 `t = log(temperature_init)`，用于缩放对比损失的 logits
- 图像和文本嵌入均经过 L2 归一化

### 3.2 图像编码器：Vision Transformer（`src/models/vit.py`）

**Patch Embedding**：
- 默认：`nn.Conv`（stride = patch_size），将图像分割为 patch
- 可选：`stem`（3层卷积渐进下采样）或线性投影

**位置编码**：
- `sincos2d`（默认）：2D 正弦余弦位置编码，支持分辨率插值
- `learned`：可学习位置嵌入

**Transformer Encoder**：
- 堆叠 `Encoder1DBlock`（MHSA + MLP + LayerNorm）
- 支持 Pre-LN（默认）

**池化方式**：
- `gap`：所有 patch token 的均值 + LayerNorm（默认）
- `tok`：CLS token
- `map`：Multi-head Attention Pooling（MAPHead）

**支持的模型规格**：

| 变体 | 隐藏维度 | 层数 | MLP维度 | 注意力头数 |
|------|---------|------|---------|-----------|
| S/16 | 384 | 12 | 1536 | 6 |
| B/16 | 768 | 12 | 3072 | 12 |
| L/16, L/14 | 1024 | 24 | 4096 | 16 |
| SoVit400m/14 | 1152 | 27 | 4304 | 16 |
| H/14 | 1280 | 32 | 5120 | 16 |

**激活检查点**（节省显存/HBM）：
- `none`：不使用
- `minimal`：仅对注意力层重计算
- `minimal_offloaded`：重计算 + 卸载到 CPU
- `minimal_flash`：结合 Flash Attention
- `full`：全层重计算

### 3.3 文本编码器：Text Transformer（`src/models/text_transformer.py`）

- Token Embedding（`nn.Embed`）+ 1D 位置编码
- 与 ViT 共享 `Encoder1DBlock` 实现
- 支持双向注意力（编码器模式）或因果掩码（解码器模式）
- 池化：`last`（取最后一个 token 的表示）
- `output_tokens=True` 时同时返回 patch token 序列（供文本解码器使用）

### 3.4 文本解码器：Autoregressive Decoder（`src/models/text_decoder.py`）

- 标准自回归 Transformer 解码器
- 输入：文本 token + 图像 patch tokens（交叉注意力）
- 输出：下一个 token 的概率分布
- 用于 CoCa 的 Caption 损失计算

---

## 4. 数据流水线

### 数据集（`src/datasets/input_pipeline.py`）

- **预训练**：DataComp（大规模网络爬取图文对）
- **评估**：ImageNet-1K、COCO、Flickr30K

### 预处理流水线（配置字符串形式）

```
inception_crop(size=84)     # 随机裁剪并缩放
| simclr_jitter_gray         # 颜色抖动 + 灰度化
| my_bert_tokenize(...)      # BERT tokenizer，最大长度 80
| get_autoreg_label          # 生成自回归标签（用于 Caption 损失）
| keep(image, labels, ...)   # 保留必要字段
```

### 分布式数据加载

```python
# 将本地 numpy 数组转换为全局 JAX 分片数组
def shard_and_put(x, sharding):
    return jax.make_array_from_single_device_arrays(
        global_shape, sharding, [jax.device_put(x_local, device) for ...]
    )
```

- 每个 TPU host 加载自己的数据分片
- `tf.data` 配置：`private_threadpool_size=48`，`max_intra_op_parallelism=1`（TPU 优化）

---

## 5. 并行策略

### 3D 设备网格

```python
# src/helpers/sharding.py
mesh = mesh_utils.create_device_mesh([data_parallelism, fsdp_parallelism, tensor_parallelism])
# 默认：[128, 1, 1]，即纯数据并行
```

| 轴 | 默认值 | 作用 |
|----|--------|------|
| `data` | 128 | 数据并行：不同设备处理不同样本 |
| `fsdp` | 1 | 全分片数据并行：参数按 `embed` 维度分片 |
| `tensor` | 1 | 张量并行：MLP/注意力头按 `tensor` 轴分片 |

### 逻辑轴到物理轴的映射

```python
logical_axis_rules = [
    ['activation_batch',  ['data', 'fsdp']],  # 激活值的 batch 维度
    ['activation_heads',  ['tensor']],          # 注意力头维度
    ['activation_embed',  ['tensor']],          # 嵌入维度
    ['mlp',               'tensor'],            # MLP 权重
    ['embed',             'fsdp'],              # 嵌入权重（FSDP）
    ['vocab',             'tensor'],            # 词表维度
]
```

### 优化器

- Adam（`scale_by_adam`），`mu_dtype='bfloat16'`（一阶矩用 bf16 存储节省内存）
- 余弦衰减学习率调度
- 梯度裁剪

---

## 6. TPU 专属代码分析

以下是项目中与 TPU 强绑定的关键代码，是 GPU 适配的主要障碍：

### 6.1 多主机初始化（`src/main_clip.py`）

```python
# TPU 专属：多主机 SPMD 初始化
jax.distributed.initialize()

# TPU 专属环境变量
os.environ['LIBTPU_INIT_ARGS'] = '--xla_tpu_spmd_rng_bit_generator_unsafe=true'

# 禁用 GPU（明确排除 GPU）
os.environ['CUDA_VISIBLE_DEVICES'] = ''
```

### 6.2 TPU Flash Attention（`src/models/text_transformer.py`）

```python
# 使用 Pallas TPU 专属 Flash Attention
from jax.experimental.pallas.ops.tpu import flash_attention

def _tpu_flash_attention(self, query, key, value, ...):
    return shard_map(
        flash_attention,
        mesh=self.mesh,
        in_specs=(P('data', 'fsdp', None, None), ...),
        out_specs=P('data', 'fsdp', None, None),
    )(query, key, value, ...)
```

### 6.3 设备网格创建（`src/helpers/sharding.py`）

```python
# 依赖 jax.experimental.mesh_utils（TPU 拓扑感知）
from jax.experimental import mesh_utils
mesh = mesh_utils.create_device_mesh([data, fsdp, tensor])
```

### 6.4 数据类型（`src/configs/openvision.py`）

```python
# bfloat16：TPU 原生支持，GPU 支持有限（A100+ 支持）
config.model.image.dtype = 'bfloat16'
config.model.text.dtype = 'bfloat16'
config.model.text_decoder_config.dtype = 'bfloat16'
```

---

## 7. GPU 适配方案

### 7.1 可行性评估

**总体结论**：项目基于 JAX/Flax，JAX 本身支持 GPU（CUDA），理论上可以适配，但需要修改若干 TPU 专属代码。工作量中等，主要集中在以下几个方面。

### 7.2 必须修改的部分

#### (1) 移除 TPU 初始化代码（`src/main_clip.py`）

```python
# 删除或条件化：
os.environ['LIBTPU_INIT_ARGS'] = '...'   # TPU 专属，GPU 上无效
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # 必须删除！这行代码会禁用所有 GPU

# jax.distributed.initialize() 在多 GPU 节点上需要不同参数：
# GPU 多节点需要指定 coordinator_address、num_processes、process_id
jax.distributed.initialize(
    coordinator_address="...",
    num_processes=num_nodes,
    process_id=node_rank,
)
```

#### (2) 替换 TPU Flash Attention（`src/models/text_transformer.py` 和 `src/models/vit.py`）

```python
# 原代码（TPU 专属）：
from jax.experimental.pallas.ops.tpu import flash_attention

# GPU 替代方案一：使用 JAX 内置（需 JAX >= 0.4.20）
# 在 nn.attention 中设置 implementation='cudnn'（需要 cuDNN 9+）

# GPU 替代方案二：使用 jax-flash-attn 或 triton flash attention
# 或直接禁用 flash attention（train.sh 中已有 use_flash_attn=False 选项）
# --config.model.image.use_flash_attn=False
# --config.model.text.use_flash_attn=False
```

**最简单的方案**：训练脚本中已有 `use_flash_attn=False` 选项，直接禁用即可，性能略有下降但功能正常。

#### (3) 设备网格创建（`src/helpers/sharding.py`）

```python
# 原代码（TPU 拓扑感知）：
from jax.experimental import mesh_utils
mesh = mesh_utils.create_device_mesh([data, fsdp, tensor])

# GPU 适配：mesh_utils 在 GPU 上也可用，但需要确认设备数量匹配
# 对于单机多 GPU（如 8xA100），设置：
# data_parallelism=8, fsdp_parallelism=1, tensor_parallelism=1
```

#### (4) bfloat16 支持

```python
# bfloat16 在 GPU 上的支持情况：
# - A100/H100：完整支持 bfloat16
# - V100/RTX 系列：不支持 bfloat16，需改为 float16

# 如果使用不支持 bfloat16 的 GPU，修改配置：
config.model.image.dtype = 'float16'      # 替换 bfloat16
config.model.text.dtype = 'float16'
config.model.text_decoder_config.dtype = 'float16'
# 注意：float16 训练稳定性较差，可能需要调整损失缩放
```

### 7.3 建议修改的部分

#### (5) 数据加载优化（`src/datasets/input_pipeline.py`）

```python
# TPU 优化配置，GPU 上可能需要调整：
options.threading.private_threadpool_size = 48  # 可根据 CPU 核数调整
options.threading.max_intra_op_parallelism = 1  # GPU 上可以放开此限制
```

#### (6) 批大小调整

TPU v3-128 有 128 个核心，GPU 集群通常核心数更少：
- 8xA100 单节点：`data_parallelism=8`，批大小相应缩小
- 梯度累积可以模拟大批量：需在 `update_fn` 中添加梯度累积逻辑

#### (7) 检查点格式

项目使用 Orbax 保存 `.npz` 格式，这与硬件无关，GPU 上可直接使用。

### 7.4 GPU 适配修改清单

| 优先级 | 文件 | 修改内容 |
|--------|------|---------|
| 必须 | `src/main_clip.py` | 删除 `CUDA_VISIBLE_DEVICES=''`，删除 `LIBTPU_INIT_ARGS` |
| 必须 | `src/main_clip.py` | 修改 `jax.distributed.initialize()` 参数 |
| 必须 | `scripts/train.sh` | 设置 `use_flash_attn=False`（或替换为 GPU Flash Attn） |
| 必须 | `scripts/train.sh` | 调整 `data_parallelism` 为实际 GPU 数量 |
| 建议 | `src/configs/openvision.py` | 若用 V100，将 `bfloat16` 改为 `float16` |
| 建议 | `src/datasets/input_pipeline.py` | 调整 tf.data 线程配置 |
| 可选 | `src/models/text_transformer.py` | 替换 TPU Flash Attention 为 GPU 版本 |

### 7.5 GPU 训练启动脚本示例

```bash
#!/bin/bash
# GPU 适配版训练脚本（单机 8xA100）

export data_parallelism=8
export fsdp_parallelism=1
export tensor_parallelism=1
export BATCH_FACTOR=0.5  # 8 GPU vs 128 TPU cores，批大小缩小 16 倍

# 使用 torchrun 风格的多进程启动（JAX 多 GPU）
python3 -m src.main_clip \
  --config=${TRAIN_CONFIG}:... \
  --config.model.image.use_flash_attn=False \
  --config.model.text.use_flash_attn=False \
  --config.model.image.dtype=bfloat16 \   # A100 支持 bfloat16
  ...
```

### 7.6 潜在问题与注意事项

1. **内存压力**：TPU HBM 与 GPU VRAM 架构不同，FSDP 分片策略可能需要重新调整
2. **通信后端**：JAX 在 GPU 上使用 NCCL，在 TPU 上使用 gRPC/ICI，多节点 GPU 需要正确配置 NCCL
3. **XLA 编译时间**：GPU 上 XLA JIT 编译时间可能比 TPU 更长
4. **数值稳定性**：float16 相比 bfloat16 动态范围更小，大批量训练时可能出现梯度溢出，建议使用 `optax.scale_by_loss_scale` 或 `jax.lax.with_sharding_constraint` 配合损失缩放
5. **批大小影响**：原始训练使用 32K 批大小，GPU 上批大小大幅缩小会影响对比学习效果，建议使用梯度累积或增大 `local_loss` 的 gather 范围
