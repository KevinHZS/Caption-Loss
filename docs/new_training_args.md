# 新增训练参数说明

本文档说明近期围绕 caption loss、short caption、projector 训练和 visual token pooling 新增或重命名的训练参数。

## 参数总览

### `long_loss_weight`

- 默认值：`1.0`
- 类别：全局图文对比损失
- 含义：控制 long caption 图文对比损失 loss_long 进入总 loss 的权重。
- 使用方式：设为 0.0 时可以关闭 long caption 对比损失，只保留其它已启用的 loss。
- 注意事项：这个参数不控制 LLM caption decoder 的交叉熵；它只作用于 CLIP/SigLIP 风格的 long text 对比学习分支。
- git 首次出现：0d269aa 2026-04-28 support function of freezing projector

### `short_loss_weight`

- 默认值：`1.0`
- 类别：全局图文对比损失
- 含义：控制 short caption 图文对比损失 loss_short 进入总 loss 的权重。
- 使用方式：需要降低或关闭 short caption 的对比学习影响时，设置 SHORT_LOSS_WEIGHT 环境变量或直接传 --short_loss_weight。
- 注意事项：只有 use_short_caption_contrastive_loss=True 且 batch 中构造了 short_text 时才会产生 loss_short。
- git 首次出现：当前工作区新增或重命名，尚未在 git 历史中找到提交

### `use_short_caption_contrastive_loss`

- 默认值：`True`
- 类别：数据与对比损失开关
- 含义：是否启用 short_caption 对应的图文对比损失路径。
- 使用方式：设为 False 时，训练不会为 short_caption 构造 short_text 对比输入，也不会计算 loss_short。
- 注意事项：这个参数只表示 short caption 的对比损失开关，不表示 LLM caption decoder 的 short caption 交叉熵。
- git 首次出现：9fadf3c 2026-04-29 Add short caption LLM loss and clarify contrastive flag

### `long_caption_loss_weight`

- 默认值：`0.0`
- 类别：LLM caption decoder 交叉熵
- 含义：控制 long caption 输入 LLM caption decoder 后得到的交叉熵 loss_caption 的权重。
- 使用方式：caption 对齐训练通常设置为 1.0；设为 0.0 时不会构造 long caption 的 LLM labels。若 short_caption_loss_weight 也为 0.0，则不会加载 LLM decoder。
- 注意事项：这是原 caption_loss_weight 的明确命名版本，用来避免和 short caption CE 混淆。
- git 首次出现：当前工作区新增或重命名，尚未在 git 历史中找到提交

### `short_caption_loss_weight`

- 默认值：`0.0`
- 类别：LLM caption decoder 交叉熵
- 含义：控制 short caption 输入同一个 LLM caption decoder 后得到的交叉熵 loss_caption_short 的权重。
- 使用方式：如果要同时用 long/short caption 做 LLM 语义空间对齐，可以设置为 0.5 或其它非零值。
- 注意事项：它和 short_loss_weight 不同：short_caption_loss_weight 是 LLM CE，short_loss_weight 是图文对比损失。
- git 首次出现：9fadf3c 2026-04-29 Add short caption LLM loss and clarify contrastive flag

### `llm_model_path`

- 默认值：`None`
- 类别：LLM caption decoder
- 含义：冻结 LLM 的本地路径，用于把 vision encoder/projector 输出的 visual token 接到 LLM token embedding 空间做 caption CE。
- 使用方式：只有 long_caption_loss_weight 或 short_caption_loss_weight 大于 0 时才需要设置。
- 注意事项：LLM 参数冻结，通常训练 projector 和可选的 vision encoder。
- git 首次出现：023b1ab 2026-04-21 add the implementation caption loss of qwen3-1.7b

### `llm_gradient_checkpointing`

- 默认值：`False`
- 类别：LLM caption decoder
- 含义：是否对冻结 LLM 开启 gradient checkpointing，以降低显存占用。
- 使用方式：caption CE 显存压力大时设为 True。
- 注意事项：虽然 LLM 冻结，但为了把梯度传回 visual token/projector/vision encoder，前向图仍然需要保留必要梯度路径。
- git 首次出现：e275286 2026-04-23 fix: 1. add llm.gradient_checkpointing_enable

### `caption_pool_2x2_tokens`

- 默认值：`False`
- 类别：LLM caption decoder visual token 压缩
- 含义：是否在送入 LLM caption decoder 前，把 vision encoder 的 patch token 按 2x2 平均池化为 1 个 visual token。
- 使用方式：图像 patch token 很长导致 LLM CE 显存过高时设为 True。
- 注意事项：池化只影响 caption decoder 输入序列长度，不改变 vision encoder 的原始输出本身。
- git 首次出现：b94c3c3 2026-04-24 1. support 2 * 2 visual feature compression

### `train_projector_only`

- 默认值：`False`
- 类别：训练参数冻结
- 含义：只训练 llm_caption_decoder.projector，冻结 vision encoder、text encoder、LLM 和其它模块。
- 使用方式：第一阶段先把 projector 对齐到冻结 LLM 语义空间时设为 True。
- 注意事项：不能和 freeze_projector=True 同时使用。
- git 首次出现：8a5cc22 2026-04-24 fix: 1.support projector trained only 2. support load projector checkpoint

### `freeze_projector`

- 默认值：`False`
- 类别：训练参数冻结
- 含义：冻结 llm_caption_decoder.projector，只训练其它可训练模块，例如 vision encoder。
- 使用方式：已预训练 projector 后，只想让 caption loss 继续优化 vision encoder 时设为 True。
- 注意事项：不能和 train_projector_only=True 同时使用。
- git 首次出现：0d269aa 2026-04-28 support function of freezing projector

### `load_projector_from`

- 默认值：`None`
- 类别：checkpoint 加载
- 含义：从指定 projector checkpoint 目录或文件加载 projector 权重。
- 使用方式：直接传 projector 子目录路径，不会自动在 output_dir 或 checkpoint-* 里追加 projector/。
- 注意事项：用于 projector-only 预训练后接联合训练。
- git 首次出现：8a5cc22 2026-04-24 fix: 1.support projector trained only 2. support load projector checkpoint

## loss 命名关系

- `loss_long`：long caption 的图文对比损失，受 `long_loss_weight` 控制。
- `loss_short`：short caption 的图文对比损失，受 `short_loss_weight` 控制。
- `loss_caption`：long caption 经过冻结 LLM caption decoder 后的交叉熵，受 `long_caption_loss_weight` 控制。
- `loss_caption_short`：short caption 经过冻结 LLM caption decoder 后的交叉熵，受 `short_caption_loss_weight` 控制。

## 常见配置

- 只用 long caption CE 训练 projector/vision encoder：`--long_loss_weight 0.0 --short_loss_weight 0.0 --long_caption_loss_weight 1.0`。
- 同时使用 long/short caption CE：`--long_caption_loss_weight 1.0 --short_caption_loss_weight 0.5`。
- 关闭 short caption 对比损失但保留 short caption CE：`--use_short_caption_contrastive_loss False --short_caption_loss_weight 0.5`。
