import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoConfig


def pool_caption_image_patch_tokens(
    image_patch_tokens: torch.Tensor,
    pixel_attention_mask: torch.Tensor,
    spatial_shapes: torch.Tensor,
):
    batch_size, _, hidden_dim = image_patch_tokens.shape
    max_height = int(spatial_shapes[:, 0].max().item())
    max_width = int(spatial_shapes[:, 1].max().item())
    dense_tokens = image_patch_tokens.new_zeros(batch_size, max_height, max_width, hidden_dim)
    dense_mask = pixel_attention_mask.new_zeros(batch_size, max_height, max_width)

    for batch_index in range(batch_size):
        height = int(spatial_shapes[batch_index, 0].item())
        width = int(spatial_shapes[batch_index, 1].item())
        real_token_count = height * width
        dense_tokens[batch_index, :height, :width] = image_patch_tokens[batch_index, :real_token_count].reshape(
            height, width, hidden_dim
        )
        dense_mask[batch_index, :height, :width] = pixel_attention_mask[batch_index, :real_token_count].reshape(
            height, width
        )

    pooled_height = (max_height + 1) // 2
    pooled_width = (max_width + 1) // 2
    pad_height = pooled_height * 2 - max_height
    pad_width = pooled_width * 2 - max_width

    padded_tokens = F.pad(dense_tokens, (0, 0, 0, pad_width, 0, pad_height))
    padded_mask = F.pad(dense_mask, (0, pad_width, 0, pad_height))

    token_blocks = padded_tokens.reshape(batch_size, pooled_height, 2, pooled_width, 2, hidden_dim)
    token_blocks = token_blocks.permute(0, 1, 3, 2, 4, 5).reshape(batch_size, pooled_height, pooled_width, 4, hidden_dim)

    mask_blocks = padded_mask.reshape(batch_size, pooled_height, 2, pooled_width, 2)
    mask_blocks = mask_blocks.permute(0, 1, 3, 2, 4).reshape(batch_size, pooled_height, pooled_width, 4)

    mask_weights = mask_blocks.to(dtype=image_patch_tokens.dtype)
    mask_counts = mask_weights.sum(dim=-1, keepdim=True)
    pooled_tokens = (token_blocks * mask_weights.unsqueeze(-1)).sum(dim=3)
    pooled_tokens = pooled_tokens / mask_counts.clamp_min(1.0)
    pooled_tokens = pooled_tokens.reshape(batch_size, pooled_height * pooled_width, hidden_dim)
    pooled_attention_mask = (mask_counts.squeeze(-1) > 0).reshape(batch_size, pooled_height * pooled_width)
    pooled_attention_mask = pooled_attention_mask.to(dtype=pixel_attention_mask.dtype)

    return pooled_tokens, pooled_attention_mask


class LLMCaptionDecoder(nn.Module):
    """
    LLM-based caption decoder.
    Vision tokens are projected via MLP to LLM hidden dim,
    then prepended to caption token embeddings as input to a frozen LLM.
    Only the MLP projector is trainable; LLM is always frozen.
    """

    def __init__(self, vis_hidden_dim: int, llm_model_path: str, caption_pool_2x2_tokens: bool = False):
        super().__init__()
        llm_config = AutoConfig.from_pretrained(llm_model_path)
        llm_hidden_dim = llm_config.hidden_size
        self.caption_pool_2x2_tokens = caption_pool_2x2_tokens

        self.projector = nn.Sequential(
            nn.Linear(vis_hidden_dim, llm_hidden_dim),
            nn.GELU(),
            nn.Linear(llm_hidden_dim, llm_hidden_dim),
        )

        self.llm = AutoModelForCausalLM.from_pretrained(llm_model_path, dtype=torch.bfloat16)
        for param in self.llm.parameters():
            param.requires_grad_(False)

    def forward(
        self,
        image_patch_tokens: torch.Tensor,   # [B, N, D_vis]
        pixel_attention_mask: torch.Tensor, # [B, N], 1=valid 0=pad
        spatial_shapes: torch.Tensor,       # [B, 2], [H, W]
        llm_input_ids: torch.Tensor,        # [B, L]
        llm_attention_mask: torch.Tensor,   # [B, L]
    ):                                     # ([B, N+L, vocab_size], N)
        if self.caption_pool_2x2_tokens:
            if pixel_attention_mask is None or spatial_shapes is None:
                raise ValueError("pixel_attention_mask and spatial_shapes are required when caption_pool_2x2_tokens=True")
            image_patch_tokens, pixel_attention_mask = pool_caption_image_patch_tokens(
                image_patch_tokens=image_patch_tokens,
                pixel_attention_mask=pixel_attention_mask,
                spatial_shapes=spatial_shapes,
            )

        visual_token_count = image_patch_tokens.shape[1]

        # project vision tokens to LLM space
        vis_embeds = self.projector(image_patch_tokens)  # [B, N, D_llm]

        # get LLM text token embeddings
        txt_embeds = self.llm.model.embed_tokens(llm_input_ids)  # [B, L, D_llm]

        # concat: [vision_tokens | text_tokens]
        inputs_embeds = torch.cat([vis_embeds, txt_embeds], dim=1)  # [B, N+L, D_llm]
        attention_mask = torch.cat([
            pixel_attention_mask.long(),
            llm_attention_mask.long(),
        ], dim=1)  # [B, N+L]

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
        )
        return outputs.logits, visual_token_count  # [B, N+L, vocab_size], N
