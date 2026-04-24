import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig


def pool_caption_image_patch_tokens(
    image_patch_tokens: torch.Tensor,
    pixel_attention_mask: torch.Tensor,
    spatial_shapes: torch.Tensor,
):
    batch_size, _, hidden_dim = image_patch_tokens.shape
    pooled_tokens_per_image = []
    pooled_masks_per_image = []
    max_pooled_tokens = 0

    for batch_index in range(batch_size):
        height = int(spatial_shapes[batch_index, 0].item())
        width = int(spatial_shapes[batch_index, 1].item())
        real_token_count = height * width

        image_tokens = image_patch_tokens[batch_index, :real_token_count].reshape(height, width, hidden_dim)
        image_mask = pixel_attention_mask[batch_index, :real_token_count].reshape(height, width).bool()

        pooled_blocks = []
        pooled_mask = []
        for row_start in range(0, height, 2):
            for col_start in range(0, width, 2):
                token_block = image_tokens[row_start : row_start + 2, col_start : col_start + 2].reshape(-1, hidden_dim)
                mask_block = image_mask[row_start : row_start + 2, col_start : col_start + 2].reshape(-1)
                if mask_block.any():
                    pooled_blocks.append(token_block[mask_block].mean(dim=0))
                    pooled_mask.append(1)
                else:
                    pooled_blocks.append(image_patch_tokens.new_zeros(hidden_dim))
                    pooled_mask.append(0)

        pooled_tokens = torch.stack(pooled_blocks, dim=0)
        pooled_attention_mask = torch.tensor(pooled_mask, device=pixel_attention_mask.device, dtype=pixel_attention_mask.dtype)

        pooled_tokens_per_image.append(pooled_tokens)
        pooled_masks_per_image.append(pooled_attention_mask)
        max_pooled_tokens = max(max_pooled_tokens, pooled_tokens.shape[0])

    padded_tokens = image_patch_tokens.new_zeros(batch_size, max_pooled_tokens, hidden_dim)
    padded_attention_mask = pixel_attention_mask.new_zeros(batch_size, max_pooled_tokens)

    for batch_index, (pooled_tokens, pooled_attention_mask) in enumerate(zip(pooled_tokens_per_image, pooled_masks_per_image)):
        current_token_count = pooled_tokens.shape[0]
        padded_tokens[batch_index, :current_token_count] = pooled_tokens
        padded_attention_mask[batch_index, :current_token_count] = pooled_attention_mask

    return padded_tokens, padded_attention_mask


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
