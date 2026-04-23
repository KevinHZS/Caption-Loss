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

        self.llm = AutoModelForCausalLM.from_pretrained(llm_model_path, dtype=torch.bfloat16)
        for param in self.llm.parameters():
            param.requires_grad_(False)

    def forward(
        self,
        image_patch_tokens: torch.Tensor,   # [B, N, D_vis]
        pixel_attention_mask: torch.Tensor, # [B, N], 1=valid 0=pad
        llm_input_ids: torch.Tensor,        # [B, L]
        llm_attention_mask: torch.Tensor,   # [B, L]
    ) -> torch.Tensor:                      # [B, N+L, vocab_size]
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
        return outputs.logits  # [B, N+L, vocab_size]