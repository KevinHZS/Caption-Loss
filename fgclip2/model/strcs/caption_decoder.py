import torch
import torch.nn as nn
import torch.nn.functional as F


class CaptionDecoderLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(
        self,
        q: torch.Tensor,               # [B, T, D]
        kv: torch.Tensor,              # [B, N+L, D]
        causal_mask: torch.Tensor,     # [T, T]
        kv_key_padding_mask: torch.Tensor,  # [B, N+L], True = ignore
    ) -> torch.Tensor:
        # causal self-attention
        residual = q
        q = self.norm1(q)
        q, _ = self.self_attn(q, q, q, attn_mask=causal_mask)
        q = residual + q

        # cross-attention to image+text kv
        residual = q
        q = self.norm2(q)
        q, _ = self.cross_attn(q, kv, kv, key_padding_mask=kv_key_padding_mask)
        q = residual + q

        # ffn
        residual = q
        q = self.norm3(q)
        q = residual + self.ffn(q)

        return q


class CaptionDecoder(nn.Module):
    """
    CoCa-style caption decoder (Plan A).
    Query: learnable tokens (not conditioned on text input).
    K/V: concat(image_patch_tokens, text_token_embs) after projection.

    TODO (Plan C): use text token embeddings directly as query.
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        num_layers: int = 6,
        num_heads: int = 12,
        num_learnable_tokens: int = 196,
        vocab_size: int = 32000,
    ):
        super().__init__()
        self.num_learnable_tokens = num_learnable_tokens

        self.image_proj = nn.Linear(hidden_dim, hidden_dim)
        self.text_proj = nn.Linear(hidden_dim, hidden_dim)
        self.learnable_tokens = nn.Parameter(
            torch.randn(num_learnable_tokens, hidden_dim) * 0.02
        )
        self.layers = nn.ModuleList(
            [CaptionDecoderLayer(hidden_dim, num_heads) for _ in range(num_layers)]
        )
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(
        self,
        image_patch_tokens: torch.Tensor,       # [B, N, D]
        text_token_embs: torch.Tensor,          # [B, L, D]
        pixel_attention_mask: torch.Tensor,     # [B, N], 1=valid 0=pad
    ) -> torch.Tensor:                          # [B, T, vocab_size]
        B = image_patch_tokens.shape[0]
        device = image_patch_tokens.device

        # project to decoder space
        img_kv = self.image_proj(image_patch_tokens)   # [B, N, D]
        txt_kv = self.text_proj(text_token_embs)        # [B, L, D]
        kv = torch.cat([img_kv, txt_kv], dim=1)         # [B, N+L, D]

        # key_padding_mask: True = position to ignore
        L = text_token_embs.shape[1]
        txt_mask = torch.ones(B, L, dtype=torch.bool, device=device)  # all valid
        img_mask = pixel_attention_mask.bool()           # [B, N], True=valid
        # MultiheadAttention key_padding_mask: True = ignore → invert
        kv_key_padding_mask = ~torch.cat([img_mask, txt_mask], dim=1)  # [B, N+L]

        # learnable tokens as query
        q = self.learnable_tokens.unsqueeze(0).expand(B, -1, -1)  # [B, T, D]
        q = q.contiguous()

        # causal mask for self-attention [T, T]
        T = self.num_learnable_tokens
        causal_mask = torch.triu(
            torch.full((T, T), float('-inf'), device=device), diagonal=1
        )

        for layer in self.layers:
            q = layer(q, kv, causal_mask, kv_key_padding_mask)

        return self.lm_head(q)  # [B, T, vocab_size]