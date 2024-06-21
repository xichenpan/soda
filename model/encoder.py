from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import SiglipVisionModel


def SDPAforward(self):
    def head_to_batch_dim(tensor: torch.Tensor, head_size, out_dim: int = 4) -> torch.Tensor:
        r"""
        Reshape the tensor from `[batch_size, seq_len, dim]` to `[batch_size, seq_len, heads, dim // heads]` `heads` is
        the number of heads initialized while constructing the `Attention` class.

        Args:
            tensor (`torch.Tensor`): The tensor to reshape.
            out_dim (`int`, *optional*, defaults to `3`): The output dimension of the tensor. If `3`, the tensor is
                reshaped to `[batch_size * heads, seq_len, dim // heads]`.

        Returns:
            `torch.Tensor`: The reshaped tensor.
        """
        if tensor.ndim == 3:
            batch_size, seq_len, dim = tensor.shape
            extra_dim = 1
        else:
            batch_size, extra_dim, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size, seq_len * extra_dim, head_size, dim // head_size)
        tensor = tensor.permute(0, 2, 1, 3)

        if out_dim == 3:
            tensor = tensor.reshape(batch_size * head_size, seq_len * extra_dim, dim // head_size)

        return tensor

    def forward(
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = head_to_batch_dim(query, self.num_heads)
        key = head_to_batch_dim(key, self.num_heads)
        value = head_to_batch_dim(value, self.num_heads)

        out = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=self.dropout,
                                             scale=self.scale)
        # B * head * len * dim

        out = out.permute(0, 2, 1, 3).flatten(2)
        out = self.out_proj(out)

        return out, None

    return forward


class Encoder(nn.Module):
    """
    An encoder network (image -> feature_dim)
    """

    def __init__(self, gradient_checkpointing):
        super(Encoder, self).__init__()
        self.model = SiglipVisionModel.from_pretrained("google/siglip-so400m-patch14-384")
        # replace the attention layer with our own implementation
        for layer in self.model.vision_model.encoder.layers:
            module = layer.self_attn
            module.forward = SDPAforward(module)

        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

    def forward(self, x):
        return self.model(x, output_attentions=False, output_hidden_states=False).pooler_output
