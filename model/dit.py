from typing import Any
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.embeddings import PatchEmbed
from diffusers.models.embeddings import Timesteps, TimestepEmbedding
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import is_torch_version
from torch import nn
from transformers.models.llama.modeling_llama import LlamaMLP, LlamaAttention, LlamaConfig

from vlb_loss import GaussianDiffusion


def _chunked_feed_forward(ff: nn.Module, hidden_states: torch.Tensor, chunk_dim: int, chunk_size: int):
    # "feed_forward_chunk_size" can be used to save memory
    if hidden_states.shape[chunk_dim] % chunk_size != 0:
        raise ValueError(
            f"`hidden_states` dimension to be chunked: {hidden_states.shape[chunk_dim]} has to be divisible by chunk size: {chunk_size}. Make sure to set an appropriate `chunk_size` when calling `unet.enable_forward_chunking`."
        )

    num_chunks = hidden_states.shape[chunk_dim] // chunk_size
    ff_output = torch.cat(
        [ff(hid_slice) for hid_slice in hidden_states.chunk(num_chunks, dim=chunk_dim)],
        dim=chunk_dim,
    )
    return ff_output


class LatentEmbedding(nn.Module):
    def __init__(self, n_channels):
        """
        * `n_channels` is the number of dimensions in the embedding
        """
        super().__init__()
        self.n_channels = n_channels

    def forward(self, z, drop_mask):
        """
        * `z` is the latent code
        * `drop_mask`: mask out the condition if drop_mask == 1
        """
        drop_mask = drop_mask[:, None]
        drop_mask = drop_mask.repeat(1, self.n_channels)
        drop_mask = 1 - drop_mask  # need to flip 0 <-> 1
        z = z * drop_mask
        return z


class BasicTransformerBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            num_attention_heads: int,
            attention_head_dim: int,
            dropout=0.0,
            num_embeds_ada_norm: Optional[int] = None,
            attention_bias: bool = False,
            norm_elementwise_affine: bool = True,
            norm_type: str = "ada_norm_zero",
            norm_eps: float = 1e-5,
            ff_inner_dim: Optional[int] = None,
            ff_bias: bool = False,
            layer_idx: Optional[int] = None,
    ):
        super().__init__()
        assert (num_embeds_ada_norm is not None) and norm_type == "ada_norm_zero", \
            "this class only supports ada_norm_zero."

        self.norm_type = norm_type
        self.num_embeds_ada_norm = num_embeds_ada_norm

        self.time_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True)
        )

        self.latent_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True)
        )

        # 1. Self-Attn
        self.norm1 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)

        self.attn = LlamaAttention(LlamaConfig(
            attention_dropout=dropout,
            hidden_size=dim,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_attention_heads,
            attention_bias=attention_bias,
        ),
            layer_idx=layer_idx,
        )

        self.norm2 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)

        self.ff = LlamaMLP(LlamaConfig(
            hidden_size=dim,
            intermediate_size=ff_inner_dim,
            mlp_bias=ff_bias,
        ))

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int = 0):
        # Sets chunk feed-forward
        self._chunk_size = chunk_size
        self._chunk_dim = dim

    @staticmethod
    def modulate(x, shift, scale):
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            timestep: Optional[torch.LongTensor] = None,
            z_latents: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        # 0. Self-Attention
        time_shift_msa, time_scale_msa, time_gate_msa, time_shift_mlp, time_scale_mlp, time_gate_mlp = (
            self.time_modulation(
                timestep
            ).chunk(6, dim=1))

        latent_shift_msa, latent_scale_msa, latent_gate_msa, latent_shift_mlp, latent_scale_mlp, latent_gate_mlp = (
            self.latent_modulation(
                z_latents
            ).chunk(6, dim=1))

        norm_hidden_states = self.modulate(self.norm1(hidden_states), time_shift_msa, time_scale_msa)
        norm_hidden_states = self.modulate(norm_hidden_states, latent_shift_msa, latent_scale_msa)

        attn_output = self.attn(
            norm_hidden_states,
            encoder_hidden_states=None,
            attention_mask=attention_mask,
        )
        attn_output = time_gate_msa.unsqueeze(1) * attn_output
        attn_output = latent_gate_msa.unsqueeze(1) * attn_output

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        # 4. Feed-forward
        norm_hidden_states = self.modulate(self.norm2(hidden_states), time_shift_mlp, time_scale_mlp)
        norm_hidden_states = self.modulate(norm_hidden_states, latent_shift_mlp, latent_scale_mlp)

        if self._chunk_size is not None:
            # "feed_forward_chunk_size" can be used to save memory
            ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
        else:
            ff_output = self.ff(norm_hidden_states)

        ff_output = time_gate_mlp.unsqueeze(1) * ff_output

        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        return hidden_states


class DiTTransformer2DModel(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
            self,
            num_attention_heads: int = 16,
            attention_head_dim: int = 72,
            in_channels: int = 4,
            out_channels: int = 8,
            num_layers: int = 28,
            dropout: float = 0.0,
            attention_bias: bool = True,
            sample_size: int = 64,
            patch_size: int = 2,
            num_embeds_ada_norm: Optional[int] = 1000,
            norm_type: str = "ada_norm_zero",
            norm_elementwise_affine: bool = False,
            norm_eps: float = 1e-5,
    ):
        super().__init__()

        # Validate inputs.
        if norm_type != "ada_norm_zero":
            raise NotImplementedError(
                f"Forward pass is not implemented when `patch_size` is not None and `norm_type` is '{norm_type}'."
            )
        elif norm_type == "ada_norm_zero" and num_embeds_ada_norm is None:
            raise ValueError(
                f"When using a `patch_size` and this `norm_type` ({norm_type}), `num_embeds_ada_norm` cannot be None."
            )

        # Set some common variables used across the board.
        self.attention_head_dim = attention_head_dim
        self.inner_dim = num_attention_heads * attention_head_dim
        self.out_channels = in_channels if out_channels is None else out_channels
        self.gradient_checkpointing = False

        # 2. Initialize the position embedding and transformer blocks.
        self.height = sample_size
        self.width = sample_size

        self.patch_size = patch_size
        self.pos_embed = PatchEmbed(
            height=sample_size,
            width=sample_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=self.inner_dim,
        )

        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=1)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=num_embeds_ada_norm)
        self.latent_embedder = LatentEmbedding(n_channels=self.inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    self.inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    num_embeds_ada_norm=num_embeds_ada_norm,
                    attention_bias=attention_bias,
                    norm_type=norm_type,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                    layer_idx=layer_idx,
                )
                for layer_idx in range(num_layers)
            ]
        )

        # 3. Output blocks.
        self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.time_modulation = nn.Linear(self.inner_dim, 2 * self.inner_dim)
        self.latent_modulation = nn.Linear(self.inner_dim, 2 * self.inner_dim)
        self.proj_out = nn.Linear(
            self.inner_dim, patch_size * patch_size * self.out_channels
        )
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.pos_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        nn.init.constant_(self.pos_embed.proj.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.timestep_embedder.linear_1.weight, std=0.02)
        nn.init.normal_(self.timestep_embedder.linear_2.weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.time_modulation[-1].weight, 0)
            nn.init.constant_(block.time_modulation[-1].bias, 0)
            nn.init.constant_(block.latent_modulation[-1].weight, 0)
            nn.init.constant_(block.latent_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.time_modulation.weight, 0)
        nn.init.constant_(self.time_modulation.bias, 0)
        nn.init.constant_(self.latent_modulation.weight, 0)
        nn.init.constant_(self.latent_modulation.bias, 0)
        nn.init.constant_(self.proj_out.weight, 0)
        nn.init.constant_(self.proj_out.bias, 0)

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def forward(
            self,
            hidden_states: torch.Tensor,
            timestep: torch.LongTensor,
            z_latents: torch.Tensor,
    ):
        # 1. Input
        hidden_states = self.pos_embed(hidden_states)

        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_states.dtype))  # (N, D)

        # 2. Blocks
        for block in self.transformer_blocks:
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    None,
                    None,
                    None,
                    timesteps_emb,
                    z_latents,
                    **ckpt_kwargs,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    timestep=timesteps_emb,
                    latents=z_latents,
                )

        # 3. Output
        shift, scale = self.time_modulation(F.silu(timesteps_emb)).chunk(2, dim=1)
        hidden_states = self.modulate(hidden_states, shift, scale)
        shift, scale = self.latent_modulation(F.silu(z_latents)).chunk(2, dim=1)
        hidden_states = self.modulate(hidden_states, shift, scale)
        hidden_states = self.proj_out(hidden_states)

        # unpatchify
        height = width = int(hidden_states.shape[1] ** 0.5)
        hidden_states = hidden_states.reshape(
            shape=(-1, height, width, self.patch_size, self.patch_size, self.out_channels)
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(-1, self.out_channels, height * self.patch_size, width * self.patch_size)
        )

        return output


class BottleneckDiTLLaMA(nn.Module):
    def __init__(
            self
    ):
        super().__init__()
        self.transformer = DiTTransformer2DModel()
        self.noise_scheduler = GaussianDiffusion.from_pretrained("facebook/DiT-XL-2-256")
        self.transformer.train()
        self.transformer.enable_gradient_checkpointing()

    def forward(
            self,
            latents,
            z_latents,
    ):
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()
        noise = torch.randn_like(latents)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        z_latents = torch.tensor(z_latents, device=latents.device)
        model_pred = self.transformer(noisy_latents, timesteps, z_latents)
        model_output, model_var_values = torch.split(model_pred, self.transformer.config.in_channels, dim=1)
        frozen_out = torch.cat([model_output.detach(), model_var_values], dim=1)

        loss = (
                F.mse_loss(model_output.float(), noise.float(), reduction="mean") +
                self.noise_scheduler._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=latents,
                    x_t=noisy_latents,
                    t=timesteps,
                    clip_denoised=False,
                )["output"]
        )

        return loss


if __name__ == "__main__":
    model = BottleneckDiTLLaMA()
