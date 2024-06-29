from typing import Any, Dict, Optional, Tuple, Union, List

import torch
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.embeddings import PatchEmbed
from diffusers.models.embeddings import Timesteps, TimestepEmbedding
from diffusers.models.modeling_utils import ModelMixin
from diffusers.schedulers import DDPMScheduler, DDIMScheduler
from diffusers.utils import is_torch_version
from diffusers.utils.torch_utils import randn_tensor
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.models.llama.modeling_llama import LlamaMLP, LlamaAttention, LlamaConfig, Cache, apply_rotary_pos_emb, \
    repeat_kv

from vlb_loss import GaussianDiffusion


class LlamaSdpaAttention(LlamaAttention):
    """
    Llama attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `LlamaAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Adapted from LlamaAttention.forward
    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            use_cache: bool = False,
            cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=None,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=False,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


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


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


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

        self.latent_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.num_embeds_ada_norm, 6 * dim, bias=True)
        )

        # 1. Self-Attn
        self.norm1 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)

        self.attn = LlamaSdpaAttention(LlamaConfig(
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
            intermediate_size=ff_inner_dim if ff_inner_dim is not None else int(dim * 4),
            mlp_bias=ff_bias,
        ))

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int = 0):
        # Sets chunk feed-forward
        self._chunk_size = chunk_size
        self._chunk_dim = dim

    def forward(
            self,
            hidden_states,
            attention_mask,
            z_latents,
            time_shift_msa, time_scale_msa, time_gate_msa, time_shift_mlp, time_scale_mlp, time_gate_mlp,
    ) -> torch.Tensor:
        # 0. Self-Attention
        latent_shift_msa, latent_scale_msa, latent_gate_msa, latent_shift_mlp, latent_scale_mlp, latent_gate_mlp = (
            self.latent_modulation(
                z_latents
            ).chunk(6, dim=1))

        norm_hidden_states = modulate(self.norm1(hidden_states), time_shift_msa, time_scale_msa)
        norm_hidden_states = modulate(norm_hidden_states, latent_shift_msa, latent_scale_msa)

        cache_position = torch.arange(
            0, norm_hidden_states.shape[1], device=norm_hidden_states.device
        )
        position_ids = cache_position.unsqueeze(0)

        attn_output = self.attn(
            norm_hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
        )[0]
        attn_output = time_gate_msa.unsqueeze(1) * attn_output
        attn_output = latent_gate_msa.unsqueeze(1) * attn_output

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        # 4. Feed-forward
        norm_hidden_states = modulate(self.norm2(hidden_states), time_shift_mlp, time_scale_mlp)
        norm_hidden_states = modulate(norm_hidden_states, latent_shift_mlp, latent_scale_mlp)

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
            num_embeds_ada_norm: Optional[int] = 1152,
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

        self.time_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(num_embeds_ada_norm, 6 * self.inner_dim, bias=True)
        )

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
        self.final_time_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(num_embeds_ada_norm, 2 * self.inner_dim, bias=True)
        )
        self.final_latent_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(num_embeds_ada_norm, 2 * self.inner_dim, bias=True)
        )
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

        nn.init.constant_(self.time_modulation[-1].weight, 0)
        nn.init.constant_(self.time_modulation[-1].bias, 0)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.transformer_blocks:
            nn.init.constant_(block.latent_modulation[-1].weight, 0)
            nn.init.constant_(block.latent_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_time_modulation[-1].weight, 0)
        nn.init.constant_(self.final_time_modulation[-1].bias, 0)
        nn.init.constant_(self.final_latent_modulation[-1].weight, 0)
        nn.init.constant_(self.final_latent_modulation[-1].bias, 0)
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
        timesteps_emb = self.timestep_embedder(timesteps_proj)  # (N, D)

        time_shift_msa, time_scale_msa, time_gate_msa, time_shift_mlp, time_scale_mlp, time_gate_mlp = (
            self.time_modulation(
                timesteps_emb
            ).chunk(6, dim=1)
        )

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
                    z_latents,
                    time_shift_msa, time_scale_msa, time_gate_msa, time_shift_mlp, time_scale_mlp, time_gate_mlp,
                    **ckpt_kwargs,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    None,
                    z_latents,
                    time_shift_msa, time_scale_msa, time_gate_msa, time_shift_mlp, time_scale_mlp, time_gate_mlp,
                )

        # 3. Output
        shift, scale = self.final_time_modulation(F.silu(timesteps_emb)).chunk(2, dim=1)
        hidden_states = modulate(hidden_states, shift, scale)
        shift, scale = self.final_latent_modulation(F.silu(z_latents)).chunk(2, dim=1)
        hidden_states = modulate(hidden_states, shift, scale)
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


class BottleneckDiTLLaMAConfig(PretrainedConfig):
    model_type = "bottleneck-dit-llama"

    def __init__(
            self,
            num_embeds_ada_norm: int,
            gradient_checkpointing: bool = True,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_embeds_ada_norm = num_embeds_ada_norm
        self.gradient_checkpointing = gradient_checkpointing


class BottleneckDiTLLaMA(PreTrainedModel):
    def __init__(
            self,
            config: BottleneckDiTLLaMAConfig,
    ):
        super().__init__(config)
        self.transformer = DiTTransformer2DModel(num_embeds_ada_norm=config.num_embeds_ada_norm)
        self.noise_scheduler = DDPMScheduler.from_pretrained("facebook/DiT-XL-2-256", subfolder="scheduler")
        self.scheduler = DDIMScheduler.from_pretrained("facebook/DiT-XL-2-256", subfolder="scheduler")
        self.vlb_loss = GaussianDiffusion(
            alphas=self.noise_scheduler.alphas,
            alphas_cumprod=self.noise_scheduler.alphas_cumprod,
            betas=self.noise_scheduler.betas,
        )
        self.transformer.train()
        self.transformer.enable_xformers_memory_efficient_attention()
        if config.gradient_checkpointing:
            self.transformer.enable_gradient_checkpointing()

    def forward(
            self,
            latents,
            z_latents,
    ):
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device,
                                  dtype=torch.long)
        noise = torch.randn_like(latents, device=latents.device)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        model_pred = self.transformer(noisy_latents, timesteps, z_latents)
        model_output, model_var_values = torch.split(model_pred, self.transformer.config.in_channels, dim=1)
        frozen_out = torch.cat([model_output.detach(), model_var_values], dim=1)

        loss = (
            F.mse_loss(model_output.float(), noise.float(), reduction="mean")
            # +
            # self.vlb_loss._vb_terms_bpd(
            #     models=lambda *args, r=frozen_out: r,
            #     x_start=latents,
            #     x_t=noisy_latents,
            #     t=timesteps,
            #     clip_denoised=False,
            # )["output"].mean()
        )

        return loss

    def sample(
            self,
            z_latents,
            guidance_scale: float = 4.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            num_inference_steps: int = 50,
    ):
        batch_size = z_latents.shape[0]
        latent_size = self.transformer.config.sample_size
        latent_channels = self.transformer.config.in_channels

        latents = randn_tensor(
            shape=(batch_size, latent_channels, latent_size, latent_size),
            generator=generator,
            device=z_latents.device,
            dtype=self.transformer.dtype,
        )
        latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1 else latents

        z_latents_null = torch.zeros_like(z_latents, device=z_latents.device)
        z_latents_input = torch.cat([z_latents, z_latents_null], 0) if guidance_scale > 1 else z_latents

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)
        for t in self.scheduler.timesteps:
            if guidance_scale > 1:
                half = latent_model_input[: len(latent_model_input) // 2]
                latent_model_input = torch.cat([half, half], dim=0)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            timesteps = t
            if not torch.is_tensor(timesteps):
                # This would be a good case for the `match` statement (Python 3.10+)
                is_mps = latent_model_input.device.type == "mps"
                if isinstance(timesteps, float):
                    dtype = torch.float32 if is_mps else torch.float64
                else:
                    dtype = torch.int32 if is_mps else torch.int64
                timesteps = torch.tensor([timesteps], dtype=dtype, device=latent_model_input.device)
            elif len(timesteps.shape) == 0:
                timesteps = timesteps[None].to(latent_model_input.device)
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timesteps = timesteps.expand(latent_model_input.shape[0])
            # predict noise model_output
            noise_pred = self.transformer(
                latent_model_input, timestep=timesteps, z_latents=z_latents_input
            )

            # perform guidance
            if guidance_scale > 1:
                eps, rest = noise_pred[:, :latent_channels], noise_pred[:, latent_channels:]
                cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)

                half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
                eps = torch.cat([half_eps, half_eps], dim=0)

                noise_pred = torch.cat([eps, rest], dim=1)

            # learned sigma
            if self.transformer.config.out_channels // 2 == latent_channels:
                model_output, _ = torch.split(noise_pred, latent_channels, dim=1)
            else:
                model_output = noise_pred

            # compute previous image: x_t -> x_t-1
            latent_model_input = self.scheduler.step(model_output, t, latent_model_input).prev_sample

        if guidance_scale > 1:
            latents, _ = latent_model_input.chunk(2, dim=0)
        else:
            latents = latent_model_input

        return latents
