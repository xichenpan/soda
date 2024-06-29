import torch
from diffusers.pipelines.pipeline_utils import numpy_to_pil
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel


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


class SODAConfig(PretrainedConfig):
    model_type = 'soda'

    def __init__(self, drop_prob=42, **kwargs):
        super().__init__(**kwargs)
        self.drop_prob = drop_prob


class SODA(PreTrainedModel):
    config_class = SODAConfig

    def __init__(self, config, encoder, vae, decoder):
        super().__init__(config)
        self.encoder = encoder
        self.vae = vae
        self.decoder = decoder
        self.latent_embedder = LatentEmbedding(n_channels=decoder.config.num_embeds_ada_norm)

        self.freeze_modules([self.encoder, self.vae])

    def freeze_modules(self, modules):
        ''' Freeze the specified modules.

            Args:
                modules: The list of modules to be frozen.
        '''
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False

    def forward(self, x_source, x_target):
        ''' Training with simple noise prediction loss.

            Args:
                x_source: The augmented image tensor.
                x_target: The augmented image tensor ranged in `[0, 1]`.
            Returns:
                The simple MSE loss.
        '''
        # get the dtype of encoder params
        with torch.no_grad():
            x_target = self.vae.encode(x_target).latent_dist.sample()
            x_target = x_target * self.vae.config.scaling_factor
            # 0 for conditional, 1 for unconditional
            mask = torch.bernoulli(torch.zeros(x_target.shape[0], device=x_target.device) + self.config.drop_prob)
            z = self.encoder(x_source)

        z = self.latent_embedder(z, mask)
        loss = self.decoder(x_target, z)

        return {"loss": loss}

    def sample_images(self, x_source):
        z = self.encoder(x_source)
        latents = self.decoder.sample(z)
        latents = 1 / self.vae.config.scaling_factor * latents
        samples = self.vae.decode(latents).sample
        samples = (samples / 2 + 0.5).clamp(0, 1)
        samples = samples.cpu().permute(0, 2, 3, 1).float().numpy()
        samples = numpy_to_pil(samples)
        return samples
