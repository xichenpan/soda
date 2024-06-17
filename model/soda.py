import torch
import torch.nn as nn


class SODA(nn.Module):
    def __init__(self, encoder, vae, decoder, drop_prob, dtype):
        ''' SODA proposed by "SODA: Bottleneck Diffusion Models for Representation Learning", and \
            DDPM proposed by "Denoising Diffusion Probabilistic Models", as well as \
            DDIM sampler proposed by "Denoising Diffusion Implicit Models".

            Args:
                encoder: A network (e.g. ResNet) which performs image->latent mapping.
                vae: A network (e.g. VAE) which performs image->latent mapping.
                decoder: A network (e.g. UNet) which performs same-shape mapping.
            Parameters:
                betas, n_T, drop_prob
        '''
        super(SODA, self).__init__()
        self.encoder = encoder
        self.vae = vae
        self.decoder = decoder
        self.drop_prob = drop_prob
        self.dtype = dtype

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
        x_target = self.vae.encode(x_target.to(self.dtype)).latent_dist.sample()
        x_target = x_target * self.vae.config.scaling_factor

        # 0 for conditional, 1 for unconditional
        mask = torch.bernoulli(torch.zeros(x_target.shape[0]) + self.drop_prob).to(x_target.device, self.dtype)

        z = self.encoder(x_source.to(self.dtype))
        loss = self.decoder(x_target, z, mask)

        return {"loss": loss}
