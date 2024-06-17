import os

import torch
import torch.nn as nn


def normalize_to_neg_one_to_one(img):
    # [0.0, 1.0] -> [-1.0, 1.0]
    return img * 2 - 1


class SODA(nn.Module):
    def __init__(self, encoder, vae, decoder, device, drop_prob, **kwargs):
        ''' SODA proposed by "SODA: Bottleneck Diffusion Models for Representation Learning", and \
            DDPM proposed by "Denoising Diffusion Probabilistic Models", as well as \
            DDIM sampler proposed by "Denoising Diffusion Implicit Models".

            Args:
                encoder: A network (e.g. ResNet) which performs image->latent mapping.
                vae: A network (e.g. VAE) which performs image->latent mapping.
                decoder: A network (e.g. UNet) which performs same-shape mapping.
                device: The CUDA device that tensors run on.
            Parameters:
                betas, n_T, drop_prob
        '''
        super(SODA, self).__init__()
        self.encoder = encoder.to(device)
        self.vae = vae.to(device)
        self.decoder = decoder.to(device)
        self.device = device
        self.drop_prob = drop_prob

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
        x_target = normalize_to_neg_one_to_one(x_target)
        x_target = self.vae.encode(x_target.to(self.decoder.dtype)).latent_dist.sample()
        x_target = x_target * self.vae.config.scaling_factor

        x_noised, t, noise = self.perturb(x_target, t=None)

        # 0 for conditional, 1 for unconditional
        mask = torch.bernoulli(torch.zeros(x_noised.shape[0]) + self.drop_prob).to(self.device)

        z = self.encoder(x_source)

        B, C = x_t.shape[:2]
        model_output, model_var_values = torch.split(model_output, C, dim=1)
        # Learn the variance using the variational bound, but don't let
        # it affect our mean prediction.
        frozen_out = torch.cat([model_output.detach(), model_var_values], dim=1)
        terms["vb"] = self._vb_terms_bpd(
            model=lambda *args, r=frozen_out: r,
            x_start=x_start,
            x_t=x_t,
            t=t,
            clip_denoised=False,
        )["output"]

        target = noise
        assert model_output.shape == target.shape == x_start.shape
        terms["mse"] = mean_flat((target - model_output) ** 2)
        terms["loss"] = terms["mse"] + terms["vb"]

        return loss

    def encode(self, x, norm=False):
        z = self.encoder(x)
        if norm:
            z = torch.nn.functional.normalize(z)
        return z
