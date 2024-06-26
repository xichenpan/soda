import torch
from transformers import PretrainedConfig, PreTrainedModel


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
            mask = torch.bernoulli(torch.zeros(x_target.shape[0]) + self.config.drop_prob)
            z = self.encoder(x_source)

        loss = self.decoder(x_target, z, mask)

        return {"loss": loss}
