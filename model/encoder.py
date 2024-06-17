import torch.nn as nn
from transformers import SiglipVisionModel


class Encoder(nn.Module):
    """
    An encoder network (image -> feature_dim)
    """

    def __init__(self, gradient_checkpointing):
        super(Encoder, self).__init__()
        self.model = SiglipVisionModel.from_pretrained("google/siglip-so400m-patch14-384")
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

    def forward(self, x):
        return self.model(x, output_attentions=False, output_hidden_states=False).pooler_output
