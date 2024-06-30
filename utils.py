import torch


class AddGaussianNoise:
    def __init__(self, sigma=0.10):
        self.sigma = sigma

    def __call__(self, tensor):
        assert isinstance(tensor, torch.Tensor)
        dtype = tensor.dtype

        tensor = tensor.float()
        out = tensor + self.sigma * torch.randn_like(tensor)

        if out.dtype != dtype:
            out = out.to(dtype)
        return out


class ProcessorWrapper:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, tensor):
        return self.processor(tensor, return_tensors="pt")['pixel_values'].squeeze(0)
