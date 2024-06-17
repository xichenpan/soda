import os
from dataclasses import dataclass, field
from typing import List, Union

import torch
import transformers
from datasets import load_dataset
from diffusers.models import AutoencoderKL
from huggingface_hub import login
from torchvision import transforms
from transformers import SiglipImageProcessor
from transformers import Trainer

from model.soda import SODA
from model.dit import BottleneckDiTLLaMA
from model.encoder import Encoder
from diffusers.training_utils import EMAModel

login(token="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")
os.environ["WANDB_PROJECT"] = "SODA"


class AddGaussianNoise():
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


@dataclass
class ModelArguments:
    # betas: List[float] = field(default_factory=lambda: [1.0e-4, 0.02])
    betas = "cosine"
    n_T: int = 1000
    drop_prob: float = 0.1
    z_channels: int = 1152


@dataclass
class DataArguments:
    source_image_size: int = 385
    target_image_size: int = 512


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    output_dir: str = '/fsx-project/xichenpan/output'
    overwrite_output_dir: bool = True
    eval_strategy: str = 'no'
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    optim: str = 'adamw_torch_fused'
    max_steps: int = int(1e10)
    learning_rate: float = 5e-5
    weight_decay: float = 0.05
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 0.5
    lr_scheduler_type: str = 'constant_with_warmup'
    warmup_steps: int = 5000
    logging_dir: str = '/fsx-project/xichenpan/log'
    logging_steps: int = 100
    save_steps: int = 5000
    save_total_limit: int = 30
    restore_callback_states_from_checkpoint: bool = True
    seed: int = 42
    data_seed: int = 42
    bf16: bool = True
    dataloader_num_workers: int = 4
    dataloader_persistent_workers: bool = True
    remove_unused_columns: bool = False
    run_name: str = 'test'
    report_to: str = 'wandb'
    # gradient_checkpointing: bool = True


if __name__ == "__main__":
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    device = "cuda:%d" % local_rank

    assert data_args.target_image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = data_args.target_image_size // 8
    encoder = Encoder()
    decoder = BottleneckDiTLLaMA()
    model = SODA(
        encoder=encoder,
        vae=AutoencoderKL.from_pretrained("stabilityai/sdxl-vae"),
        decoder=decoder,
        drop_prob=model_args.drop_prob,
        device=device,
    )

    if local_rank == 0:
        params = sum(p.numel() for p in encoder.parameters() if p.requires_grad) / 1e6
        print(f"encoder # params: {params:.1f}")
        params = sum(p.numel() for p in decoder.parameters() if p.requires_grad) / 1e6
        print(f"decoder # params: {params:.1f}")

    model.to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    processor = SiglipImageProcessor.from_pretrained("google/siglip-so400m-patch14-384")

    # for encoder, at the training
    source_transform = transforms.Compose([
        transforms.RandomApply([
            transforms.RandomResizedCrop(data_args.source_image_size),
            transforms.RandomHorizontalFlip(),
        ], p=0.95),
        transforms.Resize(data_args.source_image_size),
        transforms.CenterCrop(data_args.source_image_size),
        transforms.RandomApply([
            transforms.RandAugment(),
        ], p=0.65),
        ProcessorWrapper(processor),
        AddGaussianNoise(),
    ])
    # for decoder, at the training
    target_transform = transforms.Compose([
        transforms.RandomApply([
            transforms.RandomResizedCrop(data_args.target_image_size),
            transforms.RandomHorizontalFlip(),
        ], p=0.95),
        transforms.Resize(data_args.target_image_size),
        transforms.CenterCrop(data_args.target_image_size),
        transforms.RandomApply([
            transforms.RandAugment(),
        ], p=0.65),
        transforms.ToTensor(),
    ])


    def process_func(batch):
        images = [img.convert("RGB") for img in batch["image"]]
        return {
            "x_source": [source_transform(img) for img in images],
            "x_target": [target_transform(img) for img in images],
        }


    # If the dataset is gated/private, make sure you have run huggingface-cli login
    dataset = load_dataset("ILSVRC/imagenet-1k", trust_remote_code=True, cache_dir="/fsx-project/xichenpan/.cache",
                           split="validation")
    dataset = dataset.to_iterable_dataset(num_shards=len(dataset) // 1024)
    dataset = dataset.map(process_func, batched=True, batch_size=training_args.per_device_train_batch_size,
                          remove_columns=["image", "label"])
    dataset = dataset.with_format("torch")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    #
    #
    # def save_model_hook(models, weights, output_dir):
    #     if trainer.args.local_rank in [-1, 0]:
    #         ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))
    #
    #         for i, model in enumerate(models):
    #             model.save_pretrained(os.path.join(output_dir, "unet"))
    #
    #             # make sure to pop weight so that corresponding model is not saved again
    #             weights.pop()
    #
    #
    # def load_model_hook(models, input_dir):
    #     load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DConditionModel)
    #     ema_unet.load_state_dict(load_model.state_dict())
    #     ema_unet.to(accelerator.device)
    #     del load_model
    #
    #     for _ in range(len(models)):
    #         # pop models so that they are not loaded again
    #         model = models.pop()
    #
    #         # load diffusers style into model
    #         load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
    #         model.register_to_config(**load_model.config)
    #
    #         model.load_state_dict(load_model.state_dict())
    #         del load_model
    #
    #
    # trainer.register_save_state_pre_hook(save_model_hook)
    # trainer.register_load_state_pre_hook(load_model_hook)

    trainer.train()
