import os
from dataclasses import dataclass

import torch
import transformers
from diffusers.models import AutoencoderKL
from huggingface_hub import login
from torchvision import transforms
from transformers import SiglipImageProcessor
from transformers import Trainer

from data.dataset import get_dataset
from model.dit import BottleneckDiTLLaMA
from model.encoder import Encoder
from model.soda import SODA
from utils import ProcessorWrapper, AddGaussianNoise

login(token="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")
os.environ["WANDB_PROJECT"] = "SODA"


@dataclass
class ModelArguments:
    drop_prob: float = 0.1
    encoder_id: str = "google/siglip-large-patch16-256"


@dataclass
class DataArguments:
    source_image_size: int = 256
    target_image_size: int = 256


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    # deepspeed = "./deepspeed_config.json"
    output_dir: str = '~/xichenpan/output'
    data_dir: str = '~/xichenpan/.cache'
    overwrite_output_dir: bool = True
    eval_strategy: str = 'no'
    per_device_train_batch_size: int = 320
    gradient_accumulation_steps: int = 1
    optim: str = 'adamw_torch_fused'
    max_steps: int = int(1e10)
    learning_rate: float = 1e-4
    weight_decay: float = 0.05
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 0.5
    lr_scheduler_type: str = 'constant_with_warmup'
    warmup_steps: int = 5000
    logging_dir: str = '~/xichenpan/log'
    logging_steps: int = 1
    save_steps: int = 500
    save_total_limit: int = 30
    restore_callback_states_from_checkpoint: bool = True
    seed: int = 42
    data_seed: int = 42
    bf16: bool = True
    dataloader_num_workers: int = 24
    dataloader_persistent_workers: bool = True
    dataloader_drop_last: bool = True
    dataloader_prefetch_factor: int = 2
    remove_unused_columns: bool = False
    run_name: str = 'test'
    report_to: str = 'wandb'
    ddp_find_unused_parameters: bool = False
    _gradient_checkpointing: bool = True


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

    assert data_args.target_image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = data_args.target_image_size // 8
    encoder = Encoder(model_args.encoder_id, training_args._gradient_checkpointing)
    decoder = BottleneckDiTLLaMA(
        num_embeds_ada_norm=encoder.model.config.hidden_size,
        gradient_checkpointing=training_args._gradient_checkpointing
    )
    model = SODA(
        encoder=encoder.to(compute_dtype),
        vae=AutoencoderKL.from_pretrained("stabilityai/sdxl-vae", torch_dtype=compute_dtype),
        decoder=decoder.to(compute_dtype),
        drop_prob=model_args.drop_prob,
        dtype=compute_dtype,
    )

    if local_rank == 0:
        params = sum(p.numel() for p in encoder.parameters() if p.requires_grad) / 1e6
        print(f"encoder # params: {params:.1f}")
        params = sum(p.numel() for p in decoder.parameters() if p.requires_grad) / 1e6
        print(f"decoder # params: {params:.1f}")

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    processor = SiglipImageProcessor.from_pretrained(model_args.encoder_id)

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


    def process_func(item):
        return source_transform(item["image"]), target_transform(item["image"])


    def collate_fn(batch):
        source, target = [list(x) for x in zip(*batch)]
        return {
            "x_source": torch.stack(source),
            "x_target": torch.stack(target)
        }


    dataset = get_dataset(
        process_fn=process_func,
        data_dir=training_args.data_dir,
        seed=training_args.data_seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn,
    )

    trainer.train()
