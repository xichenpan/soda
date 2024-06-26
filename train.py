import os
from dataclasses import dataclass

import datasets
import torch
import transformers
from accelerate import Accelerator
from datasets import load_dataset
from diffusers.models import AutoencoderKL
from huggingface_hub import login
from torchvision import transforms
from transformers import SiglipImageProcessor

from model.dit import BottleneckDiTLLaMAConfig, BottleneckDiTLLaMA
from model.encoder import EncoderConfig, Encoder
from model.soda import SODAConfig, SODA
from utils import ProcessorWrapper, AddGaussianNoise

datasets.config.IN_MEMORY_MAX_SIZE = 1800000000000

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
class TrainingArguments:
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
    mixed_precision: str = 'bf16'
    dataloader_num_workers: int = 24
    dataloader_persistent_workers: bool = True
    dataloader_drop_last: bool = True
    dataloader_prefetch_factor: int = 2
    remove_unused_columns: bool = False
    run_name: str = 'test'
    report_to: str = 'wandb'
    ddp_find_unused_parameters: bool = False
    gradient_checkpointing: bool = True


if __name__ == "__main__":
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    assert data_args.target_image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = data_args.target_image_size // 8

    encoder = Encoder(
        EncoderConfig(
            encoder_id=model_args.encoder_id,
            gradient_checkpointing=training_args.gradient_checkpointing
        )
    )
    decoder = BottleneckDiTLLaMA(
        BottleneckDiTLLaMAConfig(
            num_embeds_ada_norm=encoder.model.config.hidden_size,
            gradient_checkpointing=training_args.gradient_checkpointing
        )
    )
    model = SODA(
        config=SODAConfig(
            drop_prob=model_args.drop_prob
        ),
        encoder=encoder,
        vae=AutoencoderKL.from_pretrained("stabilityai/sdxl-vae"),
        decoder=decoder
    )

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


    def filter_func(batch):
        return [img is not None for img in batch["image"]]


    def process_func(batch):
        images = [img.convert("RGB") for img in batch["image"]]
        return [source_transform(img) for img in images], [target_transform(img) for img in images]


    # If the dataset is gated/private, make sure you have run huggingface-cli login
    dataset = load_dataset("ILSVRC/imagenet-1k", trust_remote_code=True, cache_dir=training_args.data_dir,
                           split="train", keep_in_memory=True)
    dataset.info.task_templates = None
    dataset = dataset.to_iterable_dataset(num_shards=128)
    dataset = dataset.filter(filter_func, batched=True, batch_size=training_args.per_device_train_batch_size)
    dataset = dataset.map(process_func, batched=True, batch_size=training_args.per_device_train_batch_size,
                          remove_columns=["image", "label"])
    dataset = dataset.shuffle(seed=training_args.data_seed)
    dataset = dataset.with_format("torch")

    training_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=training_args.per_device_train_batch_size,
        num_workers=training_args.dataloader_num_workers,
        persistent_workers=training_args.dataloader_persistent_workers,
        drop_last=training_args.dataloader_drop_last,
        prefetch_factor=training_args.dataloader_prefetch_factor,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_args.learning_rate,
        betas=(training_args.adam_beta1, training_args.adam_beta2),
        eps=training_args.adam_epsilon,
        weight_decay=training_args.weight_decay,
    )

    scheduler = transformers.get_scheduler(
        training_args.lr_scheduler_type,
        optimizer,
        num_warmup_steps=training_args.warmup_steps,
    )

    accelerator = Accelerator(
        mixed_precision=training_args.mixed_precision,

    )

    device = accelerator.device
    model, optimizer, training_dataloader, scheduler = accelerator.prepare(
        model, optimizer, training_dataloader, scheduler
    )

    for batch in training_dataloader:
        optimizer.zero_grad()
        x_source, x_target = batch
        x_source = x_source.to(device)
        x_target = x_target.to(device)
        loss = model(x_source, x_target)
        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()

        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            training_args.output_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
        )
