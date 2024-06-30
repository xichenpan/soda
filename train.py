import logging
import os
from dataclasses import dataclass

import datasets
import torch
import transformers
from datasets import load_dataset
from diffusers.models import AutoencoderKL
from huggingface_hub import login
from torchvision import transforms
from transformers import SiglipImageProcessor
from transformers.trainer_utils import get_last_checkpoint

from data.dataset import get_dataset
from models.dit import BottleneckDiTLLaMAConfig, BottleneckDiTLLaMA
from models.encoder import EncoderConfig, Encoder
from models.soda import SODAConfig, SODA
from trainer import SODATrainer
from utils import ProcessorWrapper, AddGaussianNoise

login(token="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")
os.environ["WANDB_PROJECT"] = "SODA"
logger = logging.getLogger(__name__)

datasets.config.IN_MEMORY_MAX_SIZE = 200_000_000_000


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
    output_dir: str = '/data/home/xichenpan/output'
    data_dir: str = '/data/home/xichenpan/.cache'
    eval_strategy: str = 'steps'
    eval_steps: int = 500
    per_device_train_batch_size: int = 72
    per_device_eval_batch_size: int = 4
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
    logging_dir: str = '/data/home/xichenpan/log'
    logging_steps: int = 1
    save_steps: int = 2000
    save_total_limit: int = 30
    restore_callback_states_from_checkpoint: bool = True
    seed: int = 42
    data_seed: int = 42
    bf16: bool = True
    dataloader_num_workers: int = os.getenv("OMP_NUM_THREADS", 12)
    dataloader_persistent_workers: bool = True
    dataloader_drop_last: bool = True
    dataloader_prefetch_factor: int = 2
    remove_unused_columns: bool = False
    run_name: str = 'test'
    report_to: str = 'wandb'
    ddp_find_unused_parameters: bool = False
    _gradient_checkpointing: bool = False
    overwrite_output_dir: bool = True


if __name__ == "__main__":
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    global local_rank
    local_rank = training_args.local_rank

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

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
        return source_transform(item["image"].convert("RGB")), target_transform(item["image"].convert("RGB"))


    def collate_fn(batch):
        source, target = [list(x) for x in zip(*batch)]
        return {
            "x_source": torch.stack(source),
            "x_target": torch.stack(target)
        }


    train_dataset = get_dataset(
        process_fn=process_func,
        data_dir=training_args.data_dir,
        seed=training_args.data_seed,
    )


    def eval_process_func(batch):
        images = [img.convert("RGB") for img in batch["image"]]
        return {
            "x_source": [source_transform(img) for img in images],
        }


    eval_dataset = load_dataset(
        "ILSVRC/imagenet-1k", trust_remote_code=True, cache_dir=training_args.data_dir,
        split="validation"
    )
    eval_dataset = eval_dataset.select(range(32))
    eval_dataset = eval_dataset.map(eval_process_func, batched=True,
                                    batch_size=training_args.per_device_train_batch_size,
                                    remove_columns=["image", "label"])
    eval_dataset = eval_dataset.with_format("torch")

    trainer = SODATrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
    )

    trainer.train(resume_from_checkpoint=last_checkpoint)
