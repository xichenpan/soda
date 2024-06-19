conda install \
  pytorch=2.3 pytorch-cuda=12.1 \
  torchvision torchaudio \
  --strict-channel-priority \
  --override-channels \
  -c https://aws-ml-conda.s3.us-west-2.amazonaws.com \
  -c pytorch \
  -c nvidia \
  -c conda-forge
pip install -U transformers[deepspeed]
pip install -U accelerate
pip install -U diffusers
pip install -U datasets
pip install -U xformers
pip install wandb
pip install timm