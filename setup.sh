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
pip install git+https://github.com/microsoft/infinibatch
pip install OmegaConf
pip install numpy==1.26.4

conda install cupy pkg-config libjpeg-turbo opencv numba -c conda-forge -c pytorch
conda update ffmpeg
pip install ffcv
