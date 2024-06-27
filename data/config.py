from omegaconf import OmegaConf
import os

cur_dir = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(cur_dir, 'config.yaml')

dataconfig = OmegaConf.load(config_path)
