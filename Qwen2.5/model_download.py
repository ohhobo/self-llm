import torch
from modelscope import snapshot_download
import os
model_dir = snapshot_download('qwen/Qwen2.5-7B-Instruct', cache_dir='./models', revision='master')
