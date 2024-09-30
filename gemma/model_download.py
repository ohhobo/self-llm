import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer
import os

model_dir = snapshot_download('Lucachen/gemma2b', cache_dir='../models', revision='master')
