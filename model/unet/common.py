import os
import random

import numpy as np
import torch
from einops import repeat


def expand_to_batch(tensor, desired_size):
    tile = desired_size // tensor.shape[0]
    return repeat(tensor, 'b ... -> (b tile) ...', tile=tile)
  # einops.repeat 允许对元素重新排序并以任意组合重复它们。此操作包括重复、平铺、广播功能的功能。


def init_random_seed(seed, gpu=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if gpu:
        torch.backends.cudnn.deterministic = True
