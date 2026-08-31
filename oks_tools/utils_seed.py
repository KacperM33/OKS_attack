import os
import random
import numpy as np
import torch
from mmengine.runner import set_random_seed

def set_deterministic_seed(seed=42):
    print(f"\n[INFO] Setting global random seed to: {seed}\n")

    # python
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # numpy
    np.random.seed(seed)

    # pytorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # OpenMMLab
    set_random_seed(seed)