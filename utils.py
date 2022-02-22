import numpy as np
import scipy.sparse as sp
import torch
import random


# 设置随机种子 以便于结果可以复显
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True