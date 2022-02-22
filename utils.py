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



# 存放需要的小功能函数
import os

def ensure_dir(file_path):
    '''
    :param file_path:  创建文件夹
    :return: 检查 并 创建
    '''
    try:
        os.mkdir(file_path)
    except FileExistsError:
        print("Folder already found")