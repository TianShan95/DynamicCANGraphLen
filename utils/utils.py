import numpy as np
import scipy.sparse as sp
import torch
import random
import os


# 设置随机种子 以便于结果可以复显
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True


# 生成一定数量的随机数
def random_list(start, stop, length):
    if length >= 0:
        length = int(length)
    start, stop = (int(start), int(stop)) if start <= stop else (int(stop), int(start))
    random_list = []
    for i in range(length):
        random_list.append(random.randint(start, stop))
    return random_list


# 存放需要的小功能函数
def ensure_dir(file_path):
    '''
    :param file_path:  创建文件夹
    :return: 检查 并 创建
    '''
    try:
        os.mkdir(file_path)
    except FileExistsError:
        print("Folder already found")