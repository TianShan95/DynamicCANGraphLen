# import random
#
# graph_len_ = random.randint(0, 1)
# print(graph_len_)
# import torch
# # # mode_path = '/Users/aaron/Hebut/征稿_图像信息安全_20211130截稿/源程序/图塌缩分类/graphModelForRl/0.804806_better_model_2022-02-21 11_03_28_totalEpoch_5_epoch_4_ps_10_gs_300_nor_1_gs_50.pth'
# model_path = '/Users/aaron/Hebut/征稿_图像信息安全_20211130截稿/源程序/图塌缩分类/log/drive-download-20220221T024621Z-001/graphSize_300_Normlize_True_20220220_112631_log/0.8443854995579133_better_model_2022-02-20 11_36_25_totalEpoch_200_epoch_0_ps_10_gs_300_nor_1.pth'
# print(torch.load(model_path, map_location=torch.device('cpu')))

import random
from random import shuffle

random.seed(1)
print(random.randint(0, 2))
print(random.randint(0, 2))

l = [x for x in range(10)]
print(l)
shuffle(l)
print(l)
shuffle(l)
print(l)