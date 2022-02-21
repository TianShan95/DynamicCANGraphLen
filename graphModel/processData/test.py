import os

# file = '/Users/aaron/Hebut/Bin.Cao/车辆网络安全/数据集/Car_Hacking_Challenge_Dataset_rev20Mar2021/0_Preliminary/0_Training/Pre_train_D_0.csv'
# # dir = os.path.dirname(file)
# base_name = os.path.basename(os.path.splitext(file)[0])[-3:]
# print(base_name)

# l = list()
# a = {1:'a', 2:'b', 3:'c'}
b = {7:'d', 5:'e', 6:'f', 2:'g'}
for i in b:
    print(i)
# l.append(a)
# l.append(b)
# n = 0
# c = sum([len(i) for i in l])
# print(c)

# print(a)
# a.clear()
# print(a)

# i = 10
# print(i)
# for i in range(2):
#     print(i)
#
# for i in range(2):
#     print(i)


# import collections
#
# a = collections.OrderedDict()
# a['10'] = 'a'
# a['4'] = 'b'
# a['1'] = 'c'
# print(a)
# for key, value in a.items():
#     print(key + ' ', value)
# print('###')
# for i in range(2):
#     print(i)
#
# print(bool(0))
# import pandas as pd
#
# node_label_df = pd.read_csv('node_label_dict.txt', sep='  ', usecols=['label', 'id'], engine='python')
# result_dic = node_label_df.groupby('id')['label'].apply(list).to_dict()
#
# print(result_dic)

origin_can_dir_ = '/Users/aaron/git_project/eigenpooling/data/Car_Hacking_Challenge_Dataset_rev20Mar2021/' \
                  '0_Preliminary/0_Training/Pre_train_D_1.csv'
n = 0
with open(origin_can_dir_) as f:
    for line in f:
        n += 1
        print(line)
        if n == 10:
            break