# 把 can 数据 处理为 图数据
# 固定长度{300}的 can 数据 构成 一个图
# 本文件独立运行 用来根据 原始can报文数据 生成 图网络数据集
import pandas as pd
import os
import copy
import collections


# 处理 数据集 Car_Hacking_Challenge_Dataset_rev20Mar2021
class OnlyGraphData:
    def __init__(self, can_csv_dir, fixed_len):
        self.fixed_len = fixed_len  # 指定 每个图需要的 can 数据长度
        col_names = ['Arbitration_ID', 'Class']
        self.df = pd.read_csv(can_csv_dir, usecols=col_names)  # 得到全部数据
        print(f'全部报文长度为: {len(self.df)}')
        print(f'共产生 {len(self.df)/fixed_len} 个图数据')
        # output_ds_dir = os.path.dirname(can_csv_dir)  # 放在 与 原始csv 同路径下
        self.output_name_suffix = os.path.dirname(can_csv_dir) + '/' + os.path.basename(os.path.splitext(can_csv_dir)[0])

    def get_a(self):

        # 节点相关
        can_id_list = self.df.get("Arbitration_ID").values  # 取出 ID 得到numpy类型 数据
        can_type_list = self.df.get("Class").values  # 取出 can 数据 的 报文类型 数据

        # 转换为 十六进制ID 为键 十进制编号 为键值 的 节点字典
        # 字典为有序字典
        hex2dec_dict = collections.OrderedDict()
        # 存放 每个图的 十六进制ID 和 十进制编号的 字典 列表
        # 列表每个元素 代表了一个图的 十六进制ID 和 十进制编号的 对应关系
        hex2dec_dict_list = list()
        # print('###')
        # print(len(can_id_list))
        # 同一个图内 ID相同 转换到的 十进制(节点编号) 也相同
        # 不同图 即使ID相同 转换到的 十进制(节点编号) 不相同
        graph_num = 0  # 遍历到的图的 位置
        read_done = False  # 标志 数据是否读取完成
        while True:
            for i in range(self.fixed_len):  # 逐个图编号节点 # 遍历所有的 can ID 转换为 十进制
                try:
                    if can_id_list[graph_num*self.fixed_len+i] not in hex2dec_dict.keys():  # 如果 此十六进制 ID 在此图中 未出现过 则需要新的节点编号
                        # ID 的节点标签是 递增的
                        a = sum([len(h2d_dic) for h2d_dic in hex2dec_dict_list]) + len(hex2dec_dict) + 1
                        # print(f'a: {a}')
                        hex2dec_dict[can_id_list[graph_num*self.fixed_len+i]] = a # 新出现一个 十六进制ID 则加入新键值对 键值 为 递增 节点标签 从 0 递增
                except IndexError:
                    # 如果最后一个图的数据 不够组成一个图 则舍弃
                    print('can报文 文件读取完毕')
                    read_done = True
                    break
            # 文件读取完成 则退出 while循环
            if read_done:
                break
            # 读取完一个 图数据 把读取到的对应字典加入 列表
            hex2dec_dict_list.append(copy.deepcopy(hex2dec_dict))
            # 清空字典
            hex2dec_dict.clear()
            # 进行下一个图的 遍历
            graph_num += 1

        # 遍历 hex2dec_dict_list 里的每个字典 并输出文件
        # 每个字典的 键值(编号) 都应该是 递增 1 的
        # 输出文件 用来检验 遍历字典的顺序 是按照 字典的加入顺序进行的
        # 若第一列的数字 按行递增 1 则检验通过
        with open(self.output_name_suffix+'_check_node_num_dict.txt', 'w+') as f:
            for i in hex2dec_dict_list:
                for key, value in i.items():
                    # print(str(value) + '  ' + str(key) + '\r\n')
                    f.write(str(value) + '  ' + str(key) + '\r\n')
        f.close()

        # 节点标签 虽然节点的编号不同 但是可能用有相同的节点标签
        # 节点标签从 1 开始
        # 因为 节点的编号因为在不同的图中 所以相同 CAN ID 的节点却有不同的节点编号 但是 他们的节点标签却是相同的
        # 遍历 全部数据 得到 CAN ID 和 节点标签的对应字典 键名：十六进制 字符串型 CAN ID 键值：十进制 int 型 节点标签
        with open(self.output_name_suffix+'_node_labels.txt', 'w+') as f:
            node_label_dict = collections.OrderedDict()
            for i in can_id_list:  # 遍历每条数据的 ID
                if i not in node_label_dict.keys():
                    node_label_dict[i] = len(node_label_dict) + 1
            for graph_nodes_dic in hex2dec_dict_list:  # 遍历每个图
                for node_num in graph_nodes_dic:  # 遍历图里的 节点编号 和 十六进制CAN ID 对；根据CAN ID 来写对应 的 节点标签
                    f.write(str(node_label_dict[node_num]) + '\n')
        f.close()

        # 检验步骤
        # 输出 十六进制报文ID 和 节点标签的对应关系
        # 如果节点标签列 每行递增 1 则通过检查
        with open(self.output_name_suffix+'_check_node_label_dict.txt', 'w+') as f:
            for key, value in node_label_dict.items():
                # print(str(value) + '  ' + str(key) + '\r\n')
                f.write(str(value) + '  ' + str(key) + '\r\n')
        f.close()


        # 获得 DS_A 边 文件
        graph_num = 0  # 遍历到的图的 位置
        graph_adj_list = list()  # 储存全部图的 连接边 每个元素表示一个图的连接关系
        adj_list = list()  # 图的边 每个元素是一个元祖 表示一条边
        read_done = False
        while True:
            for i in range(self.fixed_len):  # 逐个图 遍历 节点的连接关系
                try:

                    if (hex2dec_dict_list[graph_num][can_id_list[graph_num*self.fixed_len+i]], hex2dec_dict_list[graph_num][can_id_list[graph_num*self.fixed_len+i+1]]) not in adj_list:  # 如果边没在 列表中 则 添加入 列表
                        adj_list.append((hex2dec_dict_list[graph_num][can_id_list[graph_num*self.fixed_len+i]], hex2dec_dict_list[graph_num][can_id_list[graph_num*self.fixed_len+i+1]]))
                    # print("###")
                except KeyError:
                    # print('%%%')
                    # 如果最后一个图的数据 不够组成一个图 则舍弃
                    if i == self.fixed_len - 1:
                        continue
                    else:
                        print(f'KeyError 边 构造完毕 共构造了 {graph_num} 个图')
                        read_done = True
                        break
                except IndexError:
                    print(f'IndexError 边 构造完毕 共构造了 {graph_num} 个图')
                    read_done = True
                    break
                #     print('keyError_A')
                #     print(i)
                #     print(graph_num)
                #     print(graph_num*self.fixed_len+i)
                #     print(hex2dec_dict_list[graph_num][can_id_list[graph_num*self.fixed_len+i]])
                #     print(hex2dec_dict_list[graph_num][can_id_list[graph_num*self.fixed_len+i+1]])

            # 文件读取完成 则退出 while循环
            if read_done:
                break
            # 读取完一个 图数据 把读取到的边列表 加入 列表
            # print(f'adj_list: \n{adj_list}')
            graph_adj_list.append(copy.deepcopy(adj_list))
            # 清空列表
            adj_list.clear()
            # 进行下一个图的 遍历
            graph_num += 1

        # 写入 边文件
        with open(self.output_name_suffix + '_A.txt', 'w+') as f:
            for adj_list_ in graph_adj_list:  # 遍历每个图的边列表
                for edge_ in adj_list_:
                    f.write(str(edge_[0]) + ', ' + str(edge_[1]) + '\n')
        f.close()


        # 获得 图表示文件 DS_graph_indicator
        # 虽然每个 图数据是固定长度的 can 报文数据 但是并不代表每个图的节点数相同
        # 可以使用 十六进制对应十进制的 字典来获取
        # 图标识 从 1 开始 有多少个图 就到 多少
        with open(self.output_name_suffix + '_graph_indicator.txt', 'w+') as f:
            for graph_index_, h2d_dic_ in enumerate(hex2dec_dict_list):  # 遍历 每个图
                # print(len(h2d_dic_))
                for i in range(len(h2d_dic_)):  # 该图有多少个节点 每个节点都加上属于哪个图的标识
                    f.write(str(graph_index_ + 1) + '\n')
        f.close()


        # 获得 图标签文件
        # 图标签 正常报文 1 异常报文 2
        with open(self.output_name_suffix + '_graph_labels.txt', 'w+') as f:
            for graph_num in range(len(hex2dec_dict_list)):  # 遍历所有图的报文
                for i in range(self.fixed_len):  # 遍历一个图的 所有报文类型 只要出现 attack 类型 则报文类型 为 attack
                    node_label = 1  # 设定初始标签为 正常报文
                    if can_type_list[graph_num*self.fixed_len+i] == 'Attack':
                        node_label = 2
                        # 若报文里出现一帧异常报文 则该图的标签是 2
                        break
                # 遍历完一个图 写入 文件
                f.write(str(node_label) + '\n')
        print(f'graph_label文件写入完毕 共 {graph_num+1} 个图')
        f.close()


if __name__ == '__main__':
    csv_dir = '/Users/aaron/PycharmProjects/DynamicCANGraphLen/graphModel/data/Car_Hacking_Challenge_Dataset_rev20Mar2021/0_Preliminary/0_Training/Pre_train_D_0.csv'
    p = OnlyGraphData(can_csv_dir=csv_dir, fixed_len=300)
    p.get_a()

    # # 查看 _A 文件
    # df_a = pd.read_csv('/Users/aaron/PycharmProjects/DynamicCANGraphLen/graphModel/data/Car_Hacking_Challenge_Dataset_rev20Mar2021/0_Preliminary/0_Training/D_1_A.txt')
    # print(df_a.tail())
    #
    # # 查看 节点编号 和 CAN ID 对应文件
    # df_a = pd.read_csv('/Users/aaron/PycharmProjects/DynamicCANGraphLen/graphModel/data/Car_Hacking_Challenge_Dataset_rev20Mar2021/0_Preliminary/0_Training/D_1_check_node_num_dict.txt', sep='  ')
    # print(df_a.tail())