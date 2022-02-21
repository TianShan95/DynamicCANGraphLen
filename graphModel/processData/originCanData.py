import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


class OriginCanData:
    def __init__(self, origin_can_dir, max_nodes_num):
        col_name_list = ['Arbitration_ID', 'Class']  # 需要取出 文件 的 列名称
        self.df = pd.read_csv(origin_can_dir, usecols=col_name_list)  # 读取 can报文 csv源文件
        self.point = 0  # 记录 此次取出的can数据 结束 位置
        self.max_nodes_num = max_nodes_num
        self.data_total_len = len(self.df)
        # print(self.df.head())
        # print(self.df.tail())

    def get_ds_a(self, len_can):

        done = False
        graph = None
        df = self.df.iloc[self.point:self.point+len_can, :]  # 取出 特定长度 的can报文

        if not done:  # 如果数据没有 读完
            # 图相关
            graph_label = 0  # 标识图标签 0：正常报文 1：入侵报文

            # 边相关
            adj_list = list()  # 存储 每条 连接边
            edge_weight_list = list()  # 存储 每条边的权重 每重复连接一次 加1

            # 节点相关
            can_id = df.get("Arbitration_ID").values  # 取出 ID 得到numpy类型 数据
            can_type = df.get("Class").values  # 取出 can 数据 的 报文类型 数据

            # print(type(can_id))
            print(f'取出原始can报文 {self.point} - {self.point+len_can-1} 长度为 {len(can_id)}')
            # print(df)
            # print('***')

            # 转换为 十六进制ID 为键 十进制ID 为键值 的 节点字典
            hex2dec_dict = dict()
            # print('###')
            # print(len(can_id))
            for i in range(len_can):
                try:
                    if can_id[i] not in hex2dec_dict.keys():  # 如果 此十六进制 ID 未出现过
                        hex2dec_dict[can_id[i]] = len(hex2dec_dict)  # 新出现一个 十六进制ID 则加入新键值对 键值 为 递增 节点标签 从 0 递增
                except IndexError:
                    print(i)
                    print('can报文 文件读取完毕')
                    done = True

            # 得到 图实例 所需要的数据
            for i in range(len_can):  # 遍历 本次 选出 的 报文数据
                try:
                    # 图标签
                    if can_type[i] != 'Normal':  # 如果含有入侵数据 则 判定为 入侵报文
                        graph_label = 1

                    # 节点
                    # 图的节点标签 直接 从 节点十六进制 到 十进制 的字典得到

                    # 边
                    if (hex2dec_dict[can_id[i]], hex2dec_dict[can_id[i+1]]) not in adj_list:  # 如果边没在 列表中 则 添加入 列表
                        adj_list.append((hex2dec_dict[can_id[i]], hex2dec_dict[can_id[i+1]]))

                    # 以下代码为 考虑 边权重 情况
                    #     edge_weight_list.append(1)
                    # else:  # 边已经 被储存在 列表
                    #     print(f'节点 {i} {adj_list.index((can_id[i], can_id[i+1]))} 重复连接')
                    #     edge_weight_list[adj_list.index((can_id[i], can_id[i+1]))] += 1

                except IndexError:

                    pass  # 当遍历到 数据 最后一个点 会触发异常 退出循环
            print(f'边列表:\n {adj_list}')
            print(f'边数{len(adj_list)}')
            # print(edge_weight_list)
            # print(len(edge_weight_list))
            print(f'节点标签{hex2dec_dict.values()}')

            # 实例化 图
            graph = nx.from_edgelist(adj_list)  # 从边 列表 构造 图
            graph.graph['label'] = graph_label
            print(f'报文类型: {graph_label}')
            # Plot the graph 可视化建立的 图实例
            # nx.draw(graph, with_labels=True, font_weight='bold')
            # plt.show()

            # 添加图标签 使用十进制的 图标签 的 one-hot 作为 图节点标签
            for u in graph.nodes():
                node_label_one_hot = [0] * self.max_nodes_num  # 图标签 种类数 个零 的 一个 列表 保持每次训练的一致性 把节点特征维度设为最大节点数
                node_label_one_hot[u] = 1
                graph.nodes[u]['label'] = node_label_one_hot

            self.point += len_can

            # for u in graph.nodes():
            #     print(graph.nodes[u]['label'])

        return graph, done

    def test(self, can_len):
        df = self.df.iloc[179346:179349, :]  # 取出 特定长度 的can报文
        print(df)
        print(len(self.df))


if __name__ == '__main__':
    origin_can_dir_ = '/Users/aaron/git_project/eigenpooling/data/Car_Hacking_Challenge_Dataset_rev20Mar2021/' \
                      '0_Preliminary/0_Training/Pre_train_D_1.csv'
    p = OriginCanData(origin_can_dir_,300)
    # for i in range(10000):
    # p.get_ds_a(100)
    p.test(2)
