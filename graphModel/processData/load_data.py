import networkx as nx
import numpy as np
import scipy as sc
import os
import re


# 本文件 在单独训练 坍塌图 网络 提供 图特征的
def read_graphfile(datadir, dataname, max_nodes=None):
    ''' Read data from https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets
        graph index starts with 1 in file

    Returns:
        List of networkx objects with graph and node labels
    '''
    prefix = os.path.join(datadir, dataname)

    # 获取 节点属于哪个图 字典
    filename_graph_indic = prefix + '_graph_indicator.txt'
    # index of graphs that a given node belongs to
    # 字典的键名是节点的编号 从1开始 递增1 键值为 该节点属于的图编号（文件图编号从 1 开始）
    # 图编号 从几开始 和文件中数字相同
    graph_indic = {}
    with open(filename_graph_indic) as f:
        i = 1
        for line in f:
            line = line.strip("\n")
            graph_indic[i] = int(line)
            i += 1
    # print(graph_indic)
    # print('######')

    # 获取 节点标签 列表
    # 列表的(角标+1) 标识 节点编号(从1开始) 元素标识 该节点编号对应的 标签
    # 默认文件的 节点标签元素 从 1 开始
    # 列表 节点标签 元素 从 0 开始
    filename_nodes = prefix + '_node_labels.txt'
    node_labels = []
    try:
        with open(filename_nodes) as f:
            for line in f:
                line = line.strip("\n")
                node_labels += [int(line) - 1]
        # 得到共有多少中节点标签
        num_unique_node_labels = max(node_labels) + 1
        print(f'共有 {num_unique_node_labels} 个节点标签')
    except IOError:
        print('No node labels')

    # 获取节点属性列表
    # 暂时忽略
    # filename_node_attrs = prefix + '_node_attributes.txt'
    # node_attrs = []
    # try:
    #     with open(filename_node_attrs) as f:
    #         for line in f:
    #             line = line.strip("\s\n")
    #             attrs = [float(attr) for attr in re.split("[,\s]+", line) if not attr == '']
    #             node_attrs.append(np.array(attrs))
    # except IOError:
    #     print('No node attributes')

    # 获取图标签列表
    # 图标签 从0开始
    # 列表的（角标+1）代表图编号（图编号从1开始） 元素标识该编号图的标签
    # 无论数据文件的中的图标签最小 1开始 或者 从0 开始 得到的图标签列表从 0 开始
    label_has_zero = False
    filename_graphs = prefix + '_graph_labels.txt'
    graph_labels = []
    with open(filename_graphs) as f:
        for line in f:
            line = line.strip("\n")
            val = int(line)
            if val == 0:
                label_has_zero = True
            graph_labels.append(val - 1)
    graph_labels = np.array(graph_labels)
    if label_has_zero:
        graph_labels += 1

    # 获取边字典 和 每个图包含节点编号字典
    filename_adj = prefix + '_A.txt'
    # 该字典共有 图个数个 键值对
    # 键值是一个列表 标识 每个图的 所有的连接关系
    # 键值（列表）的元素是元祖 每个元祖标识 相连的两个节点
    # 键名表示 图编号 从 1 开始
    adj_list = {i: [] for i in range(1, len(graph_labels) + 1)}
    # 该字典表示 每个图所包含的节点编号
    # 键名是图编号 键值是一个列表 该图包含的节点编号 列表里的元素是 int 型
    # index_graph = {i: [] for i in range(1, len(graph_labels) + 1)}
    # 统计该数据集 所有的 边数
    num_edges = 0
    with open(filename_adj) as f:
        for line in f:
            line = line.strip("\n").split(",")
            e0, e1 = (int(line[0].strip(" ")), int(line[1].strip(" ")))
            adj_list[graph_indic[e0]].append((e0, e1))
            # index_graph[graph_indic[e0]] += [e0, e1]
            num_edges += 1
    # for k in index_graph.keys():
        # 节点编号去重
        # 节点编号 从 0 开始 递增 1
        # index_graph[k] = [u - 1 for u in set(index_graph[k])]

    # 建立图对象 并存储为列表
    graphs = []
    for i in range(1, 1 + len(adj_list)):
        # indexed from 1 here
        G = nx.from_edgelist(adj_list[i])
        if max_nodes is not None and G.number_of_nodes() > max_nodes:
            continue

        # add features and labels
        G.graph['label'] = graph_labels[i - 1]
        for u in G.nodes():
            # u 是节点编号 从 1 开始
            # 创建 节点标签的 独热编码
            if len(node_labels) > 0:
                node_label_one_hot = [0] * num_unique_node_labels
                node_label = node_labels[u - 1]
                # 节点标签从0开始 所以节点标签为0 则 独热编码的第零的元素为 1
                node_label_one_hot[node_label] = 1
                G.nodes[u]['label'] = node_label_one_hot
        #     if len(node_attrs) > 0:
        #         G.node[u]['feat'] = node_attrs[u - 1]
        # if len(node_attrs) > 0:
        #     G.graph['feat_dim'] = node_attrs[0].shape[0]

        # relabeling
        mapping = {}
        it = 0
        if float(nx.__version__) < 2.0:
            for n in G.nodes():
                mapping[n] = it
                it += 1
        else:
            for n in G.nodes:
                # print('###')
                # print(G.nodes[n]['label'])
                mapping[n] = it
                it += 1

        # indexed from 0
        # 把每个图的 节点编号 从 0 开始
        graphs.append(nx.relabel_nodes(G, mapping))

        # 最终得到的 1.图标签从0开始 2.节点编号从0开始 3.节点标签以独热编码表示

    return graphs

