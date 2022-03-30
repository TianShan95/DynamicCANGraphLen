import networkx as nx
import numpy as np
import torch
import torch.utils.data
from sklearn import preprocessing
import copy


class GraphSampler(torch.utils.data.Dataset):
    ''' Sample graphs and nodes in graph
    '''

    def __init__(self, G_list, graphs_coarsen, num_pool_matrix, num_pool_final_matrix, features='default', normalize=True,
                 assign_feat='default', max_num_nodes=0, norm='l2'):
        self.adj_all = []
        self.len_all = []
        self.feature_all = []
        self.label_all = []
        self.graphs_coarsen = graphs_coarsen
        self.num_pool_matrix = num_pool_matrix
        self.num_pool_final_matrix = num_pool_final_matrix
        self.norm = norm

        self.assign_feat_all = []

        if max_num_nodes == 0:
            self.max_num_nodes = max(G.number_of_nodes() for G in G_list)
        else:
            self.max_num_nodes = max_num_nodes  # 指定的最大节点数目

        # print(f'graph_origin: {graph_origin}')
        # print(graph_origin.nodes[0]['feat'])
        self.feat_dim = G_list[0].nodes[0]['feat'].shape[0]  # 特征维度

        for G in G_list:
            adj = np.array(nx.to_numpy_matrix(G))
            if normalize:  # adg 正则化
                sqrt_deg = np.diag(1.0 / np.sqrt(np.sum(adj, axis=0, dtype=float).squeeze()))
                adj = np.matmul(np.matmul(sqrt_deg, adj), sqrt_deg)  # 得到 重归一化 形式的 拉普拉斯矩阵
            self.adj_all.append(adj)
            self.len_all.append(G.number_of_nodes())  # 节点数量
            self.label_all.append(G.graph['label'])  # 图标签
            if features == 'default':
                f = np.zeros((self.max_num_nodes, self.feat_dim), dtype=float)
                for i, u in enumerate(G.nodes()):  # 遍历图中所有的节点 但是特征矩阵初始化是以最大的节点大小初始化的 所以结尾的行存在全零行
                    f[i, :] = G.nodes[u]['feat']
                self.feature_all.append(copy.deepcopy(f))
            elif features == 'id':
                self.feature_all.append(np.identity(self.max_num_nodes))
            elif features == 'deg-num':
                degs = np.sum(np.array(adj), 1)
                degs = np.expand_dims(np.pad(degs, [0, self.max_num_nodes - G.number_of_nodes()], 0),
                                      axis=1)
                self.feature_all.append(degs)
            elif features == 'struct':
                self.max_deg = 10
                degs = np.sum(np.array(adj), 1).astype(int)
                degs[degs > 10] = 10
                feat = np.zeros((len(degs), self.max_deg + 1))
                feat[np.arange(len(degs)), degs] = 1
                degs = np.pad(feat, ((0, self.max_num_nodes - G.number_of_nodes()), (0, 0)),
                              'constant', constant_values=0)

                clusterings = np.array(list(nx.clustering(G).values()))
                clusterings = np.expand_dims(np.pad(clusterings,
                                                    [0, self.max_num_nodes - G.number_of_nodes()],
                                                    'constant'),
                                             axis=1)
                g_feat = np.hstack([degs, clusterings])
                if 'feat' in G_list.node[0]:
                    node_feats = np.array([G.node[i]['feat'] for i in range(G.number_of_nodes())])
                    node_feats = np.pad(node_feats, ((0, self.max_num_nodes - G.number_of_nodes()), (0, 0)),
                                        'constant')
                    g_feat = np.hstack([g_feat, node_feats])

                self.feature_all.append(g_feat)

            # print('feature shapoe 1..1.', self.feature_all[0].shape)

            if assign_feat == 'id':
                self.assign_feat_all.append(
                    np.hstack((np.identity(self.max_num_nodes), self.feature_all[-1])))
            else:
                self.assign_feat_all.append(self.feature_all[-1])

        self.feat_dim = self.feature_all[0].shape[1]
        self.assign_feat_dim = self.assign_feat_all[0].shape[1]


    def __len__(self):
        return len(self.adj_all)

    def __getitem__(self, idx):
        adj = self.adj_all[idx]
        num_nodes = adj.shape[0]
        adj_padded = np.zeros((self.max_num_nodes, self.max_num_nodes))
        adj_padded[:num_nodes, :num_nodes] = adj

        graph = self.graphs_coarsen[idx]

        return_dic = {'adj': adj_padded,
                      'feats': self.feature_all[idx].copy(),
                      'label': self.label_all[idx],
                      'num_nodes': num_nodes,
                      'assign_feats': self.assign_feat_all[idx].copy()}

        for i in range(len(graph.graphs) - 1):  # graph.graphs 是列表 存储邻接矩阵 和 图塌缩后的 邻接矩阵
            ind = i + 1
            adj_key = 'adj_pool_' + str(ind)
            num_nodes_key = 'num_nodes_' + str(ind)
            num_nodes_ = graph.graphs[ind].shape[0]
            return_dic[num_nodes_key] = num_nodes_
            adj_padded_ = np.zeros((self.max_num_nodes, self.max_num_nodes))
            adj_padded_[:num_nodes_, :num_nodes_] = graph.graphs[ind].todense().astype(float)

            return_dic[adj_key] = adj_padded_
        for i in range(len(graph.layer2pooling_matrices)):  # 遍历 池化 矩阵
            if i == len(graph.layer2pooling_matrices) - 1:  # 最后一层 池化
                for j in range(self.num_pool_final_matrix):
                    pool_adj_key = 'pool_adj_' + str(i) + '_' + str(j)
                    pool_adj = graph.layer2pooling_matrices[i][j]

                    pool_adj_padded = np.zeros((self.max_num_nodes, self.max_num_nodes))

                    if self.norm == 'l1':
                        pool_adj = pool_adj.todense().astype(float)
                        pool_adj = preprocessing.normalize(pool_adj, norm=self.norm, axis=0)
                        pool_adj_padded[:pool_adj.shape[0], : pool_adj.shape[1]] = pool_adj
                    else:
                        pool_adj_padded[:pool_adj.shape[0], : pool_adj.shape[1]] = pool_adj.todense().astype(float)
                    return_dic[pool_adj_key] = pool_adj_padded
            else:
                for j in range(self.num_pool_matrix):  # 每次池化 需要池化矩阵的个数
                    pool_adj_key = 'pool_adj_' + str(i) + '_' + str(j)
                    pool_adj = graph.layer2pooling_matrices[i][j]

                    pool_adj_padded = np.zeros((self.max_num_nodes, self.max_num_nodes))

                    if self.norm == 'l1':
                        pool_adj = pool_adj.todense().astype(float)
                        pool_adj = preprocessing.normalize(pool_adj, norm=self.norm, axis=0)
                        pool_adj_padded[:pool_adj.shape[0], : pool_adj.shape[1]] = pool_adj
                    else:
                        pool_adj_padded[:pool_adj.shape[0], : pool_adj.shape[1]] = pool_adj.todense().astype(float)
                    return_dic[pool_adj_key] = pool_adj_padded

        return return_dic
