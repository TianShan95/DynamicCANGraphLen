from graphModel.processData.prepare_data import prepare_data
from graphModel.evaluate import evaluate
from graphModel.train import train

import networkx as nx
from graphModel.coarsen_pooling_with_last_eigen_padding import Graphs as gp
from graphModel import encoders
# import graphModel.gen.feat as featgen
import numpy as np
from graphModel.processData import originCanData
import torch
import hiddenlayer as hl
# import matplotlib.pyplot as plt
import os
from utils.logger import logger


class Task:
    def __init__(self, args, device):
        self.origin_can_obj = originCanData.OriginCanData(args)
        self.args = args
        self.pool_sizes = [int(i) for i in self.args.pool_sizes.split('_')]  # 池化时 每个簇的 子图大小
        # 从指定路径 load 模型
        logger.info(f'模型所在路径: {args.graph_model_path}')
        logger.info(f'模型是否存在: {os.path.isfile(args.graph_model_path)}')

        self.device = device

        if args.graph_model_path:
            if device == 'cpu':
                self.model = torch.load(args.graph_model_path, map_location=torch.device('cpu'))  # 模型 变量 会在 benchmark_task_val 首次调用时定义
                # 模型加载参数
                match_string_len = len("model")
                # 模型加载 参数
                last_char_index = args.graph_model_path.rfind("model")
                net_state_dict = torch.load(args.graph_model_path[:last_char_index] +
                                            "para" + args.graph_model_path[last_char_index+match_string_len:],
                                            map_location=torch.device('cpu'))
                self.model.load_state_dict(net_state_dict)

            elif device == 'cuda':
                self.model = torch.load(args.graph_model_path)  # 模型 变量 会在 benchmark_task_val 首次调用时定义
                # 模型加载参数
                match_string_len = len("model")
                # 模型加载 参数
                last_char_index = args.graph_model_path.rfind("model")
                net_state_dict = torch.load(args.graph_model_path[:last_char_index] +
                                            "para" + args.graph_model_path[last_char_index + match_string_len:])
                self.model.load_state_dict(net_state_dict)
        else:
            logger.info('============= 图模型从头训练 =================')
            pool_sizes = [int(i) for i in args.pool_sizes.split('_')]
            pred_hidden_dims = [int(i) for i in args.pred_hidden.split('_')]
            self.model = encoders.WavePoolingGcnEncoder(args.input_dim, args.hidden_dim, args.output_dim,
                                                        args.num_classes, args.num_gc_layers, args.num_pool_matrix,
                                                        args.num_pool_final_matrix, pool_sizes=pool_sizes,
                                                        pred_hidden_dims=pred_hidden_dims,
                                                        concat=args.concat, bn=args.bn,
                                                        dropout=args.dropout, mask=args.mask, args=args, device=device)

        # 定义优化器
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr, weight_decay=args.weight_decay)

        self.history1 = hl.History()
        self.canvas1 = hl.Canvas()

        print('self.pool_sizes: ', self.pool_sizes)
        
    def benchmark_task_val(self, feat, pred_hidden_dims, len_can):
        '''
        self.args:
            epoch: 代数
            self.args: 输入的参数
            feat: 特征
            pred_hidden_dims: 预测网络隐藏层 维度
            device:  cpu 或者 cuda
            len_can: 报文长度
            self.origin_can_obj: 原始 can 报文数据 对象
            mode: train or test
            first: 是否第一次调用 第一次调用需要定义一次模型 之后再调用则不需要再次定义模型 模型变量 self.model
        '''

        sample_graph, train_done, val_done = self.origin_can_obj.get_ds_a(len_can)  # 取出 指定长度(此动作)的数据 并 转换为 图对象 输出是否完成信号

        after_gcn_vector = None
        reward = 0
        label = None
        pred = None
        graph_loss = 0
        # val_done 是最后的结束标志
        if not val_done:
            adj = nx.adjacency_matrix(sample_graph)  # 大图 邻接矩阵
            coarsen_graph = gp(adj.todense().astype(float), self.pool_sizes)  # 实例化 要进行塌缩的 图
            coarsen_graph.coarsening_pooling(self.args.normalize)  # 进行 图 塌缩

            # 指定 图 特征
            # 使用节点 标签特征
            for u in sample_graph.nodes():
                sample_graph.nodes[u]['feat'] = np.array(sample_graph.nodes[u]['label'])


            # 生成训练数据
            train_data = prepare_data(sample_graph, coarsen_graph, self.args, max_nodes=self.args.max_nodes)

            if train_done:
                # # 送入模型 得到执行此动作(选出这些数量的报文)的 状态向量
                # # 在进行表示学习时 不进行模型更新
                after_gcn_vector, reward, label, pred = evaluate(train_data, self.model, self.args, device=self.device)
            else:
                # 送入模型 得到执行此动作(选出这些数量的报文)的 状态向量
                # 在进行表示学习时 进行模型更新
                after_gcn_vector, reward, label, pred, graph_loss = train(train_data, self.model, self.args, self.optimizer, device=self.device)


        return after_gcn_vector, reward, train_done, val_done, label, pred, graph_loss
