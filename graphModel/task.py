from graphModel.processData.prepare_data import prepare_data
from graphModel.model.train import train
from graphModel.model.evaluate import evaluate
import networkx as nx
from graphModel.model.coarsen_pooling_with_last_eigen_padding import Graphs as gp
from graphModel.model import encoders
import graphModel.gen.feat as featgen
import numpy as np
from graphModel.args import graph_model_train_log_file, graph_model_test_log_file
from graphModel.processData import originCanData
import torch
import hiddenlayer as hl
import matplotlib.pyplot as plt
import os


class Task:
    def __init__(self, args):
        self.origin_can_obj = originCanData.OriginCanData(args.origin_can_datadir, args.max_nodes)
        self.args = args
        self.pool_sizes = [int(i) for i in self.args.pool_sizes.split('_')]  # 池化时 每个簇的 子图大小
        self.model = None  # 模型 变量 会在 benchmark_task_val 首次调用时定义
        self.history = hl.History()
        self.canvas = hl.Canvas()
        self.loss_list = []
        print('self.pool_sizes: ', self.pool_sizes)
        
    def benchmark_task_val(self, epoch, step, feat, pred_hidden_dims, device, len_can,  mode, first):
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
    
        sample_graph, done = self.origin_can_obj.get_ds_a(len_can)  # 取出 指定长度的数据 并 转换为 图对象

        adj = nx.adjacency_matrix(sample_graph)  # 大图 邻接矩阵
        coarsen_graph = gp(adj.todense().astype(float), self.pool_sizes)  # 实例化 要进行塌缩的 图
        coarsen_graph.coarsening_pooling(self.args.normalize)  # 进行 图 塌缩

        # 指定 图 特征
        # if feat == 'node-feat' and 'feat_dim' in sample_graph.graph:
        #     print('Using node features')
        #     input_dim = sample_graph.graph['feat_dim']
        # elif feat == 'node-label' and 'label' in sample_graph.nodes[0]:
        #     print('Using node labels')
        # 使用节点 标签特征
        for u in sample_graph.nodes():
            sample_graph.nodes[u]['feat'] = np.array(sample_graph.nodes[u]['label'])
        # else:
        #     print('Using constant labels')
        #     featgen_const = featgen.ConstFeatureGen(np.ones(self.args.input_dim, dtype=float))  # 若没有 选择 则使用 G.nodes() 作为特征
        #     featgen_const.gen_node_features(sample_graph)

        # 生成训练数据
        train_data, input_dim = prepare_data(sample_graph, coarsen_graph, self.args, test_graphs=[], max_nodes=self.args.max_nodes)
        # print(f'input_dim:{input_dim}')
        if first:

            # 首次调用 定义模型
            self.model = encoders.WavePoolingGcnEncoder(input_dim, self.args.hidden_dim, self.args.output_dim, self.args.num_classes,
                                               self.args.num_gc_layers, self.args.num_pool_matrix, self.args.num_pool_final_matrix,
                                               pool_sizes=self.pool_sizes, pred_hidden_dims=pred_hidden_dims, concat=self.args.concat,
                                               bn=self.args.bn, dropout=self.args.dropout, mask=self.args.mask, args=self.args, device=device)

            # 判断是否需要 从本地 加载模型 参数
            # 只有训练 指定需要 不加载 其他情况都需要加载
            if mode == 'test' or self.args.load:
                self.model.load_state_dict(torch.load(self.args.directory) + 'graph.pth')

        # after_gcn_vector 是图卷积之后得到的 一维特征向量 作为强化学习下一次的 状态向量
        after_gcn_vector = None
        reward = 0
        after_train_model = None
        if mode == 'train':
            with open(graph_model_train_log_file, 'a') as f:
                f.write('====================================================================================\n')
            after_train_model, after_gcn_vector, reward, loss = train(epoch, train_data, self.model, self.args, log_file=graph_model_train_log_file, device=device)
            # 训练时 实时显示 loss 变化曲线
            self.history.log((epoch, step), train_loss=loss)
            self.loss_list.append(loss)
            with self.canvas:
                self.canvas.draw_plot(self.history['train_loss'])
            print(f'选取报文位置为 ( {self.origin_can_obj.point} - {self.origin_can_obj.point+len_can-1} ) / {self.origin_can_obj.data_total_len}')
            print(f'长度为 {len_can}')
            print(self.args.origin_can_datadir)

            # 数据 扫瞄完毕 画出loss图线 并保存
            if done:
                plt_loss_file_name = 'loss' + os.path.splitext(os.path.basename(self.args.origin_can_datadir))[0]

                step_all_individual_list = [i for i in range(0, len(self.loss_list))]
                print(step_all_individual_list)

                plt.plot(step_all_individual_list, self.loss_list)
                plt.xlabel('steps')
                plt.ylabel('train_loss')
                plt.title('train_loss figure')
                plt.savefig(self.args.plt_out_dir + plt_loss_file_name, dpi=300, bbox_inches='tight')

        elif mode == 'test':
            with open(graph_model_test_log_file, 'a') as f:
                f.write('====================================================================================\n')
            after_gcn_vector, reward = evaluate(train_data, self.model, self.args, log_file=graph_model_test_log_file, device=device)

        print(f'reward: {reward}')



        # print(type(after_gcn_vector))
        print('结束 图部分')
        return after_gcn_vector, reward, done, after_train_model
