import random
from graphModel.processData.graph_sampler import GraphSampler
import torch


def prepare_data(graph, graphs_coarsen, args,  max_nodes=0):
    '''
    :param graph:  原始图
    :param graphs_coarsen:  包括原始图和塌缩矩阵
    :param args:
    :param max_nodes:
    :return:
    '''

    dataset_sampler = GraphSampler(graph, graphs_coarsen, args.num_pool_matrix, args.num_pool_final_matrix,
                                   normalize=False, max_num_nodes=max_nodes,
                                   features=args.feature_type, norm=args.norm)  # args.norm 正则化的形式
    train_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers)


    return train_dataset_loader
