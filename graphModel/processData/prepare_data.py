import random
from graphModel.processData.graph_sampler import GraphSampler
import torch


def prepare_data(graphs, coarsen_graphs, args,  max_nodes=0):
    '''
    :param graphs:  原始图
    :param coarsen_graphs:  包括原始图和塌缩矩阵
    :param args:
    :param max_nodes:
    :return:
    '''
    zip_list = list(zip(graphs, coarsen_graphs))
    random.Random(args.seed).shuffle(zip_list)
    graphs, coarsen_graphs = zip(*zip_list)

    dataset_sampler = GraphSampler(graphs, coarsen_graphs, args.num_pool_matrix, args.num_pool_final_matrix,
                                   normalize=False, max_num_nodes=max_nodes,
                                   features=args.feature_type, norm=args.norm)  # args.norm 正则化的形式
    train_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.graph_batchsize,
        shuffle=True,
        num_workers=args.graph_num_workers)


    return train_dataset_loader
