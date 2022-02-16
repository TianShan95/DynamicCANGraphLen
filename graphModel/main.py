import random
import torch
import numpy as np
from  graphModel import task
from graphModel.args import arg_parse


def main():
    prog_args = arg_parse()
    seed = 1
    print(prog_args)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    print('bmname: ', prog_args.bmname)
    print('num_classes: ', prog_args.num_classes)
    # print('method: ', prog_args.method)
    print('batch_size: ', prog_args.batch_size)
    print('num_pool_matrix: ', prog_args.num_pool_matrix)
    print('num_pool_final_matrix: ', prog_args.num_pool_final_matrix)
    print('epochs: ', prog_args.num_epochs)
    print('learning rate: ', prog_args.lr)
    print('num of gc layers: ', prog_args.num_gc_layers)
    print('output_dim: ', prog_args.output_dim)
    print('hidden_dim: ', prog_args.hidden_dim)
    print('pred_hidden: ', prog_args.pred_hidden)
    # print('if_transpose: ', prog_args.if_transpose)
    print('dropout: ', prog_args.dropout)
    print('weight_decay: ', prog_args.weight_decay)
    print('shuffle: ', prog_args.shuffle)
    print('Using batch normalize: ', prog_args.bn)
    print('Using feat: ', prog_args.feat)
    print('Using mask: ', prog_args.mask)
    print('Norm for eigens: ', prog_args.norm)
    # print('Combine pooling results: ', prog_args.pool_m)
    print('With test: ', prog_args.with_test)

    if torch.cuda.is_available() and prog_args.device == 'cuda':
        device = 'cuda'
    else:
        device = 'cpu'

    print('Device: ', device)
    pred_hidden_dims = [int(i) for i in prog_args.pred_hidden.split('_')]

    if prog_args.bmname is not None:  # 如果指定了 数据集
        graph_len_ = random.randint(12, 300)  # 第一次随机 图的长度 给出强化学习的 初始 state
        next_state_vector = task.benchmark_task_val(prog_args,  prog_args.feat, pred_hidden_dims, device, prog_args.origin_can_datadir, graph_len_)



if __name__ == "__main__":
    main()
