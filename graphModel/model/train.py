import torch
import time
from torch.autograd import Variable
import torch.nn as nn
from graphModel.model.evaluate import evaluate


def train(epoch, dataset, model, args, mask_nodes=True, log_file=None, device='cpu'):
    '''
    :param epoch:
    :param dataset:
    :param model:
    :param args:
    :param mask_nodes:
    :param log_file:
    :param device:
    :return:
    '''

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    iter = 0
    best_val_result = {
        'epoch': 0,
        'loss': 0,
        'acc': 0}
    test_result = {
        'epoch': 0,
        'loss': 0,
        'acc': 0}
    train_accs = []
    train_epochs = []
    best_val_accs = []
    best_val_epochs = []
    test_accs = []
    test_epochs = []
    val_accs = []
    after_gcn_vector = None
    # for epoch in range(args.num_epochs):
    begin_time = time.time()
    avg_loss = 0.0
    model.train()
    reward = 0
    for batch_idx, data in enumerate(dataset):  # dataset 里只有一个数据

        time1 = time.time()
        model.zero_grad()  # 梯度 置 0

        adj = Variable(data['adj'].float(), requires_grad=False).to(device)  # adj矩阵（如需要正则矩阵 前面已正则化
        h0 = Variable(data['feats'].float(), requires_grad=False).to(device)
        label = Variable(data['label'].long()).to(device)
        batch_num_nodes = data['num_nodes'].int().numpy() if mask_nodes else None  #
        # assign_input = Variable(processData['assign_feats'].float(), requires_grad=False).to(device)

        # if args.method == 'wave':
        adj_pooled_list = []  # 池化后的超级节点间的 邻接矩阵
        batch_num_nodes_list = []
        pool_matrices_dic = dict()
        pool_sizes = [int(i) for i in args.pool_sizes.split('_')]  # 图坍塌是 每个簇的 节点个数


        # 得到 图塌缩后的 邻接矩阵
        # 得到 图塌缩后的 节点个数
        # 得到 图塌缩操作的 池化矩阵
        for i in range(len(pool_sizes)):  # pool_sizes = [10]
            ind = i + 1
            # 图塌缩后的 超级节点 之间的 邻接矩阵
            adj_key = 'adj_pool_' + str(ind)
            adj_pooled_list.append(Variable(data[adj_key].float(), requires_grad=False).to(device))
            # 图塌缩后的 节点个数
            num_nodes_key = 'num_nodes_' + str(ind)
            batch_num_nodes_list.append(data[num_nodes_key])
            # 图塌缩操作的 池化矩阵
            # 第 i 次 池化操作 的 池化矩阵列表  # 因为可以定义每次池化操作 需要池化矩阵的个数 所以每次池化的池化矩阵需要一个列表存储
            pool_matrices_list = []
            for j in range(args.num_pool_matrix):  # 每次池化 池化矩阵的 个数
                pool_adj_key = 'pool_adj_' + str(i) + '_' + str(j)
                pool_matrices_list.append(Variable(data[pool_adj_key].float(), requires_grad=False).to(device))
            pool_matrices_dic[i] = pool_matrices_list

        # 因为 图塌缩阶段 已经把图塌缩到 一个节点 本段代码无意义
        pool_matrices_list = []
        if args.num_pool_final_matrix > 0:
            for j in range(args.num_pool_final_matrix):
                pool_adj_key = 'pool_adj_' + str(ind) + '_' + str(j)
                pool_matrices_list.append(Variable(data[pool_adj_key].float(), requires_grad=False).to(device))
            pool_matrices_dic[ind] = pool_matrices_list

        time2 = time.time()

        # 执行 图分类
        ypred, after_gcn_vector = model(h0, adj, adj_pooled_list, batch_num_nodes, batch_num_nodes_list, pool_matrices_dic)
        # else:
        #     ypred = model(h0, adj, batch_num_nodes, assign_x=assign_input)
        # if not args.method == 'soft-assign' or not args.linkpred:
        loss = model.loss(ypred, label)
        _, pre_label = torch.max(ypred, 1)
        # print('图分类结果是：')
        # print(ypred)
        # print(label)
        # print(pre_label)
        if pre_label == label:
            reward = 1
        # else:
        #     loss = model.loss(ypred, label, adj, batch_num_nodes)
        loss.backward()

        time3 = time.time()

        nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        iter += 1
        loss = loss.item()


    elapsed = time.time() - begin_time

    if log_file is not None:
        with open(log_file, 'a') as f:
            f.write('Epoch: ' + str(epoch) + '-----------------------------\n')
            f.write('Train_loss: ' + str(loss) + '\n')

    return model, after_gcn_vector, reward, loss
