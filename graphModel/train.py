import torch
import time
from torch.autograd import Variable
import torch.nn as nn


def train(dataset, model, args, optimizer, mask_nodes=True, device='cpu'):

    model.train()  # 模型训练模式

    for batch_idx, data in enumerate(dataset):

        # 模型 梯度 置0
        model.zero_grad()

        adj = Variable(data['adj'].float(), requires_grad=False).to(device)
        h0 = Variable(data['feats'].float(), requires_grad=False).to(device)
        label = Variable(data['label'].long()).to(device)
        batch_num_nodes = data['num_nodes'].int().numpy() if mask_nodes else None
        # assign_input = Variable(data['assign_feats'].float(), requires_grad=False).to(device)

        # if args.method == 'wave':
        adj_pooled_list = []
        batch_num_nodes_list = []
        pool_matrices_dic = dict()
        pool_sizes = [int(i) for i in args.pool_sizes.split('_')]
        for i in range(len(pool_sizes)):
            ind = i + 1
            adj_key = 'adj_pool_' + str(ind)
            adj_pooled_list.append(Variable(data[adj_key].float(), requires_grad=False).to(device))
            num_nodes_key = 'num_nodes_' + str(ind)
            batch_num_nodes_list.append(data[num_nodes_key])

            pool_matrices_list = []
            for j in range(args.num_pool_matrix):
                pool_adj_key = 'pool_adj_' + str(i) + '_' + str(j)

                pool_matrices_list.append(Variable(data[pool_adj_key].float(), requires_grad=False).to(device))

            pool_matrices_dic[i] = pool_matrices_list

        pool_matrices_list = []
        if args.num_pool_final_matrix > 0:

            for j in range(args.num_pool_final_matrix):
                pool_adj_key = 'pool_adj_' + str(ind) + '_' + str(j)

                pool_matrices_list.append(Variable(data[pool_adj_key].float(), requires_grad=False).to(device))

            pool_matrices_dic[ind] = pool_matrices_list


        ypred, after_gcn_vector = model(h0, adj, adj_pooled_list, batch_num_nodes, batch_num_nodes_list, pool_matrices_dic)
        # else:
        #     ypred = model(h0, adj, batch_num_nodes, assign_x=assign_input)
        # if not args.method == 'soft-assign' or not args.linkpred:
        loss = model.loss(ypred, label)
        # else:
        #     loss = model.loss(ypred, label, adj, batch_num_nodes)
        loss.backward()


        # nn.utils.clip_grad_norm(model.parameters(), args.clip)
        optimizer.step()

        pre_label = torch.argmax(ypred, 1)
        ypred_dim0, ypred_dim1 = ypred.shape

        # ypred_np = ypred.cpu().detach().numpy()
        # logger.info(f'pred: {pre_label.item()}; {labels[batch_idx].astype(int)[0] == pre_label.item()}')

        # 制定 reward
        # print(f'pre_label: {pre_label.item()}')
        # print(f'label: {label.item()}')
        # all_reward = 0
        # for singleCAN in range(ypred_dim0):
        #     if pre_label[singleCAN] == label[singleCAN]:
        #         reward = abs(ypred[0, 0] - ypred[0, 1])
        #     else:
        #         reward = - abs(ypred[0, 0] - ypred[0, 1])*10  # 加大对错误的惩罚
        #
        #     all_reward += reward
        #
        # all_reward = all_reward / args.graph_batchsize

        count_correct = 0
        for singleCAN in range(ypred_dim0):
            if pre_label[singleCAN] == label[singleCAN]:
                count_correct += 1

        count_correct = count_correct / args.graph_batchsize


        # return  after_gcn_vector, all_reward.cpu().detach().numpy().tolist(), label.cpu().detach().numpy().tolist(), pre_label.cpu().detach().numpy().tolist(), loss.item()

        return  after_gcn_vector, count_correct, label.cpu().detach().numpy().tolist(), pre_label.cpu().detach().numpy().tolist(), loss.item()
