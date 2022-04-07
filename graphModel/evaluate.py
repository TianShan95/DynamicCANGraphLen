import torch
import numpy as np
from torch.autograd import Variable
import sklearn.metrics as metrics
from utils.logger import logger


def evaluate(dataset, model, args, mask_nodes=True, device='cpu'):

    model.eval()  # 模型训练模式

    for batch_idx, data in enumerate(dataset):

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

        loss = model.loss(ypred, label)

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
        #         reward = - abs(ypred[0, 0] - ypred[0, 1]) * 10  # 加大对错误的惩罚
        #
        #     all_reward += reward
        count_correct = 0
        for singleCAN in range(ypred_dim0):
            if pre_label[singleCAN] == label[singleCAN]:
                count_correct += 1

        count_correct = count_correct / args.graph_batchsize

        return  after_gcn_vector, count_correct, label.cpu().detach().numpy().tolist(), pre_label.cpu().detach().numpy().tolist(), loss.item()


def evaluate1(dataset, model, args, device):

    # 载入 模型参数
    model.eval()
    # labels = []
    after_gcn_vector = None
    reward = 0
    for batch_idx, data in enumerate(dataset):
        adj = Variable(data['adj'].float(), requires_grad=False).to(device)
        h0 = Variable(data['feats'].float()).to(device)
        label = Variable(data['label'].long()).to(device)

        batch_num_nodes = data['num_nodes'].int().numpy()

        # 报文标签 输出到 log 文件
        # with open(log_out_file, 'a') as f:
        #     f.write(f'graph_label: {labels[batch_idx].astype(int)[0]}\n')
        # logger.info(f'graph_label: {labels[batch_idx].astype(int)[0]}')

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

        # print(f'h0.shape {h0.shape})')
        # print(f'adj.shape {adj.shape})')
        ypred, after_gcn_vector = model(h0, adj, adj_pooled_list, batch_num_nodes, batch_num_nodes_list, pool_matrices_dic)

        # else:
        #     ypred = model(h0, adj, batch_num_nodes, assign_x=assign_input)
        _, pre_label = torch.max(ypred, 1)

        ypred_np = ypred.cpu().detach().numpy()
        # logger.info(f'pred: {pre_label.item()}; {labels[batch_idx].astype(int)[0] == pre_label.item()}')

        # 制定 reward
        if pre_label == label:
            reward = abs(ypred_np[0, 0] - ypred[0, 1])
        else:
            reward = - abs(ypred_np[0, 0] - ypred[0, 1]) * 100  # 加大对错误的惩罚


    # preds = []
    # preds.append(indices.cpu().data.numpy())
    # labels = np.hstack(labels)
    # preds = np.hstack(preds)
    #
    # result = {'prec': metrics.precision_score(labels, preds, average='macro'),
    #           'recall': metrics.recall_score(labels, preds, average='macro'),
    #           'acc': metrics.accuracy_score(labels, preds),
    #           'F1': metrics.f1_score(labels, preds, average="micro")}
        return after_gcn_vector, reward, label.item(), pre_label.item()

