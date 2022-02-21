import torch
import numpy as np
from torch.autograd import Variable
import sklearn.metrics as metrics


def evaluate(dataset, model, args, max_num_examples=None, log_file=None, device='cpu'):

    # 载入 模型参数
    model.eval()
    labels = []
    after_gcn_vector = None
    reward = 0
    for batch_idx, data in enumerate(dataset):
        adj = Variable(data['adj'].float(), requires_grad=False).to(device)
        h0 = Variable(data['feats'].float()).to(device)
        labels.append(data['label'].long().numpy())
        batch_num_nodes = data['num_nodes'].int().numpy()


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
        print(f'ypred: {ypred_np}')

        if pre_label == pre_label:
            reward = abs(ypred_np[0, 0] - ypred[0, 1])
        else:
            reward = - abs(ypred_np[0, 0] - ypred[0, 1])

        if max_num_examples is not None:
            if (batch_idx + 1) * args.batch_size > max_num_examples:
                break

    # preds = []
    # preds.append(indices.cpu().data.numpy())
    # labels = np.hstack(labels)
    # preds = np.hstack(preds)
    #
    # result = {'prec': metrics.precision_score(labels, preds, average='macro'),
    #           'recall': metrics.recall_score(labels, preds, average='macro'),
    #           'acc': metrics.accuracy_score(labels, preds),
    #           'F1': metrics.f1_score(labels, preds, average="micro")}
    return after_gcn_vector, reward


def evaluate1(dataset, model, args, name='Validation', max_num_examples=None, device='cpu'):
    model.eval()

    labels = []
    preds = []
    for batch_idx, data in enumerate(dataset):
        adj = Variable(data['adj'].float(), requires_grad=False).to(device)
        h0 = Variable(data['feats'].float()).to(device)
        labels.append(data['label'].long().numpy())
        batch_num_nodes = data['num_nodes'].int().numpy()

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

        ypred = model(h0, adj, adj_pooled_list, batch_num_nodes, batch_num_nodes_list, pool_matrices_dic)

        # else:
        #     ypred = model(h0, adj, batch_num_nodes, assign_x=assign_input)
        _, indices = torch.max(ypred, 1)
        preds.append(indices.cpu().data.numpy())

        if max_num_examples is not None:
            if (batch_idx + 1) * args.batch_size > max_num_examples:
                break

    labels = np.hstack(labels)
    preds = np.hstack(preds)

    result = {'prec': metrics.precision_score(labels, preds, average='macro'),
              'recall': metrics.recall_score(labels, preds, average='macro'),
              'acc': metrics.accuracy_score(labels, preds),
              'F1': metrics.f1_score(labels, preds, average="micro")}
    return result
