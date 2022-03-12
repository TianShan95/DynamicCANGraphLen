import argparse
from utils.utils import ensure_dir
import time


def arg_parse():
    parser = argparse.ArgumentParser(description='Arguments.')

    # 数据集相关
    parser.add_argument('--bmname', dest='bmname', help='Name of the benchmark dataset')

    # 强化学习相关
    parser.add_argument('--mode', default='train', type=str)  # mode = 'train' or 'test'
    parser.add_argument('--tau', default=0.005, type=float)  # target smoothing coefficient
    parser.add_argument('--target_update_interval', default=1, type=int)
    parser.add_argument('--train_epoch', default=1, type=int)  # 训练代数
    # parser.add_argument('--test_iteration', default=5, type=int)

    parser.add_argument('--learning_rate', default=3e-4, type=float)
    parser.add_argument('--gamma', default=0.99, type=int)  # discounted factor
    parser.add_argument('--capacity', default=1000, type=int)  # replay buffer size 1000
    parser.add_argument('--batch_size', default=100, type=int)  # mini batch size  100
    parser.add_argument('--seed', default=1, type=int)


    # 定义 并创建 此次实验的 log 文件夹
    time_mark = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    # parser.add_argument('--model_store_dir', default='../rl_model_store/' + time_mark + '_', type=str)
    parser.add_argument('--model_load_dir', default='', type=str)
    parser.add_argument('--directory', default='../experiment/Rl_20220312_174654_multiDim_log/epoch_0_20220312_214619', type=str)

    parser.add_argument('--out_dir', type=str,
                        help='out_dir', default='../experiment')  # 实验结果输出文件夹

    # optional parameters
    parser.add_argument('--activation', default='Relu', type=str)
    # parser.add_argument('--log_interval', default=50, type=int)  # 打印 log 的间隔
    parser.add_argument('--load', default=False, type=bool)  # load model  在训练时 强化学习是否加载模型
    # parser.add_argument('--render_interval', default=100, type=int)  # after render_interval, the env.render() will work

    parser.add_argument('--policy_delay', default=2, type=int)
    # parser.add_argument('--policy_noise', default=0.2, type=float)  #噪声相关
    # parser.add_argument('--noise_clip', default=0.5, type=float)  # 噪声相关
    # parser.add_argument('--exploration_noise', default=0.1, type=float)  # 对于强化学习输出的 选择can长度加入噪声

    parser.add_argument('--train-ratio', dest='train_ratio', type=float,
                        help='Ratio of number of graphs training set to all graphs.')
    parser.add_argument('--test-ratio', dest='test_ratio', type=float,
                        help='Ratio of number of graphs testing set to all graphs.')

    # 图网络部分
    parser.add_argument('--max-nodes', dest='max_nodes', type=int,
            help='Maximum number of nodes (ignore graghs with nodes exceeding the number.')

    parser.add_argument('--graph_model_path',
                        default='../experiment/20220311_215200_randGen_1_Normlize_True_concat_1_log/0.85_better_model__totalEpoch_300_epoch_78_ps_10_nor_1.pth',
                        type=str)
    parser.add_argument('--num_workers', dest='num_workers', type=int,
            help='Number of workers to load processData.')
    parser.add_argument('--feature', dest='feature_type',
            help='Feature used for encoder. Can be: id, deg')
    parser.add_argument('--hidden-dim', dest='hidden_dim', type=int,
            help='Hidden dimension')
    parser.add_argument('--output-dim', dest='output_dim', type=int,
            help='Output dimension')
    parser.add_argument('--num-classes', dest='num_classes', type=int,
            help='Number of label classes')  # 图网络输出的 维度
    parser.add_argument('--num-gc-layers', dest='num_gc_layers', type=int,
            help='Number of graph convolution layers before each pooling')  # 卷积层数目
    parser.add_argument('--dropout', dest='dropout', type=float,
            help='Dropout rate.')
    parser.add_argument('--nobias', dest='bias', action='store_const',
            const=False, default=True,
            help='Whether to add bias. Default to True.')
    parser.add_argument('--origin-can-datadir', dest='origin_can_datadir',
                        help='Directory where origin can dataset is located')  # can 数据的路径
    parser.add_argument('--pool_sizes', type=str,
                        help='pool_sizes', default='10')  # 分簇时 每个簇里节点的个数
    # parser.add_argument('--group_sizes', type=str,
    #                     help='group_sizes', default='10')  # 分簇时 分的簇的个数
    parser.add_argument('--num_pool_matrix', type=int,
                        help='num_pooling_matrix', default=1)

    parser.add_argument('--weight_decay', type=float,
                        help='weight_decay', default=0.0)

    parser.add_argument('--choice_graph_len_mode', type=int,
                        help='choice graphLen mode', default=0)  # 0: 只选择最可能的维度 1: 从前5个中选最大 2: 从前5个中随机选

    parser.add_argument('--num_pool_final_matrix', type=int,
                        help = 'number of final pool matrix', default=0)
    parser.add_argument('--normalize', type = int,
                        help='nomrlaized laplacian or not', default=1)
    parser.add_argument('--pred_hidden', type=str,
                        help='pred_hidden', default='50')
    parser.add_argument('--concat', type=int,
                        help='whether concat', default=1)  # 是否把每层卷积的特征向量都拼接起来 输出给图部分的预测层
    parser.add_argument('--feat', type=str,
                        help='which feat to use', default='node-label')
    parser.add_argument('--mask', type=int,
                        help='mask or not', default=1)
    parser.add_argument('--norm', type=str,
                        help='Norm for eigens', default='l2')
    parser.add_argument('--con_final', type=int,
                        help='con_final', default=1)
    parser.add_argument('--device', type=str,
                        help='cpu or cuda', default='cpu')

    # 使用 Car_Hacking_Challenge_Dataset_rev20Mar2021 数据集
    # 生成数据集需要修改的参数
    parser.add_argument('--dataset_name', type=str,
                        help='dynamic or static', default='Car_Hacking_Challenge_Dataset_rev20Mar2021')  # 0 or 1 or 2

    parser.add_argument('--ds', type=list,
                        help='dynamic or static', default=['D'])  # D or S 车辆动态报文 或者 车辆静止报文
    parser.add_argument('--csv_num', nargs='+', type=int,
                        help='csv num', default=[1, 2])  # 0 or 1 or 2  # csv文件标号
    parser.add_argument('--msg_smallest_num', type=int,
                        help='the smallest num of msg of a graph', default=50)  # 强化学习 每个步骤取一个图 构成这个图报文最小的条数
    parser.add_argument('--msg_biggest_num', type=int,
                        help='the biggest num of msg of a graph', default=300)  # 强化学习 每个步骤取一个图 构成这个图报文最大的条数


    # 实验结果发送邮件相关
    parser.add_argument('--username', type=str,
                        help='user name', default="976362661@qq.com")  # 用户名
    parser.add_argument('--password', type=str,
                        help='password', default="vsgbohogiqerbcji")  # 授权码
    parser.add_argument('--sender', type=str,
                        help='sender email', default='976362661@qq.com')  # 发送邮箱
    parser.add_argument('--receivers', type=str,
                        help='recervers email', default='ts_951117aaron@163.com')  # 接收邮箱
    parser.add_argument('--smtp_server', type=str,
                        help='smtp server', default="smtp.qq.com")  # smtp 服务器
    parser.add_argument('--port', type=int,
                        help='smtp port', default=465)  # smtp 端口

    # 显卡平台选择 涉及数据保存
    parser.add_argument('--gpu_device', type=str,
                        help='gpu platform', default="colab")  # colab or share_gpu


    parser.set_defaults(max_nodes=81,
                        feature_type='default',
                        lr=0.001,
                        train_ratio=0.8,
                        batch_size=64,  # 一个图的选择是一个动作
                        num_epochs=20,
                        num_workers=2,
                        input_dim=10,
                        hidden_dim=20,
                        output_dim=20,
                        num_classes=2,  # 正常报文和入侵报文
                        num_gc_layers=3,
                        dropout=0.0,
                        bmname='Pre_train',
                        origin_can_datadir='../data/Car_Hacking_Challenge_Dataset_rev20Mar2021/0_Preliminary/0_Training/',
                       )

    return parser.parse_args()

args_graph = arg_parse()
