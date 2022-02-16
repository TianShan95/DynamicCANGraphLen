import argparse
from graphModel.utils import ensure_dir


def arg_parse():
    parser = argparse.ArgumentParser(description='Arguments.')
    parser.add_argument('--bmname', dest='bmname', help='Name of the benchmark dataset')
    parser.add_argument('--max-nodes', dest='max_nodes', type=int,
            help='Maximum number of nodes (ignore graghs with nodes exceeding the number.')
    parser.add_argument('--lr', dest='lr', type=float,
            help='Learning rate.')
    parser.add_argument('--clip', dest='clip', type=float,
            help='Gradient clipping.')
    parser.add_argument('--batch-size', dest='batch_size', type=int,
            help='Batch size.')
    parser.add_argument('--epochs', dest='num_epochs', type=int,
            help='Number of epochs to train.')
    parser.add_argument('--train-ratio', dest='train_ratio', type=float,
            help='Ratio of number of graphs training set to all graphs.')
    parser.add_argument('--test-ratio', dest='test_ratio', type=float,
            help='Ratio of number of graphs testing set to all graphs.')
    parser.add_argument('--num_workers', dest='num_workers', type=int,
            help='Number of workers to load processData.')
    parser.add_argument('--feature', dest='feature_type',
            help='Feature used for encoder. Can be: id, deg')
    parser.add_argument('--input-dim', dest='input_dim', type=int,
            help='Input feature dimension')
    parser.add_argument('--hidden-dim', dest='hidden_dim', type=int,
            help='Hidden dimension')
    parser.add_argument('--output-dim', dest='output_dim', type=int,
            help='Output dimension')
    parser.add_argument('--num-classes', dest='num_classes', type=int,
            help='Number of label classes')
    parser.add_argument('--num-gc-layers', dest='num_gc_layers', type=int,
            help='Number of graph convolution layers before each pooling')
    parser.add_argument('--nobn', dest='bn', action='store_const',
            const=False, default=True,
            help='Whether batch normalization is used')
    parser.add_argument('--dropout', dest='dropout', type=float,
            help='Dropout rate.')
    parser.add_argument('--nobias', dest='bias', action='store_const',
            const=False, default=True,
            help='Whether to add bias. Default to True.')
    parser.add_argument('--datadir', dest='datadir',
            help='Directory where benchmark is located')
    parser.add_argument('--origin-can-datadir', dest='origin_can_datadir',
                        help='Directory where origin can dataset is located')


    parser.add_argument('--pool_sizes', type=str,
                        help='pool_sizes', default='10')
    parser.add_argument('--num_pool_matrix', type=int,
                        help='num_pooling_matrix', default=1)
    parser.add_argument('--min_nodes', type=int,
                        help='min_nodes', default = 12)

    parser.add_argument('--weight_decay', type=float,
                        help='weight_decay', default=0.0)
    parser.add_argument('--num_pool_final_matrix', type = int,
                        help = 'number of final pool matrix', default = 0)

    parser.add_argument('--normalize', type = int,
                        help = 'nomrlaized laplacian or not', default = 0)
    parser.add_argument('--pred_hidden', type = str,
                        help = 'pred_hidden', default = '50')

    parser.add_argument('--out_dir', type = str,
                        help = 'out_dir', default = 'experiment')
    parser.add_argument('--num_shuffle', type = int,
                        help = 'total num_shuffle', default = 10)
    parser.add_argument('--shuffle', type = int,
                        help = 'which shuffle, choose from 0 to 9', default=0)
    parser.add_argument('--concat', type = int,
                        help = 'whether concat', default=1)
    parser.add_argument('--feat', type = str,
                        help = 'which feat to use', default='node-label')
    parser.add_argument('--mask', type = int,
                        help = 'mask or not', default = 1)
    parser.add_argument('--norm', type = str,
                        help = 'Norm for eigens', default = 'l2')

    parser.add_argument('--directory', default='graphModel/graph_model_store/', type=str)
    parser.add_argument('--load', default=False, type=bool)


    parser.add_argument('--with_test', type = int,
                        help = 'with test or not', default = 0)
    parser.add_argument('--con_final', type = int,
                        help = 'con_final', default = 1)
    parser.add_argument('--device', type = str,
                        help='cpu or cuda', default='cpu')
    parser.set_defaults(max_nodes=300,
                        feature_type='default',
                        datadir='processData',
                        lr=0.001,
                        clip=2.0,
                        batch_size=1,  # 一个图的选择是一个动作
                        num_epochs=1000,
                        train_ratio=0.8,
                        test_ratio=0.1,
                        num_workers=0,
                        input_dim=10,
                        hidden_dim=20,
                        output_dim=20,
                        num_classes=2,  # 正常报文和入侵报文
                        num_gc_layers=3,
                        dropout=0.0,
                        bmname='data_can',
                        origin_can_datadir='graphModel/data/Car_Hacking_Challenge_Dataset_rev20Mar2021/0_Preliminary/0_Training/Pre_train_D_0.csv',
                        log_out_dir='log/',
                        # pred_hidden = ,
                       )

    ensure_dir(parser.parse_args().log_out_dir)
    return parser.parse_args()

args_graph = arg_parse()
graph_model_train_log_file = args_graph.log_out_dir + '/graph_model_train.log'
graph_model_test_log_file = args_graph.log_out_dir + '/graph_model_test.log'
