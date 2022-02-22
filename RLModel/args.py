import argparse
import time


def arg_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', default='train', type=str)  # mode = 'train' or 'test'
    parser.add_argument('--tau', default=0.005, type=float)  # target smoothing coefficient
    parser.add_argument('--target_update_interval', default=1, type=int)
    parser.add_argument('--iteration', default=5, type=int)

    parser.add_argument('--learning_rate', default=3e-4, type=float)
    parser.add_argument('--gamma', default=0.99, type=int)  # discounted factor
    parser.add_argument('--capacity', default=100, type=int)  # replay buffer size
    parser.add_argument('--num_iteration', default=1, type=int)  # num of  games
    parser.add_argument('--batch_size', default=64, type=int)  # mini batch size
    parser.add_argument('--seed', default=1, type=int)

    # 定义 并创建 此次实验的 log 文件夹
    time_mark = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    parser.add_argument('--model_store_dir', default='../rl_model_store/' + time_mark, type=str)

    parser.add_argument('--model_load_dir', default='', type=str)

    parser.add_argument('--directory', default='../rl_model_log/', type=str)

    # optional parameters
    parser.add_argument('--num_hidden_layers', default=2, type=int)
    parser.add_argument('--sample_frequency', default=256, type=int)
    parser.add_argument('--activation', default='Relu', type=str)
    parser.add_argument('--render', default=False, type=bool)  # show UI or not
    parser.add_argument('--log_interval', default=50, type=int)  #
    parser.add_argument('--load', default=False, type=bool)  # load model
    # parser.add_argument('--render_interval', default=100, type=int)  # after render_interval, the env.render() will work
    parser.add_argument('--policy_noise', default=0.2, type=float)
    parser.add_argument('--noise_clip', default=0.5, type=float)
    parser.add_argument('--policy_delay', default=2, type=int)
    parser.add_argument('--exploration_noise', default=0.1, type=float)
    parser.add_argument('--max_episode', default=2000, type=int)
    parser.add_argument('--print_log', default=5, type=int)


    return parser.parse_args()

args_RL = arg_parse()