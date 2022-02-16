import os, sys, random
import numpy as np

import torch

from graphModel.args import arg_parse as arg_parse_gp
args_graphModel = arg_parse_gp()

from RLModel.args import arg_parse as arg_parse_RL
args_RL = arg_parse_RL()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
from RLModel.model.td3 import TD3


# Set seeds
# env.seed(args.seed)
# torch.manual_seed(args.seed)
# np.random.seed(args.seed)
script_name = os.path.basename(__file__)

# 选取 CAN 报文数据 数量范围
get_node_num_max = 300
get_node_num_min = 12
# 状态维度
# 这里的状态表示向量分别是 第一次图卷积操作（做了三次卷积 每次卷积产生 20 维向量） 图塌缩后 第二次卷积 同样是 产生 3 * 20 维
state_dim = ((args_graphModel.num_gc_layers-1) * args_graphModel.hidden_dim + args_graphModel.output_dim) * 2  # 60 * 2
# 动作维度
action_dim = get_node_num_max - get_node_num_min  # 288
# 模型存放目录
directory = './exp_RL_model/'
'''
Implementation of TD3 with pytorch 
Original paper: https://arxiv.org/abs/1802.09477
Not the author's implementation !
'''


def main():
    agent = TD3(state_dim, action_dim, 1)
    ep_r = 0
    from graphModel import task
    pred_hidden_dims = [int(i) for i in args_graphModel.pred_hidden.split('_')]
    graph_len_ = random.randint(12, 300)  # 第一次随机 图的长度 给出强化学习的 初始 state


    if args_RL.mode == 'test':
        pass
    #     agent.load()
    #     for i in range(args.iteration):
    #         state = env.reset()
    #         for t in count():
    #             action = agent.select_action(state)
    #             next_state, reward, done, info = env.step(np.float32(action))
    #             ep_r += reward
    #             env.render()
    #             if done or t ==2000 :
    #                 print("Ep_i \t{}, the ep_r is \t{:0.2f}, the step is \t{}".format(i, ep_r, t))
    #                 break
    #             state = next_state

    elif args_RL.mode == 'train':
        print("====================================")
        print("Collection Experience...")
        print("====================================")

        # 判断文件夹 是否存在
        # print(args_graphModel.origin_can_datadir)
        # print(os.path.exists(args_graphModel.origin_can_datadir))
        # print(os.path.exists('expmain.pyPendulum-v0./events.out.tfevents.1644981321.MacBook-Pro.local'))
        # print(os.listdir('../graphModel/data/Car_Hacking_Challenge_Dataset_rev20Mar2021/0_Preliminary/'))


        if args_RL.load: agent.load()
        for i in range(args_RL.num_iteration):
            # state = env.reset()
            state, _ , _ = task.benchmark_task_val(args_graphModel, args_graphModel.feat, pred_hidden_dims, device, args_graphModel.origin_can_datadir, graph_len_)
            for t in range(1):
            # 这里 会在 图操作中 传回一个 数据读完的标志 然后停止训练

                action = agent.select_action(state)  # 从 现在的 状态 得到一个动作
                action = np.argmax(action)  # 得到动作
                print(f'action: {action}')
                # print(action)

                # break
                # action = action + np.random.normal(0, args.exploration_noise, size=action_dim)  # 动作加入 噪音
                # action = action.clip(env.action_space.low, env.action_space.high)
                # next_state, reward, done, info = env.step(action)
                next_state, reward, done = task.benchmark_task_val(args_graphModel, args_graphModel.feat, pred_hidden_dims, device, args_graphModel.origin_can_datadir, action)
                ep_r += reward
    #             if args.render and i >= args.render_interval : env.render()
                agent.memory.push((state, next_state, action, reward, np.float(done)))
    #             if i+1 % 10 == 0:
    #                 print('Episode {},  The memory size is {} '.format(i, len(agent.memory.storage)))
                if len(agent.memory.storage) >= args_RL.capacity-1:
                    agent.update(10)
    #
                state = next_state
                if done or t == args_RL.max_episode - 1:
                    agent.writer.add_scalar('ep_r', ep_r, global_step=i)
                    if i % args_RL.print_log == 0:
                        print("Ep_i \t{}, the ep_r is \t{:0.2f}, the step is \t{}".format(i, ep_r, t))
                    ep_r = 0
                    break

            if i % args_RL.log_interval == 0:
                agent.save()

    else:
        raise NameError("mode wrong!!!")

if __name__ == '__main__':
    main()
