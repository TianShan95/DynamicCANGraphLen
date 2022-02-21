import os, sys, random
import numpy as np
import torch
from graphModel.args import args_graph
from RLModel.args import args_RL
from RLModel.model.td3 import TD3
from graphModel.task import Task
import time


# Set seeds
# env.seed(args.seed)
# torch.manual_seed(args.seed)
# np.random.seed(args.seed)


# 选取 CAN 报文数据 数量范围
get_node_num_max = 300
get_node_num_min = 50
# 状态维度
# 这里的状态表示向量分别是 第一次图卷积操作（做了三次卷积 每次卷积产生 20 维向量） 图塌缩后 第二次卷积 同样是 产生 3 * 20 维
state_dim = ((args_graph.num_gc_layers - 1) * args_graph.hidden_dim + args_graph.output_dim) * 2  # 60 * 2
# 动作维度
action_dim = get_node_num_max - get_node_num_min  # 288
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print('use device: ', device)

'''
Implementation of TD3 with pytorch 
Original paper: https://arxiv.org/abs/1802.09477
Not the author's implementation !
'''


def main():

    agent = TD3(state_dim, action_dim, 1)
    ep_r = 0
    # 实例化 图任务
    graph_task = Task(args_graph, device)
    pred_hidden_dims = [int(i) for i in args_graph.pred_hidden.split('_')]
    graph_len_ = random.randint(12, 300)  # 第一次随机 图的长度 [12-300] 闭空间 给出强化学习的 初始 state

    # 定义此次实验的 log 文件夹
    time_mark = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_out_dir = args_graph.out_dir + '/' + 'Rl_' + time_mark + '_log/'
    if not os.path.exists(log_out_dir):
        os.makedirs(log_out_dir, exist_ok=True)
    # 定义 并创建 log 文件
    log_out_file = log_out_dir + 'Rl_' + time_mark + '.txt'

    with open(log_out_file, 'a') as f:
        f.write("RL 参数\n")
        f.write(f'{args_RL}\n')
        f.write("图部分参数\n")
        f.write(f'{args_graph}\n')

    if args_RL.mode == 'test':

        # 载入 模型
        agent.load()
        # 记录 图模型 执行 步数
        graph_step = 0
        for i in range(args_RL.iteration):
            # 随机获取 初始状态
            state, _, _, _ = graph_task.benchmark_task_val(i, graph_step, args_graph.feat, pred_hidden_dims, graph_len_, 'test', first=True)

            while True:
                action = agent.select_action(state)  # 从 现在的 状态 得到一个动作 维度是 报文长度可选择数量
                # 图操作 步数 自增 1
                graph_step += 1
                # 下个状态 奖励 是否完成
                next_state, reward, done, trained_model = graph_task.benchmark_task_val(i, graph_step, args_graph.feat, pred_hidden_dims, action,'test', first=False)
                # 累加 奖励
                ep_r += reward
                # 更新 状态
                state = next_state

                # 数据读取完毕
                # if done:
                #     agent.writer.add_scalar('ep_r', ep_r, global_step=i)
                #     if i % args_RL.print_log == 0:
                #         print("Ep_i \t{}, the ep_r is \t{:0.2f}, the step is \t{}".format(i, ep_r, t))
                #     ep_r = 0
                #     break

    elif args_RL.mode == 'train':
        print("====================================")
        print("Collection Experience...")
        print("====================================")

        # 若指定 载入模型参数 则 载入
        if args_RL.load: agent.load()
        # 记录 图模型 执行 步数
        graph_step = 0

        for i in range(args_RL.num_iteration):  # epoch
            # 随机获取 初始状态
            state, _ , _, = graph_task.benchmark_task_val(i, graph_step, args_graph.feat, pred_hidden_dims, graph_len_, 'train', first=True)

            # print('### main.py state.shape ###', state.shape)
            # for t in range(1):
            while True:

                action = agent.select_action(state)  # 从 现在的 状态 得到一个动作 报文长度可选择数量
                action = action + np.random.normal(0.2, args_RL.exploration_noise, size=action.shape[0])
                # action = action.clip(env.action_space.low, env.action_space.high)
                print('### main.py action.shape ###', action.shape)

                # print(f'action: {action}')
                # print(action)

                # 图操作 步数 自增 1
                graph_step += 1
                # 下个状态 奖励 是否完成
                next_state, reward, done = graph_task.benchmark_task_val(i, graph_step, args_graph.feat, pred_hidden_dims, action, 'train', first=False)
                with open(log_out_file, 'a') as f:
                    f.write(f'epoch: {i}; graph_step: {graph_step}; reward: {reward}')
                # 累加 奖励
                ep_r += reward
                print(f'epoch: {i}; graph_step: {graph_step}; reward: {reward}; ep_r: {ep_r}')


                agent.memory.push((state.cpu().data.numpy().flatten(), next_state.cpu().data.numpy().flatten(), action, reward, np.float(done)))
    #             if i+1 % 10 == 0:
    #                 print('Episode {},  The memory size is {} '.format(i, len(agent.memory.storage)))
                if len(agent.memory.storage) >= args_RL.capacity-1:
                    agent.update(10, log_out_file)  # 使用经验回放 更新网络

                # 更新 状态
                state = next_state

                # 数据读取完毕
                if done:
                    agent.writer.add_scalar('ep_r', ep_r, global_step=i)
                    # if i % args_RL.print_log == 0:
                    #     print("Ep_i \t{}, the ep_r is \t{:0.2f}, the step is \t{}".format(i, ep_r, t))
                    ep_r = 0
                    break

            # 保存 模型
            # if i % args_RL.log_interval == 0:
            agent.save()

    else:
        raise NameError("mode wrong!!!")

if __name__ == '__main__':
    main()
