import os, sys, random
import numpy as np
import torch
from args import arg_parse

from RLModel.model.td3 import TD3
from graphModel.task import Task
from utils import setup_seed
import time

import warnings
warnings.filterwarnings('ignore')  # 忽略警告

prog_args = arg_parse()
# Set seeds
setup_seed(prog_args.seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('use device: ', device)

# 状态维度
# 这里的状态表示向量分别是 第一次图卷积操作（做了三次卷积 每次卷积产生 20 维向量） 图塌缩后 第二次卷积 同样是 产生 3 * 20 维
state_dim = ((prog_args.num_gc_layers - 1) * prog_args.hidden_dim + prog_args.output_dim) * 2  # 60 * 2
# 动作维度
action_dim = prog_args.msg_biggest_num - prog_args.msg_smallest_num  # 每个图可选取报文长度的范围


'''
Implementation of TD3 with pytorch 
Original paper: https://arxiv.org/abs/1802.09477
Not the author's implementation !
'''
'''
每个 can 长度的选项都是 一个维度
每次识别 can 报文的长度 就是 强化学习网络的输出维度
'''


def main():

    agent = TD3(state_dim, action_dim, 1, prog_args)
    # 累加奖励
    ep_r = 0
    # 实例化 图任务
    graph_task = Task(prog_args, device)
    pred_hidden_dims = [int(i) for i in prog_args.pred_hidden.split('_')]
    # 第一次随机 图的长度 [50-300] 闭空间 给出强化学习的 初始 state
    graph_len_ = random.randint(prog_args.msg_smallest_num, prog_args.msg_biggest_num)

    # 定义此次实验的 log 文件夹
    time_mark = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_out_dir = prog_args.out_dir + '/' + 'Rl_' + time_mark + '_multiDim_log/'
    if not os.path.exists(log_out_dir):
        os.makedirs(log_out_dir, exist_ok=True)
    # 定义 并创建 log 文件
    log_out_file = log_out_dir + 'Rl_' + time_mark + '.txt'
    print(f'log 路径: {log_out_file}')

    with open(log_out_file, 'w+') as f:
        f.write(f'{prog_args}\n')

    if prog_args.mode == 'test':

        # 载入 模型
        agent.load()
        # 记录 图模型 执行 步数
        graph_step = 0
        # for i in range(prog_args.train_epoch):
        # 随机获取 初始状态
        state, _, _ = graph_task.benchmark_task_val(prog_args.feat, pred_hidden_dims, graph_len_, log_out_file)

        while True:
            action = agent.select_action(state)  # 从 现在的 状态 得到一个动作 维度是 报文长度可选择数量
            # 图操作 步数 自增 1
            graph_step += 1
            # 下个状态 奖励 是否完成
            len_can = np.argmax(action) + prog_args.msg_smallest_num  # 得到 下一块 数据的长度
            next_state, reward, done = graph_task.benchmark_task_val(prog_args.feat, pred_hidden_dims, len_can, log_out_file)
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

    elif prog_args.mode == 'train':
        print("====================================")
        print("Collection Experience...")
        print("====================================")

        # 若指定 载入模型参数 则 载入
        if prog_args.load:
            print(f'加载模型 {prog_args.model_load_dir}')
            with open(log_out_file, 'a') as f:
                f.write(f'加载模型... {prog_args.model_load_dir}\n')
            agent.load()


        for i in range(prog_args.train_epoch):  # epoch
            # 随机获取 初始状态
            state, _ , _ = graph_task.benchmark_task_val(prog_args.feat, pred_hidden_dims, graph_len_, log_out_file)
            # 记录 图模型 执行 步数
            graph_step = 0
            # 记录正确预测的 报文 个数
            pred_correct = 0
            # print('### main.py state.shape ###', state.shape)
            # for t in range(1):
            while True:

                # 强化学习网络
                print("====================================" * 3)
                action = agent.select_action(state)  # 从 现在的 状态 得到一个动作 报文长度可选择数量
                # action = action + np.random.normal(0.2, args_RL.exploration_noise, size=action.shape[0])  # 给强化学习的输出加入噪声
                # action = action.clip(env.action_space.low, env.action_space.high)
                # print('### main.py action.shape ###', action.shape)
                # print(f'action: {action}')
                # print(action)

                # 图操作 步数 自增 1
                graph_step += 1
                # 下个状态 奖励 是否完成
                # 选取 前5 个最大的可能里 选择报文数最大的
                len_can = max(action.argsort()[::-1][0:5]) + prog_args.msg_smallest_num
                # len_can = np.argmax(action) + prog_args.msg_smallest_num  # 得到 下一块 数据的长度
                next_state, reward, done = graph_task.benchmark_task_val(prog_args.feat, pred_hidden_dims, len_can, log_out_file)

                # 数据读取完毕 跳出本轮
                if done:
                    agent.writer.add_scalar('ep_r', ep_r, global_step=i)
                    # if i % args_RL.print_log == 0:
                    #     print("Ep_i \t{}, the ep_r is \t{:0.2f}, the step is \t{}".format(i, ep_r, t))
                    ep_r = 0
                    break

                # 累加 奖励
                ep_r += reward
                if reward > 0:
                    pred_correct += 1
                with open(log_out_file, 'a') as f:
                    f.write("====================================\n")
                    f.write(f'epoch: {i}; graph_step: {graph_step}; len_can: {len_can}; reward: {reward}; ep_r: {ep_r}; acc: {pred_correct/graph_step:.4f}\n')

                print(f'epoch: {i}; graph_step: {graph_step}; len_can: {len_can}; reward: {reward}; ep_r: {ep_r}; acc: {pred_correct/graph_step:.4f}\n')

                # 存入 经验
                agent.memory.push((state.cpu().data.numpy().flatten(), next_state.cpu().data.numpy().flatten(), action, reward, np.float(done)))
    #             if i+1 % 10 == 0:
    #                 print('Episode {},  The memory size is {} '.format(i, len(agent.memory.storage)))
                if len(agent.memory.storage) >= prog_args.capacity-1:
                    agent.update(10, log_out_file)  # 使用经验回放 更新网络

                # 更新 状态
                state = next_state

                # # 保存 模型
                # if graph_step % args_RL.log_interval == 0:
                #     agent.save()
                #     break
            # 保存 模型
            print(f'epoch {i} over.')
            agent.save(log_out_dir)
            time_mark = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            with open(log_out_file, 'a') as f:
                f.write(time_mark + '\n')
                f.write('END')

    else:
        raise NameError("mode wrong!!!")

if __name__ == '__main__':
    main()
