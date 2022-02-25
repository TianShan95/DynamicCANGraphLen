import os, sys, random
import platform
import numpy as np
import torch
import time
import logging
import traceback
# 工程参数
from args import arg_parse
# 强化学习
from RLModel.model.td3 import TD3
# 图网络
from graphModel.task import Task
# 功能函数
from utils.utils import setup_seed
from utils.packResult import packresult
from utils.sendMail import send_email
from utils.logger import logger
# 警告处理
import warnings
warnings.filterwarnings('ignore')  # 忽略警告

# 参数初始化
prog_args = arg_parse()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('use device: ', device)

# 设置随机种子 方便实验复现
setup_seed(prog_args.seed)

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
global sys_cst_time, train_times, avg_Q1_loss, avg_Q2_loss  # 系统时间


def main():

    global sys_cst_time, train_times, avg_Q1_loss, avg_Q2_loss

    # 如果系统是 linux 则对系统时区进行设置
    # 避免日志文件中的日期 不是大陆时区
    if platform.system().lower() == 'linux':
        print("linux")
        os.system("cp /usr/share/zoneinfo/Asia/Shanghai /etc/localtime")
        sys_cst_time = os.popen('date').read()
        print(f'系统时间: {sys_cst_time}')

    agent = TD3(state_dim, action_dim, 1, prog_args)
    # 累加奖励
    ep_r = 0
    # 实例化 图任务
    graph_task = Task(prog_args, device)
    pred_hidden_dims = [int(i) for i in prog_args.pred_hidden.split('_')]

    # 授时之后 系统时间更改 但是python获取的时间会有延迟
    # 验证时间 python 取到的时间是否和系统相符
    # timestr = 'Fri Feb 25 17:35:08 CST 2022'
    while True:
        # python时间 和 系统时间 同步 退出
        if abs(time.mktime(time.strptime(sys_cst_time.strip('\n'), '%a %b %d %H:%M:%S CST %Y')) - time.time()) < 120:
            break
        time.sleep(0.5)
    print(f'python 时间同步完成')

    # 定义此次实验的 log 文件夹
    time_mark = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_out_dir = prog_args.out_dir + '/' + 'Rl_' + time_mark + '_multiDim_log/'
    if not os.path.exists(log_out_dir):
        os.makedirs(log_out_dir, exist_ok=True)

    # 定义 并创建 log 文件
    log_out_file = log_out_dir + 'Rl_' + time_mark + '.log'

    print(f'输出 log 文件路径: {log_out_file}')

    # 配置日志 输出格式
    handler = logging.FileHandler(log_out_file)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    # log 写入 参数
    logger.info(f'{prog_args}')

    error = None  # 表示实验是否发生异常
    retrain = False  # 表示模型是否是从开开始训练 False: 从头训练 True: 继续训练

    if prog_args.mode == 'test':

        # 载入 模型
        agent.load()
        # 记录 图模型 执行 步数
        graph_step = 0
        # for i in range(prog_args.train_epoch):
        # 随机获取 初始状态
        # 第一次随机 图的长度 [50-300] 闭空间 给出强化学习的 初始 state
        graph_len_ = random.randint(prog_args.msg_smallest_num, prog_args.msg_biggest_num)

        state, _, _ = graph_task.benchmark_task_val(prog_args.feat, pred_hidden_dims, graph_len_)

        while True:
            action = agent.select_action(state)  # 从 现在的 状态 得到一个动作 维度是 报文长度可选择数量
            # 图操作 步数 自增 1
            graph_step += 1
            # 下个状态 奖励 是否完成
            len_can = np.argmax(action) + prog_args.msg_smallest_num  # 得到 下一块 数据的长度
            next_state, reward, done = graph_task.benchmark_task_val(prog_args.feat, pred_hidden_dims, len_can)
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
                # f.close()
            logger.info(f'加载模型... {prog_args.model_load_dir}\n')
            agent.load()  # 载入模型
        else:
            with open(log_out_file, 'a') as f:
                f.write(f'模型从头开始训练\n')
                # f.close()
            logger.info(f'模型从头开始训练\n')


        try:
            for i in range(prog_args.train_epoch):  # epoch

                # 第一次随机 图的长度 [50-300] 闭空间 给出强化学习的 初始 state
                graph_len_ = random.randint(prog_args.msg_smallest_num, prog_args.msg_biggest_num)
                # 随机获取 初始状态
                state, _ , _, _, _ = graph_task.benchmark_task_val(prog_args.feat, pred_hidden_dims, graph_len_)
                # print(f'随机得到的状态是 {state}')
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

                    len_can = 0
                    # 选取 前5 个最大的可能里 选择报文数最大的
                    if prog_args.choice_graph_len_mode == 0:
                        len_can = np.argmax(action) + prog_args.msg_smallest_num  # 得到 下一块 数据的长度
                    elif prog_args.choice_graph_len_mode == 1:
                        # 选取 前5 个最大的可能里 选择报文数最大的
                        len_can = max(action.argsort()[::-1][0:5]) + prog_args.msg_smallest_num
                    elif prog_args.choice_graph_len_mode == 2:
                        # 在 前 5 个最大的可能里 随机选择一个报文长度
                        alter = random.randint(0, 4)
                        len_can = action.argsort()[::-1][0:5][alter] + prog_args.msg_smallest_num

                    next_state, reward, done, label, pred = graph_task.benchmark_task_val(prog_args.feat, pred_hidden_dims, len_can)

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
                    # 结果写入 log
                    logger.info(f'epoch: {i:<3}; step: {graph_step:<8}; label: {label}; pred: {pred}; len: {len_can:<3}; reward: {reward:<8.3f}; acc: {pred_correct/graph_step:<4.1f}; trainTimes: {train_times}; avg_Q1_loss: {avg_Q1_loss:.2f}; avg_Q2_loss: {avg_Q2_loss:.2f}; ep_r: {ep_r:.2f}')

                    # 存入 经验
                    agent.memory.push((state.cpu().data.numpy().flatten(), next_state.cpu().data.numpy().flatten(), action, reward, np.float(done)))
        #             if i+1 % 10 == 0:
        #                 print('Episode {},  The memory size is {} '.format(i, len(agent.memory.storage)))
                    if len(agent.memory.storage) >= prog_args.capacity-1:
                        train_times, avg_Q1_loss, avg_Q2_loss = agent.update(10)  # 使用经验回放 更新网络

                    # 更新 状态
                    state = next_state

                    # 短期退出 epoch 验证 程序可运行行
                    if graph_step > 20:
                        print(f'大于 20步')
                        print(f'i {i}')
                        break
                    #     raise Exception

                    # # 保存 模型
                    # if graph_step % args_RL.log_interval == 0:
                    #     agent.save()
                    #     break
                # 跳出whileTrue 结束epoch 保存模型
                agent.save(i, log_out_dir)

                # # 结束一次 epoch 发送一次邮件 防止 colab 突然停止
                # content = f'{time.strftime("%Y%m%d_%H%M%S", time.localtime())} END\n' \
                #           f'epoch: {i}\n'\
                #           f'retrain: {retrain}\n'
                # resultfile = packresult(log_out_dir[:-1], i)  # 1.传入log路径参数 去掉最后的 / 2. 训练结束的代数
                # send_email(prog_args.username, prog_args.password, prog_args.sender, prog_args.receivers,
                #            prog_args.smtp_server, prog_args.port, content, resultfile)

        except Exception as e:  # 捕捉所有异常
            print(f'发生异常 {e}')
            error = e

        finally:
            # 异常信息写入 log
            logger.warning(f'error: {error}')
            # 程序执行失败信息写入 log
            traceback.print_exc()
            logger.warning(f"执行失败信息: {traceback.format_exc()}")
            # 无论实验是否执行完毕 都把结果发送邮件
            # 跑完所有的 epoch 打包实验结果 返回带 .zip 的文件路径
            print(f'正在打包结果文件夹  {log_out_dir}')
            agent.save(i, log_out_dir)  # 保存 最新的模型参数
            resultfile = packresult(log_out_dir[:-1], i)  # 1.传入log路径参数 去掉最后的 / 2. 训练结束的代数
            print(f'打包完毕')
            # 发送邮件
            print(f'正在发送邮件...')
            content = f'platform: {prog_args.gpu_device}'\
                      f'{time.strftime("%Y%m%d_%H%M%S", time.localtime())} END\n' \
                      f'retrain: {retrain}\n'\
                      f'error: {error}\n'

            send_email(prog_args.username, prog_args.password, prog_args.sender, prog_args.receivers, prog_args.smtp_server, prog_args.port, content,resultfile)
            print(f'发送邮件完毕')

            # 如果是在 share_gpu 上运行的 把数据都拷贝到 oss 个人数据下
            if prog_args.gpu_device == 'share_gpu':
                # 全部打包
                resultfile = packresult(log_out_dir[:-1], i, allfile=True)  # 1.传入log路径参数 去掉最后的 / 2. 训练结束的代数
                os.system(f"oss cp {resultfile} oss://backup/")
                print('关机...')
                os.system("shutdown")


    else:
        raise NameError("mode wrong!!!")

if __name__ == '__main__':
    main()
