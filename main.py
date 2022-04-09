import os, sys, random
import platform
import numpy as np
import torch
import time
import logging
import traceback
# 工程参数
import utils.utils
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
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from utils.WrapperModel import Wrapper as Model_Wrapper

# 参数初始化
prog_args = arg_parse()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('use device: ', device)

# 设置随机种子 方便实验复现
setup_seed(prog_args.seed)

# 状态维度
# 这里的状态表示向量分别是 第一次图卷积操作（做了三次卷积 每次卷积产生 20 维向量） 图塌缩后 第二次卷积 同样是 产生 3 * 20 维
# state_dim = ((prog_args.num_gc_layers - 1) * prog_args.hidden_dim + prog_args.output_dim) * 2  # 60 * 2
state_dim = prog_args.state_dim
# 动作维度
action_dim = prog_args.msg_biggest_num - prog_args.msg_smallest_num + 1  # 每个图可选取报文长度的范围 动作空间 左闭右闭 [200, 500]

'''
Implementation of TD3 with pytorch 
Original paper: https://arxiv.org/abs/1802.09477
Not the author's implementation !
'''
'''
每个 can 长度的选项都是 一个维度
每次识别 can 报文的长度 就是 强化学习网络的输出维度
'''
global sys_cst_time
global train_times, avg_Q1_loss, avg_Q2_loss  # 系统时间


def main():
    # 获取当地时间
    time_mark = time.strftime("%Y%m%d_%H%M%S", time.localtime())

    global sys_cst_time, train_times, avg_Q1_loss, avg_Q2_loss
    train_times = 0
    avg_Q1_loss = 0
    avg_Q2_loss = 0

    # 记录本次实验的输入参数
    with open('../experiment/exp-record.txt', 'a') as f:
        f.write(time_mark + '\t' + '\t'.join(sys.argv) + '\n')
        f.close()
    # 定义此次实验的 log 文件夹
    log_out_dir = prog_args.out_dir + '/' + 'Rl_' + time_mark + '_multiDim_log/'
    if not os.path.exists(log_out_dir):
        os.makedirs(log_out_dir, exist_ok=True)

    # 如果系统是 linux 则对系统时区进行设置
    # 避免日志文件中的日期 不是大陆时区
    # if platform.system().lower() == 'linux':
    #     print("linux")
    #     os.system("cp /usr/share/zoneinfo/Asia/Shanghai /etc/localtime")
    #     sys_cst_time = os.popen('date').read()
    #     print(f'系统时间: {sys_cst_time}')
    # 授时之后 系统时间更改 但是python获取的时间会有延迟
    # 验证时间 python 取到的时间是否和系统相符
    # timestr = 'Fri Feb 25 17:35:08 CST 2022'
    # while True:
    #     # python时间 和 系统时间 同步 退出
    #     if abs(time.mktime(time.strptime(sys_cst_time.strip('\n'), '%a %b %d %H:%M:%S CST %Y')) - time.time()) < 120:
    #         break
    #     time.sleep(0.5)
    # print(f'python 时间同步完成')

    agent = TD3(state_dim, action_dim, 1, prog_args, log_out_dir)  # 创建 agent 对象

    # tensorboard 可视化 actor 和 critic
    Wrapper = Model_Wrapper(agent.actor, agent.critic_1)
    agent.writer.add_graph(Wrapper, [torch.zeros([1,  prog_args.state_dim]).to(device), torch.zeros([1, prog_args.msg_biggest_num - prog_args.msg_smallest_num + 1]).to(device)])

    # 累加奖励
    ep_r = 0
    # 实例化 图任务
    graph_task = Task(prog_args, device)
    pred_hidden_dims = [int(i) for i in prog_args.pred_hidden.split('_')]

    # 定义 并创建 log 文件
    log_out_file = log_out_dir + 'Rl_' + time_mark + '.log'
    # 配置日志 输出格式
    handler = logging.FileHandler(log_out_file)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    # log 写入 参数
    logger.info('\t'.join(sys.argv))  # 写入运行程序时输入的参数
    logger.info(f'输出 log 文件路径: {log_out_file}')
    logger.info(f'{prog_args}')
    logger.info(f'图模型- 学习率: {prog_args.graph_lr}')
    logger.info(f'强化学习学习率: {prog_args.reforce_lr}')

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
        graph_len_list = random.randint(prog_args.msg_smallest_num, prog_args.msg_biggest_num)

        state, _, _ = graph_task.benchmark_task_val(prog_args.feat, pred_hidden_dims, graph_len_list)

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
        logger.info("====================================")
        logger.info("Collection Experience...")
        logger.info("====================================")

        # 若指定强化学习 载入模型参数 则 载入
        if prog_args.load:
            logger.info(f'加载模型 {prog_args.model_load_dir}')
            agent.load()  # 载入模型
        else:
            logger.info(f'模型从头开始训练\n')

        # 不需要每个 epoch 都重新赋值的变量
        last_val_acc = 0  # 记录上一轮的验证精度 如果精度上升则保存模型
        val_acc = 0  # 验证精度
        train_acc = 0  # 训练精度

        try:
            for i in range(prog_args.train_epoch):  # epoch

                # 第一次随机 图的长度 [50-300] 闭空间 并且是batchsize个给出强化学习的 初始 state
                graph_len_list = utils.utils.random_list(prog_args.msg_smallest_num, prog_args.msg_biggest_num, prog_args.graph_batchsize)
                # 随机获取 初始状态 next_state, reward, train_done, val_done, label, pred, graph_loss
                state, _ , _, _, _, _, _ = graph_task.benchmark_task_val(prog_args.feat, pred_hidden_dims, graph_len_list)
                # print(f'随机得到的状态是 {state}')
                # 记录 图模型 执行 步数
                step = 0
                graph_train_step = 0
                graph_val_step = prog_args.graph_batchsize  # 因为模型读取train数据读完，未跳出读取 val 数据，所以需要赋予 batchsize 大小的初值
                # 记录正确预测的 报文 个数
                pred_train_correct = 0
                pred_val_correct = 0
                train_done = False
                states_np = None

                # states_np = np.array([], [])  # 存储每次图卷积网络输出的状态向量

                while True:
                    # 强化学习网络
                    # 串行把 batchsize 大小的数据输入 强化学习 得到batchsize大小个 can 数据长度
                    len_can_list = []
                    actions = []
                    for singleCan in range(prog_args.graph_batchsize):
                        action = agent.select_action(state[singleCan], p=True)  # 从 现在的 状态 得到一个动作 报文长度可选择数量
                        # print('aaron55')
                        #
                        # agent.writer.add_graph(Wrapper, [torch.unsqueeze(state[singleCan], dim=0), torch.unsqueeze(torch.from_numpy(action).to(device), dim=0)])

                        # action = action + np.random.normal(0, prog_args.exploration_noise, size=action.shape[0])
                        # action = action.clip(-1, 1)

                        # len_can = 0
                        # 选取 前5 个最大的可能里 选择报文数最大的
                        # if prog_args.choice_graph_len_mode == 0:
                        #     len_can = np.argmax(action) + prog_args.msg_smallest_num  # 得到 下一块 数据的长度
                        if np.random.uniform() > prog_args.epsilon:  # choosing action
                            len_can = np.random.randint(prog_args.msg_smallest_num, prog_args.msg_biggest_num)
                        else:
                            # len_can = np.argmax(action) + prog_args.msg_smallest_num  # 得到 下一块 数据的长度
                            # np.random.choice 是左闭右开 所以加 1
                            len_can = np.random.choice(prog_args.msg_biggest_num - prog_args.msg_smallest_num + 1, 1, action.tolist())[0] + prog_args.msg_smallest_num

                        # elif prog_args.choice_graph_len_mode == 1:
                        #     # 选取 前5 个最大的可能里 选择报文数最大的
                        #     len_can = max(action.argsort()[::-1][0:5]) + prog_args.msg_smallest_num
                        # elif prog_args.choice_graph_len_mode == 2:
                        #     # 在 前 5 个最大的可能里 随机选择一个报文长度
                        #     alter = random.randint(0, 4)
                        #     len_can = action.argsort()[::-1][0:5][alter] + prog_args.msg_smallest_num

                        action_store = [1 if _ == (len_can - prog_args.msg_smallest_num) else 0 for _ in range(prog_args.msg_biggest_num - prog_args.msg_smallest_num + 1)]

                        # 输出 softmax 各个动作的概率
                        actions.append(action_store)

                        # 训练阶段
                        if not train_done:
                            # 图操作 训练步数 自增 1
                            graph_train_step += 1
                        else:
                            # 图操作 验证步数自增 1
                            graph_val_step += 1
                        step += 1  # 总 step

                        # if step % 1000 == 0:  #  每1000步输出 状态热力图
                        #     fig, ax = plt.subplots(figsize=(10, 10))
                        #     sns.heatmap(states_np, cmap="YlGnBu")
                        #     plt.savefig(log_out_dir + '/plt_state_%d_%d' % (i, step), dpi=300, bbox_inches='tight')
                        #     plt.clf()  # 更新画布
                        #     states_np = None


                        # 把神经网络得到的长度加入列表
                        len_can_list.append(len_can)

                    next_state, reward, train_done, val_done, label, pred, graph_loss = graph_task.benchmark_task_val(prog_args.feat, pred_hidden_dims, len_can_list)

                    # 最后结束
                    if val_done:
                        # 把 train_done 和 val_done 置位
                        graph_task.origin_can_obj.train_done = False
                        graph_task.origin_can_obj.val_done = False

                        agent.writer.add_scalar('ep_r', ep_r, global_step=i)
                        # if i % args_RL.print_log == 0:
                        #     print("Ep_i \t{}, the ep_r is \t{:0.2f}, the step is \t{}".format(i, ep_r, t))
                        ep_r = 0
                        break  # break true

                    # # 实时绘制 图神经网络的loss曲线
                    # graph_task.history1.log((i, graph_train_step), graph_train_loss=graph_loss)
                    # with graph_task.canvas1:
                    #     graph_task.canvas1.draw_plot(graph_task.history1["graph_train_loss"])

                    # 训练部分
                    if not train_done:
                        # rewards = []
                        # push 经验
                        store_reward = []
                        for singleCan in range(prog_args.graph_batchsize):
                            # # 存入 经验
                            # if label[singleCan] == pred[singleCan]:
                            #     reward = abs(reward)
                            # else:
                            #     reward = -abs(reward)
                            if reward > 0.75:  # 预测准确率达 0.75 - 1
                                if label[singleCan] == pred[singleCan]:
                                    store_reward.append(100)
                                else:
                                    store_reward.append(-1)
                            elif reward > 0.5:  # 预测准确率达 0.5 - 0.75
                                if label[singleCan] == pred[singleCan]:
                                    store_reward.append(10)
                                else:
                                    store_reward.append(-100)
                            elif reward > 0.25:  # 预测准确率达 0.25 - 0.5
                                if label[singleCan] == pred[singleCan]:
                                    store_reward.append(-10)
                                else:
                                    store_reward.append(-100)
                            else :  # 预测准确率达 0.25
                                if label[singleCan] == pred[singleCan]:
                                    store_reward.append(-50)
                                else:
                                    store_reward.append(-100)

                            # 累加 奖励
                            ep_r += reward
                            # rewards.append(reward)
                            agent.memory.push((state[singleCan].cpu().data.numpy().flatten(),
                                               next_state[singleCan].cpu().data.numpy().flatten(),
                                               actions[singleCan], store_reward[-1], np.float(train_done)))
                            if len(agent.memory.storage) >= prog_args.capacity - 1:
                                train_times, avg_Q1_loss, avg_Q2_loss = agent.update(num_iteration=10)  # 使用经验回放 更新网络

                        # 计数训练时 预测正确的个数
                        for index, singlab in enumerate(label):
                            if singlab == pred[index]:
                                pred_train_correct += 1

                        # 得到训练精度
                        train_acc = pred_train_correct/graph_train_step
                        agent.writer.add_scalar('acc/train_acc', train_acc, global_step=graph_train_step)
                        agent.writer.add_scalar('Loss/graph_train_loss', graph_loss, global_step=graph_train_step)


                        # 结果写入 log
                        logger.info(f'epoch-train: {i:<3}; train-step: {graph_train_step:<6}; '
                                    f'block_{graph_task.origin_can_obj.train_index}: {graph_task.origin_can_obj.train_order[graph_task.origin_can_obj.train_index]}; '
                                    f'{graph_task.origin_can_obj.point}/{graph_task.origin_can_obj.data_train_block_len}; '
                                    f'reward: {reward:<8.3f}; '
                                    f'acc: {train_acc:<4.2f}; trainTimes: {train_times}; g_loss: {graph_loss:<8.6f}; '
                                    f'avg_Q1_loss: {avg_Q1_loss:.2f}; avg_Q2_loss: {avg_Q2_loss:.2f}; ep_r: {ep_r:.2f}')
                        logger.info(f'len_can_list: {len_can_list}')
                        logger.info(f'labe: {label}')
                        logger.info(f'pred: {pred}')
                        logger.info(f'swrd: {store_reward}')

                    # 验证部分
                    else:
                        store_reward = []
                        for singleCan in range(prog_args.graph_batchsize):
                            # # 存入 经验
                            # if label[singleCan] == pred[singleCan]:
                            #     reward = abs(reward)
                            # else:
                            #     reward = -abs(reward)
                            # # 累加 奖励
                            # ep_r += reward
                            if reward > 0.75:  # 预测准确率达 0.75 - 1
                                if label[singleCan] == pred[singleCan]:
                                    store_reward.append(100)
                                else:
                                    store_reward.append(-1)
                            elif reward > 0.5:  # 预测准确率达 0.5 - 0.75
                                if label[singleCan] == pred[singleCan]:
                                    store_reward.append(10)
                                else:
                                    store_reward.append(-100)
                            elif reward > 0.25:  # 预测准确率达 0.25 - 0.5
                                if label[singleCan] == pred[singleCan]:
                                    store_reward.append(-10)
                                else:
                                    store_reward.append(-100)
                            else:  # 预测准确率达 0.25
                                if label[singleCan] == pred[singleCan]:
                                    store_reward.append(-50)
                                else:
                                    store_reward.append(-100)

                        # 计数训练时 预测正确的个数
                        for index, singlab in enumerate(label):
                            if singlab == pred[index]:
                                pred_val_correct += 1
                        # 得到验证精度
                        val_acc = pred_val_correct/graph_val_step

                        agent.writer.add_scalar('acc/val_acc', val_acc, global_step=graph_val_step)
                        agent.writer.add_scalar('Loss/graph_val_loss', graph_loss, global_step=graph_val_step)
                        # 结果写入 log
                        logger.info(f'epoch-val: {i:<3}; step: {graph_val_step:<6}; '
                                    f'{graph_task.origin_can_obj.point}/{graph_task.origin_can_obj.data_val_len}; '
                                    f'reward: {reward:<8.3f}; '
                                    f'acc: {val_acc:<4.2f}; g_loss: {graph_loss:<8.6f}; ep_r: {ep_r:.2f}')
                        logger.info(f'len_can_list: {len_can_list}')
                        logger.info(f'labe: {label}')
                        logger.info(f'pred: {pred}')
                        logger.info(f'swrd: {store_reward}')

                    # 更新 状态
                    state = next_state

                    # if graph_train_step < 100 or states_np is None:
                    #     states_np = state.cpu().detach().numpy()
                    # else:
                    #     states_np = np.concatenate((states_np, state.cpu().detach().numpy()), axis=0)


                    # # 保存 模型
                    # if graph_step % args_RL.log_interval == 0:
                    #     agent.save()
                    #     break

                # 记录此次的训练精度 和 验证精度
                logger.info(f'epoch-{i}-over '
                            f'trian-times: {train_times} '
                            f'train_acc: {train_acc:<4.6f} '
                            f'val_acc: {val_acc:<4.6f}')
                # 跳出whileTrue 结束epoch 保存模型
                # 如果此次的验证精度上升则保存模型

                # 置位完成标识位
                graph_task.origin_can_obj.train_done = False
                graph_task.origin_can_obj.val_done = False

                if val_acc > last_val_acc:
                    # 保存本次的验证精度
                    last_val_acc = val_acc
                    # 保存强化学习模型
                    agent.save(i, str('%.4f' % val_acc), log_out_dir)
                    # 保存图模型
                    graph_model_path = log_out_dir + 'epoch_' + str(i) + '_graph_model.pth'
                    graph_model_para_path = log_out_dir + 'epoch_' + str(i) + '_graph_para.pth'
                    torch.save(graph_task.model, graph_model_path)
                    torch.save(graph_task.model.state_dict(), graph_model_para_path)

                # # 结束一次 epoch 发送一次邮件 防止 colab 突然停止
                # content = f'{time.strftime("%Y%m%d_%H%M%S", time.localtime())} END\n' \
                #           f'epoch: {i}\n'\
                #           f'retrain: {retrain}\n'
                # resultfile = packresult(log_out_dir[:-1], i)  # 1.传入log路径参数 去掉最后的 / 2. 训练结束的代数
                # send_email(prog_args.username, prog_args.password, prog_args.sender, prog_args.receivers,
                #            prog_args.smtp_server, prog_args.port, content, resultfile)

        except Exception as e:  # 捕捉所有异常
            logger.info(f'发生异常 {e}')
            error = e

        finally:
            # 异常信息写入 log
            logger.warning(f'error: {error}')
            # 程序执行失败信息写入 log
            traceback.print_exc()
            logger.warning(f"执行失败信息: {traceback.format_exc()}")
            # 无论实验是否执行完毕 都把结果发送邮件
            # 跑完所有的 epoch 打包实验结果 返回带 .zip 的文件路径
            # print(f'正在打包结果文件夹  {log_out_dir}')
            # agent.save(i, log_out_dir)  # 保存 最新的模型参数
            # resultfile = packresult(log_out_dir[:-1], i)  # 1.传入log路径参数 去掉最后的 / 2. 训练结束的代数
            # print(f'打包完毕')
            # 发送邮件
            # print(f'正在发送邮件...')
            # content = f'platform: {prog_args.gpu_device}\n'\
            #           f'{time.strftime("%Y%m%d_%H%M%S", time.localtime())} END\n' \
            #           f'retrain: {retrain}\n' \
            #           f'schedule: {graph_task.origin_can_obj.point}/{graph_task.origin_can_obj.data_total_len}\n' \
            #           f'error: {error}\n'

            # send_email(prog_args.username, prog_args.password, prog_args.sender, prog_args.receivers, prog_args.smtp_server, prog_args.port, content,resultfile)
            # print(f'发送邮件完毕')

            # # 如果是在 share_gpu 上运行的 把数据都拷贝到 oss 个人数据下
            # if prog_args.gpu_device == 'share_gpu':
            #     # 全部打包
            #     resultfile = packresult(log_out_dir[:-1], i, allfile=True)  # 1.传入log路径参数 去掉最后的 / 2. 训练结束的代数
            #     os.system(f"oss cp {resultfile} oss://backup/")
            #     print('关机...')
            #     os.system('/root/upload.sh')


    else:
        raise NameError("mode wrong!!!")

if __name__ == '__main__':
    main()
