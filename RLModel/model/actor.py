import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class Actor(nn.Module):

    def __init__(self, state_dim, action_dim, max_action, log_dir):
        super(Actor, self).__init__()

        self.layernorm = nn.LayerNorm(state_dim)

        self.fc1 = nn.Linear(state_dim, 500)
        torch.nn.Dropout(0.5)
        # self.fc2 = nn.Linear(300, 200)
        # torch.nn.Dropout(0.5)
        self.fc3 = nn.Linear(500, action_dim)

        self.max_action = max_action  # 1

        self.log_our_dir = log_dir
        # self.state_norm_np = np.array([], [])
        self.step = 0

    def forward(self, state, p=False):
        # print(state.shape)
        a = self.layernorm(state)

        if p:  # 如果是在强化学习输出过程则记录 训练过程则不记录
            self.step += 1
            if self.step == 1:
                self.state_norm_np = a.cpu().detach().numpy()
            else:
                self.state_norm_np = np.concatenate((self.state_norm_np, a.cpu().detach().numpy()), axis=0)
            if self.step % 1000 == 0:
                sns.heatmap(self.state_norm_np, cmap="YlGnBu", xticklabels=False, yticklabels=False)
                plt.savefig(self.log_our_dir + '/plt_norm_state_%d' % self.step, dpi=300, bbox_inches='tight')
                plt.clf()  # 更新画布
                self.step = 0

        a = F.relu(self.fc1(a))
        # a = F.relu(self.fc2(a))
        a = self.fc3(a)

        # a = torch.tanh(self.fc3(a)) * self.max_action
        return a
