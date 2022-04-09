import torch
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # layer norm
        # self.layernorm = nn.LayerNorm(state_dim + action_dim)

        # batch norm
        self.batchnorm = nn.BatchNorm1d(361)

        self.fc1 = nn.Linear(state_dim + action_dim, 300)
        torch.nn.Dropout(0.5)
        self.fc2 = nn.Linear(300, 100)
        torch.nn.Dropout(0.5)
        self.fc3 = nn.Linear(100, 1)

    def forward(self, state, action):

        state_action = torch.cat([state, action], 1)

        q = self.batchnorm(state_action)

        q = F.relu(self.fc1(q))
        q = F.relu(self.fc2(q))
        q = self.fc3(q)
        return q
