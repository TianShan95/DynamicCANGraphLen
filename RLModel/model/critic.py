import torch
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(state_dim + action_dim, 300)
        torch.nn.Dropout(0.5)
        self.fc2 = nn.Linear(300, 100)
        torch.nn.Dropout(0.5)
        self.fc3 = nn.Linear(100, 1)

    def forward(self, state, action):
        # print('critic forward: ')

        # print(state.shape)
        # print(state)
        # print(action.shape)
        # print(action)
        state_action = torch.cat([state, action], 1)

        q = F.relu(self.fc1(state_action))
        q = F.relu(self.fc2(q))
        q = self.fc3(q)
        return q
