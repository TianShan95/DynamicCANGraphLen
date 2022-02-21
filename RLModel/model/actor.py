import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):

    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(state_dim, 300)
        torch.nn.Dropout(0.5)
        self.fc2 = nn.Linear(300, 100)
        torch.nn.Dropout(0.5)
        self.fc3 = nn.Linear(100, action_dim)

        self.max_action = max_action

    def forward(self, state):
        # print(state.shape)
        a = F.relu(self.fc1(state))
        a = F.relu(self.fc2(a))
        a = torch.tanh(self.fc3(a)) * self.max_action
        return a
