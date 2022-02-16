import torch
from RLModel.model.actor import Actor
from RLModel.model.critic import Critic
from RLModel.model.replayBuff import Replay_buffer
from tensorboardX import SummaryWriter
from RLModel.args import args_RL
import torch.optim as optim
import torch.nn.functional as F


'''
Implementation of TD3 with pytorch 
Original paper: https://arxiv.org/abs/1802.09477
Not the author's implementation !
'''

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class TD3:
    def __init__(self, state_dim, action_dim, max_action):

        self.actor = Actor(state_dim, action_dim, max_action).to(device)  # 动作 网络
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)  # target 动作网络
        self.critic_1 = Critic(state_dim, action_dim).to(device)  #
        self.critic_1_target = Critic(state_dim, action_dim).to(device)
        self.critic_2 = Critic(state_dim, action_dim).to(device)
        self.critic_2_target = Critic(state_dim, action_dim).to(device)

        self.actor_optimizer = optim.Adam(self.actor.parameters())
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters())
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters())

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())

        self.max_action = max_action
        self.memory = Replay_buffer(args_RL.capacity)
        self.writer = SummaryWriter(args_RL.directory)
        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0

    def select_action(self, state):
        # state = torch.tensor(state.reshape(1, -1)).float().to(device)
        state = state.reshape(1, -1).clone().detach().requires_grad_(True)
        return self.actor(state).cpu().data.numpy().flatten()

    def update(self, num_iteration):

        if self.num_training % 500 == 0:
            print("====================================")
            print("model has been trained for {} times...".format(self.num_training))
            print("====================================")
        for i in range(num_iteration):
            x, y, u, r, d = self.memory.sample(args_RL.batch_size)
            state = torch.FloatTensor(x).to(device)
            action = torch.FloatTensor(u).to(device)
            next_state = torch.FloatTensor(y).to(device)
            done = torch.FloatTensor(d).to(device)
            reward = torch.FloatTensor(r).to(device)

            # Select next action according to target policy:
            # noise = torch.ones_like(action).data.normal_(0, args_RL.policy_noise).to(device)
            # noise = noise.clamp(-args_RL.noise_clip, args_RL.noise_clip)
            next_action = (self.actor_target(next_state))
            # next_action = (self.actor_target(next_state) + noise)
            # next_action = next_action.clamp(-self.max_action, self.max_action)

            # Compute target Q-value:
            target_Q1 = self.critic_1_target(next_state, next_action)
            target_Q2 = self.critic_2_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + ((1 - done) * args_RL.gamma * target_Q).detach()

            # Optimize Critic 1:
            current_Q1 = self.critic_1(state, action)
            loss_Q1 = F.mse_loss(current_Q1, target_Q)
            self.critic_1_optimizer.zero_grad()
            loss_Q1.backward()
            self.critic_1_optimizer.step()
            self.writer.add_scalar('Loss/Q1_loss', loss_Q1, global_step=self.num_critic_update_iteration)

            # Optimize Critic 2:
            current_Q2 = self.critic_2(state, action)
            loss_Q2 = F.mse_loss(current_Q2, target_Q)
            self.critic_2_optimizer.zero_grad()
            loss_Q2.backward()
            self.critic_2_optimizer.step()
            self.writer.add_scalar('Loss/Q2_loss', loss_Q2, global_step=self.num_critic_update_iteration)
            # Delayed policy updates:
            if i % args_RL.policy_delay == 0:
                # Compute actor loss:
                actor_loss = - self.critic_1(state, self.actor(state)).mean()

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                self.writer.add_scalar('Loss/actor_loss', actor_loss, global_step=self.num_actor_update_iteration)
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(((1 - args_RL.tau) * target_param.data) + args_RL.tau * param.data)

                for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
                    target_param.data.copy_(((1 - args_RL.tau) * target_param.data) + args_RL.tau * param.data)

                for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
                    target_param.data.copy_(((1 - args_RL.tau) * target_param.data) + args_RL.tau * param.data)

                self.num_actor_update_iteration += 1
        self.num_critic_update_iteration += 1
        self.num_training += 1

    def save(self):
        torch.save(self.actor.state_dict(), args_RL.directory+'actor.pth')
        torch.save(self.actor_target.state_dict(), args_RL.directory+'actor_target.pth')
        torch.save(self.critic_1.state_dict(), args_RL.directory+'critic_1.pth')
        torch.save(self.critic_1_target.state_dict(), args_RL.directory+'critic_1_target.pth')
        torch.save(self.critic_2.state_dict(), args_RL.directory+'critic_2.pth')
        torch.save(self.critic_2_target.state_dict(), args_RL.directory+'critic_2_target.pth')
        print("====================================")
        print("Model has been saved...")
        print("====================================")

    def load(self):
        self.actor.load_state_dict(torch.load(args_RL.directory + 'actor.pth'))
        self.actor_target.load_state_dict(torch.load(args_RL.directory + 'actor_target.pth'))
        self.critic_1.load_state_dict(torch.load(args_RL.directory + 'critic_1.pth'))
        self.critic_1_target.load_state_dict(torch.load(args_RL.directory + 'critic_1_target.pth'))
        self.critic_2.load_state_dict(torch.load(args_RL.directory + 'critic_2.pth'))
        self.critic_2_target.load_state_dict(torch.load(args_RL.directory + 'critic_2_target.pth'))
        print("====================================")
        print("model has been loaded...")
        print("====================================")

