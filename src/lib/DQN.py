import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
import random
import copy
from collections import deque
from itertools import count


class DQN:
    def __init__(self, n_outputs, mem_size, gamma, batch_size, network: nn.Module):
        self.n_outputs = n_outputs
        self.memory = deque(maxlen=mem_size)
        self.batch_size = batch_size
        self.gamma = gamma

        self.policy_net = network
        self.target_net = copy.deepcopy(network)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)

    def select_action(self, state, eps=0.0, get_q_values=False):
        sample = random.random()
        if sample > eps:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was found
                q_values = self.get_q_values(state)
                if get_q_values:
                    return q_values.max(1)[1].view(1, 1), q_values
                return q_values.max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.n_outputs)]], dtype=torch.long)

    def get_q_values(self, state):
        return self.policy_net(state)

    def experience_replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, n_states, dones = zip(*batch)
        state_batch = torch.cat(states)
        action_batch = torch.cat(actions)
        reward_batch = torch.cat(rewards)
        n_states = torch.cat(n_states)
        dones = torch.tensor(dones).int()

        q_values = self.policy_net(state_batch)
        n_q_values = self.target_net(n_states)

        targets = reward_batch + (1-dones)*self.gamma*n_q_values.max(1)[0].detach()
        expected_q_values = q_values.clone()
        expected_q_values[torch.arange(self.batch_size), action_batch.squeeze()] = targets

        loss = F.mse_loss(q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def push_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def load_state_dict(self, path):
        self.policy_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(torch.load(path))




