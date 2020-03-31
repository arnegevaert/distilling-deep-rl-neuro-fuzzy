import torch
from collections import deque
from lib.anfis.anfis import Anfis
from lib.anfis.optim import AnfisOptimizer
import random


def distill(teacher, anfis: Anfis, anfis_optim: AnfisOptimizer, episodes, batch_size, memory_len, env):
    memory = deque(maxlen=memory_len)
    ep_rewards, ep_losses, ep_reg_losses = [], [], []
    total_steps = 0
    eps = 0.1
    for episode in range(episodes):
        state = env.reset().reshape(1, -1)
        state = torch.tensor(state).float()
        done = False
        ep_loss, ep_reg_loss, ep_reward, ep_len = 0, 0, 0, 0
        while not done:
            total_steps += 1
            ep_len += 1
            # Get teacher values and perform action
            # Teacher was trained using original states (not normalized)
            # ANFIS uses state vectors normalized on trajectories
            with torch.no_grad():
                y_true = teacher.get_q_values(state)
                if random.random() < eps:
                    action = random.randint(0, len(y_true))
                else:
                    action = torch.argmax(anfis(state)).item()
            memory.append((state, y_true))
            n_state, r, done, _ = env.step(action)
            ep_reward += r
            n_state = torch.tensor(n_state.reshape(1, -1)).float()
            state = n_state

            # Experience replay
            if len(memory) >= batch_size:
                batch = random.sample(memory, batch_size)
                states, targets = list(zip(*batch))
                states = torch.cat(states)
                targets = torch.cat(targets)
                l, rl = anfis_optim.optimize(states, targets)
                ep_loss += l
                ep_reg_loss += rl
        ep_rewards.append(ep_reward)
        ep_losses.append(ep_loss/ep_len)
        ep_reg_losses.append(ep_reg_loss/ep_len)
        eps *= 0.9
        print(f"{episode}: {ep_reward} (avg loss: {ep_loss/ep_len}   avg reg: {ep_reg_loss/ep_len}) eps: {eps}")
    return ep_rewards, ep_losses, ep_reg_losses
