from lib.DQN import DQN
import torch
import gym
import numpy as np

#env_name = "CartPole-v1"
#weights_loc = "../saved_models/cartpole-dqn.pth"
env_name = "LunarLander-v2"
weights_loc = "../saved_models/lunarlander-dqn.pth"
#env_name = "MountainCar-v0"
#weights_loc = "../saved_models/mountaincar-dqn.pth"

env = gym.make(env_name)
n_inputs = env.observation_space.shape[0]
n_outputs = env.action_space.n

network = torch.nn.Sequential(
    torch.nn.Linear(n_inputs, 64),
    torch.nn.LayerNorm(64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, n_outputs)
)

teacher = DQN(n_outputs=n_outputs, mem_size=1000, gamma=0.99, batch_size=16, network=network)
teacher.load_state_dict(weights_loc)

ep_rewards = []
for episode in range(50):
    state = torch.tensor(env.reset().reshape(1, -1)).float()
    done = False
    ep_reward = 0
    while not done:
        with torch.no_grad():
            action = torch.argmax(teacher.get_q_values(state)).item()
        n_state, r, done, _ = env.step(action)
        ep_reward += r
        n_state = torch.tensor(n_state.reshape(1, -1)).float()
        state = n_state
    print(f"{episode}: {ep_reward}")
    ep_rewards.append(ep_reward)
print(f"Average: {np.average(ep_rewards)}, median: {np.median(ep_rewards)}")
