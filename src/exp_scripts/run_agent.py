from lib.anfis.weighted_tnorm_anfis import WeightedTnormAnfis
import torch
import gym
import numpy as np
import os
import pickle


env = gym.make("LunarLander-v2")
n_inputs = env.observation_space.shape[0]
n_outputs = env.action_space.n

root_dir = "out/lunarlander"
out_contents = os.listdir(f"{root_dir}/distill")
out_contents.sort()
idx = -1
most_recent = out_contents[idx]
d = pickle.load(open(f"{root_dir}/distill/{most_recent}/run31/after_postproc.pkl", "rb"))
anfis = WeightedTnormAnfis.load_dict(d)
anfis.mask_small_weights(0.01)


ep_rewards = []
for episode in range(50):
    state = torch.tensor(env.reset().reshape(1, -1)).float()
    done = False
    ep_reward = 0
    while not done:
        with torch.no_grad():
            action = torch.argmax(anfis(state)).item()
        n_state, r, done, _ = env.step(action)
        ep_reward += r
        n_state = torch.tensor(n_state.reshape(1, -1)).float()
        state = n_state
    print(f"{episode}: {ep_reward}")
    ep_rewards.append(ep_reward)
print(f"Average: {np.average(ep_rewards)}, median: {np.median(ep_rewards)}")
