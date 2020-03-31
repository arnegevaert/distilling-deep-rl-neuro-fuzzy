from lib.DQN import DQN
from lib.anfis.weighted_tnorm_anfis import WeightedTnormAnfis
import pandas as pd
import pickle
import sys
import json
import gym
import time
import re
import os
import torch


# GET CONFIGURATION

config_loc = "json/q_learn/lunarlander.json"
if len(sys.argv) > 1:
    config_loc = sys.argv[1]

config_file = open(config_loc, "r")
config = json.load(config_file)
env = gym.make(config["env_name"])
out_dir = f"out/{re.split('[/.]', config_loc)[-2]}/q_learn/{int(time.time())}"  # filename without extension + timestamp
n_inputs = env.observation_space.shape[0]
n_outputs = env.action_space.n

# COPY CONFIGURATION TO OUT DIR
if config["output"]:
    os.makedirs(out_dir)
    json.dump(config, open(f"{out_dir}/params.json", "w"))

all_rewards = []
for run in range(config["runs"]):
    run_rewards = []
    print(f"RUN {run+1}")
    if config["output"]:
        os.mkdir(f"{out_dir}/run{run}")
    anfis = WeightedTnormAnfis(n_inputs, n_outputs, config["rules"])
    agent = DQN(n_outputs, config["mem_len"], config["gamma"], config["batch_size"], anfis)
    eps = 1.
    for episode in range(config["episodes"]):
        state = torch.tensor(env.reset()).float().view(1, -1)
        ep_reward = 0
        done = False
        while not done:
            action = agent.select_action(state, eps)
            n_state, reward, done, _ = env.step(action.item())
            ep_reward += reward

            n_state = torch.tensor(n_state).float().view(1, -1)
            agent.push_transition(state, action, torch.tensor([reward]).float(), n_state, done)
            agent.experience_replay()
            agent.update_target_net()
            state = n_state
            eps *= 0.99
            eps = max(eps, 0.01)
        run_rewards.append(ep_reward)
        print(f"{episode}: {ep_reward} (eps = {eps})")
    all_rewards.append(run_rewards)

if config["output"]:
    arrays = [
        (all_rewards, "rewards")
    ]
    data = {
        name: pd.DataFrame(d).transpose() for (d, name) in arrays
    }
    pickle.dump(data, open(f"{out_dir}/result.pkl", "wb"))