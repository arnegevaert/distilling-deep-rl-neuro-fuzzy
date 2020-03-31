import torch
import numpy as np
from sklearn import preprocessing
from itertools import count


def initialize_from_trajectories(teacher, env, episodes, cluster_algo):
    scaler = preprocessing.MinMaxScaler()
    states, outputs = gather_trajectories(teacher, env, episodes)
    n_inputs = states.shape[1]
    trajectories = np.hstack([states, outputs])
    trajectories_norm = scaler.fit_transform(trajectories)

    centers, sigma = cluster_algo(trajectories_norm)
    sigma = np.sqrt(sigma)

    #sigma = scaler.inverse_transform(np.sqrt(sigma))
    #centers = scaler.inverse_transform(centers)

    state_scaler = preprocessing.MinMaxScaler()
    state_scaler.fit(states)
    return centers[:, :n_inputs], sigma[:, :n_inputs], centers[:, n_inputs:], state_scaler


def gather_trajectories(agent, env, episodes, verbose=False):
    states = []
    outputs = []
    for i_episode in range(episodes):
        # Initialize the environment and state
        state = torch.tensor(env.reset()).float().view(1, -1)
        ep_reward = 0
        for t in count():
            # Select and perform an action
            action, q_values = agent.select_action(state, eps=0.0, get_q_values=True)
            states.append(state.numpy())
            outputs.append(q_values.numpy())

            next_state, reward, done, _ = env.step(action.item())
            state = torch.tensor(next_state).float().view(1, -1)
            ep_reward += reward if not done else -reward

            if done:
                if verbose:
                    print(f"{i_episode}: {ep_reward}")
                break
    return np.vstack(states), np.vstack(outputs)


