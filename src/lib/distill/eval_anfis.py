import torch


def eval_anfis(anfis, env, episodes=5, render=True, verbose=True, ignore_dims=None):
    ep_rewards = []
    for e in range(episodes):
        state = torch.tensor(env.reset().reshape(1, -1)).float()
        done = False
        ep_reward = 0
        while not done:
            if render:
                env.render()
            with torch.no_grad():
                action = torch.argmax(anfis(state)).item()
            n_state, r, done, _ = env.step(action)
            ep_reward += r
            n_state = torch.tensor(n_state.reshape(1, -1)).float()
            state = n_state
        if verbose:
            print(f"{e}: {ep_reward}")
        ep_rewards.append(ep_reward)
    return ep_rewards
