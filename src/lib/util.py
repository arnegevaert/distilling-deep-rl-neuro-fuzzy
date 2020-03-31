import torch


# As described in Rusu et al. 2015: Policy Distillation
def get_temperature_kl_divergence_with_logits(tau=0.01):
    def l_f(true_logits, pred_logits):
        y_true = torch.clamp(torch.softmax(true_logits/tau, dim=-1), 1e-12, 1)
        y_pred = torch.clamp(torch.softmax(pred_logits, dim=-1), 1e-12, 1)
        return torch.sum(y_true * torch.log(y_true / y_pred), dim=-1)
    return l_f


def eval_gym(agent, env, episodes, device, render=False):
    ep_rewards = []
    for episode in range(episodes):
        state = torch.tensor(env.reset(), device=device).float().view(1, -1)
        done = False
        ep_reward = 0
        while not done:
            if render:
                env.render()
            with torch.no_grad():
                action = torch.argmax(agent(state)).item()
            n_state, r, done, _ = env.step(action)
            ep_reward += r if not done else -r
            n_state = torch.tensor(n_state, device=device).float().view(1, -1)

            state = n_state
        ep_rewards.append(ep_reward)
    return ep_rewards

