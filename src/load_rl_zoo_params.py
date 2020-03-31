import pickle
import torch
import torch.nn as nn
import gym

# Load parameters
params = pickle.load(open("saved_models/params.pkl", "rb"))
fc0_weights = params["deepq/model/action_value/fully_connected/weights:0"]
fc0_biases = params["deepq/model/action_value/fully_connected/biases:0"]
layernorm_beta = params["deepq/model/action_value/LayerNorm/beta:0"]
layernorm_gamma = params["deepq/model/action_value/LayerNorm/gamma:0"]
fc1_weights = params["deepq/model/action_value/fully_connected_1/weights:0"]
fc1_biases = params["deepq/model/action_value/fully_connected_1/biases:0"]

# Build model
model = torch.nn.Sequential(
    nn.Linear(4, 64),
    nn.LayerNorm(64),
    nn.ReLU(),
    nn.Linear(64, 2)
)

# Set model parameters
with torch.no_grad():
    # FC 0
    model[0].weight = nn.Parameter(torch.tensor(fc0_weights.transpose()))
    model[0].bias = nn.Parameter(torch.tensor(fc0_biases))
    # LN
    model[1].weight = nn.Parameter(torch.tensor(layernorm_gamma))
    model[1].bias = nn.Parameter(torch.tensor(layernorm_beta))
    # FC 1
    model[3].weight = nn.Parameter(torch.tensor(fc1_weights.transpose()))
    model[3].bias = nn.Parameter(torch.tensor(fc1_biases))

# Test model
env = gym.make("CartPole-v1")
for e in range(1):
    state = torch.tensor(env.reset()).float().view(1, -1)
    done = False
    ep_reward = 0
    while not done:
        env.render()
        with torch.no_grad():
            action = torch.argmax(model(state)).item()
        n_state, r, done, _ = env.step(action)
        ep_reward += r
        n_state = torch.tensor(n_state).float().view(1, -1)
        state = n_state
    print(f"{e}: {ep_reward}")
torch.save(model.state_dict(), 'saved_models/cartpole-dqn.pth')
