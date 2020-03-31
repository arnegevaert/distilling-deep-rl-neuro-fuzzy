from lib.DQN import DQN
from lib.anfis.weighted_tnorm_anfis import WeightedTnormAnfis
from lib.anfis.anfis import Anfis
from lib.anfis.optim import AnfisOptimizer, GradualMergeRegularizer, TNormWeightRegularizer
from lib.distill import initialize_from_trajectories, distill, eval_anfis, subtractive_clustering, gmm_clustering
from lib.util import get_temperature_kl_divergence_with_logits
import pandas as pd
import gym
import torch
import sys
import json
import re
import time
import pickle
import os

# GET CONFIGURATION

config_loc = "json/distill/lunarlander4.json"
if len(sys.argv) > 1:
    config_loc = sys.argv[1]

config_file = open(config_loc, "r")
config = json.load(config_file)
env = gym.make(config["env_name"])
out_dir = f"out/{re.split('[/.]', config_loc)[-2]}/distill/{int(time.time())}"  # filename without extension + timestamp
n_inputs = env.observation_space.shape[0]
n_outputs = env.action_space.n

ignore_dims = config.get("ignore_dims", None)
anfis_n_inputs = n_inputs if not ignore_dims else n_inputs - len(ignore_dims)

# COPY CONFIGURATION TO OUT DIR
if config["output"]:
    os.makedirs(out_dir)
    json.dump(config, open(f"{out_dir}/params.json", "w"))

# TEACHER SETUP

network = torch.nn.Sequential(
    torch.nn.Linear(n_inputs, 64),
    torch.nn.LayerNorm(64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, n_outputs)
)

# Mem size, gamma, batch size don't matter, teacher is already trained
teacher = DQN(n_outputs=n_outputs, mem_size=1000, gamma=0.99, batch_size=16, network=network)
teacher.load_state_dict(config["teacher_state_dict_loc"])

# STUDENT INIT THROUGH CLUSTERING

cluster_params = config["clustering"]
if cluster_params["type"] == "subtractive":
    cluster_algo = lambda traj: subtractive_clustering(traj, cluster_params["ra"], cluster_params["rb"],
                                                       cluster_params["eps_up"], cluster_params["eps_down"])
elif cluster_params["type"] == "gmm":
    cluster_algo = lambda traj: gmm_clustering(traj, cluster_params["n_components"])
else:
    raise Exception("Invalid cluster algorithm")

all_rewards = []
all_losses = []
all_reg_losses = []
all_rewards_after_postproc = []
for run in range(config["runs"]):
    print(f"RUN {run+1}")
    if config["output"]:
        os.mkdir(f"{out_dir}/run{run}")
    mu, sigma, seq, scaler = initialize_from_trajectories(teacher, env, config["init_episodes"], cluster_algo)
    loss_fn = get_temperature_kl_divergence_with_logits(config["kl_div_tau"])
    anfis_constructors = {"default": Anfis, "WeightedTNorm": WeightedTnormAnfis}
    anfis: Anfis = anfis_constructors[config["anfis_type"]](anfis_n_inputs, n_outputs, mu.shape[0], ignore_dims=ignore_dims)
    anfis.set_mu(mu)
    anfis.set_sigma(sigma)
    anfis.set_seq(seq)
    anfis.set_scaler(scaler)
    print(f"ANFIS created using {anfis.n_rules} rules")

    # ADD REGULARIZATION

    reg_constructors = {
        "Merge": GradualMergeRegularizer, "TNormWeight": TNormWeightRegularizer
    }
    regularizers = [reg_constructors[reg["type"]](reg["lambda"]) for reg in config["regularizers"]]
    optimizer = AnfisOptimizer(anfis, torch.optim.Adam(anfis.parameters(), lr=config["lr"]), loss_fn, regularizers)

    # DISTILLATION

    rewards, losses, reg_losses = distill(teacher, anfis, optimizer, config["distill_episodes"],
                                          config["distill_batch_size"], config["distill_mem_len"], env)

    # RECORD DISTILLATION RESULTS
    all_rewards.append(rewards)
    all_losses.append(losses)
    all_reg_losses.append(reg_losses)

    # SAVE MODEL BEFORE POSTPROCESSING
    if config["output"]:
        anf_before_postproc = anfis.serialize_dict()
        pickle.dump(anf_before_postproc, open(f"{out_dir}/run{run}/before_postproc.pkl", "wb"))

    # POSTPROCESSING

    if not config["output"]:
        anfis.describe()
    anfis_am = anfis.merge_antecedents(config["am_thresh"])
    if config["anfis_type"] == "WeightedTNorm":
        anfis_am.mask_small_weights(config["wm_thresh"])
    print()
    if not config["output"]:
        anfis_am.describe()
    rew_after_postproc = eval_anfis(anfis_am, env, episodes=25, render=False, ignore_dims=ignore_dims)
    all_rewards_after_postproc.append(rew_after_postproc)

    # SAVE MODEL AFTER POSTPROCESSING
    if config["output"]:
        anf_after_postproc = anfis.serialize_dict()
        pickle.dump(anf_after_postproc, open(f"{out_dir}/run{run}/after_postproc.pkl", "wb"))

# SAVE DISTILLATION RESULTS TO FILE
if config["output"]:
    arrays = [
        (all_rewards, "rewards"),
        (all_losses, "losses"),
        (all_reg_losses, "reg_losses"),
        (all_rewards_after_postproc, "rewards_after_postproc"),
    ]
    data = {
        name: pd.DataFrame(d).transpose() for (d, name) in arrays
    }
    pickle.dump(data, open(f"{out_dir}/result.pkl", "wb"))
