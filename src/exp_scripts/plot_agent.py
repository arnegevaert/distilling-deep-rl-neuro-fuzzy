from lib.anfis.weighted_tnorm_anfis import WeightedTnormAnfis
import os
import pickle
import matplotlib.pyplot as plt
import math
import numpy as np


agents = [
    ("cartpole", 11, ["p_x", "v_x", "angle", "v_tip"], (25, 15)),
    ("mountaincar", 1, ["p_x", "v_x"], (25, 7.5)),
     # ("lunarlander", 31, ["p_x", "p_y", "v_x", "v_y", "angle", "v_a"], (25, 30)),
    ("lunarlander4", 2, ["p_x", "p_y", "v_x", "v_y", "angle", "v_a"], (25, 22.5))
]

for env, agent_idx, dim_names, figsize in agents:
    root_dir = f"out/{env}"
    out_contents = os.listdir(f"{root_dir}/distill")
    out_contents.sort()
    idx = -1
    most_recent = out_contents[idx]
    while not os.path.exists(f"{root_dir}/distill/{most_recent}/result.pkl"):
        idx -= 1
        most_recent = out_contents[idx]
    if idx != -1:
        print(f"Used old experiment data for {env}: {most_recent}")

    d = pickle.load(open(f"{root_dir}/distill/{most_recent}/run{agent_idx}/after_postproc.pkl", "rb"))
    anfis = WeightedTnormAnfis.load_dict(d)
    anfis.mask_small_weights(0.01)

    nrows = math.ceil(anfis.n_inputs / 2)
    ncols = 2
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
    axs = axs.flatten()
    set_numbers = np.empty((anfis.n_rules, anfis.n_inputs), dtype=int)
    for dim in range(anfis.n_inputs):
        ax = axs[dim]
        ax.set_title(dim_names[dim], fontsize=35)
        ax.grid(True)
        ax.tick_params(axis="both", which="major", labelsize=20)
        mask = anfis.memb_fn_mask[dim, :, :]
        # Gather relevant mus and sigmas
        for rule in range(anfis.n_rules):
            if anfis._weight_mask[0, rule, dim]:
                # The first False entry in this row of the mask corresponds to the fuzzy set
                first_false = next(i for i in range(anfis.n_rules) if not mask[rule, i])
                # If any entry before the diagonal is False, this function has already been plotted
                if first_false == rule:
                    mu = anfis.get_mu()[rule, dim]
                    sigma = anfis.get_sigma()[rule, dim]
                    sigma = max(0.1, sigma)
                    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
                    y = np.exp(-((x - mu) / sigma) ** 2)
                    ax.plot(x, y, label=f"A{dim}_{rule}")
                    cur_left_lim, cur_right_lim = ax.get_xlim()
                    left_lim = min(-1, mu - 3 * sigma, cur_left_lim)
                    right_lim = max(1, mu + 3 * sigma, cur_right_lim)
                    ax.set_xlim(left_lim, right_lim)
                    ax.legend(fontsize=20)
                set_numbers[rule, dim] = first_false
    anfis.print_rules(set_numbers)
    print()
    fig.savefig(f"../../tex/pdf/agent-{env}.pdf", bbox_inches="tight")