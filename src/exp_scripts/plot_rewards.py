import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

for env, bl in [("Cartpole", 500), ("MountainCar", -147), ("LunarLander", 280), ("LunarLander4", 280)]:
    root_dir = f"out/{env.lower()}"
    fig, ax = plt.subplots(figsize=(20, 12))
    for exp, col in [("distill", "C0"), ("q_learn", "C1")]:
        out_contents = os.listdir(f"{root_dir}/{exp}")
        out_contents.sort()
        idx = -1
        most_recent = out_contents[idx]
        while not os.path.exists(f"{root_dir}/{exp}/{most_recent}/result.pkl"):
            idx -= 1
            most_recent = out_contents[idx]
        if idx != -1:
            print(f"Used old experiment data for {env}/{exp}: {most_recent}")
        data = pickle.load(open(f"{root_dir}/{exp}/{most_recent}/result.pkl", "rb"))

        rewards = data["rewards"]
        quantiles = rewards.quantile([.25, .5, .75], axis=1)
        x = list(range(rewards.shape[0]))

        ax.plot(x, quantiles.iloc[0, :], color=col, linewidth=0)
        ax.plot(x, quantiles.iloc[1, :], color=col, linewidth=5)
        ax.plot(x, quantiles.iloc[2, :], color=col, linewidth=0)
        ax.fill_between(x, quantiles.iloc[0, :], quantiles.iloc[2, :], alpha=0.2)
        ax.grid(True)
        ax.set_xlabel("episode", fontsize=40)
        ax.set_ylabel("reward", fontsize=40)
        ax.tick_params(axis="both", which="major", labelsize=40)
    ax.plot(x, np.ones_like(x) * bl, color="C2", linewidth=5)

    fig.savefig(f"../../tex/pdf/distill-{env}.pdf", bbox_inches="tight")
