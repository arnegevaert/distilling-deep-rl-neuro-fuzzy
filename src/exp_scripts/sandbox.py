from lib.anfis.weighted_tnorm_anfis import WeightedTnormAnfis
import os
import pickle


env = "lunarlander"
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

data = pickle.load(open(f"{root_dir}/distill/{most_recent}/result.pkl", "rb"))
means = (data["rewards_after_postproc"] > 200).mean()


def show_agent(a_idx):
    d = pickle.load(open(f"{root_dir}/distill/{most_recent}/run{a_idx}/after_postproc.pkl", "rb"))
    anfis = WeightedTnormAnfis.load_dict(d)
    anfis.mask_small_weights(0.01)
    anfis.describe()

#d = pickle.load(open(f"{root_dir}/distill/{most_recent}/run{agent_idx}/after_postproc.pkl", "rb"))
#anfis = WeightedTnormAnfis.load_dict(d)
#anfis.mask_small_weights(0.01)

