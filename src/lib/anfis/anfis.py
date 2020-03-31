import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
import math
from .util import gaussian_mf, plot_gaussian_mf, sim
import warnings


class Anfis(nn.Module):
    def __init__(self, n_inputs, n_outputs, n_rules, ignore_dims=None):
        super(Anfis, self).__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_rules = n_rules

        self.memb_fn_mask = np.ones((n_inputs, n_rules, n_rules), dtype=bool)
        for dim in range(self.memb_fn_mask.shape[0]):
            np.fill_diagonal(self.memb_fn_mask[dim, :, :], False)

        self._mu = nn.Parameter(torch.rand(1, self.n_rules, self.n_inputs), requires_grad=True)
        self._sigma = nn.Parameter(torch.rand(1, self.n_rules, self.n_inputs), requires_grad=True)
        self._seq = nn.Parameter(torch.rand(self.n_rules, self.n_outputs), requires_grad=True)
        self.scaler = None
        self.ignore_dims = ignore_dims

    def forward(self, x: torch.Tensor):
        if self.scaler:
            x = torch.tensor(self.scaler.transform(x)).float()
        # We expect shape to be (batch_size, n_inputs)
        # Need shape to be (batch_size, 1, n_inputs) for broadcasting later
        x = x.unsqueeze(1)
        univariate_memb = gaussian_mf(x, self._mu, self._sigma)
        # Take T-norm for every rule
        rule_activations = torch.min(univariate_memb, dim=-1)[0]
        # Divide each rule activation by total membership
        rule_activations = torch.div(rule_activations,
                                     torch.clamp(torch.sum(rule_activations, dim=1).unsqueeze(-1),
                                                 1e-12, 1e12))
        # Weighted sum comes down to matmul
        return rule_activations.matmul(self._seq)

    def describe(self, legend=True):
        nrows = math.ceil(self.n_inputs / 2)
        ncols = 2
        plt.figure()
        set_numbers = np.empty((self.n_rules, self.n_inputs), dtype=int)
        for dim in range(self.n_inputs):
            plt.subplot(nrows, ncols, dim + 1)
            mask = self.memb_fn_mask[dim, :, :]
            # Gather relevant mus and sigmas
            for rule in range(self.n_rules):
                # The first False entry in this row of the mask corresponds to the fuzzy set
                first_false = next(i for i in range(self.n_rules) if not mask[rule, i])
                # If any entry before the diagonal is False, this function has already been plotted
                if first_false == rule:
                    plot_gaussian_mf(self.get_mu()[rule, dim], self.get_sigma()[rule, dim], legend, f"A{rule}")
                set_numbers[rule, dim] = first_false
        plt.show()
        self.print_rules(set_numbers)

    def print_rules(self, set_numbers):
        for rule in range(self.n_rules):
            fstring = "IF " + " AND ".join([f"x{dim} IS A{set_numbers[rule, dim]}" for dim in range(self.n_inputs)]) \
                      + f" THEN y IS {self.get_seq()[rule]}"
            print(fstring)

    def set_mu(self, mu):
        if self.ignore_dims:
            mu = np.delete(mu, self.ignore_dims, 1)
        with torch.no_grad():
            self._mu.data = torch.tensor(mu).unsqueeze(0).float()

    def set_sigma(self, sigma):
        if self.ignore_dims:
            sigma = np.delete(sigma, self.ignore_dims, 1)
        with torch.no_grad():
            self._sigma.data = torch.tensor(sigma).unsqueeze(0).float()

    def set_seq(self, seq):
        with torch.no_grad():
            self._seq.data = torch.tensor(seq).float()

    def get_mu(self):
        return self._mu.detach().cpu().view(self.n_rules, self.n_inputs).numpy()

    def get_sigma(self):
        return np.abs(self._sigma.detach().cpu().view(self.n_rules, self.n_inputs).numpy())

    def get_seq(self):
        return self._seq.detach().cpu().numpy()

    def set_scaler(self, scaler):
        self.scaler = scaler

    def copy(self):
        anfis: Anfis = Anfis(self.n_inputs, self.n_outputs, self.n_rules, self.ignore_dims)
        anfis.set_mu(self.get_mu())
        anfis.set_sigma(self.get_sigma())
        anfis.set_seq(self.get_seq())
        anfis.set_scaler(self.scaler)
        return anfis

    def merge_antecedents(self, thresh):
        mu = self.get_mu()  # (n_rules, n_inputs)
        sigma = self.get_sigma()  # (n_rules, n_inputs)

        # Initialize similarity matrix for each input dimension
        similarities = []
        for i in range(self.n_inputs):
            # Initialize similarity matrix
            dim_similarities = np.zeros((mu.shape[0], mu.shape[0]))
            for j in range(mu.shape[0]):
                for k in range(j+1, mu.shape[0]):
                    dim_similarities[j, k] = sim(mu[j, i], sigma[j, i], mu[k, i], sigma[k, i])
            similarities.append(dim_similarities)
        similarities = np.array(similarities)  # (num_inputs, num_rules, num_rules)
        mask = np.ones_like(similarities, dtype=bool)  # False entries are merges that already happened or diagonal
        for dim in range(mask.shape[0]):
            np.fill_diagonal(mask[dim, :, :], False)
        merges = np.ones_like(mu, dtype=int)

        max_sim = np.amax(similarities)
        all_merged = False
        while max_sim >= thresh and not all_merged:
            # Calculate merged parameters according to (Dourado, 2004)
            idx = np.where((similarities == max_sim) & mask)
            dim, rule1, rule2 = idx[0][0], idx[1][0], idx[2][0]
            print(f"Merging {dim} {rule1} {rule2} with similarity {max_sim}")
            new_merges = merges[rule1, dim] + merges[rule2, dim]
            new_mu = (merges[rule1, dim] * mu[rule1, dim] + merges[rule2, dim] * mu[rule2, dim]) / new_merges
            new_sigma = (merges[rule1, dim] * sigma[rule1, dim] + merges[rule2, dim] * sigma[rule2, dim]) / new_merges

            # Replace values
            mask_1 = mask[dim, rule1, :]
            mask_2 = mask[dim, rule2, :]
            to_replace = np.logical_not(np.logical_and(mask_1, mask_2))  # If any entry is False, must be replaced
            mu[to_replace, dim] = new_mu
            sigma[to_replace, dim] = new_sigma
            merges[to_replace, dim] = new_merges

            # Update mask
            mask[dim, rule1, to_replace] = False
            mask[dim, rule2, to_replace] = False
            mask[dim, to_replace, rule1] = False
            mask[dim, to_replace, rule2] = False

            # Recalculate similarities
            for i in range(mu.shape[0]):
                value = sim(new_mu, new_sigma, mu[i, dim], sigma[i, dim])
                similarities[dim, (rule1, rule2), i] = value
                similarities[dim, i, (rule1, rule2)] = value
            if np.any(similarities[mask]):
                max_sim = np.amax(similarities[mask])
            else:
                all_merged = True

        result: Anfis = self.copy()
        result.set_mu(mu)
        result.set_sigma(sigma)
        result.memb_fn_mask = mask
        return result

    def serialize_dict(self):
        return {
            "state_dict": self.state_dict(),
            "scaler": self.scaler,
            "memb_fn_mask": self.memb_fn_mask,
            "n_inputs": self.n_inputs,
            "n_outputs": self.n_outputs,
            "n_rules": self.n_rules
        }

    @staticmethod
    def load_dict(d):
        result: Anfis = Anfis(d["n_inputs"], d["n_outputs"], d["n_rules"])
        result.set_scaler(d["scaler"])
        result.memb_fn_mask = d["memb_fn_mask"]
        result.load_state_dict(d["state_dict"])
        return result
