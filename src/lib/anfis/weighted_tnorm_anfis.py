import torch.nn as nn
import torch
from .anfis import Anfis
import numpy as np
import matplotlib.pyplot as plt
import math
from .util import gaussian_mf, plot_gaussian_mf


class WeightedTnormAnfis(Anfis):
    def __init__(self, n_inputs, n_outputs, n_rules, ignore_dims=None):
        super(WeightedTnormAnfis, self).__init__(n_inputs, n_outputs, n_rules, ignore_dims)
        self._tnorm_weights = nn.Parameter(torch.ones(1, self.n_rules, self.n_inputs), requires_grad=True)
        self._weight_mask = torch.ones(1, self.n_rules, self.n_inputs)
        self.masking = False

    def forward(self, x: torch.Tensor):
        if self.scaler:
            x = torch.tensor(self.scaler.transform(x)).float()
        if self.ignore_dims:
            x = x.detach().numpy()
            x = np.delete(x, self.ignore_dims, 1)
            x = torch.tensor(x)
        x = x.unsqueeze(1)
        self._clamp_weights()
        univariate_memb = gaussian_mf(x, self._mu, self._sigma)
        # Take weighted T-norm for every rule
        # Normalize per rule st maximum importance weight is 1
        normalized_tnorm_weights = self._tnorm_weights / self._tnorm_weights.max(dim=-1)[0].unsqueeze(-1)
        if self.masking:
            normalized_tnorm_weights *= self._weight_mask
        rule_activations = torch.prod(torch.pow(univariate_memb, normalized_tnorm_weights), dim=-1)
        # Divide each rule activation by total rule activation
        rule_activations = torch.div(rule_activations,
                                     torch.clamp(torch.sum(rule_activations, dim=1).unsqueeze(-1),
                                                 1e-12, 1e12))
        # Weighted sum is basically matmul
        return rule_activations.matmul(self._seq)

    def _clamp_weights(self):
        x = self._tnorm_weights.data
        x = x.clamp(1e-12, 1)
        self._tnorm_weights.data = x

    def mask_small_weights(self, thresh):
        self._weight_mask = self._tnorm_weights > thresh
        self.masking = True

    def set_tnorm_weights(self, weights):
        with torch.no_grad():
            self._tnorm_weights.data = torch.tensor(weights).unsqueeze(0).float()

    def get_tnorm_weights(self):
        return self._tnorm_weights.detach().cpu().view(self.n_rules, self.n_inputs).numpy()

    def copy(self):
        anfis: WeightedTnormAnfis = WeightedTnormAnfis(self.n_inputs, self.n_outputs, self.n_rules, self.ignore_dims)
        anfis.set_mu(self.get_mu())
        anfis.set_sigma(self.get_sigma())
        anfis.set_seq(self.get_seq())
        anfis.set_tnorm_weights(self.get_tnorm_weights())
        anfis.set_scaler(self.scaler)
        return anfis

    def print_rules(self, set_numbers):
        self._clamp_weights()
        for rule in range(self.n_rules):
            memberships = [f"x{dim} IS A{set_numbers[rule, dim]} WITH {self.get_tnorm_weights()[rule, dim]:.2f}"
                           for dim in range(self.n_inputs) if self._weight_mask[0, rule, dim] == 1]
            print("IF " + " AND ".join(memberships) + f" THEN y IS {self.get_seq()[rule]}")

    def serialize_dict(self):
        d = super(WeightedTnormAnfis, self).serialize_dict()
        d["state_dict"] = self.state_dict()
        d["weight_mask"] = self._weight_mask
        return d

    @staticmethod
    def load_dict(d):
        result: WeightedTnormAnfis = WeightedTnormAnfis(d["n_inputs"], d["n_outputs"], d["n_rules"])
        result.set_scaler(d["scaler"])
        result.memb_fn_mask = d["memb_fn_mask"]
        result.load_state_dict(d["state_dict"])
        result._weight_mask = d["weight_mask"]
        return result

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
                if not self.masking or self._weight_mask[0, rule, dim]:
                    # The first False entry in this row of the mask corresponds to the fuzzy set
                    first_false = next(i for i in range(self.n_rules) if not mask[rule, i])
                    # If any entry before the diagonal is False, this function has already been plotted
                    if first_false == rule:
                        plot_gaussian_mf(self.get_mu()[rule, dim], self.get_sigma()[rule, dim], legend, f"A{rule}")
                    set_numbers[rule, dim] = first_false
        plt.show()
        self.print_rules(set_numbers)
