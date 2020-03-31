import torch
from .anfis import Anfis
from .weighted_tnorm_anfis import WeightedTnormAnfis
from .util import mu_sigma_pairwise_squared_distance


class AnfisRegularizer:
    def __init__(self, reg_factor):
        self.reg_factor = reg_factor

    def __call__(self, anfis):
        raise NotImplementedError()


class AnfisOptimizer:
    def __init__(self, anfis: Anfis, optimizer, loss_fn, regularizers=None):
        self.anfis = anfis
        self.optimizer = optimizer
        self.regularizers = regularizers if regularizers else []
        self.loss_fn = loss_fn

    def add_regularizer(self, regularizer: AnfisRegularizer):
        self.regularizers.append(regularizer)

    def reset_regularizers(self):
        self.regularizers = []

    def optimize(self, states, targets):
        prediction = self.anfis(states)

        loss = self.loss_fn(targets, prediction).sum()
        reg_loss = 0
        for regularizer in self.regularizers:
            reg_loss += regularizer.reg_factor * regularizer(self.anfis)
        loss += reg_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if len(self.regularizers) > 0:
            reg_loss = reg_loss.detach().numpy()
        return loss.detach().numpy(), reg_loss


# Notes: reg 2i
class GradualMergeRegularizer(AnfisRegularizer):
    def __call__(self, anfis: Anfis):
        mu_pw_dists, sigma_pw_dists = mu_sigma_pairwise_squared_distance(anfis)
        mu_pw_dists = torch.min(mu_pw_dists, torch.ones_like(mu_pw_dists))
        d, n = sigma_pw_dists.shape[2], sigma_pw_dists.shape[0]
        loss = torch.sum(torch.sqrt(sigma_pw_dists) / (1 + torch.sqrt(mu_pw_dists))) / 2
        return loss / (n * d)


# L1 norm on T-norm importance weights
class TNormWeightRegularizer(AnfisRegularizer):
    def __call__(self, anfis: WeightedTnormAnfis):
        normalized_tnorm_weights = anfis._tnorm_weights / anfis._tnorm_weights.max(dim=-1)[0].unsqueeze(-1)
        return torch.sum(normalized_tnorm_weights)
