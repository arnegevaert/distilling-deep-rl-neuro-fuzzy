import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
from scipy.special import softmax
import math


def pairwise_squared_distance(x, y):
    n, d = x.shape
    x = x.unsqueeze(1).expand(n, n, d)
    y = y.unsqueeze(0).expand(n, n, d)
    return torch.pow(x - y, 2).sum(2).clamp(min=1e-5)


# TODO for loop can probably be eliminated
def mu_sigma_pairwise_squared_distance(anfis):
    mu = anfis._mu[0]  # (n_rules, n_inputs)
    sigma = anfis._sigma[0]
    result_mu = []
    result_sigma = []
    for dim in range(mu.shape[1]):
        mu_dim = mu[:, dim].view(-1, 1)
        sigma_dim = sigma[:, dim].view(-1, 1)
        result_mu.append(pairwise_squared_distance(mu_dim, mu_dim))
        result_sigma.append(pairwise_squared_distance(sigma_dim, sigma_dim))
    return torch.stack(result_mu, 2), torch.stack(result_sigma, 2)


def gaussian_mf(x: torch.tensor, mu: torch.tensor, sigma: torch.tensor):
    # Univariate Gaussian membership functions for every input
    x = (x - mu)  # Distance of center
    x = torch.div(x, sigma.abs().clamp(min=1e-12))  # Divide by variance
    x = x ** 2  # Square to make positive
    return torch.clamp(torch.exp(-x), min=1e-12)  # Exponential of negative value: [0, 1]


def plot_gaussian_mf(mu, sigma, legend, label):
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
    y = np.exp(-((x - mu) / sigma) ** 2)
    plt.plot(x, y, label=label)
    cur_left_lim, cur_right_lim = plt.xlim()
    left_lim = min(-1, mu - 3*sigma, cur_left_lim)
    right_lim = max(1, mu + 3*sigma, cur_right_lim)
    plt.xlim(left_lim, right_lim)
    plt.grid(True)
    if legend:
        plt.legend()


def sim(mu1, sigma1, mu2, sigma2):
    def _product_cardinality(_mu1, _sigma1, _mu2, _sigma2):
        # For stability in division
        _sigma1 = max(_sigma1, 1e-2)
        _sigma2 = max(_sigma2, 1e-2)
        # infinite integral of product of 2 Gaussians
        a = 1/_sigma1**2 + 1/_sigma2**2
        b = 2*(_mu1/_sigma1**2 + _mu2/_sigma2**2)
        c = - (_mu1/_sigma1)**2 - (_mu2/_sigma2)**2
        return math.sqrt(math.pi/a) * math.exp((b**2)/(4*a) + c)
    # Jaccard index based on product + prob. sum is not reflexive
    # This version is reflexive (WMAI p151)
    card_prod = _product_cardinality(mu1, sigma1, mu2, sigma2)
    card_a = _product_cardinality(mu1, sigma1, mu1, sigma1)
    card_b = _product_cardinality(mu2, sigma2, mu2, sigma2)
    return card_prod / (card_a + card_b - card_prod)
