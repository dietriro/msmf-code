import numpy as np
import scipy as sc

from scipy.stats import gamma, uniform
from scipy.stats import entropy

from msmfcode.models.cann import ContinuousAttractorNetwork


def kl(p, q):
    """Kullback-Leibler divergence D(P || Q) for discrete distributions
    Parameters
    ----------
    p, q : array-like, dtype=float, shape=n
    Discrete probability distributions.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    return np.sum(np.where(p != 0, p * np.log(p / q), 0))


def calc_kls_sizes(net: ContinuousAttractorNetwork, alpha, beta, lower_ppf=0.01, upper_ppf=0.99, num_bins=100):
    g = gamma(a=alpha, scale=beta, loc=0)
    lower_bound = g.ppf(lower_ppf)
    upper_bound = g.ppf(upper_ppf)
    bins = np.linspace(lower_bound, upper_bound, num_bins)
    bins = np.append(bins, [100])

    p_cumul = g.cdf(bins)
    p_cumul_shift = np.concatenate(([0], p_cumul[:-1]))
    bin_prob_gamma = p_cumul - p_cumul_shift

    kls = []

    for i_neuron in range(net.p.num_neurons):
        fields = net.field_sizes[i_neuron].flatten()
        fields = fields[~(fields == 0)]

        bin_nums = np.digitize(fields, bins=bins)
        bin_counts = np.unique(bin_nums, return_counts=True)

        # Calculate value count for each bin
        bin_probs = np.zeros(len(bins))
        bin_probs[bin_counts[0]] = bin_counts[1]
        bin_probs /= np.sum(bin_probs)

        kl_scp = sc.special.kl_div(bin_prob_gamma, bin_probs)
        kl_scp = np.sum(kl_scp[kl_scp != np.inf])

        # print(kl(bin_probs, bin_prob_gamma), np.sum(kl_scp))

        # kls.append(kl(bin_probs, bin_prob_gamma))
        kls.append(kl_scp)

    return kls


def calc_kls_locs(net: ContinuousAttractorNetwork, alpha, beta, lower_ppf=0.01, upper_ppf=0.99, num_bins=100):
    u = uniform(scale=beta)
    lower_bound = u.ppf(lower_ppf)
    upper_bound = u.ppf(upper_ppf)
    bins = np.linspace(lower_bound, upper_bound, num_bins)
    # bins = np.append(bins, [200])

    p_cumul = u.cdf(bins)
    p_cumul_shift = np.concatenate(([0], p_cumul[:-1]))
    bin_prob_gamma = p_cumul - p_cumul_shift

    kls = []

    for i_neuron in range(net.p.num_neurons):
        fields = net.field_locs[i_neuron].flatten()
        fields = fields[~(fields == 0)]

        bin_nums = np.digitize(fields, bins=bins)
        bin_counts = np.unique(bin_nums, return_counts=True)

        # Calculate value count for each bin
        bin_probs = np.zeros(len(bins))
        bin_probs[bin_counts[0]] = bin_counts[1]
        bin_probs /= np.sum(bin_probs)

        kl_scp = sc.special.kl_div(bin_prob_gamma, bin_probs)
        kl_scp = np.sum(kl_scp[kl_scp != np.inf])

        # print(kl(bin_probs, bin_prob_gamma), np.sum(kl_scp))

        # kls.append(kl(bin_probs, bin_prob_gamma))
        kls.append(kl_scp)

    return kls
