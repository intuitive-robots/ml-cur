import os

import numpy as np

from ml_cur.distribution.marginal import Categorical, Gaussian


class GMM:
    """
    Gaussian Mixture Model
    """

    def __init__(self, weights, means, covars):
        self._weight_distribution = Categorical(weights)
        self._components = [
            Gaussian(means[i], covars[i]) for i in range(means.shape[0])
        ]

    def sample(self, num_samples):
        w_samples = self._weight_distribution.sample(num_samples)
        samples = []
        for i in range(self.num_components):
            cns = np.count_nonzero(w_samples == i)
            if cns > 0:
                samples.append(self.components[i].sample(cns))
        return np.random.permutation(np.concatenate(samples, axis=0))

    def density(self, samples, sum_components=True):
        densities = np.stack(
            [self._components[i].density(samples) for i in range(self.num_components)],
            axis=0,
        )
        w = np.expand_dims(self.weight_distribution.probabilities(), axis=-1)

        densities[np.isinf(densities)] = np.max(
            densities[np.logical_not(np.isinf(densities))]
        )

        if sum_components:
            return np.sum(w * densities, axis=0)
        return w * densities

    def log_density(self, samples):
        return np.log(self.density(samples) + 1e-25)

    def log_likelihood(self, samples):
        return np.mean(self.log_density(samples))

    @property
    def components(self):
        return self._components

    @property
    def num_components(self):
        return len(self._components)

    @property
    def weight_distribution(self):
        return self._weight_distribution

    def add_component(self, initial_weight, initial_mean, initial_covar):
        self._weight_distribution.add_entry(initial_weight)
        self.components.append(Gaussian(initial_mean, initial_covar))

    def remove_component(self, idx):
        self._weight_distribution.remove_entry(idx)
        del self._components[idx]

    def save(self, fpath):
        means = np.stack([c.mean for c in self.components], axis=0)
        covars = np.stack([c.covar for c in self.components], axis=0)
        model_dict = {
            "weights": self.weight_distribution.probabilities(),
            "means": means,
            "covars": covars,
        }
        np.savez_compressed(fpath, **model_dict)

    @staticmethod
    def load(npz_path):
        model_dict = dict(np.load(npz_path, allow_pickle=True))
        return GMM(model_dict["weights"], model_dict["means"], model_dict["covars"])
