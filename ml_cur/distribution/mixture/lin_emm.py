import os

import numpy as np

from ml_cur.distribution.lin_conditional import LinCondGaussian, QuadSoftmax, Softmax


class LinEMM:
    """
    Linear Expert Mixture Model
    """

    def __init__(self, gating_params, component_params, covars, gating_type="affine"):
        self._context_dim = component_params.shape[1] - 1
        self._sample_dim = component_params.shape[2]
        self._gating_type = str(gating_type)

        if self._gating_type.lower() == "affine":
            self._gating_distribution = Softmax(gating_params)
        elif self._gating_type.lower() == "quad":
            self._gating_distribution = QuadSoftmax(gating_params)
        else:
            raise AssertionError("Invalid Gating Type")
        self._components = []

        for i in range(component_params.shape[0]):
            self._components.append(LinCondGaussian(component_params[i], covars[i]))

    def sample(self, contexts):
        gating_samples = self._gating_distribution.sample(contexts)
        samples = np.zeros([contexts.shape[0], self._sample_dim])
        for i, c in enumerate(self._components):
            idx = gating_samples == i
            if len(idx) > 0:
                samples[idx] = c.sample(contexts[idx])
        return samples

    def density(self, contexts, samples, sum_components=True):
        densities = [
            self._components[i].density(contexts, samples)
            for i in range(self.num_components)
        ]
        densities = np.stack(densities, axis=1)
        if sum_components:
            return np.sum(
                self.gating_distribution.probabilities(contexts) * densities, axis=1
            )
        return self._gating_distribution.probabilities(contexts) * densities

    def log_density(self, contexts, samples):
        return np.log(self.density(contexts, samples) + 1e-25)

    def log_likelihood(self, contexts, samples):
        return np.mean(self.log_density(contexts, samples))

    @property
    def components(self):
        return self._components

    @property
    def num_components(self):
        return len(self._components)

    @property
    def gating_distribution(self):
        return self._gating_distribution

    def add_component(self, contexts, initial_weights, initial_params, initial_covar):
        self._gating_distribution.add_entry(contexts, initial_weights)
        self._components.append(LinCondGaussian(initial_params, initial_covar))

    def remove_component(self, contexts, idx):
        self._gating_distribution.remove_entry(contexts, idx)
        del self._components[idx]

    @property
    def parameters(self):
        gating_params = self._gating_distribution.params
        mean_params = np.stack([c.params for c in self._components], axis=0)
        covars = np.stack([c.covar for c in self._components], axis=0)
        return gating_params, mean_params, covars, self._gating_type

    def save(self, fpath: str):
        # components
        means_comps = np.stack([c.params for c in self.components], axis=0)
        covars_comps = np.stack([c.covar for c in self.components], axis=0)

        # softmax
        gating_params = self.gating_distribution.params

        model_dict = {
            "gating_params": gating_params,
            "means_comps": means_comps,
            "covars_comps": covars_comps,
            "gating_type": self._gating_type,
        }
        # file_name = 'model'
        np.savez_compressed(fpath, **model_dict)

    @staticmethod
    def load(model_path):
        # load LinEMM
        model_dict = dict(np.load(model_path))
        model = LinEMM(
            model_dict["gating_params"],
            model_dict["means_comps"],
            model_dict["covars_comps"],
            gating_type=model_dict["gating_type"],
        )
        return model
