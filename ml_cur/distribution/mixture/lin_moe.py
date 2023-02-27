import os

import numpy as np

from ml_cur.distribution.lin_conditional import LinCondGaussian
from ml_cur.distribution.marginal import Categorical, Gaussian
from ml_cur.util.functions import log_sum_exp


class LinMOE:
    """
    Linear Mixture of Experts Model
    p(a|s) = sum_o p(a|s,o) p(s|o) p(o)

    p(a|s,o) = Expert Component = Lin. Cond. Gauss
    p(s|o) =  Context Component = Gauss
    p(o) = Mixture Weights = Categorical
    """

    def __init__(
        self,
        component_params,
        component_covars,
        context_component_means,
        context_component_covars,
    ):
        self._context_dim = component_params.shape[1] - 1
        self._sample_dim = component_params.shape[2]
        self._weight_distr = Categorical(
            np.ones(component_params.shape[0]) / component_params.shape[0]
        )

        self._components = []
        self._context_components = []

        for i in range(component_params.shape[0]):
            self._components.append(
                LinCondGaussian(component_params[i], component_covars[i])
            )
            self._context_components.append(
                Gaussian(context_component_means[i], context_component_covars[i])
            )

    def sample(self, contexts):
        gating_probs = self.gating_probs(contexts)

        thresh = np.cumsum(gating_probs, axis=1)
        thresh[:, -1] = np.ones(contexts.shape[0])
        eps = np.random.uniform(size=[contexts.shape[0], 1])
        comp_idx_samples = np.argmax(eps < thresh, axis=-1)

        samples = np.zeros((contexts.shape[0], self._sample_dim))
        for i in range(self.num_components):
            context_samples_component_i_idx = np.where(comp_idx_samples == i)[0]
            context_samples_component_i = contexts[context_samples_component_i_idx, :]
            if context_samples_component_i.shape[0] != 0:
                samples[context_samples_component_i_idx, :] = self._components[
                    i
                ].sample(context_samples_component_i)
        return samples

    def density(self, contexts, samples):
        """
        p(a|s) = sum_o p(a|s,o) p(s|o) p(o)
        """
        return np.exp(self.log_density(contexts, samples))

    def log_density(self, contexts, samples):
        """
        log p(a|s) = log sum_o p(a|s,o) p(s|o) p(o)
        """
        log_component_densities = self.log_component_densities(contexts, samples)
        log_gating_probs = self.log_gating_probs(contexts)
        exp_arg = log_component_densities + log_gating_probs
        log_density = log_sum_exp(exp_arg, axis=1)
        return log_density

    def log_likelihood(self, contexts, samples):
        return np.mean(self.log_density(contexts, samples))

    def component_densities(self, contexts, samples):
        """
        p(a|s,o) for all o
        """
        return np.exp(self.log_component_densities(contexts, samples))

    def log_component_densities(self, contexts, samples):
        """
        log p(a|s,o) for all o
        """
        n_comps = self.num_components
        log_probs = np.zeros((contexts.shape[0], n_comps))
        for i in range(n_comps):
            log_probs[:, i] = self._components[i].log_density(contexts, samples)
        return log_probs

    def component_context_densities(self, contexts):
        """
        p(s|o) for all o
        """
        return np.exp(self.log_component_context_densities(contexts))

    def log_component_context_densities(self, contexts):
        """
        log p(s|o) for all o
        """
        n_comps = self.num_components
        log_probs = np.zeros((contexts.shape[0], n_comps))
        for i in range(n_comps):
            log_probs[:, i] = self._context_components[i].log_density(contexts)
        return log_probs

    def gating_probs(self, ctxts):
        """
        p(o|s)
        """
        return np.exp(self.log_gating_probs(ctxts))

    def log_gating_probs(self, ctxts):
        """
        log p(o|s)
        """
        log_weights = self._weight_distr.log_probabilities()
        (
            log_cmp_ctxt_densities,
            log_marg_ctxt_densities,
        ) = self.log_component_marg_context_densities(ctxts)
        log_gating_probs = (
            log_cmp_ctxt_densities
            + log_weights[None, :]
            - log_marg_ctxt_densities[:, None]
        )
        return log_gating_probs

    def component_marg_context_densities(self, contexts):
        """
        log p(s|o) and log p(s)
        returning both to avoid redundant calculations
        """
        (
            log_component_context_densities,
            log_marg_context_densities,
        ) = self.log_component_marg_context_densities(contexts)
        return (
            np.exp(log_component_context_densities),
            np.exp(log_marg_context_densities),
        )

    def log_component_marg_context_densities(self, contexts):
        """
        log pi(s|o) and log pi(s)
        returning both to avoid redundant calculations
        """
        log_weights = self._weight_distr.log_probabilities()
        log_component_context_densities = self.log_component_context_densities(contexts)

        exp_arg = log_component_context_densities + log_weights[None, :]
        log_marg_context_densities = log_sum_exp(exp_arg, axis=1)
        return log_component_context_densities, log_marg_context_densities

    def log_responsibilities(self, contexts, samples):
        """
        p(o|a,s), p(o|s)
        return both because pi(o|s) is automatically calculated
        """

        log_weights = self._weight_distr.log_probabilities()
        (
            log_component_context_densities,
            log_marg_context_densities,
        ) = self.log_component_marg_context_densities(contexts)
        log_gating_probs = (
            log_component_context_densities
            + log_weights[None, :]
            - log_marg_context_densities[:, None]
        )
        log_component_densities = self.log_component_densities(contexts, samples)
        log_model_density = log_sum_exp(
            log_component_densities + log_gating_probs, axis=1
        )
        log_resps = (
            log_component_densities + log_gating_probs - log_model_density[:, None]
        )
        return log_resps, log_gating_probs

    @property
    def components(self):
        return self._components

    @property
    def context_components(self):
        return self._context_components

    @property
    def weight_distribution(self):
        return self._weight_distr

    @property
    def num_components(self):
        return len(self._components)

    @property
    def context_dim(self):
        return self._context_dim

    def add_component(
        self,
        component_params,
        component_covar,
        context_component_mean,
        context_component_covar,
        init_weight,
    ):
        self._components.append(LinCondGaussian(component_params, component_covar))
        self._context_components.append(
            Gaussian(context_component_mean, context_component_covar)
        )
        self._weight_distr.add_entry(init_weight)

    def remove_component(self, idx):
        del self._components[idx]
        del self._context_components[idx]
        self._weight_distr.remove_entry(idx)

    def update_parameters(
        self,
        component_params,
        component_covars,
        context_component_means,
        context_component_covars,
    ):
        for i in range(len(self._components)):
            self._components[i].update_parameters(
                component_params[i], component_covars[i]
            )
            self._context_components[i].update_parameters(
                context_component_means[i], context_component_covars[i]
            )

    def save(self, path: str, f_name: str):
        means_components = np.stack([c.params for c in self.components], axis=0)
        covars_components = np.stack([c.covar for c in self.components], axis=0)

        means_context_components = np.stack(
            [c.mean for c in self.context_components], axis=0
        )
        covars_context_components = np.stack(
            [c.covar for c in self.context_components], axis=0
        )

        weight_distr_probs = self.weight_distribution.probabilities

        model_dict = {
            "means_components": means_components,
            "covars_components": covars_components,
            "means_context_components": means_context_components,
            "covars_context_components": covars_context_components,
            "weights_distr_probs": weight_distr_probs,
        }

        np.savez_compressed(os.path.join(path, f_name + ".npz"), **model_dict)

    @staticmethod
    def load(npz_path):
        model_dict = dict(np.load(npz_path, allow_pickle=True))

        model = LinMOE(
            component_params=model_dict["means_components"],
            component_covars=model_dict["covars_components"],
            context_component_means=model_dict["means_context_components"],
            context_component_covars=model_dict["covars_context_components"],
        )
        model.weight_distribution._p = model_dict["weights_distr_probs"]
        return model
