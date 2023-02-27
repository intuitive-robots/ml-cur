import abc
from functools import partial

import autograd.numpy as np
from autograd import grad
from scipy.optimize import minimize
from scipy.special import logsumexp

import ml_cur.learner.ml_cur as mlcur_base
from ml_cur.util.functions import stabilize_cov, stabilize_density
from ml_cur.util.logistic_regression import lr


class MlCurDual(mlcur_base.MlCur, abc.ABC):
    def __init__(self, model, train_samples=None, train_contexts=None):
        super().__init__(model, train_samples, train_contexts)
        self._alpha = [0] * self._model.num_components
        self._min_entropy = 0

    def train(
        self,
        epochs: int,
        effective_samples: int,
        *args,
        verbose: bool = False,
        **kwargs
    ):
        self._min_entropy = np.log(effective_samples)

        for i in range(epochs):
            self.alpha_step(verbose)
            self._alpha = np.asarray(self._alpha)
            w_tilde = self._weight_step()
            ess = np.column_stack([c.probabilities() for c in self._sample_weights])
            self._m_step(ess)

    def alpha_step(self, verbose=False):
        comp_densities = self._component_wise_log_density()
        for i in range(self._model.num_components):
            self.update_entropy_scaling(i, comp_densities[i], verbose)

    def dual(self, cmp_idx, cmp_density, params):
        alpha = params[0]

        rewards = cmp_density  # Rewards are log q(x|y,z)
        max_r = np.max(rewards)
        geom_avg = np.exp(
            ((rewards - max_r) / alpha) + self.log_w_tilde()[:, cmp_idx]
        )  # Log gatings are log w^\tilde_z|i
        g = alpha * np.log(np.sum(geom_avg) + 1e-20) + max_r
        g += -alpha * self._min_entropy

        return g

    def update_entropy_scaling(self, cmp_idx, cmp_density, verbose=False):
        bounds = [(1e-6, 1e8)]
        res = minimize(
            partial(self.dual, cmp_idx, cmp_density),
            np.array([self._alpha[cmp_idx]]),
            method="L-BFGS-B",
            jac=grad(partial(self.dual, cmp_idx, cmp_density)),
            bounds=bounds,
            options={"ftol": 1e-6},
        )
        if verbose:
            print(res.x)

        self._alpha[cmp_idx] = res.x[0]


class GaussianMixtureMlCurDual(mlcur_base.GaussianMixtureMlCur, MlCurDual):
    def train(self, epochs: int, effective_samples: int, *args, **kwargs):
        return super(mlcur_base.GaussianMixtureMlCur, self).train(
            epochs, effective_samples, *args, **kwargs
        )


class LinMoeMlCurDual(mlcur_base.LinMoeMlCur, MlCurDual):
    def train(self, epochs: int, effective_samples: int, *args, **kwargs):
        return super(mlcur_base.LinMoeMlCur, self).train(
            epochs, effective_samples, *args, **kwargs
        )
