import abc

import numpy as np

from ml_cur.util.functions import stabilize_cov, stabilize_density
from ml_cur.util.logistic_regression import fit_affine_softmax, lr


class EM(abc.ABC):
    """Abstract base Class for all Expectation-Maximization (EM) based Algorithms"""

    def __init__(
        self, model, train_samples: np.ndarray = None, train_contexts: np.ndarray = None
    ):
        """EM Base Class

        Args:
            model: model to be trained
            train_samples (np.ndarray, optional): array of training samples. Defaults to None.
            train_contexts (np.ndarray, optional): array of training contexts. Defaults to None.
        """
        self._model = model
        self._train_samples = train_samples
        self._train_contexts = train_contexts

    def train(self, epochs: int, *args, **kwargs):
        """train procedure for EM algorithm.
        For e in epochs, alternately perform E and M steps.

        Args:
            epochs (int): number of training epochs
        """
        for i in range(epochs):
            ess = self._e_step()
            self._m_step(ess)

    @property
    def model(self):
        return self._model

    @abc.abstractmethod
    def _e_step(self) -> np.ndarray:
        """expectation step

        Returns:
            np.ndarray: expected sufficient statistics (ESS)
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _m_step(self, *args):
        """maximization step"""
        raise NotImplementedError


class GaussianMixtureEM(EM):
    def _e_step(self):
        densities = np.column_stack(
            [c.density(self._train_samples) for c in self._model.components]
        )
        # EM for GMMs is ill defined, components can focus on single samples, yielding infinite likelihood
        # this is clipped here
        densities[np.isinf(densities)] = np.max(
            densities[np.logical_not(np.isinf(densities))]
        )
        densities *= np.expand_dims(self._model.weight_distribution.probabilities(), 0)
        nt = np.sum(densities, -1, keepdims=True)
        responsibilities = densities / np.maximum(nt, 1e-10)

        return responsibilities

    def _m_step(self, responsibilities: np.ndarray):
        samples = np.expand_dims(self._train_samples, 1)
        responsibilities = np.expand_dims(responsibilities, -1)

        unnormal_weights = np.sum(responsibilities, axis=0, keepdims=True)
        new_weights = unnormal_weights / len(samples)
        new_means = (
            np.sum(responsibilities * samples, axis=0, keepdims=True) / unnormal_weights
        )

        diff = np.expand_dims(samples - new_means, -1)
        diff = np.matmul(diff, np.transpose(diff, [0, 1, 3, 2]))
        new_covs = np.sum(np.expand_dims(responsibilities, -1) * diff, 0, keepdims=True)
        new_covs /= np.expand_dims(unnormal_weights, -1)
        new_covs = new_covs[0]
        for i in range(len(new_covs)):
            new_covs[i] = stabilize_cov(new_covs[i])

        self._model.weight_distribution.update_parameters(new_weights[0, :, 0])
        for i in range(len(self._model.components)):
            self._model.components[i].update_parameters(new_means[0, i, :], new_covs[i])


class LinExpertMixtureEM(EM):
    def _e_step(self):
        densities = np.column_stack(
            [
                c.density(self._train_contexts, self._train_samples)
                for c in self._model.components
            ]
        )
        weights = self._model.gating_distribution.probabilities(self._train_contexts)
        densities *= weights
        densities = stabilize_density(densities)
        return densities

    def _m_step(self, responsibilities):
        contexts = self._train_contexts
        samples = self._train_samples
        contexts_bias = np.concatenate(
            [contexts, np.ones([contexts.shape[0], 1])], axis=-1
        )

        try:
            new_mixture_weights = fit_affine_softmax(
                contexts,
                responsibilities,
                initial_params=self._model.gating_distribution.params,
                maxeval=None,
            )
        except RuntimeError as e:
            print(e)
            new_mixture_weights = self._model.gating_distribution.params

        new_components_weights = []
        for k in range(self._model.num_components):
            new_components_weights.append(
                lr(contexts_bias, samples, responsibilities[:, k])
            )

        new_covars = []
        for k in range(self._model.num_components):
            diff = samples - contexts_bias @ new_components_weights[k]
            new_cov = np.dot(responsibilities[:, k] * diff.T, diff)
            new_cov /= np.sum(responsibilities[:, k])
            new_covars.append(stabilize_cov(new_cov))

        # update
        self._model.gating_distribution.update_parameters(new_mixture_weights)
        for k in range(self._model.num_components):
            c = self._model.components[k]
            c.update_parameters(new_components_weights[k], new_covars[k])
