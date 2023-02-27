import abc

import numpy as np

from ml_cur.distribution.marginal import Categorical
from ml_cur.util.functions import stabilize_cov, stabilize_density
from ml_cur.util.logistic_regression import lr


class MlCur(abc.ABC):
    def __init__(self, model, train_samples=None, train_contexts=None):
        self._model = model
        self._train_samples = train_samples
        self._train_contexts = train_contexts

        num_samples = train_samples.shape[0]
        self._sample_weights = [
            Categorical(np.ones(num_samples) / num_samples)
            for _ in range(self._model.num_components)
        ]

        self._alpha = None

    def train(self, epochs: int, alpha: float, *args, **kwargs):
        self._alpha = alpha

        for i in range(epochs):
            w_tilde = self._weight_step()
            ess = np.column_stack([c.probabilities() for c in self._sample_weights])
            self._m_step(ess)

    def log_w_tilde(self):
        # Shape (N_Samples, N_Components)
        w_tilde = np.column_stack([c.probabilities() for c in self._sample_weights])
        w_tilde = w_tilde / np.expand_dims(np.sum(w_tilde, axis=-1), axis=-1)

        log_w_tilde = np.log(w_tilde)
        return log_w_tilde

    def _weight_step(self):
        # Shape: (N_Components, N_Samples)
        componentwise_preds = self._component_wise_log_density()

        N_count = componentwise_preds.shape[0]
        alpha = self._alpha
        if isinstance(self._alpha, np.ndarray):
            N_count = 1
            alpha = self._alpha[:, None]

        componentwise_preds *= N_count / alpha

        log_w_tilde = self.log_w_tilde()

        w_oi_new = np.exp(componentwise_preds + np.swapaxes(log_w_tilde, 0, 1))
        w_oi_new_normal = stabilize_density(w_oi_new)

        for i, c in enumerate(self._sample_weights):
            c.update_parameters(w_oi_new_normal[i])
        return np.exp(log_w_tilde)

    @property
    def model(self):
        return self._model

    @abc.abstractmethod
    def _m_step(self):
        raise NotImplementedError

    @abc.abstractmethod
    def _component_wise_log_density(self):
        raise NotImplementedError


class GaussianMixtureMlCur(MlCur):
    def _component_wise_log_density(self):
        densities = self._model.density(self._train_samples, sum_components=False)
        return np.log(densities + 1e-25)

    def _m_step(self, responsibilities: np.ndarray):
        """update weight distribution and components individually"""
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


class LinMoeMlCur(MlCur):
    def _component_wise_log_density(self):
        s = self._train_contexts
        a = self._train_samples

        model = self._model

        return np.swapaxes(
            model.log_component_densities(s, a)  # p(a|s,o)
            + model.log_component_context_densities(s)  # p(s|o)
            + model.weight_distribution.log_probabilities(),  # p(o)
            0,
            1,
        )

    def _m_step(self, responsibilities):
        sample_resp, ctxt_resp = responsibilities, responsibilities
        contexts = self._train_contexts
        samples = self._train_samples

        ### Context Component Update
        ctx_responsibilities = np.expand_dims(ctxt_resp, -1)
        ext_contexts = np.expand_dims(contexts, 1)
        # ext_contexts = contexts

        unnormal_weights = np.sum(ctx_responsibilities, axis=0, keepdims=True)
        new_weights = unnormal_weights / len(samples)

        ctxt_cmp_means = (
            np.sum(ctx_responsibilities * ext_contexts, axis=0, keepdims=True)
            / unnormal_weights
        )

        ctxt_diff = np.expand_dims(ext_contexts - ctxt_cmp_means, -1)
        ctxt_diff = np.matmul(ctxt_diff, np.transpose(ctxt_diff, [0, 1, 3, 2]))

        ctxt_cmp_covars = np.sum(
            np.expand_dims(ctx_responsibilities, -1) * ctxt_diff, 0, keepdims=True
        )
        ctxt_cmp_covars /= np.expand_dims(unnormal_weights, -1)

        ctxt_cmp_covars = ctxt_cmp_covars[0]

        for i in range(len(ctxt_cmp_covars)):
            ctxt_cmp_covars[i] = stabilize_cov(ctxt_cmp_covars[i])

        ### Mixture Component Update
        contexts_bias = np.concatenate(
            [contexts, np.ones([contexts.shape[0], 1])], axis=-1
        )

        new_components_weights = []
        for k in range(self.model.num_components):
            new_components_weights.append(lr(contexts_bias, samples, sample_resp[:, k]))

        new_covars = []
        for k in range(self.model.num_components):
            diff = samples - contexts_bias @ new_components_weights[k]
            new_cov = np.dot(sample_resp[:, k] * diff.T, diff)
            new_cov /= np.sum(sample_resp[:, k])

            try:
                np.linalg.cholesky(new_cov)
            except:
                new_cov += np.eye(new_cov.shape[0]) * 1e-6

            new_covars.append(stabilize_cov(new_cov))

        self._model.weight_distribution.update_parameters(new_weights[0, :, 0])
        self._model.update_parameters(
            np.stack(new_components_weights, axis=0),
            np.stack(new_covars, axis=0),
            ctxt_cmp_means[0],
            ctxt_cmp_covars,
        )
