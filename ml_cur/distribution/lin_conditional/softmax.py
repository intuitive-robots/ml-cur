import numpy as np

import ml_cur.util.logistic_regression as lr


class Softmax:
    """Softmax Distribution"""

    def __init__(self, params):
        self._params = params
        self._feature_map = self.affine_mapping

    def sample(self, contexts):
        """sample one point from the distribution for each entry in the contexts array

        Args:
            contexts (np.ndarray): array with context points

        Returns:
            np.ndarray: sampled points
        """
        p = self.probabilities(contexts)
        thresholds = np.cumsum(p, axis=-1)
        thresholds[:, -1] = 1.0
        eps = np.random.uniform(size=[contexts.shape[0], 1])
        samples = np.argmax(eps < thresholds, axis=-1)
        return samples

    def probabilities(self, contexts):
        """computes the probablities for the each context point

        Args:
            contexts (np.ndarray): array of context points

        Returns:
            np.ndarray: probabilities
        """
        return np.exp(self.log_probabilities(contexts))

    def log_probabilities(self, contexts):
        """computes the log probability for the each context point

        Args:
            contexts (np.ndarray): array of context points

        Returns:
            np.ndarray: log probability
        """
        logits = self._feature_map(contexts, self._params)
        max_logits = np.max(logits, axis=-1, keepdims=True)
        return logits - (
            max_logits
            + np.log(np.sum(np.exp(logits - max_logits), axis=-1, keepdims=True))
        )

    @staticmethod
    def affine_mapping(x, p):
        return x @ p[:-1] + p[-1]

    @property
    def params(self):
        return self._params

    def update_parameters(self, new_params):
        """updates distribution parameters"""
        self._params = new_params

    def add_entry(self, contexts, initial_probs):
        # if len(initial_probs.shape) == 1:
        #    initial_probs = np.expand_dims(initial_probs, -1)
        # new_probs = np.concatenate([self.probabilities(contexts), initial_probs], axis=-1)
        # new_probs /= np.sum(new_probs, axis=-1, keepdims=True)
        new_params = np.concatenate(
            [self._params, np.zeros((self._params.shape[0], 1))], axis=-1
        )
        self._params = self.fit(contexts, initial_probs, new_params)

    def remove_entry(self, contexts, idx):
        probs = self.probabilities(contexts)
        new_probs = np.concatenate([probs[:, :idx], probs[:, idx + 1 :]], axis=-1)
        new_probs /= np.sum(new_probs, axis=-1, keepdims=True)
        new_params = np.concatenate(
            [self._params[:, :idx], self._params[:, idx + 1 :]], axis=-1
        )
        self._params = self.fit(contexts, new_probs, new_params)

    def fit(
        self,
        contexts,
        targets,
        initial_params,
        weights=None,
        maxeval=None,
        ftol_rel=None,
    ):
        return lr.fit_affine_softmax(
            contexts,
            targets,
            initial_params,
            weights=weights,
            maxeval=maxeval,
            ftol_rel=ftol_rel,
        )

    def copy(self):
        return Softmax(self._params)


class QuadSoftmax(Softmax):
    """Quadratic Softmax Distribution"""

    def __init__(self, params):
        super().__init__(params)
        self._feature_map = lambda x, p: lr.quad_mapping(x, p)[0]

    def fit(
        self,
        contexts,
        targets,
        initial_params,
        weights=None,
        maxeval=None,
        ftol_rel=None,
    ):
        return lr.fit_quad_softmax(
            contexts,
            targets,
            initial_params,
            weights=weights,
            maxeval=maxeval,
            ftol_rel=ftol_rel,
        )

    def copy(self):
        return QuadSoftmax(self._params)

    def add_entry(self, contexts, initial_probs):
        new_params = np.concatenate(
            [self._params, np.mean(self._params, keepdims=True, axis=-1)], axis=-1
        )
        self._params = self.fit(contexts, initial_probs, new_params)
