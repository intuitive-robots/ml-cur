import numpy as np


class LinCondGaussian:
    """
    Linear Conditional Gaussian Distribution
    """

    def __init__(self, params: np.ndarray, covar: np.ndarray):
        """Creates a linear conditional Gaussian distribution"""
        self._context_dim = params.shape[0] - 1
        self._sample_dim = params.shape[1]
        self.update_parameters(params, covar)

    def sample(self, contexts: np.ndarray) -> np.ndarray:
        """sample one point from the distribution for each entry in the contexts array

        Args:
            contexts (np.ndarray): array with context points

        Returns:
            np.ndarray: sampled points
        """
        eps = np.random.normal(size=[contexts.shape[0], self._sample_dim])
        return self.means(contexts) + eps @ self._chol_covar.T

    def density(self, contexts: np.ndarray, samples: np.ndarray) -> np.ndarray:
        """computes the density for the each context and sample pair in zip(contexts, samples)

        Args:
            contexts (np.ndarray): array of context points
            samples (np.ndarray): array of sample point

        Returns:
            np.ndarray: density
        """
        return np.exp(self.log_density(contexts, samples))

    def log_density(self, contexts: np.ndarray, samples: np.ndarray) -> np.ndarray:
        """computes the log density for the each context and sample pair in zip(contexts, samples)

        Args:
            contexts (np.ndarray): array of context points
            samples (np.ndarray): array of sample point

        Returns:
            np.ndarray: log density
        """
        norm_term = self._sample_dim * np.log(2 * np.pi) + self.covar_logdet()
        diff = samples - self.means(contexts)
        exp_term = np.sum(np.square(diff @ self._chol_precision), axis=-1)
        return -0.5 * (norm_term + exp_term)

    def log_likelihood(self, contexts, samples) -> float:
        """Computes the log-likelihood of the given context sample pairs

        Args:
            contexts (np.ndarray): array of context points
            samples (np.ndarray): array of sample point

        Returns:
            float: log-likelihood
        """
        return np.mean(self.log_density(contexts, samples))

    def update_parameters(self, params, covar):
        """updates distribution parameters"""
        try:

            try:
                chol_covar = np.linalg.cholesky(covar)
            except np.linalg.LinAlgError as _:
                chol_covar = np.linalg.cholesky(covar + 1e-6 * np.eye(covar.shape[0]))
            inv_chol_covar = np.linalg.inv(chol_covar)
            precision = inv_chol_covar.T @ inv_chol_covar
            chol_precision = np.linalg.cholesky(precision)

            self._covar = covar
            self._chol_covar = chol_covar
            self._inv_chol_covar = inv_chol_covar
            self._precision = precision
            self._chol_precision = chol_precision
            self._params = params

        except np.linalg.LinAlgError as e:
            print("Linear Conditional Gaussian Paramameter update rejected:", e)

    def covar_logdet(self) -> float:
        """computes the log determinante of the covariance matrix

        Returns:
            float: log determinante
        """
        return 2 * np.sum(np.log(np.diagonal(self._chol_covar) + 1e-25))

    def means(self, contexts):
        return contexts @ self._params[:-1] + self._params[-1]

    def covars(self, contexts):
        return np.tile(np.expand_dims(self.covar, 0), [contexts.shape[0], 1, 1])

    @property
    def params(self):
        return self._params

    @property
    def covar(self):
        return self._covar

    @property
    def precision(self):
        return self._precision

    @property
    def chol_covar(self):
        return self._chol_covar

    @property
    def chol_precision(self):
        return self._chol_precision
