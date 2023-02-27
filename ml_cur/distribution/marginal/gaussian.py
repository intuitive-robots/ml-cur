import numpy as np


class Gaussian:
    """
    Gaussian Distribution
    """

    def __init__(self, mean: np.ndarray, covar: np.ndarray):
        """Create a Gaussian Distribution

        Args:
            mean (np.ndarray): mean of the gaussian with shape (D,)
            covar (np.ndarray): full covariance matrix of the distribution with shape (D,D)
        """
        self._dim = mean.shape[-1]
        self.update_parameters(mean, covar)

    def sample(self, num_samples: int) -> np.ndarray:
        """Draw n samples from the distribution

        Args:
            num_samples (int): number of samples

        Returns:
            np.ndarray: array with samples
        """
        eps = np.random.normal(size=[num_samples, self._dim])
        return self._mean + eps @ self._chol_covar.T

    def density(self, samples: np.ndarray) -> np.ndarray:
        """Computes the density of the given samples

        Args:
            samples (np.ndarray): array with samples

        Returns:
            np.ndarray: array with density per sample
        """
        return np.exp(self.log_density(samples))

    def log_density(self, samples: np.ndarray) -> np.ndarray:
        """Computes the log density of the given samples

        Args:
            samples (np.ndarray): array with samples

        Returns:
            np.ndarray: array with density per sample
        """
        norm_term = self._dim * np.log(2 * np.pi) + self.covar_logdet()
        diff = samples - self._mean
        exp_term = np.sum(np.square(diff @ self._chol_precision), axis=-1)
        return -0.5 * (norm_term + exp_term)

    def log_likelihood(self, samples: np.ndarray) -> float:
        """Computes the log-likelihood of the given samples

        Args:
            samples (np.ndarray): array with samples

        Returns:
            float: log-likelihood
        """
        return np.mean(self.log_density(samples))

    def update_parameters(self, mean: np.ndarray, covar: np.ndarray):
        """updates distribution parameters

        Args:
            mean (np.ndarray): mean of the gaussian with Dimension 1xD
            covar (np.ndarray): full covariance matrix of the distribution
        """
        try:
            try:
                chol_covar = np.linalg.cholesky(covar)
            except np.linalg.LinAlgError as _:
                covar = np.nan_to_num(covar)
                chol_covar = np.linalg.cholesky(covar + 1e-6 * np.eye(covar.shape[0]))
            inv_chol_covar = np.linalg.inv(chol_covar)
            precision = inv_chol_covar.T @ inv_chol_covar

            self._chol_precision = np.linalg.cholesky(precision)
            self._mean = mean
            self._lin_term = precision @ mean
            self._covar = covar
            self._precision = precision

            self._chol_covar = chol_covar

        except Exception as error:
            print("Gaussian Paramameter update rejected:", error)

    def covar_logdet(self) -> float:
        """computes the log determinante of the covariance matrix

        Returns:
            float: log determinante
        """
        return 2 * np.sum(np.log(np.diagonal(self._chol_covar) + 1e-25))

    @property
    def mean(self) -> np.ndarray:
        return self._mean

    @property
    def covar(self) -> np.ndarray:
        return self._covar

    @property
    def lin_term(self):
        return self._lin_term

    @property
    def precision(self):
        return self._precision

    @property
    def chol_covar(self):
        return self._chol_covar

    @property
    def chol_precision(self):
        return self._chol_precision
