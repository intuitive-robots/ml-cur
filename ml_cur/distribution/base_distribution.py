import abc

import numpy as np


class BaseDistribution(abc.ABC):
    @abc.abstractmethod
    def sample(self, num_samples: int) -> np.ndarray:
        """Draw n samples from the distribution

        Args:
            num_samples (int): number of samples

        Returns:
            np.ndarray: array with samples
        """
        raise NotImplementedError


class ContinuousDistribution(BaseDistribution, abc.ABC):
    pass


class DiscreteDistribution(BaseDistribution, abc.ABC):
    @abc.abstractmethod
    def probabilities(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: probability array
        """
        raise NotImplementedError

    @abc.abstractmethod
    def log_probabilities(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: log probabilities array
        """
        raise NotImplementedError

    @abc.abstractmethod
    def add_entry(self, new_prob: float):
        """add a new class probability entry to the distribution

        Args:
            new_prob (float): new class probability. Will be normalized afterwards
        """
        raise NotImplementedError
