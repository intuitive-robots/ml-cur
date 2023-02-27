import numpy as np


class Categorical:
    """
    Categorical Distribution
    """

    def __init__(self, probabilities: np.ndarray):
        """Create a Categorical distribution

        Args:
            probabilities (np.ndarray): array of probabilities for all classes. Can be unnormalized.
        """
        self._p = None
        self.update_parameters(probabilities)

    def sample(self, num_samples: int) -> np.ndarray:
        """Draw n samples from the distribution

        Args:
            num_samples (int): number of samples

        Returns:
            np.ndarray: array with samples
        """
        thresholds = np.expand_dims(np.cumsum(self._p), 0)
        thresholds[0, -1] = 1.0
        eps = np.random.uniform(size=[num_samples, 1])
        samples = np.argmax(eps < thresholds, axis=-1)
        return samples

    def probabilities(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: probability array
        """
        return self._p.copy()

    def log_probabilities(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: log probabilities array
        """
        return np.log(self._p + 1e-25)

    def update_parameters(self, new_probabilities: np.ndarray):
        """updates distribution parameters, i.e. array of class probabilites

        Args:
            new_probabilities (np.ndarray): array of new class probabilities. Can be unnormalized.
        """
        self._p = new_probabilities / np.sum(new_probabilities)

    def add_entry(self, new_prob: float):
        """add a new class probability entry to the distribution

        Args:
            new_prob (float): new class probability. Will be normalized afterwards
        """
        p = np.concatenate(
            [
                self._p,
                new_prob
                * np.ones(
                    1,
                ),
            ],
            axis=0,
        )
        self.update_parameters(p)

    def remove_entry(self, idx: int):
        """removes a class entry from the distribution

        Args:
            idx (int): index to be removed

        Raises:
            IndexError: if the index is greater than the size of the probability array
        """
        if idx > self._p.size:
            raise IndexError(
                f"Invalid Index {idx} for Categorical with size {self._p.size}"
            )
        p = np.concatenate([self._p[:idx], self._p[idx + 1 :]], axis=0)
        self.update_parameters(p)
