import numpy as np

from ml_cur.distribution import factory, mixture
from ml_cur.learner import em
from ml_cur.learner import ml_cur as mlcur_learner
from ml_cur.learner import ml_cur_dual


class EmGmm:
    def __init__(
        self, n_components: int, train_iter: int, init_mode: str = "random"
    ) -> None:
        """Creates a GMM Trainer using EM

        Args:
            n_components (int): number of GMM components
            train_iter (int): number of training iterations
            init_mode (str): initialization method of the GMM. Can be "random" or "kmeans". Defaults to "random"
        """
        self._n_components = n_components
        self.train_iter = train_iter
        self.learner: em.GaussianMixtureEM = None

        if init_mode == "random":
            self.init_mode = factory.InitMode.RANDOM
        elif init_mode == "kmeans":
            self.init_mode = factory.InitMode.KMEANS
        else:
            raise ValueError("init_mode must be random or kmeans")

        self._initialized = False

    @property
    def model(self) -> mixture.GMM:
        """getter for the underlying GMM Model

        Returns:
            mixture.GMM: GMM Model
        """
        return self.learner.model

    def fit(self, train_samples: np.ndarray):
        """fits a GMM model to the training data using EM

        Args:
            train_samples (np.ndarray): array of training samples
        """
        gmm = factory.build_gmm(train_samples, self._n_components, self.init_mode, 0)
        self.learner = em.GaussianMixtureEM(gmm, train_samples)

        self._initialized = True
        self.learner.train(self.train_iter)

    def sample(self, n_samples: int) -> np.ndarray:
        """sample n_samples from the trained GMM distribution

        Args:
            n_samples (int): number of samples to draw

        Returns:
            np.ndarray: array of drawn samples
        """
        return self.model.sample(n_samples)

    def density(self, samples: np.ndarray) -> np.ndarray:
        """compute the density for the given samples

        Args:
            samples (np.ndarray): array of samples

        Returns:
            np.ndarray: array with density values
        """
        return self.model.density(samples)

    def predict_proba(self, samples: np.ndarray) -> np.ndarray:
        """compute the density for the given samples
        Wrapper around self.density() to comply with scikit learn.

        Args:
            samples (np.ndarray): array of samples

        Returns:
            np.ndarray: array with density values
        """
        return self.density(samples)

    def continue_train(self, train_iter: int):
        """continues training for additional n iterations

        Args:
            train_iter (int): number of additional training iterations

        Raises:
            RuntimeError: thrown if .fit() has not been called before, as the data is missing.
        """
        if not self._initialized:
            raise RuntimeError("Please call .fit() first")
        self.train_iter += train_iter
        self.learner.train(train_iter)

    def save_model(self, fpath: str):
        """saves the underlying GMM model to a file

        Args:
            fpath (str): path to save the model to
        """
        self.model.save(fpath)

    def load_model(self, npz_path: str):
        """loads a model from the path

        Args:
            npz_path (str): path to a saved model
        """
        gmm = mixture.GMM.load(npz_path)
        if not self._initialized:
            learner = em.GaussianMixtureEM(gmm, None)
            self.learner = learner
        self.learner.model = gmm


class MlCurGmm:
    def __init__(
        self,
        n_components: int,
        train_iter: int,
        num_active_samples: int | float = 0.3,
        static_alpha: float = None,
        init_mode: str = "random",
    ) -> None:
        """Creates a GMM Trainer using ML-Cur

        Args:
            n_components (int): number of GMM components
            train_iter (int): number of training iterations
            num_active_samples (int | float, optional): number of active samples per component.
                Can be int=absolute number, or float=proportion of training samples.
                Overriden by setting static_alpha. Defaults to 0.3.
            static_alpha (float, optional): Manual setting of a static alpha value for all components.
                Overrides num_active_samples. Defaults to None.
            init_mode (str): initialization method of the GMM. Can be "random" or "kmeans". Defaults to "random"
        """
        self._n_components = n_components
        self.train_iter = train_iter
        self._num_active_samples = num_active_samples
        self.static_alpha = static_alpha
        self.learner = None

        if init_mode == "random":
            self.init_mode = factory.InitMode.RANDOM
        elif init_mode == "kmeans":
            self.init_mode = factory.InitMode.KMEANS
        else:
            raise ValueError("init_mode must be random or kmeans")

        self._initialized = False

    @property
    def model(self) -> mixture.GMM:
        """getter for the underlying GMM model

        Returns:
            mixture.GMM: GMM Model
        """
        return self.learner.model

    def fit(self, train_samples: np.ndarray):
        """fits a GMM to the training data using ML-Cur

        Args:
            train_samples (np.ndarray): array of training samples
        """
        gmm = factory.build_gmm(train_samples, self._n_components, self.init_mode, 0)

        if self.static_alpha is not None:
            alpha_param = self.static_alpha
            learner = mlcur_learner.GaussianMixtureMlCur(gmm, train_samples)
        else:
            if self._num_active_samples < 1:
                self._num_active_samples *= train_samples.shape[0]
            alpha_param = self._num_active_samples
            learner = ml_cur_dual.GaussianMixtureMlCurDual(gmm, train_samples)

        self.learner = learner
        self._initialized = True
        self.learner.train(self.train_iter, alpha_param)

    def sample(self, n_samples: int) -> np.ndarray:
        """sample n_samples from the trained GMM distribution

        Args:
            n_samples (int): number of samples to draw

        Returns:
            np.ndarray: array of drawn samples
        """
        return self.model.sample(n_samples)

    def density(self, samples: np.ndarray) -> np.ndarray:
        """compute the density for the given samples

        Args:
            samples (np.ndarray): array of samples

        Returns:
            np.ndarray: array with density values
        """
        return self.model.density(samples)

    def predict_proba(self, samples: np.ndarray) -> np.ndarray:
        """compute the density for the given samples
        Wrapper around self.density() to comply with scikit learn.

        Args:
            samples (np.ndarray): array of samples

        Returns:
            np.ndarray: array with density values
        """
        return self.density(samples)

    def continue_train(self, train_iter: int):
        """continues training for additional n iterations

        Args:
            train_iter (int): number of additional training iterations

        Raises:
            RuntimeError: thrown if .fit() has not been called before, as the data is missing.
        """
        if not self._initialized:
            raise RuntimeError("Please call .fit() first")

        if self.static_alpha is not None:
            alpha_param = self.static_alpha
        else:
            alpha_param = self._num_active_samples
        self.learner.train(train_iter, alpha_param)

    def save_model(self, fpath: str):
        """saves the underlying GMM model to a file

        Args:
            fpath (str): path to save the model to
        """
        self.model.save(fpath)

    def load_model(self, npz_path):
        """loads a model from the path

        Args:
            npz_path (str): path to a saved model
        """
        gmm = mixture.GMM.load(npz_path)
        if not self._initialized:
            learner = ml_cur_dual.GaussianMixtureMlCurDual(gmm)
            self.learner = learner
        self.learner.model = gmm


class EmLinEmm:
    def __init__(
        self, n_components: int, train_iter: int, init_mode: str = "random"
    ) -> None:
        """Creates a Linear Expert Mixture Model (LinEMM) Trainer using EM

        Args:
            n_components (int): number of LinEMM components
            train_iter (int): number of training iterations
            init_mode (str): initialization method of the EMM. Can be "random" or "kmeans". Defaults to "random"
        """
        self._n_components = n_components
        self.train_iter = train_iter
        self.learner: em.LinExpertMixtureEM = None

        if init_mode == "random":
            self.init_mode = factory.InitMode.RANDOM
        elif init_mode == "kmeans":
            self.init_mode = factory.InitMode.KMEANS
        else:
            raise ValueError("init_mode must be random or kmeans")

        self._initialized = False

    @property
    def model(self) -> mixture.LinEMM:
        """getter for the underlying LinEMM Model

        Returns:
            mixture.LinEMM: LiNEMM Model
        """
        return self.learner.model

    def fit(self, train_samples: np.ndarray, train_contexts: np.ndarray):
        """fits a LinEMM model to the training data using EM

        Args:
            train_samples (np.ndarray): array of training samples
            train_contexts (np.ndarray): array of training contexts
        """
        emm = factory.build_lin_emm(
            train_contexts,
            train_samples,
            self._n_components,
            self.init_mode,
            0,
        )
        self.learner = em.LinExpertMixtureEM(emm, train_samples, train_contexts)
        self._initialized = True
        self.learner.train(self.train_iter)

    def sample(self, contexts: np.ndarray) -> np.ndarray:
        """sample one point for each entry in the context array

        Args:
            contexts (np.ndarray): array of context points

        Returns:
            np.ndarray: array of drawn samples
        """
        return self.model.sample(contexts)

    def density(self, contexts: np.ndarray, samples: np.ndarray) -> np.ndarray:
        """compute the density for the given context-sample pairs

        Args:
            contexts (np.ndarray): array of context points
            samples (np.ndarray): array of sample points

        Returns:
            np.ndarray: array of density values
        """
        return self.model.density(contexts, samples)

    def predict_proba(self, contexts: np.ndarray, samples: np.ndarray) -> np.ndarray:
        """compute the density for the given context-sample pairs
        Wrapper around self.density() to comply with scikit learn.

        Args:
            contexts (np.ndarray): array of context points
            samples (np.ndarray): array of sample points

        Returns:
            np.ndarray: array of density values
        """
        return self.density(contexts, samples)

    def continue_train(self, train_iter: int):
        """continues training for additional n iterations

        Args:
            train_iter (int): number of additional training iterations

        Raises:
            RuntimeError: thrown if .fit() has not been called before, as the data is missing.
        """
        if not self._initialized:
            raise RuntimeError("Please call .fit() first")

        self.learner.train(train_iter)

    def save_model(self, fpath):
        """saves the underlying LinEMM model to a file

        Args:
            fpath (str): path to save the model to
        """
        self.model.save(fpath)

    def load_model(self, npz_path):
        """loads a model from the path

        Args:
            npz_path (str): path to a saved model
        """
        emm = mixture.LinEMM.load(npz_path)
        if not self._initialized:
            learner = em.LinExpertMixtureEM(emm)
            self.learner = learner
        self.learner.model = emm


class MlCurLinMoe:
    def __init__(
        self,
        n_components: int,
        train_iter: int,
        num_active_samples: int | float = 0.3,
        static_alpha=None,
        init_mode: str = "random",
    ) -> None:
        """Creates a Linear Mixture of Experts (LinMOE) Trainer using ML-Cur

        Args:
            n_components (int): number of GMM components
            train_iter (int): number of training iterations
            num_active_samples (int | float, optional): number of active samples per component.
                Can be int=absolute number, or float=proportion of training samples.
                Overriden by setting static_alpha. Defaults to 0.3.
            static_alpha (float, optional): Manual setting of a static alpha value for all components.
                Overrides num_active_samples. Defaults to None.
            init_mode (str): initialization method of the EMM. Can be "random" or "kmeans". Defaults to "random"
        """

        self._n_components = n_components
        self.train_iter = train_iter
        self._num_active_samples = num_active_samples
        self.static_alpha = static_alpha
        self.learner = None

        if init_mode == "random":
            self.init_mode = factory.InitMode.RANDOM
        elif init_mode == "kmeans":
            self.init_mode = factory.InitMode.KMEANS
        else:
            raise ValueError("init_mode must be random or kmeans")

        self._initialized = False

    @property
    def model(self) -> mixture.LinMOE:
        """getter for the underlying LinMOE model

        Returns:
            mixture.LinMOE: LinMOE model
        """
        return self.learner.model

    def fit(self, train_samples: np.ndarray, train_contexts: np.ndarray):
        """fits a LinEMM model to the training data using EM

        Args:
            train_samples (np.ndarray): array of training samples
            train_contexts (np.ndarray): array of training contexts
        """
        moe = factory.build_lin_moe(
            train_contexts,
            train_samples,
            self._n_components,
            self.init_mode,
            0,
        )

        if self.static_alpha is not None:
            alpha_param = self.static_alpha
            learner = mlcur_learner.LinMoeMlCur(moe, train_samples, train_contexts)
        else:
            if self._num_active_samples < 1:
                self._num_active_samples *= train_samples.shape[0]
            alpha_param = self._num_active_samples
            learner = ml_cur_dual.LinMoeMlCurDual(moe, train_samples, train_contexts)

        self.learner = learner
        self._initialized = True
        self.learner.train(self.train_iter, alpha_param)

    def sample(self, contexts: np.ndarray) -> np.ndarray:
        """sample one point for each entry in the context array

        Args:
            contexts (np.ndarray): array of context points

        Returns:
            np.ndarray: array of drawn samples
        """
        return self.model.sample(contexts)

    def density(self, contexts: np.ndarray, samples: np.ndarray) -> np.ndarray:
        """compute the density for the given context-sample pairs

        Args:
            contexts (np.ndarray): array of context points
            samples (np.ndarray): array of sample points

        Returns:
            np.ndarray: array of density values
        """
        return self.model.density(contexts, samples)

    def predict_proba(self, contexts: np.ndarray, samples: np.ndarray) -> np.ndarray:
        """compute the density for the given context-sample pairs
        Wrapper around self.density() to comply with scikit learn.

        Args:
            contexts (np.ndarray): array of context points
            samples (np.ndarray): array of sample points

        Returns:
            np.ndarray: array of density values
        """
        return self.density(contexts, samples)

    def continue_train(self, train_iter: int):
        """continues training for additional n iterations

        Args:
            train_iter (int): number of additional training iterations

        Raises:
            RuntimeError: thrown if .fit() has not been called before, as the data is missing.
        """
        if not self._initialized:
            raise RuntimeError("Please call .fit() first")

        alpha_param = self._num_active_samples
        if self.static_alpha is not None:
            alpha_param = self.static_alpha

        self.learner.train(train_iter, alpha_param)

    def save_model(self, fpath):
        """saves the underlying LinMOE model to a file

        Args:
            fpath (str): path to save the model to
        """
        self.model.save(fpath)

    def load_model(self, npz_path):
        """loads a model from the path

        Args:
            npz_path (str): path to a saved model
        """
        emm = mixture.LinEMM.load(npz_path)
        if not self._initialized:
            learner = em.LinExpertMixtureEM(emm)
            self.learner = learner
        self.learner.model = emm
