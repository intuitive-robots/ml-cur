import enum
from argparse import ArgumentError

import numpy as np
from sklearn.cluster import k_means

from ml_cur.distribution.mixture import GMM, LinEMM, LinMOE
from ml_cur.util.logistic_regression import fit_affine_softmax, lr


class InitMode(enum.Enum):
    RANDOM = enum.auto()
    KMEANS = enum.auto()


def _gmm_init(samples, num_components: int, mode: InitMode, seed: int):
    np.random.seed(seed)

    if mode == InitMode.RANDOM:
        initial_weights = np.ones([num_components]) / num_components
        initial_means = samples[
            np.random.choice(len(samples), num_components, replace=False)
        ]
        initial_covars = np.tile(
            np.expand_dims(np.cov(samples.T), 0), [num_components, 1, 1]
        )
        return initial_weights, initial_means, initial_covars

    elif mode == InitMode.KMEANS:
        initial_responsibilities = np.zeros([len(samples), num_components])
        m, labels, _ = k_means(samples, num_components)
        initial_responsibilities[np.arange(len(samples)), labels] = 1.0
        cts = np.sum(initial_responsibilities, 0)
        initial_weights = cts / len(samples)
        initial_means = np.dot(initial_responsibilities.T, samples) / cts[:, np.newaxis]
        initial_covariances = np.zeros(
            [num_components, samples.shape[1], samples.shape[1]]
        )
        for i in range(num_components):
            initial_covariances[i] = np.cov(samples[labels == i], rowvar=False)
            initial_covariances[i] += 0.01 * np.eye(samples.shape[-1])
        return initial_weights, initial_means, initial_covariances
    else:
        raise ArgumentError("Unknown mode {}. GMM cannot be initialized.".format(mode))


def _emm_init(contexts, samples, num_components: int, mode: InitMode, seed: int):
    np.random.seed(seed)
    context_dim = contexts.shape[1]
    samples_dim = samples.shape[1]

    if mode == InitMode.RANDOM:
        gp = np.random.normal(size=[context_dim + 1, num_components])
        cps = np.random.normal(size=[num_components, context_dim + 1, samples_dim])
        covs = np.tile(
            # np.eye(samples_dim) * 0.5,
            np.reshape(np.cov(samples, rowvar=False), [1, samples_dim, samples_dim]),
            [num_components, 1, 1],
        )
        return gp, cps, covs

    elif mode == InitMode.KMEANS:
        initial_responsibilities = np.ones([len(contexts), num_components]) * 0.05
        m, labels, _ = k_means(
            np.concatenate([contexts, samples], axis=-1), num_components
        )
        initial_responsibilities[np.arange(len(contexts)), labels] = 0.95
        initial_responsibilities /= np.sum(initial_responsibilities, 1, keepdims=True)
        cts = np.sum(initial_responsibilities, 0)
        theta_init = np.random.randn(context_dim + 1, num_components)

        gp = fit_affine_softmax(
            contexts, initial_responsibilities, initial_params=theta_init
        )

        cps = np.zeros([num_components, context_dim + 1, samples_dim])
        contexts_bias = np.concatenate(
            [contexts, np.ones([contexts.shape[0], 1])], axis=-1
        )
        for i in range(num_components):
            cps[i] = lr(
                contexts_bias,
                samples,
                weights=initial_responsibilities[:, i, np.newaxis],
            )

        covs = np.zeros([num_components, samples_dim, samples_dim])
        for i in range(num_components):
            diff = samples - contexts_bias @ cps[i]
            covs[i] = np.dot(initial_responsibilities[:, i] * diff.T, diff) / cts[i]

        return gp, cps, covs

    else:
        raise ArgumentError("Unknown mode {}. EMM cannot be initialized.".format(mode))


def _moe_init(contexts, samples, num_components: int, mode: InitMode, seed: int):
    _, cmp_params, cmp_covs = _emm_init(contexts, samples, num_components, mode, seed)
    _, ctxt_cmp_means, ctxt_cmp_covars = _gmm_init(contexts, num_components, mode, seed)

    return cmp_params, cmp_covs, ctxt_cmp_means, ctxt_cmp_covars


def build_gmm(samples, num_components: int, mode: InitMode, seed: int):
    return GMM(*_gmm_init(samples, num_components, mode, seed))


def build_lin_emm(contexts, samples, num_components: int, mode: InitMode, seed: int):
    return LinEMM(*_emm_init(contexts, samples, num_components, mode, seed))


def build_lin_moe(contexts, samples, num_components: int, mode: InitMode, seed: int):
    return LinMOE(*_moe_init(contexts, samples, num_components, mode, seed))
