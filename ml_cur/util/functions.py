import numpy as np


def log_sum_exp(exp_arg, axis):
    exp_arg_use = exp_arg.copy()
    max_arg = np.max(exp_arg_use)
    exp_arg_use = np.clip(exp_arg_use - max_arg, -700, 700)
    return max_arg + np.log(np.sum(np.exp(exp_arg_use), axis=axis))


def stabilize_cov(covar: np.ndarray) -> np.ndarray:
    """Ensures Covariance is positive definite"""
    fact = 1e-10
    if not is_pd(covar):
        covar = 0.5 * (covar + covar.T)
    while (not is_pd(covar)) and fact < 10e10:
        covar += fact * np.eye(len(covar))
        fact *= 1.1
    return covar


def is_pd(mat: np.ndarray) -> bool:
    """Checks whether given matrix is positive definite"""
    try:
        np.linalg.cholesky(mat)
        return True
    except np.linalg.LinAlgError:
        return False


def stabilize_density(densities):
    densities = np.nan_to_num(densities)
    densities += 1e-120
    densities /= np.sum(densities, -1, keepdims=True)
    densities = np.clip(densities, 1e-120, 1 - 1e-120)
    densities /= np.sum(densities, -1, keepdims=True)
    return densities
