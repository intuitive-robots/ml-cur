import nlopt
import numpy as np


def _affine_objective(params, ret_grad, params_shape, contexts, targets, weights):
    params = np.reshape(params, params_shape)

    logits = contexts @ params[:-1] + params[-1]

    max_logits = np.max(logits, axis=-1, keepdims=True)
    normalized_logits = logits - (
        max_logits + np.log(np.sum(np.exp(logits - max_logits), axis=-1, keepdims=True))
    )
    pred = np.exp(normalized_logits)

    sample_wise_loss = np.sum(
        targets * normalized_logits, axis=-1, keepdims=True
    )  # * log(pred)
    loss = -np.sum(weights * sample_wise_loss)

    tmp = pred - targets
    grad = np.zeros(params_shape)

    for i in range(contexts.shape[-1]):
        grad[i] = np.sum(weights * contexts[:, i : i + 1] * tmp, axis=0)
    grad[-1] = np.sum(weights * tmp, axis=0)
    ret_grad[:] = np.reshape(grad, -1)
    return loss


def fit_affine_softmax(
    contexts, targets, initial_params, weights=None, maxeval=None, ftol_rel=None
):
    if weights is None:
        weights = np.ones([contexts.shape[0], 1]) / contexts.shape[0]
    if len(weights.shape) == 1:
        weights = np.expand_dims(weights, 1)

    params_shape = initial_params.shape

    obj = lambda p, g: _affine_objective(p, g, params_shape, contexts, targets, weights)

    opt = nlopt.opt(nlopt.LD_LBFGS, int(np.prod(params_shape)))
    if maxeval is not None:
        opt.set_maxeval(maxeval)
    if ftol_rel is not None:
        opt.set_ftol_rel(ftol_rel)
    opt.set_min_objective(obj)
    res = opt.optimize(np.reshape(initial_params, -1))
    return np.reshape(res, params_shape)


def quad_mapping(x, p):
    sample_dim = x.shape[1]
    num_samples = x.shape[0]
    num_components = p.shape[1]

    chol_idxs = np.tril_indices(sample_dim)
    all_logits = np.zeros((num_samples, num_components))
    all_chols = np.zeros((num_components, sample_dim, sample_dim))

    for comp_idx in range(num_components):
        chol = np.zeros((sample_dim, sample_dim))
        chol[chol_idxs] = p[: -(sample_dim + 1), comp_idx]

        quad_term = -np.sum(np.square(x @ chol), axis=-1)
        lin_term = x @ p[-(sample_dim + 1) : -1, comp_idx]
        logits = quad_term + lin_term + p[-1:, comp_idx]

        all_logits[:, comp_idx] = logits
        all_chols[comp_idx, :, :] = chol

    return all_logits, all_chols


def _quad_objective(params, ret_grad, params_shape, contexts, targets, weights, n):
    params = np.reshape(params, params_shape)
    n_comps = params_shape[1]

    logits, all_chols = quad_mapping(contexts, params)
    max_logits = np.max(logits, axis=-1, keepdims=True)
    normalized_logits = logits - (
        max_logits + np.log(np.sum(np.exp(logits - max_logits), axis=-1, keepdims=True))
    )
    pred = np.exp(normalized_logits)

    sample_wise_loss = np.sum(
        targets * normalized_logits, axis=-1, keepdims=True
    )  # * log(pred)
    loss = -np.sum(weights * sample_wise_loss)

    tmp = pred - targets
    grad = np.zeros(params_shape)
    contexts_outer = -2 * np.einsum("ij, jk->jik", contexts.T, contexts)
    lti_chol_indices = np.tril_indices(n)
    for i in range(n_comps):
        grad_chol_tmp = contexts_outer[:] @ all_chols[i, :, :]
        grad_chol_tmp = np.einsum(
            "i,ijk->ijk", weights[:, 0] * tmp[:, i], grad_chol_tmp
        )
        grad_chol_tmp = np.sum(grad_chol_tmp, axis=0)
        grad_chol_tmp = grad_chol_tmp[lti_chol_indices]
        grad[: -(n + 1), i] = grad_chol_tmp

        grad[-(n + 1) : -1, i] = np.einsum(
            "i, i, ij->j", weights[:, 0], tmp[:, i], contexts
        )

        grad[-1, i] = np.sum(weights[:, 0] * tmp[:, i], axis=0)
    ret_grad[:] = np.reshape(grad, -1)
    return loss


def get_squared_quad_feat_ind(quad_feat_dim):
    # this function should be applied only to one sample !
    # quad_feat_dim = the number of indices of the flattened quadratic matrix A == context dim
    indices = []
    accu = -1
    for i in range(quad_feat_dim):
        accu += i + 1
        indices.append(accu)
    return np.array(indices)


def create_bounds_array(quad_feat_dim, feat_dim, out_dim):
    # create bounds only for one component and then stack them together
    # get the indices of the squared features (they have to be positive definite
    squared_indices = list(get_squared_quad_feat_ind(quad_feat_dim))
    all_bounds = []
    for i in range(feat_dim):
        if i in squared_indices:
            all_bounds.append((1e-6, 1e12))
        else:
            all_bounds.append((-1e12, 1e12))
    all_bounds = np.expand_dims(np.array(all_bounds), 1)
    all_bounds = np.tile(all_bounds, [1, out_dim, 1])
    return np.reshape(all_bounds, [-1, 2])


def fit_quad_softmax(
    contexts, targets, initial_params, weights=None, maxeval=None, ftol_rel=None
):
    if weights is None:
        weights = np.ones([contexts.shape[0], 1]) / contexts.shape[0]
    if len(weights.shape) == 1:
        weights = np.expand_dims(weights, 1)

    params_shape = initial_params.shape
    n = contexts.shape[1]
    all_bounds = create_bounds_array(
        quad_feat_dim=n,
        feat_dim=initial_params.shape[0],
        out_dim=initial_params.shape[1],
    )

    obj = lambda p, g: _quad_objective(
        p, g, params_shape, contexts, targets, weights, n
    )
    opt = nlopt.opt(nlopt.LD_LBFGS, int(np.prod(params_shape)))
    opt.set_lower_bounds(all_bounds[:, 0])
    opt.set_upper_bounds(all_bounds[:, 1])

    if maxeval is not None:
        opt.set_maxeval(maxeval)
    if ftol_rel is not None:
        opt.set_ftol_rel(ftol_rel)
    opt.set_min_objective(obj)
    # try:
    res = opt.optimize(np.reshape(initial_params, -1))
    return np.reshape(res, params_shape)


def lr(context_bias, samples, weights=None):
    if weights is None:
        weights = np.ones(context_bias.shape[0]) / context_bias.shape[0]

    reg_mat = 1e-10 * np.eye(context_bias.shape[-1])
    reg_mat[-1, -1] = 0.0
    wcb = np.reshape(weights, [-1, 1]) * context_bias
    try:
        sol = np.linalg.solve(wcb.T @ context_bias + reg_mat, wcb.T @ samples)
    except np.linalg.LinAlgError as e:
        print(e)
        sol = np.linalg.lstsq(wcb.T @ context_bias + reg_mat, wcb.T @ samples)[0]
    return sol
