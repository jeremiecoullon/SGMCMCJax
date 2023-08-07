from typing import Callable, Tuple

import jax.numpy as jnp
import optax
from jax import jit, lax, random
from jax.tree_util import tree_map

from .gradient_estimation import build_gradient_estimation_fn
from .util import build_grad_log_post, progress_bar_scan


def build_optax_optimizer(
    optimizer: optax.GradientTransformation,
    loglikelihood: Callable,
    logprior: Callable,
    data: Tuple,
    batch_size: int,
) -> Callable:
    """build Optax optimizer. See Optax: https://github.com/deepmind/optax

    Args:
        optimizer (optax.GradientTransformation): Optax GradientTransformation object
        loglikelihood (Callable): log-likelihood for a single data point
        logprior (Callable): log-prior for a single data point
        data (Tuple): tuple of data. It should either have a single array (for unsupervised problems) or have two arrays (for supervised problems)
        batch_size (int): batch size

    Returns:
        Callable: optimizer function with signature:
            Args:
                key (PRNGKey): random key

                Niters (int): number of iterations

                params_IC (PyTree): initial parameters
            Returns:
                PyTree: final parameters

                jnp.ndarray: array of log-posterior values during the optimization
    """
    grad_log_post = build_grad_log_post(loglikelihood, logprior, data, with_val=True)
    estimate_gradient, _ = build_gradient_estimation_fn(grad_log_post, data, batch_size)

    @jit
    def body(carry, i):
        key, state, params = carry
        key, subkey = random.split(key)
        (lp_val, param_grad), _ = estimate_gradient(i, subkey, params)
        neg_param_grad = tree_map(lambda x: -x, param_grad)
        updates, state = optimizer.update(neg_param_grad, state)
        params = optax.apply_updates(params, updates)
        return (key, state, params), lp_val

    def run_optimizer(key, Niters, params):
        state = optimizer.init(params)
        body_pbar = progress_bar_scan(Niters)(body)
        (key, state, params), logpost_array = lax.scan(
            body_pbar, (key, state, params), jnp.arange(Niters)
        )
        return params, logpost_array

    return run_optimizer
