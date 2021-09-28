from typing import Tuple, Callable
from jax.tree_util import tree_map
from jax.experimental.optimizers import adam
from jax import jit, lax, random
import jax.numpy as jnp

from .gradient_estimation import build_gradient_estimation_fn
from .util import progress_bar_scan, build_grad_log_post


def build_adam_optimizer(dt: float, loglikelihood: Callable, logprior: Callable, data: Tuple, batch_size: int) -> Callable:
    grad_log_post = build_grad_log_post(loglikelihood, logprior, data, with_val=True)
    estimate_gradient, _ = build_gradient_estimation_fn(grad_log_post, data, batch_size)

    init_fn, update, get_params = adam(dt)

    @jit
    def body(carry, i):
        key, state = carry
        key, subkey = random.split(key)
        (lp_val, param_grad), _ = estimate_gradient(i, subkey, get_params(state))
        neg_param_grad = tree_map(lambda x: -x, param_grad)
        state = update(i, neg_param_grad, state)
        return (key, state), lp_val

    def run_adam(key, Niters, params_IC):
        state = init_fn(params_IC)
        body_pbar = progress_bar_scan(Niters)(body)
        (_, state_opt), logpost_array = lax.scan(body_pbar, (key, state), jnp.arange(Niters))
        return get_params(state_opt), logpost_array
    return run_adam
