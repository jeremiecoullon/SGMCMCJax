from typing import Tuple, Callable

import jax.numpy as jnp
from jax import grad, jit, vmap, lax, value_and_grad
from jax.experimental import host_callback
from tqdm.auto import tqdm

def build_grad_log_post(loglikelihood: Callable, logprior: Callable, data: Tuple, with_val: bool = False) -> Callable:
    if len(data)==1:
        batch_loglik = jit(vmap(loglikelihood, in_axes=(None, 0)))
    elif len(data)==2:
        batch_loglik = jit(vmap(loglikelihood, in_axes=(None, 0,0)))
    else:
        raise ValueError("Data must be a tuple of size 1 or 2")

    Ndata = data[0].shape[0]
    def log_post(param, *args):
        return logprior(param) + Ndata*jnp.mean(batch_loglik(param, *args), axis=0)

    if with_val:
        grad_log_post = jit(value_and_grad(log_post))
    else:
        grad_log_post = jit(grad(log_post))
    return grad_log_post

def run_loop(f, state, xs, compiled=True):
    """
    Loop over an iterable and keep only the final state
    the function `f` should return `(state, None)`
    compiled: whether or not to run lax.scan or a Python loop
    """
    if compiled:
        state, _ = lax.scan(f, state, xs)
        return state
    else:
        for x in xs:
            state, _ = f(state, x)
        return state


def progress_bar_scan(num_samples, message=None):
    "Progress bar for a JAX scan"
    if message is None:
            message = f"Running for {num_samples:,} iterations"
    tqdm_bars = {}

    def _define_tqdm(arg, transform):
        tqdm_bars[0] = tqdm(range(num_samples))
        tqdm_bars[0].set_description(message, refresh=False)

    def _update_tqdm(arg, transform):
        tqdm_bars[0].update(arg)

    def _close_tqdm(arg, transform):
        tqdm_bars[0].close()

    def _update_progress_bar(iter_num, print_rate):
        "Updates tqdm progress bar of a JAX scan or loop"
        _ = lax.cond(
            iter_num == 0,
            lambda _: host_callback.id_tap(_define_tqdm, print_rate, result=iter_num),
            lambda _: iter_num,
            operand=None,
        )

        _ = lax.cond(
            iter_num % print_rate == 0,
            lambda _: host_callback.id_tap(_update_tqdm, print_rate, result=iter_num),
            lambda _: iter_num,
            operand=None,
        )

        _ = lax.cond(
            iter_num == num_samples-1,
            lambda _: host_callback.id_tap(_close_tqdm, print_rate, result=iter_num),
            lambda _: iter_num,
            operand=None,
        )

    def _progress_bar_scan(func):
        """Decorator that adds a progress bar to `body_fun` used in `lax.scan`.
        Note that `body_fun` must either be looping over `np.arange(num_samples)`,
        or be looping over a tuple who's first element is `np.arange(num_samples)`
        This means that `iter_num` is the current iteration number
        """
        print_rate = int(num_samples / 20)

        def wrapper_progress_bar(carry, x):
            if type(x) is tuple:
                iter_num, *_ = x
            else:
                iter_num = x
            _update_progress_bar(iter_num, print_rate)
            return func(carry, x)

        return wrapper_progress_bar

    return _progress_bar_scan
