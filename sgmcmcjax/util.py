from typing import Any, Callable, Iterable, Tuple, Union

import jax.numpy as jnp
from jax import grad, jit, lax, value_and_grad, vmap
from jax.experimental import host_callback
from tqdm.auto import tqdm


def build_grad_log_post(
    loglikelihood: Callable, logprior: Callable, data: Tuple, with_val: bool = False
) -> Callable:
    """Build the gradient of the log-posterior.
    The returned function has signature:

    grad_lost_post (Callable)
        Args:
            param (Pytree): parameters to evaluate the log-posterior at
            args: data (either minibatch or fullbatch) to pass in to the log-likelihood
        Returns:
            gradient of the log-posterior (PyTree), and optionally the value of the log-posterior (float)

    Args:
        loglikelihood (Callable): log-likelihood for a single data point
        logprior (Callable): log-prior for a single data point
        data (Tuple): tuple of data. It should either have a single array (for unsupervised problems) or have two arrays (for supervised problems)
        with_val (bool, optional): Whether or not the returned function also inclues the value of the log-posterior as well as the value of the gradient. Defaults to False.

    Raises:
        ValueError: the 'data' argument should either be a tuple of size 1 or 2

    Returns:
        Callable: The gradient of the log-posterior
    """
    if len(data) == 1:
        batch_loglik = jit(vmap(loglikelihood, in_axes=(None, 0)))
    elif len(data) == 2:
        batch_loglik = jit(vmap(loglikelihood, in_axes=(None, 0, 0)))
    else:
        raise ValueError("'data' must be a tuple of size 1 or 2")

    Ndata = data[0].shape[0]

    def log_post(param, *args):
        return logprior(param) + Ndata * jnp.mean(batch_loglik(param, *args), axis=0)

    if with_val:
        grad_log_post = jit(value_and_grad(log_post))
    else:
        grad_log_post = jit(grad(log_post))
    return grad_log_post


def run_loop(f: Callable, state: Any, xs: Iterable, compiled: bool = True) -> Any:
    """Loop over an iterable and keep only the final state
    the function `f` should return `(state, None)`
    compiled: whether or not to run lax.scan or a Python loop

    Args:
        f (Callable): function to apply at every iteration
        state (Any): state of the system
        xs (iter): Iteratable to loop over
        compiled (Bool, optional): whether or not the loop is performed with lax.scan or not. Otherwise run a native Python loop. Defaults to True.

    Returns:
        Any: final state of the system
    """
    if compiled:
        state, _ = lax.scan(f, state, xs)
        return state
    else:
        for x in xs:
            state, _ = f(state, x)
        return state


def progress_bar_scan(num_samples: int, message: Union[None, str] = None) -> Callable:
    """Decorator factory to build a tqdm progress bar using lax.scan.
    This returns a decorator to apply to the 'body' function used in lax.scan

    Args:
        num_samples (int): number of samples to run lax.scan
        message (Union[None, str]): message to display in the progress bar. Defaults to f'Running for {num_samples:,} iterations'

    Returns:
        Callable: decorator to apply to the body function used in lax.scan
    """

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
            iter_num == num_samples - 1,
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
