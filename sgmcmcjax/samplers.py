import functools
from functools import partial
from typing import Any, Callable, NamedTuple, Optional, Tuple

import jax.numpy as jnp
from jax import jit, lax, random
from tqdm.auto import tqdm

from .kernels import (
    build_badodab_kernel,
    build_badodabCV_kernel,
    build_baoab_kernel,
    build_psgld_kernel,
    build_sghmc_kernel,
    build_sghmc_SVRG_kernel,
    build_sghmcCV_kernel,
    build_sgld_kernel,
    build_sgld_SVRG_kernel,
    build_sgldCV_kernel,
    build_sgnht_kernel,
    build_sgnhtCV_kernel,
)
from .types import DiffusionState, PRNGKey, PyTree, SamplerState, SVRGState
from .util import progress_bar_scan


def _build_compiled_sampler(
    init_fn: Callable[[PRNGKey, PyTree], SamplerState],
    kernel: Callable[[int, PRNGKey, SamplerState], SamplerState],
    get_params: Callable[[SamplerState], PyTree],
    pbar: bool = True,
) -> Callable:
    """Build generic compiled sampler

    Args:
        init_fn (Callable[[PRNGKey, PyTree], SamplerState]): function to initialise the state of chain
        kernel (Callable[[int, PRNGKey, SamplerState], SamplerState]): transition kernel
        get_params (Callable[[SamplerState], PyTree]): functions that gets the target parameters from the state
        pbar (bool, optional): whether or not to display the progress bar. Defaults to True.

    Returns:
        Callable: sampling function with the same signature as kernel
    """

    @partial(jit, static_argnums=(1,))
    def sampler(key, Nsamples, params):
        def body(carry, i):
            key, state = carry
            key, subkey = random.split(key)
            state = kernel(i, subkey, state)
            return (key, state), get_params(state)

        key, subkey = random.split(key)
        state = init_fn(subkey, params)

        lebody = progress_bar_scan(Nsamples)(body) if pbar else body
        (_, _), samples = lax.scan(lebody, (key, state), jnp.arange(Nsamples))
        return samples

    return sampler


def _build_noncompiled_sampler(
    init_fn: Callable[[PRNGKey, PyTree], SamplerState],
    kernel: Callable[[int, PRNGKey, SamplerState], SamplerState],
    get_params: Callable[[SamplerState], PyTree],
    pbar: bool = True,
) -> Callable:
    """Build generic non-compiled sampler

    Args:
        init_fn (Callable[[PRNGKey, PyTree], SamplerState]): function to initialise the state of chain
        kernel (Callable[[int, PRNGKey, SamplerState], SamplerState]): transition kernel
        get_params (Callable[[SamplerState], PyTree]): functions that gets the target parameters from the state
        pbar (bool, optional): whether or not to display the progress bar. Defaults to True.

    Returns:
        Callable: sampling function with the same signature as kernel
    """

    def sampler(key, Nsamples, params):
        samples = []
        key, subkey = random.split(key)
        state = init_fn(subkey, params)

        _tqdm = tqdm(range(Nsamples)) if pbar else range(Nsamples)

        for i in _tqdm:
            key, subkey = random.split(key)
            state = kernel(i, subkey, state)
            samples.append(get_params(state))

        return samples

    return sampler


def sgmcmc_sampler(build_kernel_fn: Callable) -> Callable:
    """Decorator that turns a kernel factory into a sampler factory.

    These samplers have exactly the same signatures as the kernels they're built from (see sgmcmcjax.kernels).

    Args:
        build_kernel_fn (Callable): kernel factory

    Returns:
        Callable: sampling function with the same signature as build_kernel_fn
    """

    @functools.wraps(build_kernel_fn)
    def wrapper(*args, **kwargs):
        compiled = kwargs.pop("compiled", True)
        pbar = kwargs.pop("pbar", True)
        init_fn, my_kernel, get_params = build_kernel_fn(*args, **kwargs)
        if compiled:
            return _build_compiled_sampler(init_fn, my_kernel, get_params, pbar=pbar)
        else:
            return _build_noncompiled_sampler(
                init_fn, jit(my_kernel), get_params, pbar=pbar
            )

    return wrapper


build_sgld_sampler = sgmcmc_sampler(build_sgld_kernel)
build_sgldCV_sampler = sgmcmc_sampler(build_sgldCV_kernel)
build_sgld_SVRG_sampler = sgmcmc_sampler(build_sgld_SVRG_kernel)
build_psgld_sampler = sgmcmc_sampler(build_psgld_kernel)
build_sghmc_sampler = sgmcmc_sampler(build_sghmc_kernel)
build_sghmcCV_sampler = sgmcmc_sampler(build_sghmcCV_kernel)
build_sghmc_SVRG_sampler = sgmcmc_sampler(build_sghmc_SVRG_kernel)
build_baoab_sampler = sgmcmc_sampler(build_baoab_kernel)
build_sgnht_sampler = sgmcmc_sampler(build_sgnht_kernel)
build_sgnhtCV_sampler = sgmcmc_sampler(build_sgnhtCV_kernel)
build_badodab_sampler = sgmcmc_sampler(build_badodab_kernel)
build_badodabCV_sampler = sgmcmc_sampler(build_badodabCV_kernel)
