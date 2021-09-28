from typing import NamedTuple, Union, Any, Callable, Optional, Tuple
from functools import partial
import jax.numpy as jnp
from jax import lax, jit, random
from tqdm.auto import tqdm
import functools

from .kernels import build_sgld_kernel, build_psgld_kernel, build_sgldCV_kernel, build_sgld_SVRG_kernel
from .kernels import build_sghmc_kernel, build_sghmcCV_kernel, build_sghmc_SVRG_kernel, build_baoab_kernel
from .kernels import build_sgnht_kernel, build_badodab_kernel, build_sgnhtCV_kernel, build_badodabCV_kernel
from .util import progress_bar_scan
from .types import PyTree, PRNGKey, SamplerState, SVRGState, DiffusionState

"""
Samplers: sgld, sgld_CV, sgld_SVRG, psgld, sghmc, sghmc_CV, sghmc_SVRG
"""

def _build_compiled_sampler(init_fn: Callable[[PRNGKey, PyTree], SamplerState],
                            my_kernel: Callable[[int, PRNGKey, SamplerState], SamplerState],
                            get_params: Callable[[SamplerState], PyTree],
                            pbar: bool = True) -> Callable:
    "Build generic compiled sampler"
    @partial(jit, static_argnums=(1,))
    def sampler(key, Nsamples, params):

        def body(carry, i):
            key, state = carry
            key, subkey = random.split(key)
            state = my_kernel(i, subkey, state)
            return (key, state), get_params(state)

        key, subkey = random.split(key)
        state = init_fn(subkey, params)

        lebody = progress_bar_scan(Nsamples)(body) if pbar else body
        (_, _), samples = lax.scan(lebody, (key, state), jnp.arange(Nsamples))
        return samples
    return sampler

def _build_noncompiled_sampler(init_fn: Callable[[PRNGKey, PyTree], SamplerState],
                            my_kernel: Callable[[int, PRNGKey, SamplerState], SamplerState],
                            get_params: Callable[[SamplerState], PyTree],
                            pbar: bool = True) -> Callable:
    "Build generic non-compiled sampler"
    def sampler(key, Nsamples, params):
        samples = []
        key, subkey = random.split(key)
        state = init_fn(subkey, params)

        _tqdm = tqdm(range(Nsamples)) if pbar else range(Nsamples)

        for i in _tqdm:
            key, subkey = random.split(key)
            state = my_kernel(i, subkey, state)
            samples.append(get_params(state))

        return samples
    return sampler

def sgmcmc_sampler(build_sampler_fn):
    """
    Decorator that turns a kernel factory into a sampler factory
    """
    @functools.wraps(build_sampler_fn)
    def wrapper(*args, **kwargs):
        compiled = kwargs.pop('compiled', True)
        pbar = kwargs.pop('pbar', True)
        init_fn, my_kernel, get_params = build_sampler_fn(*args, **kwargs)
        if compiled:
            return _build_compiled_sampler(init_fn, my_kernel, get_params, pbar=pbar)
        else:
            return _build_noncompiled_sampler(init_fn, jit(my_kernel), get_params, pbar=pbar)

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
