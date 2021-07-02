import jax.numpy as jnp
from jax import lax, jit, partial, random
from tqdm.auto import tqdm
import functools

from .kernels import build_sgld_kernel, build_psgld_kernel, build_sgldCV_kernel, build_sgld_SVRG_kernel
from .kernels import build_sghmc_kernel, build_sghmcCV_kernel, build_sghmc_SVRG_kernel, build_baoab_kernel
from .kernels import build_sgnht_kernel, build_badodab_kernel
from .util import progress_bar_scan

"""
Samplers: sgld, sgld_CV, sgld_SVRG, psgld, sghmc, sghmc_CV, sghmc_SVRG
"""

def _build_compiled_sampler(init_fn, my_kernel, get_params):
    "Build generic compiled sampler"
    @partial(jit, static_argnums=(1,))
    def sampler(key, Nsamples, params):

        @progress_bar_scan(Nsamples)
        def body(carry, i):
            key, state = carry
            key, subkey = random.split(key)
            state = my_kernel(i, subkey, state)
            return (key, state), get_params(state)

        state = init_fn(params)
        (_, _), samples = lax.scan(body, (key, state), jnp.arange(Nsamples))
        return samples
    return sampler

def _build_noncompiled_sampler(init_fn, my_kernel, get_params):
    "Build generic non-compiled sampler"
    def sampler(key, Nsamples, params):
        samples = []
        state = init_fn(params)

        for i in tqdm(range(Nsamples)):
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
        init_fn, my_kernel, get_params = build_sampler_fn(*args, **kwargs)
        if compiled:
            return _build_compiled_sampler(init_fn, my_kernel, get_params)
        else:
            return _build_noncompiled_sampler(init_fn, my_kernel, get_params)

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
build_badodab_sampler = sgmcmc_sampler(build_badodab_kernel)
