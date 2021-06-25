import jax.numpy as jnp
from jax import lax, jit, partial, random
from tqdm.auto import tqdm

from .kernels import build_sgld_kernel, build_psgld_kernel, build_sgldCV_kernel, build_sgld_SVRG_kernel
from .kernels import build_sghmc_kernel, build_sghmcCV_kernel, build_sghmc_SVRG_kernel#, build_sgnht_kernel
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

        for k in tqdm(range(Nsamples)):
            key, subkey = random.split(key)
            state = my_kernel(0, subkey, state)
            samples.append(get_params(state))

        return samples
    return sampler


def build_sgld_sampler(dt, loglikelihood, logprior, data, batch_size, compiled=True):
    init_fn, my_kernel, get_params = build_sgld_kernel(dt, loglikelihood, logprior, data, batch_size)
    if compiled:
        return _build_compiled_sampler(init_fn, my_kernel, get_params)
    else:
        return _build_noncompiled_sampler(init_fn, my_kernel, get_params)

def build_sgldCV_sampler(dt, loglikelihood, logprior, data, batch_size, centering_value, compiled=True):
    init_fn, my_kernel, get_params = build_sgldCV_kernel(dt, loglikelihood, logprior,
                                        data, batch_size, centering_value)
    if compiled:
        return _build_compiled_sampler(init_fn, my_kernel, get_params)
    else:
        return _build_noncompiled_sampler(init_fn, my_kernel, get_params)

def build_sgld_SVRG_sampler(dt, loglikelihood, logprior, data, batch_size, centering_value, update_rate, compiled=True):
    init_fn, my_kernel, get_params = build_sgld_SVRG_kernel(dt, loglikelihood, logprior, data,
                                        batch_size, centering_value, update_rate)
    if compiled:
        return _build_compiled_sampler(init_fn, my_kernel, get_params)
    else:
        return _build_noncompiled_sampler(init_fn, my_kernel, get_params)

def build_psgld_sampler(dt, loglikelihood, logprior, data, batch_size, compiled=True):
    init_fn, my_kernel, get_params = build_psgld_kernel(dt, loglikelihood, logprior,
                                        data, batch_size)
    if compiled:
        return _build_compiled_sampler(init_fn, my_kernel, get_params)
    else:
        return _build_noncompiled_sampler(init_fn, my_kernel, get_params)

def build_sghmc_sampler(dt, L, loglikelihood, logprior, data, batch_size,
                        alpha=0.01, compiled=True):
    init_fn, my_kernel, get_params = build_sghmc_kernel(dt, L, loglikelihood, logprior,
                                        data, batch_size, alpha)
    if compiled:
        return _build_compiled_sampler(init_fn, my_kernel, get_params)
    else:
        return _build_noncompiled_sampler(init_fn, my_kernel, get_params)

def build_sghmcCV_sampler(dt, L, loglikelihood, logprior, data, batch_size,
                        centering_value, alpha=0.01, compiled=True):
    init_fn, my_kernel, get_params = build_sghmcCV_kernel(dt, L, loglikelihood, logprior,
                                        data, batch_size, centering_value, alpha)
    if compiled:
        return _build_compiled_sampler(init_fn, my_kernel, get_params)
    else:
        return _build_noncompiled_sampler(init_fn, my_kernel, get_params)

def build_sghmc_SVRG_sampler(dt, L, loglikelihood, logprior, data, batch_size,
                        centering_value, update_rate, alpha=0.01, compiled=True):
    init_fn, my_kernel, get_params = build_sghmc_SVRG_kernel(dt, L, loglikelihood, logprior,
                                data, batch_size, centering_value, update_rate, alpha)
    if compiled:
        return _build_compiled_sampler(init_fn, my_kernel, get_params)
    else:
        return _build_noncompiled_sampler(init_fn, my_kernel, get_params)
