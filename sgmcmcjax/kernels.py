from typing import NamedTuple, Union, Any, Callable, Optional

import jax.numpy as jnp
from jax import jit, lax, random

from .diffusions import sgld, psgld, sgldAdam, sghmc, baoab, sgnht, badodab, DiffusionState
from .gradient_estimation import build_gradient_estimation_fn, build_gradient_estimation_fn_CV, build_gradient_estimation_fn_SVRG, SVRGState
from .util import build_grad_log_post, run_loop
from .types import PyTree, PRNGKey, SamplerState, SVRGState


def _build_langevin_kernel(init_fn_diffusion: Callable, update_diffusion: Callable, get_params_diffusion: Callable, estimate_gradient: Callable):
    "build generic kernel"

    def init_fn(key: PRNGKey, params:PyTree):
        diffusion_state = init_fn_diffusion(params)
        state = SamplerState(diffusion_state, params) # pass in params in place of param_grads as you don't have it yet
        param_grad, svrg_state = estimate_gradient(0, key, state, get_params_diffusion)
        return SamplerState(diffusion_state, param_grad, svrg_state, None)

    def kernel(i: int, key: PRNGKey, state: SamplerState) -> SamplerState:
        diffusion_state, param_grad, svrg_state, grad_info = state
        k1, k2 = random.split(key)
        diffusion_state = update_diffusion(i, k1, param_grad, diffusion_state)
        param_grad, svrg_state = estimate_gradient(
                                                i,
                                                k2,
                                                SamplerState(diffusion_state, param_grad, svrg_state, grad_info),
                                                get_params_diffusion
                                            )
        return SamplerState(diffusion_state, param_grad, svrg_state, grad_info)

    def get_params(state: SamplerState):
        return get_params_diffusion(state.diffusion_state)

    return init_fn, kernel, get_params


# generic SVRG kernel builders:
# def _build_langevin_SVRG_kernel(update: Callable, get_params: Callable, estimate_gradient: Callable):
#     "build generic overdamped SVRG kernel"
#
#     @jit
#     def kernel(i, key, state):
#         state_params, state_svrg = state
#         k1, k2 = random.split(key)
#         param_grad, state_svrg = estimate_gradient(i, k1, SamplerState(diffusion_state, param_grad, state_svrg, grad_info), get_params(state_params))
#         state_params = update(i, k2, param_grad, state_params)
#         return (state_params, state_svrg)
#
#     return kernel



def _build_sghmc_kernel(L, update, get_params, resample_momentum, estimate_gradient, compiled_leapfrog=True):
    "Build generic sghmc kernel"

    def sghmc_kernel(i, key, state):
        def body(state, key):
            k1, k2 = random.split(key)
            g, _ = estimate_gradient(i, k1, SamplerState(diffusion_state, param_grad), get_params)
            state = update(i, k2, g, state)
            return state, None

        k1, k2 = random.split(key)
        state = resample_momentum(i, k1, state)
        keys = random.split(k2, L)
        state = run_loop(body, state, keys, compiled_leapfrog)
        return state
    if compiled_leapfrog:
        return jit(sghmc_kernel)
    else:
        return sghmc_kernel


def _build_sghmc_SVRG_kernel(L, update, get_params, resample_momentum, estimate_gradient, compiled_leapfrog=True):
    "Build generic sghmc SVRG kernel"
    def body(state, x):
        i, key = x
        state_params, state_svrg = state
        k1, k2 = random.split(key)
        g, state_svrg = estimate_gradient(i, k1, SamplerState(diffusion_state, param_grad, state_svrg), get_params)
        state_params = update(i, k2, g, state_params)
        return (state_params, state_svrg), None

    def sghmc_kernel(i, key, state):
        k1, k2 = random.split(key)
        state_params, state_svrg = state
        state_params = resample_momentum(i, k1, state_params)
        keys = random.split(k2, L)
        state = (state_params, state_svrg)
        state = run_loop(body, state, (jnp.arange(L), keys), compiled_leapfrog)
        return state

    return sghmc_kernel

def _build_palindrome_kernel(update1, update2, get_params, estimate_gradient):
    "build generic palindrome kernel"

    @jit
    def kernel(i, key, state):
        state_params, param_grad = state
        k1, k2, k3 = random.split(key, 3)
        state_params = update1(i, k1, param_grad, state_params)
        param_grad, _ = estimate_gradient(i, k1, SamplerState(diffusion_state, param_grad, state_svrg), get_params)
        state_params = update2(i, k3, param_grad, state_params)
        return (state_params, param_grad)

    return kernel

# =======
# kernels
# =======

def build_sgld_kernel(dt, loglikelihood, logprior, data, batch_size):
    grad_log_post = build_grad_log_post(loglikelihood, logprior, data)
    estimate_gradient = build_gradient_estimation_fn(grad_log_post, data, batch_size)
    init_fn, sgld_kernel, get_params = _build_langevin_kernel(*sgld(dt), estimate_gradient)
    return init_fn, sgld_kernel, get_params

def build_sgldCV_kernel(dt, loglikelihood, logprior, data, batch_size, centering_value):
    grad_log_post = build_grad_log_post(loglikelihood, logprior, data)
    estimate_gradient = build_gradient_estimation_fn_CV(grad_log_post, data, batch_size, centering_value)
    init_fn, sgldCV_kernel, get_params = _build_langevin_kernel(*sgld(dt), estimate_gradient)
    return init_fn, sgldCV_kernel, get_params

def build_sgld_SVRG_kernel(dt, loglikelihood, logprior, data, batch_size, centering_value, update_rate):
    grad_log_post = build_grad_log_post(loglikelihood, logprior, data)
    estimate_gradient = build_gradient_estimation_fn_SVRG(grad_log_post, data, batch_size, centering_value, update_rate)
    init_fn, sgldSVRG_kernel, get_params = _build_langevin_kernel(*sgld(dt), estimate_gradient)
    return init_fn, sgldSVRG_kernel, get_params


# def build_sgld_SVRG_kernel(dt, loglikelihood, logprior, data, batch_size, centering_value, update_rate):
#     grad_log_post = build_grad_log_post(loglikelihood, logprior, data)
#     init_fn, update, get_params = sgld(dt)
    # estimate_gradient, state_SVRG = build_gradient_estimation_fn_SVRG(grad_log_post, data,
    #                                                       batch_size, centering_value, update_rate)
    # sgld_kernel = _build_langevin_SVRG_kernel(update, get_params, estimate_gradient)

#     def new_init_fn(key, params):
#         return (init_fn(params), state_SVRG)
#
#     def new_get_params(state):
#         state_params, _ = state
#         return get_params(state_params)
#
#     return new_init_fn, sgld_kernel, new_get_params

def build_psgld_kernel(dt, loglikelihood, logprior, data, batch_size, alpha=0.99, eps=1e-5):
    grad_log_post = build_grad_log_post(loglikelihood, logprior, data)
    estimate_gradient = build_gradient_estimation_fn(grad_log_post, data, batch_size)
    init_fn, sgld_kernel, get_params = _build_langevin_kernel(*psgld(dt, alpha, eps), estimate_gradient)
    return init_fn, sgld_kernel, get_params


def build_sgldAdam_kernel(dt, loglikelihood, logprior, data, batch_size, beta1=0.9, beta2=0.999, eps=1e-8):
    grad_log_post = build_grad_log_post(loglikelihood, logprior, data)
    estimate_gradient = build_gradient_estimation_fn(grad_log_post, data, batch_size)
    init_fn, sgldAdam_kernel, get_params = _build_langevin_kernel(*sgldAdam(dt, beta1, beta2, eps), estimate_gradient)
    return init_fn, sgldAdam_kernel, get_params

def build_sghmc_kernel(dt, L, loglikelihood, logprior, data, batch_size, alpha=0.01, compiled_leapfrog=True):
    grad_log_post = build_grad_log_post(loglikelihood, logprior, data)
    init_fn, update, get_params, resample_momentum = sghmc(dt, alpha)
    estimate_gradient = build_gradient_estimation_fn(grad_log_post, data, batch_size)
    sghmc_kernel = _build_sghmc_kernel(L, update, get_params, resample_momentum,
                                    estimate_gradient, compiled_leapfrog=compiled_leapfrog)
    def new_init_fn(key, params):
        return init_fn(params)

    return new_init_fn, sghmc_kernel, get_params

def build_sghmcCV_kernel(dt, L, loglikelihood, logprior, data, batch_size,
                            centering_value, alpha=0.01, compiled_leapfrog=True):
    grad_log_post = build_grad_log_post(loglikelihood, logprior, data)
    init_fn, update, get_params, resample_momentum = sghmc(dt, alpha)
    estimate_gradient = build_gradient_estimation_fn_CV(grad_log_post, data, batch_size, centering_value)
    sghmc_kernel = _build_sghmc_kernel(L, update, get_params, resample_momentum,
                            estimate_gradient, compiled_leapfrog=compiled_leapfrog)

    def new_init_fn(key, params):
        return init_fn(params)

    return new_init_fn, sghmc_kernel, get_params


def build_sghmc_SVRG_kernel(dt, L, loglikelihood, logprior, data, batch_size,
                                centering_value, update_rate, alpha=0.01, compiled_leapfrog=True):
    grad_log_post = build_grad_log_post(loglikelihood, logprior, data)
    init_fn, update, get_params, resample_momentum = sghmc(dt, alpha)
    estimate_gradient = build_gradient_estimation_fn_SVRG(grad_log_post, data,
                                                          batch_size, centering_value, update_rate)
    sghmc_kernel = _build_sghmc_SVRG_kernel(L, update, get_params, resample_momentum,
                            estimate_gradient, compiled_leapfrog=compiled_leapfrog)

    def new_init_fn(key, params):
        return (init_fn(params), state_SVRG)

    def new_get_params(state):
        state_params, _ = state
        return get_params(state_params)

    return new_init_fn, sghmc_kernel, new_get_params


def build_baoab_kernel(dt, gamma, loglikelihood, logprior, data, batch_size, tau=1):
    grad_log_post = build_grad_log_post(loglikelihood, logprior, data)
    init_fn, update1, update2, get_params = baoab(dt, gamma)
    estimate_gradient = build_gradient_estimation_fn(grad_log_post, data, batch_size)
    baoab_kernel = _build_palindrome_kernel(update1, update2, get_params, estimate_gradient)

    def new_init_fn(key, params):
        param_grads = grad_log_post(params, *data)
        return (init_fn(params), param_grads)

    def new_get_params(state):
        state_params, _ = state
        return get_params(state_params)

    return new_init_fn, baoab_kernel, new_get_params

def build_sgnht_kernel(dt, loglikelihood, logprior, data, batch_size, a=0.01):
    grad_log_post = build_grad_log_post(loglikelihood, logprior, data)
    estimate_gradient = build_gradient_estimation_fn(grad_log_post, data, batch_size)
    init_fn, sgnht_kernel, get_params = _build_langevin_kernel(*sgnht(dt, a), estimate_gradient)
    return init_fn, sgnht_kernel, get_params


def build_badodab_kernel(dt, loglikelihood, logprior, data, batch_size, a=0.01):
    grad_log_post = build_grad_log_post(loglikelihood, logprior, data)
    init_fn, update1, update2, get_params = badodab(dt, a)
    estimate_gradient = build_gradient_estimation_fn(grad_log_post, data, batch_size)
    badodab_kernel = _build_palindrome_kernel(update1, update2, get_params, estimate_gradient)

    def new_init_fn(key, params):
        param_grads = grad_log_post(params, *data)
        return (init_fn(params), param_grads)

    def new_get_params(state):
        state_params, _ = state
        return get_params(state_params)

    return new_init_fn, badodab_kernel, new_get_params
