from collections import namedtuple
import functools
import jax.numpy as jnp
from jax import jit, lax, random

from .diffusions import sgld, psgld, sgldAdam, sghmc, baoab, sgnht, badodab
from .gradient_estimation import build_gradient_estimation_fn, build_gradient_estimation_fn_CV, build_gradient_estimation_fn_SVRG
from .util import build_grad_log_post

def _build_langevin_kernel(update, get_params, estimate_gradient):
    "build generic kernel"
    @jit
    def kernel(i, key, state):
        k1, k2 = random.split(key)
        param_grad = estimate_gradient(k1, get_params(state))
        state = update(i, k2, param_grad, state)
        return state

    return kernel


def _build_sghmc_kernel(L, update, get_params, resample_momentum, estimate_gradient, compiled_leapfrog=True):
    "Build generic sghmc kernel"

    def sghmc_kernel(i, key, state):
        def body(state, key):
            k1, k2 = random.split(key)
            g = estimate_gradient(k1, get_params(state))
            state = update(i, k2, g, state)
            return state, None

        k1, k2 = random.split(key)
        state = resample_momentum(i, k1, state)
        keys = random.split(k2, L)
        if compiled_leapfrog:
            state, _ = lax.scan(body, state, keys)
            return state
        else:
            for key in keys:
                state, _ = body(state, key)
            return state
    if compiled_leapfrog:
        return jit(sghmc_kernel)
    else:
        return sghmc_kernel

# generic SVRG kernel builders:
def _build_langevin_SVRG_kernel(update, get_params, estimate_gradient):
    "build generic overdamped SVRG kernel"

    @jit
    def kernel(i, key, state):
        state_params, state_svrg = state
        k1, k2 = random.split(key)
        param_grad, state_svrg = estimate_gradient(k1, get_params(state_params), i, state_svrg)
        state_params = update(i, k2, param_grad, state_params)
        return (state_params, state_svrg)

    return kernel


def _build_sghmc_SVRG_kernel(L, update, get_params, resample_momentum, estimate_gradient, compiled_leapfrog=True):
    "Build generic sghmc SVRG kernel"
    def body(state, x):
        i, key = x
        state_params, state_svrg = state
        k1, k2 = random.split(key)
        g, state_svrg = estimate_gradient(k1, get_params(state_params), i, state_svrg)
        state_params = update(i, k2, g, state_params)
        return (state_params, state_svrg), None

    def sghmc_kernel(i, key, state):
        k1, k2 = random.split(key)
        state_params, state_svrg = state
        state_params = resample_momentum(i, k1, state_params)
        keys = random.split(k2, L)
        state = (state_params, state_svrg)
        if compiled_leapfrog:
            state, _ = lax.scan(body, state, (jnp.arange(L), keys))
            return state
        else:
            for key in keys:
                state, _ = body(state, key)
            return state

    return sghmc_kernel

def _build_palindrome_kernel(update1, update2, get_params, estimate_gradient):
    "build generic palindrome kernel"

    @jit
    def kernel(i, key, state):
        state_params, param_grad = state
        k1, k2 = random.split(key)
        state_params = update1(i, k2, param_grad, state_params)
        param_grad = estimate_gradient(k1, get_params(state_params))
        state_params = update2(i, k2, param_grad, state_params) # the random key k2 isn't used in update2
        return (state_params, param_grad)

    return kernel

# =======
# kernels
# =======

def build_sgld_kernel(dt, loglikelihood, logprior, data, batch_size):
    grad_log_post = build_grad_log_post(loglikelihood, logprior, data)
    init_fn, update, get_params = sgld(dt)
    estimate_gradient = build_gradient_estimation_fn(grad_log_post, data, batch_size)
    sgld_kernel = _build_langevin_kernel(update, get_params, estimate_gradient)
    return init_fn, sgld_kernel, get_params


def build_sgldCV_kernel(dt, loglikelihood, logprior, data, batch_size, centering_value):
    grad_log_post = build_grad_log_post(loglikelihood, logprior, data)
    init_fn, update, get_params = sgld(dt)
    estimate_gradient = build_gradient_estimation_fn_CV(grad_log_post, data, batch_size, centering_value)
    sgldCV_kernel = _build_langevin_kernel(update, get_params, estimate_gradient)
    return init_fn, sgldCV_kernel, get_params


def build_sgld_SVRG_kernel(dt, loglikelihood, logprior, data, batch_size, centering_value, update_rate):
    grad_log_post = build_grad_log_post(loglikelihood, logprior, data)
    init_fn, update, get_params = sgld(dt)
    estimate_gradient, state_SVRG = build_gradient_estimation_fn_SVRG(grad_log_post, data,
                                                          batch_size, centering_value, update_rate)
    sgld_kernel = _build_langevin_SVRG_kernel(update, get_params, estimate_gradient)

    def new_init_fn(params):
        return (init_fn(params), state_SVRG)

    def new_get_params(state):
        state_params, _ = state
        return get_params(state_params)

    return new_init_fn, sgld_kernel, new_get_params

def build_psgld_kernel(dt, loglikelihood, logprior, data, batch_size, alpha=0.99, eps=1e-5):
    grad_log_post = build_grad_log_post(loglikelihood, logprior, data)
    init_fn, update, get_params = psgld(dt, alpha, eps)
    estimate_gradient = build_gradient_estimation_fn(grad_log_post, data, batch_size)
    sgld_kernel = _build_langevin_kernel(update, get_params, estimate_gradient)
    return init_fn, sgld_kernel, get_params


def build_sgldAdam_kernel(dt, loglikelihood, logprior, data, batch_size, beta1=0.9, beta2=0.999, eps=1e-8):
    grad_log_post = build_grad_log_post(loglikelihood, logprior, data)
    init_fn, update, get_params = sgldAdam(dt, beta1, beta2, eps)
    estimate_gradient = build_gradient_estimation_fn(grad_log_post, data, batch_size)
    sgldAdam_kernel = _build_langevin_kernel(update, get_params, estimate_gradient)
    return init_fn, sgldAdam_kernel, get_params

def build_sghmc_kernel(dt, L, loglikelihood, logprior, data, batch_size, alpha=0.01, compiled_leapfrog=True):
    grad_log_post = build_grad_log_post(loglikelihood, logprior, data)
    init_fn, update, get_params, resample_momentum = sghmc(dt, alpha)
    estimate_gradient = build_gradient_estimation_fn(grad_log_post, data, batch_size)
    sghmc_kernel = _build_sghmc_kernel(L, update, get_params, resample_momentum,
                                    estimate_gradient, compiled_leapfrog=compiled_leapfrog)
    return init_fn, sghmc_kernel, get_params

def build_sghmcCV_kernel(dt, L, loglikelihood, logprior, data, batch_size,
                            centering_value, alpha=0.01, compiled_leapfrog=True):
    grad_log_post = build_grad_log_post(loglikelihood, logprior, data)
    init_fn, update, get_params, resample_momentum = sghmc(dt, alpha)
    estimate_gradient = build_gradient_estimation_fn_CV(grad_log_post, data, batch_size, centering_value)
    sghmc_kernel = _build_sghmc_kernel(L, update, get_params, resample_momentum,
                            estimate_gradient, compiled_leapfrog=compiled_leapfrog)
    return init_fn, sghmc_kernel, get_params


def build_sghmc_SVRG_kernel(dt, L, loglikelihood, logprior, data, batch_size,
                                centering_value, update_rate, alpha=0.01, compiled_leapfrog=True):
    grad_log_post = build_grad_log_post(loglikelihood, logprior, data)
    init_fn, update, get_params, resample_momentum = sghmc(dt, alpha)
    estimate_gradient, state_SVRG = build_gradient_estimation_fn_SVRG(grad_log_post, data,
                                                          batch_size, centering_value, update_rate)
    sghmc_kernel = _build_sghmc_SVRG_kernel(L, update, get_params, resample_momentum,
                            estimate_gradient, compiled_leapfrog=compiled_leapfrog)

    def new_init_fn(params):
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

    def new_init_fn(params):
        param_grads = grad_log_post(params, *data)
        return (init_fn(params), param_grads)

    def new_get_params(state):
        state_params, _ = state
        return get_params(state_params)

    return new_init_fn, baoab_kernel, new_get_params

def build_sgnht_kernel(dt, loglikelihood, logprior, data, batch_size, a=0.01):
    grad_log_post = build_grad_log_post(loglikelihood, logprior, data)
    init_fn, update, get_params = sgnht(dt, a)
    estimate_gradient = build_gradient_estimation_fn(grad_log_post, data, batch_size)
    sgnht_kernel = _build_langevin_kernel(update, get_params, estimate_gradient)
    return init_fn, sgnht_kernel, get_params


def build_badodab_kernel(dt, loglikelihood, logprior, data, batch_size, a=0.01):
    grad_log_post = build_grad_log_post(loglikelihood, logprior, data)
    init_fn, update1, update2, get_params = badodab(dt, a)
    estimate_gradient = build_gradient_estimation_fn(grad_log_post, data, batch_size)
    badodab_kernel = _build_palindrome_kernel(update1, update2, get_params, estimate_gradient)

    def new_init_fn(params):
        param_grads = grad_log_post(params, *data)
        return (init_fn(params), param_grads)

    def new_get_params(state):
        state_params, _ = state
        return get_params(state_params)

    return new_init_fn, badodab_kernel, new_get_params
