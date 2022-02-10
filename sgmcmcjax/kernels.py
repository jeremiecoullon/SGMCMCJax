from typing import Any, Callable, NamedTuple, Optional, Tuple

import jax.numpy as jnp
from jax import jit, lax, random

from .diffusions import badodab, baoab, psgld, sghmc, sgld, sgldAdam, sgnht
from .gradient_estimation import (SVRGState, build_gradient_estimation_fn,
                                  build_gradient_estimation_fn_CV,
                                  build_gradient_estimation_fn_SVRG)
from .types import DiffusionState, PRNGKey, PyTree, SamplerState, SVRGState
from .util import build_grad_log_post, run_loop


def _build_langevin_kernel(
    init_fn_diffusion: Callable,
    update_diffusion: Any,
    get_params_diffusion: Callable,
    estimate_gradient: Callable,
    init_gradient: Callable,
) -> Tuple[
    Callable[[PRNGKey, PyTree], SamplerState],
    Callable[[int, PRNGKey, SamplerState], SamplerState],
    Callable[[SamplerState], PyTree],
]:
    """Build a general Langevin kernel

    Args:
        init_fn_diffusion (Callable): create the initial state of the diffusion
        update_diffusion (Any): updates the state of the diffusion
        get_params_diffusion (Callable): gets the parameters from the state of the diffusion
        estimate_gradient (Callable): estimates the gradient of the log-posterior for the current parameters
        init_gradient (Callable): initialise the state of the gradient

    Returns:
        Tuple[ Callable[[PRNGKey, PyTree], SamplerState], Callable[[int, PRNGKey, SamplerState], SamplerState], Callable[[SamplerState], PyTree] ]: [description]
    """

    # Check whether the diffusion is a palindrome (ie: splitting scheme)
    if type(update_diffusion) == tuple:
        palindrome = True
        update_diffusion, update_diffusion2 = update_diffusion
    else:
        palindrome = False

    def init_fn(key: PRNGKey, params: PyTree):
        diffusion_state = init_fn_diffusion(params)
        param_grad, svrg_state = init_gradient(key, params)
        return SamplerState(diffusion_state, param_grad, svrg_state, None)

    def kernel(i: int, key: PRNGKey, state: SamplerState) -> SamplerState:
        diffusion_state, param_grad, svrg_state, grad_info = state
        k1, k2 = random.split(key)

        diffusion_state = update_diffusion(i, k1, param_grad, diffusion_state)
        param_grad, svrg_state = estimate_gradient(
            i, k2, get_params_diffusion(diffusion_state), svrg_state
        )
        if palindrome:
            diffusion_state = update_diffusion2(i, k1, param_grad, diffusion_state)

        return SamplerState(diffusion_state, param_grad, svrg_state, grad_info)

    def get_params(state: SamplerState):
        return get_params_diffusion(state.diffusion_state)

    return init_fn, kernel, get_params


def _build_sghmc_kernel(
    init_fn_diffusion: Callable,
    update_diffusion: Callable,
    get_params_diffusion: Callable,
    resample_momentum: Callable,
    estimate_gradient: Callable,
    init_gradient: Callable,
    L: int,
    compiled_leapfrog: bool = True,
) -> Tuple[
    Callable[[PRNGKey, PyTree], SamplerState],
    Callable[[int, PRNGKey, SamplerState], SamplerState],
    Callable[[SamplerState], PyTree],
]:
    """Build a sghmc kernel

    Args:
        init_fn_diffusion (Callable): create the initial state of the diffusion
        update_diffusion (Any): updates the state of the diffusion
        get_params_diffusion (Callable): gets the parameters from the state of the diffusion
        resample_momentum (Callable): [description]
        estimate_gradient (Callable): [description]
        init_gradient (Callable): [description]
        L (int): number of leapfrog steps
        compiled_leapfrog (bool, optional): whether or not the integrator is compiled. Defaults to True.

    Returns:
        Tuple[ Callable[[PRNGKey, PyTree], SamplerState], Callable[[int, PRNGKey, SamplerState], SamplerState], Callable[[SamplerState], PyTree] ]: [description]
    """

    init_fn, langevin_kernel, get_params = _build_langevin_kernel(
        init_fn_diffusion,
        update_diffusion,
        get_params_diffusion,
        estimate_gradient,
        init_gradient,
    )

    def sghmc_kernel(i: int, key: PRNGKey, state: SamplerState) -> SamplerState:
        def body(state: SamplerState, key: PRNGKey) -> Tuple[SamplerState, None]:
            return langevin_kernel(i, key, state), None

        k1, k2 = random.split(key)
        diffusion_state = resample_momentum(i, k1, state.diffusion_state)
        state = SamplerState(
            diffusion_state, state.param_grad, state.svrg_state, state.grad_info
        )

        keys = random.split(k2, L)
        state = run_loop(body, state, keys, compiled_leapfrog)
        return state

    return init_fn, sghmc_kernel, get_params


### kernels


def build_sgld_kernel(
    dt: float, loglikelihood: Callable, logprior: Callable, data: Tuple, batch_size: int
) -> Tuple[Callable, Callable, Callable]:
    """Build kernel for SGLD

    Args:
        dt (float): [description]
        loglikelihood (Callable): [description]
        logprior (Callable): [description]
        data (Tuple): [description]
        batch_size (int): [description]

    Returns:
        Tuple[ Callable, Callable, Callable ]: An (init_fun, kernel, get_params) triple.
    """
    grad_log_post = build_grad_log_post(loglikelihood, logprior, data)
    estimate_gradient, init_gradient = build_gradient_estimation_fn(
        grad_log_post, data, batch_size
    )
    init_fn, sgld_kernel, get_params = _build_langevin_kernel(
        *sgld(dt), estimate_gradient, init_gradient
    )
    return init_fn, sgld_kernel, get_params


def build_sgldCV_kernel(
    dt: float,
    loglikelihood: Callable,
    logprior: Callable,
    data: Tuple,
    batch_size: int,
    centering_value: PyTree,
) -> Tuple[Callable, Callable, Callable]:
    """Build SGLD-CV kernel

    Args:
        dt (float): [description]
        loglikelihood (Callable): [description]
        logprior (Callable): [description]
        data (Tuple): [description]
        batch_size (int): [description]
        centering_value (PyTree): [description]

    Returns:
        Tuple[ Callable, Callable, Callable ]: An (init_fun, kernel, get_params) triple.
    """
    grad_log_post = build_grad_log_post(loglikelihood, logprior, data)
    estimate_gradient, init_gradient = build_gradient_estimation_fn_CV(
        grad_log_post, data, batch_size, centering_value
    )
    init_fn, sgldCV_kernel, get_params = _build_langevin_kernel(
        *sgld(dt), estimate_gradient, init_gradient
    )
    return init_fn, sgldCV_kernel, get_params


def build_sgld_SVRG_kernel(
    dt: float,
    loglikelihood: Callable,
    logprior: Callable,
    data: Tuple,
    batch_size: int,
    update_rate: int,
) -> Tuple[Callable, Callable, Callable]:
    """build SGLD-SVRG kernel

    Args:
        dt (float): step size
        loglikelihood (Callable): log-likelihood for a single data point
        logprior (Callable): log-prior for a single data point
        data (Tuple): [description]
        batch_size (int): batch size
        update_rate (int): how often to update the centering value in the gradient estimator

    Returns:
        Tuple[Callable, Callable, Callable]: An (init_fun, kernel, get_params) triple.
    """
    grad_log_post = build_grad_log_post(loglikelihood, logprior, data)
    estimate_gradient, init_gradient = build_gradient_estimation_fn_SVRG(
        grad_log_post, data, batch_size, update_rate
    )
    init_fn, sgldSVRG_kernel, get_params = _build_langevin_kernel(
        *sgld(dt), estimate_gradient, init_gradient
    )
    return init_fn, sgldSVRG_kernel, get_params


def build_psgld_kernel(
    dt: float, loglikelihood, logprior, data, batch_size, alpha=0.99, eps=1e-5
):
    grad_log_post = build_grad_log_post(loglikelihood, logprior, data)
    estimate_gradient, init_gradient = build_gradient_estimation_fn(
        grad_log_post, data, batch_size
    )
    init_fn, sgld_kernel, get_params = _build_langevin_kernel(
        *psgld(dt, alpha, eps), estimate_gradient, init_gradient
    )
    return init_fn, sgld_kernel, get_params


def build_sgldAdam_kernel(
    dt, loglikelihood, logprior, data, batch_size, beta1=0.9, beta2=0.999, eps=1e-8
):
    grad_log_post = build_grad_log_post(loglikelihood, logprior, data)
    estimate_gradient, init_gradient = build_gradient_estimation_fn(
        grad_log_post, data, batch_size
    )
    init_fn, sgldAdam_kernel, get_params = _build_langevin_kernel(
        *sgldAdam(dt, beta1, beta2, eps), estimate_gradient, init_gradient
    )
    return init_fn, sgldAdam_kernel, get_params


def build_sgnht_kernel(dt, loglikelihood, logprior, data, batch_size, a=0.01):
    grad_log_post = build_grad_log_post(loglikelihood, logprior, data)
    estimate_gradient, init_gradient = build_gradient_estimation_fn(
        grad_log_post, data, batch_size
    )
    init_fn, sgnht_kernel, get_params = _build_langevin_kernel(
        *sgnht(dt, a), estimate_gradient, init_gradient
    )
    return init_fn, sgnht_kernel, get_params


def build_sgnhtCV_kernel(
    dt, loglikelihood, logprior, data, batch_size, centering_value, a=0.01
):
    grad_log_post = build_grad_log_post(loglikelihood, logprior, data)
    estimate_gradient, init_gradient = build_gradient_estimation_fn_CV(
        grad_log_post, data, batch_size, centering_value
    )
    init_fn, sgnht_kernel, get_params = _build_langevin_kernel(
        *sgnht(dt, a), estimate_gradient, init_gradient
    )
    return init_fn, sgnht_kernel, get_params


def build_baoab_kernel(dt, gamma, loglikelihood, logprior, data, batch_size, tau=1):
    grad_log_post = build_grad_log_post(loglikelihood, logprior, data)
    estimate_gradient, init_gradient = build_gradient_estimation_fn(
        grad_log_post, data, batch_size
    )
    init_fn, baoab_kernel, get_params = _build_langevin_kernel(
        *baoab(dt, gamma, tau), estimate_gradient, init_gradient
    )
    return init_fn, baoab_kernel, get_params


def build_badodab_kernel(dt, loglikelihood, logprior, data, batch_size, a=0.01):
    grad_log_post = build_grad_log_post(loglikelihood, logprior, data)
    estimate_gradient, init_gradient = build_gradient_estimation_fn(
        grad_log_post, data, batch_size
    )
    init_fn, baoab_kernel, get_params = _build_langevin_kernel(
        *badodab(dt, a), estimate_gradient, init_gradient
    )
    return init_fn, baoab_kernel, get_params


def build_badodabCV_kernel(
    dt, loglikelihood, logprior, data, batch_size, centering_value, a=0.01
):
    grad_log_post = build_grad_log_post(loglikelihood, logprior, data)
    estimate_gradient, init_gradient = build_gradient_estimation_fn_CV(
        grad_log_post, data, batch_size, centering_value
    )
    init_fn, baoab_kernel, get_params = _build_langevin_kernel(
        *badodab(dt, a), estimate_gradient, init_gradient
    )
    return init_fn, baoab_kernel, get_params


# sghmc kernels


def build_sghmc_kernel(
    dt, L, loglikelihood, logprior, data, batch_size, alpha=0.01, compiled_leapfrog=True
):
    grad_log_post = build_grad_log_post(loglikelihood, logprior, data)
    estimate_gradient, init_gradient = build_gradient_estimation_fn(
        grad_log_post, data, batch_size
    )
    init_fn, sghmc_kernel, get_params = _build_sghmc_kernel(
        *sghmc(dt, alpha),
        estimate_gradient,
        init_gradient,
        L,
        compiled_leapfrog=compiled_leapfrog
    )
    return init_fn, sghmc_kernel, get_params


def build_sghmcCV_kernel(
    dt,
    L,
    loglikelihood,
    logprior,
    data,
    batch_size,
    centering_value,
    alpha=0.01,
    compiled_leapfrog=True,
):
    grad_log_post = build_grad_log_post(loglikelihood, logprior, data)
    estimate_gradient, init_gradient = build_gradient_estimation_fn_CV(
        grad_log_post, data, batch_size, centering_value
    )
    init_fn, sghmc_kernel, get_params = _build_sghmc_kernel(
        *sghmc(dt, alpha),
        estimate_gradient,
        init_gradient,
        L,
        compiled_leapfrog=compiled_leapfrog
    )
    return init_fn, sghmc_kernel, get_params


def build_sghmc_SVRG_kernel(
    dt,
    L,
    loglikelihood,
    logprior,
    data,
    batch_size,
    update_rate,
    alpha=0.01,
    compiled_leapfrog=True,
):
    grad_log_post = build_grad_log_post(loglikelihood, logprior, data)
    estimate_gradient, init_gradient = build_gradient_estimation_fn_SVRG(
        grad_log_post, data, batch_size, update_rate
    )
    init_fn, sghmc_kernel, get_params = _build_sghmc_kernel(
        *sghmc(dt, alpha),
        estimate_gradient,
        init_gradient,
        L,
        compiled_leapfrog=compiled_leapfrog
    )
    return init_fn, sghmc_kernel, get_params
