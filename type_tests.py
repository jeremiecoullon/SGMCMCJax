from typing import Callable, NamedTuple, Tuple, Union, Any
from tqdm.auto import tqdm

import jax.numpy as jnp
from jax import random, jit

from sgmcmcjax.diffusions import sgld
from sgmcmcjax.diffusions import DiffusionState
from sgmcmcjax.gradient_estimation import build_gradient_estimation_fn, SVRGState, build_gradient_estimation_fn_CV, build_gradient_estimation_fn_SVRG
from sgmcmcjax.util import build_grad_log_post, run_loop
from sgmcmcjax.types import PyTree

# import model and create dataset
from docs.nbs.models.logistic_regression import gen_data, loglikelihood, logprior

key = random.PRNGKey(42)
dim = 10
Ndata = 100000

theta_true, X, y_data = gen_data(key, dim, Ndata)

data = (X, y_data)

class SamplerState(NamedTuple):
    diffusion_state: DiffusionState
    param_grad: PyTree
    svrg_state: Union[SVRGState, None] = None
    grad_info: Union[Any, None] = None

# mystate = SamplerState(theta_true, 8)

# print(mystate)


# # standard gradient estimator
# def build_gradient_estimation_fn(grad_log_post, data, batch_size):
#     assert type(data) == tuple
#     N_data, *_ = data[0].shape
#     # make sure data has jax arrays rather than numpy arrays
#     data = tuple([jnp.array(elem) for elem in data])
#
#     @jit
#     def estimate_gradient(key, param):
#         if (batch_size is None) or batch_size == N_data:
#             return grad_log_post(param, *data)
#         else:
#             idx_batch = random.choice(key=key, a=jnp.arange(N_data), shape=(batch_size,))
#             minibatch_data = tuple([elem[idx_batch] for elem in data])
#             return grad_log_post(param, *minibatch_data)
#     return estimate_gradient

# current setup
# def _build_langevin_kernel(update, get_params, estimate_gradient):
#     "build generic kernel"
#     @jit
#     def kernel(i, key, state):
#         k1, k2 = random.split(key)
#         param_grad = estimate_gradient(k1, get_params(state))
#         state = update(i, k2, param_grad, state)
#         return state
#
#     return kernel
#
#
# def build_sgld_kernel(dt, loglikelihood, logprior, data, batch_size):
#     grad_log_post = build_grad_log_post(loglikelihood, logprior, data)
#     init_fn, update, get_params = sgld(dt)
#     estimate_gradient = build_gradient_estimation_fn(grad_log_post, data, batch_size)
#     sgld_kernel = _build_langevin_kernel(update, get_params, estimate_gradient)
#     return init_fn, sgld_kernel, get_params


# ========
def _build_langevin_kernel(init_fn_diffusion, update_diffusion, get_params_diffusion, estimate_gradient):
    "build generic kernel"

    def init_fn(key, params):
        diffusion_state = init_fn_diffusion(params)
        param_grad = estimate_gradient(key, get_params_diffusion(diffusion_state))
        return SamplerState(diffusion_state, param_grad, None, None)

    def kernel(i, key, state):
        diffusion_state, param_grad, svrg_state, grad_info = state
        k1, k2 = random.split(key)
        diffusion_state = update_diffusion(i, k1, param_grad, diffusion_state)
        param_grad = estimate_gradient(k2, get_params_diffusion(diffusion_state))
        return SamplerState(diffusion_state, param_grad, svrg_state, grad_info)

    def get_params(state):
        return get_params_diffusion(state.diffusion_state)

    return init_fn, kernel, get_params


def build_sgld_kernel(dt, loglikelihood, logprior, data, batch_size):
    grad_log_post = build_grad_log_post(loglikelihood, logprior, data)
    # init_fn_diffusion, update_diffusion, get_params_diffusion = sgld(dt)
    # estimate_gradient = build_gradient_estimation_fn(grad_log_post, data, batch_size)
    # init_fn, sgld_kernel, get_params = _build_langevin_kernel(init_fn_diffusion, update_diffusion, get_params_diffusion, estimate_gradient)
    estimate_gradient = build_gradient_estimation_fn(grad_log_post, data, batch_size)
    init_fn, sgld_kernel, get_params = _build_langevin_kernel(*sgld(dt), estimate_gradient)
    return init_fn, sgld_kernel, get_params



# ================
# Run sampler

batch_size = int(0.01*X.shape[0])
dt = 1e-5

init_fn, my_kernel, get_params = build_sgld_kernel(dt, loglikelihood, logprior, data, batch_size)
key = random.PRNGKey(0)
Nsamples = 1000
samples = []

state = init_fn(key, theta_true)

for i in tqdm(range(Nsamples)):
    key, subkey = random.split(key)
    state = jit(my_kernel)(i, subkey, state)
    samples.append(get_params(state))

samples = jnp.array(samples)
print(samples[-1])
