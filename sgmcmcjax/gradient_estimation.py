from typing import Any, Tuple, Callable
from collections import namedtuple

import jax.numpy as jnp
from jax import random, jit, lax
from jax.tree_util import tree_flatten, tree_unflatten, tree_map, tree_multimap
from .types import PyTree, PRNGKey, SamplerState, SVRGState


# standard gradient estimator
def build_gradient_estimation_fn(grad_log_post, data, batch_size):
    assert type(data) == tuple
    N_data, *_ = data[0].shape
    # make sure data has jax arrays rather than numpy arrays
    data = tuple([jnp.array(elem) for elem in data])

    @jit
    # def estimate_gradient(key, param):
    def estimate_gradient(i: int, key: PRNGKey, state: SamplerState, get_params_diffusion: Callable) -> Tuple[PyTree, Any]:
       # TODO: make this signature the same for all gradient estimators, fix tests
       param = get_params_diffusion(state.diffusion_state)
       if (batch_size is None) or batch_size == N_data:
           return grad_log_post(param, *data), None
       else:
           idx_batch = random.choice(key=key, a=jnp.arange(N_data), shape=(batch_size,))
           minibatch_data = tuple([elem[idx_batch] for elem in data])
           return grad_log_post(param, *minibatch_data), None # return a tuple with None to be compatible with SVRG

    return estimate_gradient

# Control variates
def build_gradient_estimation_fn_CV(grad_log_post, data, batch_size, centering_value):
    assert type(data) == tuple
    N_data, *_ = data[0].shape
    # make sure data has jax arrays rather than numpy arrays
    data = tuple([jnp.array(elem) for elem in data])

    fb_grad_center = grad_log_post(centering_value, *data)
    flat_fb_grad_center, tree_fb_grad_center = tree_flatten(fb_grad_center)
    update_fn = lambda c,g,gc: c + g - gc

    @jit
    # def estimate_gradient(key, param):
    def estimate_gradient(i: int, key: PRNGKey, state: SamplerState, get_params_diffusion: Callable) -> Tuple[PyTree, Any]:
        param = get_params_diffusion(state.diffusion_state)
        idx_batch = random.choice(key=key, a=jnp.arange(N_data), shape=(batch_size,))
        minibatch_data = tuple([elem[idx_batch] for elem in data])

        param_grad = grad_log_post(param, *minibatch_data)
        grad_center = grad_log_post(centering_value, *minibatch_data)
        flat_param_grad, tree_param_grad = tree_flatten(param_grad)
        flat_grad_center, tree_grad_center = tree_flatten(grad_center)
        new_flat_param_grad = tree_multimap(update_fn, flat_fb_grad_center, flat_param_grad, flat_grad_center)
        param_grad = tree_unflatten(tree_param_grad, new_flat_param_grad)
        return param_grad, None

    return estimate_gradient

# # SVRG
# SVRGState = namedtuple("SVRGState", ['centering_value', 'update_rate', 'fb_grad_center'],
#                        defaults=(None, None, None))

def build_gradient_estimation_fn_SVRG(grad_log_post, data, batch_size, centering_value, update_rate):
    assert type(data) == tuple
    N_data, *_ = data[0].shape
    # make sure data has jax arrays rather than numpy arrays
    data = tuple([jnp.array(elem) for elem in data])
    update_fn = lambda c,g,gc: c + g - gc

    def update_centering_value(svrg_state: SVRGState, param: PyTree) -> SVRGState:
        fb_grad_center = grad_log_post(param, *data)
        flat_fb_grad_center, tree_fb_grad_center = tree_flatten(fb_grad_center)
        svrg_state = SVRGState(param, svrg_state.update_rate, flat_fb_grad_center)
        return svrg_state

    @jit
    # def estimate_gradient(key, param, i, state):
    def estimate_gradient(i: int, key: PRNGKey, state: SamplerState, get_params_diffusion: Callable) -> Tuple[PyTree, Any]:
        param = get_params_diffusion(state.diffusion_state)
        svrg_state = state.svrg_state
        svrg_state = lax.cond(i%svrg_state.update_rate == 0,
                    lambda _: update_centering_value(svrg_state, param),
                    lambda _ : svrg_state,
                    None
                )

        idx_batch = random.choice(key=key, a=jnp.arange(N_data), shape=(batch_size,))
        minibatch_data = tuple([elem[idx_batch] for elem in data])

        param_grad = grad_log_post(param, *minibatch_data)
        grad_center = grad_log_post(svrg_state.centering_value, *minibatch_data)
        flat_param_grad, tree_param_grad = tree_flatten(param_grad)
        flat_grad_center, tree_grad_center = tree_flatten(grad_center)
        new_flat_param_grad = tree_multimap(update_fn, svrg_state.fb_grad_center, flat_param_grad, flat_grad_center)
        param_grad = tree_unflatten(tree_param_grad, new_flat_param_grad)
        return param_grad, svrg_state

    state = update_centering_value(SVRGState(None, update_rate, None), centering_value)
    return estimate_gradient, state
