from typing import Any, Tuple, Callable
from collections import namedtuple

import jax.numpy as jnp
from jax import random, jit, lax, partial
from jax.tree_util import tree_flatten, tree_unflatten, tree_map, tree_multimap
from .types import PyTree, PRNGKey, SamplerState, SVRGState


# standard gradient estimator
def build_gradient_estimation_fn(grad_log_post, data, batch_size):
    assert type(data) == tuple
    N_data, *_ = data[0].shape
    data = tuple([jnp.array(elem) for elem in data]) # this makes sure data has jax arrays rather than numpy arrays

    def init_gradient(key: PRNGKey, param: PyTree):
        idx_batch = random.choice(key=key, a=jnp.arange(N_data), shape=(batch_size,))
        minibatch_data = tuple([elem[idx_batch] for elem in data])
        param_grad = grad_log_post(param, *minibatch_data)
        return param_grad, SVRGState()

    # def estimate_gradient(key, param):
    @partial(jit, static_argnums=(3,))
    def estimate_gradient(i: int, key: PRNGKey, state: SamplerState, get_params_diffusion: Callable) -> Tuple[PyTree, SVRGState]:
       param = get_params_diffusion(state.diffusion_state)
       if (batch_size is None) or batch_size == N_data:
           return grad_log_post(param, *data), None
       else:
           return init_gradient(key, param)

    return estimate_gradient, init_gradient

# Control variates
def build_gradient_estimation_fn_CV(grad_log_post, data, batch_size, centering_value):
    assert type(data) == tuple
    N_data, *_ = data[0].shape
    data = tuple([jnp.array(elem) for elem in data]) # this makes sure data has jax arrays rather than numpy arrays

    fb_grad_center = grad_log_post(centering_value, *data)
    flat_fb_grad_center, tree_fb_grad_center = tree_flatten(fb_grad_center)
    update_fn = lambda c,g,gc: c + g - gc

    def init_gradient(key: PRNGKey, param: PyTree):
        idx_batch = random.choice(key=key, a=jnp.arange(N_data), shape=(batch_size,))
        minibatch_data = tuple([elem[idx_batch] for elem in data])

        param_grad = grad_log_post(param, *minibatch_data)
        grad_center = grad_log_post(centering_value, *minibatch_data)
        flat_param_grad, tree_param_grad = tree_flatten(param_grad)
        flat_grad_center, tree_grad_center = tree_flatten(grad_center)
        new_flat_param_grad = tree_multimap(update_fn, flat_fb_grad_center, flat_param_grad, flat_grad_center)
        param_grad = tree_unflatten(tree_param_grad, new_flat_param_grad)
        return param_grad, SVRGState()

    @partial(jit, static_argnums=(3,))
    def estimate_gradient(i: int, key: PRNGKey, state: SamplerState, get_params_diffusion: Callable) -> Tuple[PyTree, SVRGState]:
        param = get_params_diffusion(state.diffusion_state)
        return init_gradient(key, param)

    return estimate_gradient, init_gradient

def build_gradient_estimation_fn_SVRG(grad_log_post: Callable, data, batch_size, centering_value, update_rate: int):
    assert type(data) == tuple
    N_data, *_ = data[0].shape
    data = tuple([jnp.array(elem) for elem in data]) # this makes sure data has jax arrays rather than numpy arrays
    update_fn = lambda c,g,gc: c + g - gc

    def update_centering_value(param: PyTree) -> SVRGState:
        fb_grad_center = grad_log_post(param, *data)
        flat_fb_grad_center, tree_fb_grad_center = tree_flatten(fb_grad_center)
        svrg_state = SVRGState(param, update_rate, flat_fb_grad_center)
        return svrg_state

    def init_gradient(key: PRNGKey, param: PyTree) -> Tuple[PyTree, SVRGState]:
        fb_grad_center = grad_log_post(param, *data)
        flat_fb_grad_center, tree_fb_grad_center = tree_flatten(fb_grad_center)
        svrg_state = SVRGState(param, update_rate, flat_fb_grad_center)
        return fb_grad_center, svrg_state

    @partial(jit, static_argnums=(3,))
    def estimate_gradient(i: int, key: PRNGKey, state: SamplerState, get_params_diffusion: Callable) -> Tuple[PyTree, SVRGState]:
        param = get_params_diffusion(state.diffusion_state)
        svrg_state = state.svrg_state
        svrg_state = lax.cond(i%update_rate == 0,
                    lambda _: update_centering_value(param),
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

    return estimate_gradient, init_gradient
