from collections import namedtuple
import functools
import jax.numpy as jnp
from jax import grad, jit, vmap, lax, random, partial
from jax.tree_util import tree_flatten, tree_unflatten, tree_map, register_pytree_node
from jax._src.util import safe_map, unzip2

map = safe_map

SGMCMCState = namedtuple("SGMCMCState", ['packed_state', 'tree_def', 'subtree_defs'])

register_pytree_node(
    SGMCMCState,
    lambda xs: ((xs.packed_state,), (xs.tree_def, xs.subtree_defs)),
    lambda data, xs : SGMCMCState(xs[0], data[0], data[1]))


def sgmcmc(sampler_maker):

    @functools.wraps(sampler_maker)
    def tree_sampler_maker(*args, **kwargs):
        try:
            is_sghmc = False
            init_fn, update, get_params = sampler_maker(*args, **kwargs)
        except ValueError:
            is_sghmc = True
            init_fn, update, get_params, resample_momentum = sampler_maker(*args, **kwargs)

        @functools.wraps(init_fn)
        def tree_init(x0_tree):
            x0_flat, tree = tree_flatten(x0_tree)
            initial_states = [init_fn(x0) for x0 in x0_flat]
            states_flat, subtrees = unzip2(map(tree_flatten, initial_states))
            return SGMCMCState(states_flat, tree, subtrees)

        @functools.wraps(update)
        def tree_update(i, key, grad_tree, sampler_state):
            states_flat, tree, subtrees = sampler_state
            grad_flat, tree2 = tree_flatten(grad_tree)
            if tree2 != tree:
                msg = ("sampler update function was passed a gradient tree that did "
                "not match the parameter tree structure with which it was "
                "initialized: parameter tree {} and grad tree {}.")
                raise TypeError(msg.format(tree, tree2))
            states = map(tree_unflatten, subtrees, states_flat)
            keys = random.split(key, len(states))
            new_states = map(partial(update, i), keys, grad_flat, states)
            new_states_flat, subtrees2 = unzip2(map(tree_flatten, new_states))
            for subtree, subtree2 in zip(subtrees, subtrees2):
                if subtree2 != subtree:
                    msg = ("sampler update function produced an output structure that "
                     "did not match its input structure: input {} and output {}.")
                    raise TypeError(msg.format(subtree, subtree2))
            return SGMCMCState(new_states_flat, tree, subtrees)

        @functools.wraps(get_params)
        def tree_get_params(sampler_state):
            states_flat, tree, subtrees = sampler_state
            states = map(tree_unflatten, subtrees, states_flat)
            params = map(get_params, states)
            return tree_unflatten(tree, params)

        if is_sghmc:
            @functools.wraps(resample_momentum)
            def tree_resample_momentum(k, state):
                x_tree = tree_get_params(state)
                x_flat, tree = tree_flatten(x_tree)
                keys = random.split(k, len(x_flat))
                states = [resample_momentum(key, x) for (key, x) in zip(keys, x_flat)]
                states_flat, subtree = unzip2(map(tree_flatten, states))
                return SGMCMCState(states_flat, tree, subtree)

            return tree_init, tree_update, tree_get_params, tree_resample_momentum
        else:
            return tree_init, tree_update, tree_get_params
    return tree_sampler_maker


@sgmcmc
def sgld(dt):

    def init_fn(x):
        return x

    def update(i, k, g, x):
        return x + dt*g + jnp.sqrt(2*dt)*random.normal(k, shape=jnp.shape(x))

    def get_params(x):
        return x

    return init_fn, update, get_params

@sgmcmc
def psgld(dt, alpha=0.99, eps=1e-5):
    "https://arxiv.org/pdf/1512.07666.pdf"

    def init_fn(x):
        v = jnp.zeros_like(x)
        return x, v

    def update(i, k, g, state):
        x, v = state
        v = alpha*v + (1-alpha)*jnp.square(g)
        G = 1./(jnp.sqrt(v)+eps)
        return x + dt*0.5*G*g + jnp.sqrt(dt*G)*random.normal(k, shape=jnp.shape(x)), v

    def get_params(state):
        x, _ = state
        return x

    return init_fn, update, get_params

@sgmcmc
def sgldAdam(dt, beta1=0.9, beta2=0.999, eps=1e-8):
    "https://arxiv.org/abs/2105.13059"

    def init_fn(x):
        m = jnp.zeros_like(x)
        v = jnp.zeros_like(x)
        return x, m, v

    def update(i, k, g, state):
        x,m,v = state
        m = beta1*m + (1-beta1)*g
        v = beta2*v + (1-beta2)*jnp.square(g)
        m_hat = m/(1-beta1**(i+1))
        v_hat = v/(1-beta2**(i+1))
        adapt_dt = dt/(jnp.sqrt(v_hat) + eps)

        return x + adapt_dt*0.5*m_hat + jnp.sqrt(adapt_dt)*random.normal(key=k, shape=jnp.shape(x)), m, v

    def get_params(state):
        x, _, _ = state
        return x

    return init_fn, update, get_params

@sgmcmc
def sghmc(dt, alpha=0.01):
    "https://arxiv.org/abs/1402.4102"

    def init_fn(x):
        v = jnp.zeros_like(x)
        return x, v

    def update(i, k, g, state):
        x, v = state
        x = x + v
        v = v + dt*g - alpha*v + jnp.sqrt(2*alpha*dt)*random.normal(k, shape=jnp.shape(x))
        return x, v

    def get_params(state):
        x, _ = state
        return x

    def resample_momentum(k, x):
        v = jnp.sqrt(dt)*random.normal(k, shape=jnp.shape(x))
        return x, v

    return init_fn, update, get_params, resample_momentum
