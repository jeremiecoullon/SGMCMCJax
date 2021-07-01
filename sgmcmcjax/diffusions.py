from collections import namedtuple
import functools
import jax.numpy as jnp
from jax import grad, jit, vmap, lax, random, partial
from jax.tree_util import tree_flatten, tree_unflatten, tree_map, register_pytree_node
from jax._src.util import safe_map, unzip2

map = safe_map

DiffusionState = namedtuple("DiffusionState", ['packed_state', 'tree_def', 'subtree_defs'])

register_pytree_node(
    DiffusionState,
    lambda xs: ((xs.packed_state,), (xs.tree_def, xs.subtree_defs)),
    lambda data, xs : DiffusionState(xs[0], data[0], data[1]))

def diffusion_factory(is_palindrome=False, is_sghmc=False):
    def _diffusion(sampler_maker):

        @functools.wraps(sampler_maker)
        def tree_sampler_maker(*args, **kwargs):
            if is_sghmc and (not is_palindrome):
                init_fn, update, get_params, resample_momentum = sampler_maker(*args, **kwargs)
            elif (not is_sghmc) and is_palindrome:
                init_fn, update, update2, get_params = sampler_maker(*args, **kwargs)
            elif is_sghmc and is_palindrome:
                init_fn, update, update2, get_params, resample_momentum = sampler_maker(*args, **kwargs)
            else:
                init_fn, update, get_params = sampler_maker(*args, **kwargs)

            @functools.wraps(init_fn)
            def tree_init(x0_tree):
                x0_flat, tree = tree_flatten(x0_tree)
                initial_states = [init_fn(x0) for x0 in x0_flat]
                states_flat, subtrees = unzip2(map(tree_flatten, initial_states))
                return DiffusionState(states_flat, tree, subtrees)

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
                return DiffusionState(new_states_flat, tree, subtrees)

            @functools.wraps(get_params)
            def tree_get_params(sampler_state):
                states_flat, tree, subtrees = sampler_state
                states = map(tree_unflatten, subtrees, states_flat)
                params = map(get_params, states)
                return tree_unflatten(tree, params)

            if is_sghmc:
                @functools.wraps(resample_momentum)
                def tree_resample_momentum(i, k, state):
                    x_tree = tree_get_params(state)
                    x_flat, tree = tree_flatten(x_tree)
                    keys = random.split(k, len(x_flat))
                    states = [resample_momentum(i, key, x) for (key, x) in zip(keys, x_flat)]
                    states_flat, subtree = unzip2(map(tree_flatten, states))
                    return DiffusionState(states_flat, tree, subtree)


            if is_palindrome:
                @functools.wraps(update2)
                def tree_update2(i, key, grad_tree, sampler_state):
                    states_flat, tree, subtrees = sampler_state
                    grad_flat, tree2 = tree_flatten(grad_tree)
                    if tree2 != tree:
                        msg = ("sampler update2 function was passed a gradient tree that did "
                        "not match the parameter tree structure with which it was "
                        "initialized: parameter tree {} and grad tree {}.")
                        raise TypeError(msg.format(tree, tree2))
                    states = map(tree_unflatten, subtrees, states_flat)
                    keys = random.split(key, len(states))
                    new_states = map(partial(update2, i), keys, grad_flat, states)
                    new_states_flat, subtrees2 = unzip2(map(tree_flatten, new_states))
                    for subtree, subtree2 in zip(subtrees, subtrees2):
                        if subtree2 != subtree:
                            msg = ("sampler update2 function produced an output structure that "
                             "did not match its input structure: input {} and output {}.")
                            raise TypeError(msg.format(subtree, subtree2))
                    return DiffusionState(new_states_flat, tree, subtrees)

            if is_sghmc and (not is_palindrome):
                return tree_init, tree_update, tree_get_params, tree_resample_momentum
            elif (not is_sghmc) and is_palindrome:
                return tree_init, tree_update, tree_update2, tree_get_params
            elif is_sghmc and is_palindrome:
                return tree_init, tree_update, tree_update2, tree_get_params, tree_resample_momentum
            else:
                return tree_init, tree_update, tree_get_params
        return tree_sampler_maker
    return _diffusion

diffusion = diffusion_factory()
diffusion_sghmc = diffusion_factory(is_sghmc=True)
diffusion_palindrome = diffusion_factory(is_palindrome=True)

def constant(step_size):
    def schedule(i):
        return step_size
    return schedule

def welling_teh_schedule(a,b, gamma=0.55):
    "Polynomial schedule from https://www.ics.uci.edu/~welling/publications/papers/stoclangevin_v6.pdf"
    def schedule(i):
        return a*(b+i)**(-gamma)
    return schedule

def make_schedule(scalar_or_schedule):
    if callable(scalar_or_schedule):
        return scalar_or_schedule
    elif jnp.ndim(scalar_or_schedule) == 0:
        return constant(scalar_or_schedule)
    else:
        raise TypeError(type(scalar_or_schedule))


@diffusion
def sgld(dt):
    "https://www.ics.uci.edu/~welling/publications/papers/stoclangevin_v6.pdf"
    dt = make_schedule(dt)

    def init_fn(x):
        return x

    def update(i, k, g, x):
        return x + dt(i)*g + jnp.sqrt(2*dt(i))*random.normal(k, shape=jnp.shape(x))

    def get_params(x):
        return x

    return init_fn, update, get_params

@diffusion
def psgld(dt, alpha=0.99, eps=1e-5):
    "https://arxiv.org/pdf/1512.07666.pdf"
    dt = make_schedule(dt)

    def init_fn(x):
        v = jnp.zeros_like(x)
        return x, v

    def update(i, k, g, state):
        x, v = state
        v = alpha*v + (1-alpha)*jnp.square(g)
        G = 1./(jnp.sqrt(v)+eps)
        return x + dt(i)*0.5*G*g + jnp.sqrt(dt(i)*G)*random.normal(k, shape=jnp.shape(x)), v

    def get_params(state):
        x, _ = state
        return x

    return init_fn, update, get_params

@diffusion
def sgldAdam(dt, beta1=0.9, beta2=0.999, eps=1e-8):
    "https://arxiv.org/abs/2105.13059"
    dt = make_schedule(dt)

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
        adapt_dt = dt(i)/(jnp.sqrt(v_hat) + eps)

        return x + adapt_dt*0.5*m_hat + jnp.sqrt(adapt_dt)*random.normal(key=k, shape=jnp.shape(x)), m, v

    def get_params(state):
        x, _, _ = state
        return x

    return init_fn, update, get_params

@diffusion_sghmc
def sghmc(dt, alpha=0.01, beta=0):
    "https://arxiv.org/abs/1402.4102"
    dt = make_schedule(dt)

    def init_fn(x):
        v = jnp.zeros_like(x)
        return x, v

    def update(i, k, g, state):
        x, v = state
        x = x + v
        v = v + dt(i)*g - alpha*v + jnp.sqrt(2*(alpha - beta)*dt(i))*random.normal(k, shape=jnp.shape(x))
        return x, v

    def get_params(state):
        x, _ = state
        return x

    def resample_momentum(i, k, x):
        v = jnp.sqrt(dt(i))*random.normal(k, shape=jnp.shape(x))
        return x, v

    return init_fn, update, get_params, resample_momentum

@diffusion_palindrome
def baoab(dt, gamma, tau=1):
    dt = make_schedule(dt)

    def init_fn(x):
        v = jnp.zeros_like(x)
        return x, v

    def update1(i, k, g, state):
        x, v = state

        v = v + dt(i)*0.5*g
        x = x + v*dt(i)*0.5

        c1 = jnp.exp(-gamma*dt(i))
        c2 = jnp.sqrt(1 - c1**2)
        v = c1*v + tau*c2*random.normal(k, shape=jnp.shape(v))

        x = x + v*dt(i)*0.5

        return x, v

    def update2(i, k, g, state):
        x, v = state
        v = v + dt(i)*0.5*g
        return x, v

    def get_params(state):
        x, _ = state
        return x

    return init_fn, update1, update2, get_params

@diffusion
def sgnht(dt, a=0.01):
    "http://people.ee.duke.edu/~lcarin/sgnht-4.pdf: Algorithm 2"
    dt = make_schedule(dt)

    def init_fn(x):
        v = jnp.zeros_like(x)
        alpha = a
        return x, v, alpha

    def initial_momentum(kv):
        "sample momentum at the first iteration"
        k, v = kv
        key, subkey = random.split(k)
        v = jnp.sqrt(dt(0))*random.normal(subkey, shape=v.shape)
        return key, v

    def update(i, k, g, state):
        x, v, alpha = state
        k,v = lax.cond(i==0,
                initial_momentum,
                lambda kv: (k,v),
                (k,v)
            )
        v = v - alpha*v + dt(i)*g + jnp.sqrt(2*a*dt(i))*random.normal(k, shape=jnp.shape(x))
        x = x + v
        alpha = alpha + (jnp.linalg.norm(v)**2)/v.size - dt(i)
        return x, v, alpha

    def get_params(state):
        x, _, _ = state
        return x

    return init_fn, update, get_params

@diffusion_palindrome
def badodab(dt, a=0.01):
    "https://arxiv.org/abs/1505.06889"
    dt = make_schedule(dt)

    def init_fn(x):
        v = jnp.zeros_like(x)
        alpha = a
        return x, v, alpha

    def update(i, k, g, state):
        x, v, alpha = state

        dt2 = dt(i)/2
        mu = 1.
        sigma = 1.

        v = v + dt2*g
        x = x + dt2*v
        alpha = alpha + (dt2/mu)*(jnp.linalg.norm(v) - v.size)

        c1 = jnp.exp(-alpha*dt(i))
        c2 = jnp.where(alpha==0, jnp.sqrt(dt(i)), jnp.sqrt(jnp.abs((1-c1**2)/(2*alpha))))
        v = c1*v + c2*sigma*random.normal(k, shape=jnp.shape(v))

        alpha = alpha + (dt2/mu)*(jnp.linalg.norm(v) - v.size)
        x = x + dt2*v
        return x, v, alpha

    def update2(i, k, g, state):
        x, v, alpha = state
        v = v + dt(i)*0.5*g
        return x, v, alpha

    def get_params(state):
        x, _, _ = state
        return x

    return init_fn, update, update2, get_params
