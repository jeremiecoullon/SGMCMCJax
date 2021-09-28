import functools
from functools import partial
from jax import random
from jax.tree_util import tree_flatten, tree_unflatten
from jax._src.util import safe_map, unzip2
from .types import DiffusionState

map = safe_map

def diffusion_factory(is_palindrome=False, is_sghmc=False):
    def _diffusion(sampler_maker):

        @functools.wraps(sampler_maker)
        def tree_sampler_maker(*args, **kwargs):
            if is_sghmc and (not is_palindrome):
                init_fn, update, get_params, resample_momentum = sampler_maker(*args, **kwargs)
            elif (not is_sghmc) and is_palindrome:
                init_fn, (update, update2), get_params = sampler_maker(*args, **kwargs)
            elif is_sghmc and is_palindrome:
                init_fn, (update, update2), get_params, resample_momentum = sampler_maker(*args, **kwargs)
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
                return tree_init, (tree_update, tree_update2), tree_get_params
            elif is_sghmc and is_palindrome:
                return tree_init, (tree_update, tree_update2), tree_get_params, tree_resample_momentum
            else:
                return tree_init, tree_update, tree_get_params
        return tree_sampler_maker
    return _diffusion

diffusion = diffusion_factory()
diffusion_sghmc = diffusion_factory(is_sghmc=True)
diffusion_palindrome = diffusion_factory(is_palindrome=True)
