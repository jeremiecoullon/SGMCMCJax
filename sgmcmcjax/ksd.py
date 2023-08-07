import jax.numpy as jnp
from jax import jit, lax, vmap

from .types import Array


@jit
def k_0_fun(
    parm1: Array,
    parm2: Array,
    gradlogp1: Array,
    gradlogp2: Array,
    c: float = 1.0,
    beta: float = -0.5,
) -> float:
    """KSD kernel with the IMQ kernel and the 2 norm: http://proceedings.mlr.press/v70/gorham17a/gorham17a.pdf

    Args:
        parm1 (Array): sampled parameter 1
        parm2 (Array): sampled parameter 2
        gradlogp1 (Array): gradient of sampled parameter 1
        gradlogp2 (Array): gradient of sampled parameter 2
        c (float, optional): intercept parameter in the IMQ kernel. Defaults to 1.
        beta (float, optional): exponent parameter in the IMQ kernel. Defaults to -0.5.

    Returns:
        float: value of kernel for the pair of samples
    """
    diff = parm1 - parm2
    dim = parm1.shape[0]
    base = c**2 + jnp.dot(diff, diff)
    term1 = jnp.dot(gradlogp1, gradlogp2) * base**beta
    term2 = -2 * beta * jnp.dot(gradlogp1, diff) * base ** (beta - 1)
    term3 = 2 * beta * jnp.dot(gradlogp2, diff) * base ** (beta - 1)
    term4 = -2 * dim * beta * (base ** (beta - 1))
    term5 = -4 * beta * (beta - 1) * base ** (beta - 2) * jnp.sum(jnp.square(diff))
    return term1 + term2 + term3 + term4 + term5


_batch_k_0_fun_rows = jit(vmap(k_0_fun, in_axes=(None, 0, None, 0, None, None)))


@jit
def imq_KSD(samples: Array, grads: Array) -> Array:
    """Kernel Stein Discrepancy with IMQ kernel

    Args:
        samples (Array): MCMC samples
        grads (Array): gradients of the MCMC samples

    Returns:
        float: estimate of the KSD
    """
    c, beta = 1.0, -0.5
    N = samples.shape[0]

    # we use lax.scan rather than a nested vmap as the latter becomes very slow for high dimensional problems with lots of samples.
    def body_ksd(le_sum, x):
        my_sample, my_grad = x
        le_sum += jnp.sum(
            _batch_k_0_fun_rows(my_sample, samples, my_grad, grads, c, beta)
        )
        return le_sum, None

    le_sum, _ = lax.scan(body_ksd, 0.0, (samples, grads))
    return jnp.sqrt(le_sum) / N
