from jax import jit, vmap, lax
import jax.numpy as jnp

@jit
def k_0_fun(parm1, parm2, gradlogp1, gradlogp2, c=1., beta=-0.5):
    """
    KSD kernel with the 2 norm
    """
    diff = parm1-parm2
    dim = parm1.shape[0]
    base = (c**2 + jnp.dot(diff, diff))
    term1 = jnp.dot(gradlogp1,gradlogp2)*base**beta
    term2 = -2*beta * jnp.dot(gradlogp1, diff) * base**(beta-1)
    term3 = 2*beta * jnp.dot(gradlogp2, diff) * base**(beta-1)
    term4 = -2*dim*beta*(base**(beta-1))
    term5 = -4*beta* (beta-1)*base**(beta-2)*jnp.sum(jnp.square(diff))
    return term1 + term2 + term3 + term4 + term5


batch_k_0_fun_rows = jit(vmap(k_0_fun, in_axes=(None,0,None,0,None,None)))

@jit
def imq_KSD(sgld_samples, sgld_grads):
    """
    KSD with imq kernel
    """
    c, beta = 1., -0.5
    N = sgld_samples.shape[0]

    def body_ksd(le_sum, x):
        my_sample, my_grad = x
        le_sum += jnp.sum(batch_k_0_fun_rows(my_sample, sgld_samples, my_grad, sgld_grads, c, beta))
        return le_sum, None

    le_sum, _ = lax.scan(body_ksd, 0., (sgld_samples, sgld_grads))
    return jnp.sqrt(le_sum)/N
