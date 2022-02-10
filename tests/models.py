import jax.numpy as jnp
from jax import random


# Parameter shape: array
def loglikelihood_array(theta, x):
    return -0.5 * jnp.dot(x - theta, x - theta)


def logprior_array(theta):
    return -0.5 * jnp.dot(theta, theta) * 0.01


# Parameter shape: list of 2 arrays
def loglikelihood_list_array(theta, x):
    param1, param2 = theta
    return -0.5 * jnp.dot(x - param1, x - param1) - 0.1 * jnp.dot(
        x - param2, x - param2
    )


def logprior_list_array(theta):
    param1, param2 = theta
    return -0.001 * jnp.dot(param1, param1) - 0.001 * jnp.dot(param2, param2)


# generate dataset
N, D = 1000, 5
key = random.PRNGKey(0)
X_data = random.normal(key, shape=(N, D))
