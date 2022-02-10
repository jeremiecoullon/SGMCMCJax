# Sample from multivariate Diagonal Gaussian using SGLD

import jax.numpy as jnp
from jax import random

from sgmcmcjax.samplers import build_sgld_sampler


def loglikelihood(theta, x):
    return -0.5 * jnp.dot(x - theta, x - theta)


def logprior(theta):
    return -0.5 * jnp.dot(theta, theta) * 0.01


# generate dataset
N, D = 10_000, 100
key = random.PRNGKey(0)
mu_true = random.normal(key, (D,))
X_data = random.normal(key, shape=(N, D)) + mu_true

# build sampler
batch_size = int(0.1 * N)
dt = 1e-5
sampler = build_sgld_sampler(dt, loglikelihood, logprior, (X_data,), batch_size)

# run sampler
Nsamples = 10_000
samples = sampler(key, Nsamples, jnp.zeros(D))

# test
print(samples.shape)
mu_est = jnp.mean(samples, axis=0)
print(mu_est.shape)
assert jnp.allclose(mu_est, mu_true, atol=1e-1)
print("test passed")
