
# MAP estimate for multivariate Diagonal Gaussian

import jax.numpy as jnp
from jax import random
from sgmcmcjax.optimizer import build_adam_optimizer, build_optax_optimizer
import optax

def loglikelihood(theta, x):
    return -0.5*jnp.dot(x-theta, x-theta)

def logprior(theta):
    return -0.5*jnp.dot(theta, theta)*0.01

# generate dataset
N, D = 10_000, 100
key = random.PRNGKey(0)
mu_true = random.normal(key, (D,))
X_data = random.normal(key, shape=(N, D)) + mu_true

# Adam
batch_size = int(0.1*N)
dt = 1e-2
opt = optax.adam(learning_rate=dt)
optimizer = build_optax_optimizer(opt, loglikelihood, logprior, (X_data,), batch_size)
# optimizer = build_adam_optimizer(dt, loglikelihood, logprior, (X_data,), batch_size)

Nsamples = 10_000
params, log_post_list = optimizer(key, Nsamples, jnp.zeros(D))
print(log_post_list.shape)
print(params.shape)
assert jnp.allclose(params, mu_true, atol=1e-1)
print('test passed')
