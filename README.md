# SGMCMCJax



SGMCMCJax is a lightweight library of stochastic gradient Markov chain Monte Carlo (SGMCMC) algorithms. The aim is to include both standard samplers (SGLD, SGHMC) as well as state of the art samplers (SVRG-langevin, others, ...).

The target audience for this library is researchers and practitioners: simply plug in your JAX model and easily obtain samples.

## Example usage

We show the basic usage with the following example of estimating the mean of a D-dimensional Gaussian from data using a Gaussian prior.

```python
import jax.numpy as jnp
from jax import random
from sgmcmcjax.samplers import build_sgld_sampler


# define model in JAX
def loglikelihood(theta, x):
    return -0.5*jnp.dot(x-theta, x-theta)

def logprior(theta):
    return -0.5*jnp.dot(theta, theta)*0.01

# generate dataset
N, D = 10_000, 100
key = random.PRNGKey(0)
X_data = random.normal(key, shape=(N, D))

# build sampler
batch_size = int(0.1*N)
dt = 1e-5
my_sampler = build_sgld_sampler(dt, loglikelihood, logprior, (X_data,), batch_size)

# run sampler
Nsamples = 10_000
samples = my_sampler(key, Nsamples, jnp.zeros(D))
```

## Samplers

### SGLD

[Stochastic gradient Langevin dynamics](https://www.ics.uci.edu/~welling/publications/papers/stoclangevin_v6.pdf) (SGLD) is the most basic SGMCMC algorithm.

**Pros:** simple to code up and understand and fast.

**Cons:** The variance estimated from samples quickly becomes too large as the batch size decreases  

### SGLD-CV

[SGLD with control variates](https://arxiv.org/abs/1706.05439): the same update as SGLD but with a better estimate for the gradient

**Pros:** The gradient estimate is much more accurate for log-concave posteriors for only a small added computational cost

**Cons:** The gradient estimate will lose accuracy for posteriors that are not log-concave.  You also need to obtain the centering value (by optimising the posterior) before running the sampler

### SVRG-langevin


### SGHMC


### SGHMC-CV


### SGHMC-SVRG
