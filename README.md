# SGMCMCJax

[**Quickstart**](#example-usage) | [**Samplers**](#samplers) | [**Documentation**](https://sgmcmcjax.readthedocs.io/en/latest/index.html)

SGMCMCJax is a lightweight library of stochastic gradient Markov chain Monte Carlo (SGMCMC) algorithms. The aim is to include both standard samplers (SGLD, SGHMC) as well as state of the art samplers while requiring only JAX to run.

The target audience for this library is researchers and practitioners: simply plug in your JAX model and easily obtain samples.

[![DOI](https://joss.theoj.org/papers/10.21105/joss.04113/status.svg)](https://doi.org/10.21105/joss.04113)

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

## Ask a question or open an issue

Please open issues on [Github Issue Tracker](https://github.com/jeremiecoullon/SGMCMCJax/issues), or ask a question in the [Discussion section](https://github.com/jeremiecoullon/SGMCMCJax/discussions) on Github.


## Samplers

The library includes several SGMCMC algorithms with their pros and cons briefly discussed in the [documentation](https://sgmcmcjax.readthedocs.io/en/latest/all_samplers.html).

The current list of samplers is:

- SGLD
- SGLD-CV
- SVRG-Langevin
- SGHMC
- SGHMC-CV
- SVRG-SGHMC
- pSGLD
- SGLDAdam
- BAOAB
- SGNHT
- SGNHT-CV
- BADODAB
- BADODAB-CV


## Installation

Create a virtual environment and either install a stable version using pip or install the development version.

### Stable version

To install the latest stable version run:

```
pip install sgmcmcjax
```

### Development version

To install the development version run:

```
git clone https://github.com/jeremiecoullon/SGMCMCJax.git
cd SGMCMCJax
python -m pip install -e .
```
Then run the tests with `pip install -r requirements-dev.txt; make`

To run code style checks: `make lint`

## Citing SGMCMCJax

Please use the following reference to cite this repository:

```
@article{Coullon2022,
  doi = {10.21105/joss.04113},
  url = {https://doi.org/10.21105/joss.04113},
  year = {2022},
  publisher = {The Open Journal},
  volume = {7},
  number = {72},
  pages = {4113},
  author = {Jeremie Coullon and Christopher Nemeth},
  title = {SGMCMCJax: a lightweight JAX library for stochastic gradient Markov chain Monte Carlo algorithms},
  journal = {Journal of Open Source Software}
}
```

