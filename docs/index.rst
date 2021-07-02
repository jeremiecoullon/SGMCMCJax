.. SGMCMCJax documentation master file, created by
   sphinx-quickstart on Fri Jun 25 10:50:26 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to SGMCMCJax's documentation!
=====================================

SGMCMCJax is a lightweight library of stochastic gradient Markov chain Monte Carlo (SGMCMC) algorithms. The aim is to include both standard samplers (SGLD, SGHMC) as well as state of the art samplers (SVRG-langevin, others, ...).

The target audience for this library is researchers and practitioners: simply plug in your JAX model and easily obtain samples.

You can find the source code `on Github`_.

.. _on Github: https://github.com/jeremiecoullon/SGMCMCJax

"Hello World" example
---------------------

Estimate the mean of a Gaussian using SGLD::

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

Table of contents
-----------------


.. toctree::
    :maxdepth: 1
    :caption: How to use SGMCMCJax

    overview
    nbs/sampler
    nbs/kernel
    nbs/diffusion

.. toctree::
     :maxdepth: 1
     :caption: Examples

     nbs/gaussian
     nbs/logistic_regression
     nbs/BNN

.. toctree::
      :maxdepth: 1
      :caption: Samplers

      all_samplers

.. toctree::
      :maxdepth: 1
      :caption: Add a sampler to SGMCMCJax

      add_a_sampler
      add_sampler_special_cases
