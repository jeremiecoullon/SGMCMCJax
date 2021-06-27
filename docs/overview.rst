Overview
========

There are 3 ways to use `SGMCMCJax` which correspond to 3 levels of abstraction:


Option 1: Build a sampler function:
-----------------------------------

The first way is to build a sampling function given some hyperparameters (such as step size and batch size)::

  my_sampler = build_sgld_sampler(dt, loglikelihood, logprior, data, batch_size)

You can then call this function to obtain some samples::

  samples = my_sampler(key, Nsamples, theta_true)

See `this notebook`_ for a full example.

.. _this notebook: nbs/sampler.ipynb



Option 2: Build a transition kernel:
---------------------------------------------

Going down a level of abstraction: we can build the transition kernel for the sampler::

  init_fn, my_kernel, get_params = build_sgld_kernel(dt, loglikelihood, logprior, data, batch_size)

By doing this we obtain 3 functions:

- `init_fn`: this function takes in the initial parameter and returns a `state` object
- `update`: takes in the iteration number, random key, gradient, and state. It returns the updated state
- `get_params`: takes in a `state` object and returns the parameter


We can now write the loop ourselves and update the state using the kernel function. Note that we must also split the random key, and save the samples ourselves::

  state = init_fn(jnp.zeros(10))

  for i in tqdm(range(Nsamples)):
    key, subkey = random.split(key)
    state = my_kernel(i, subkey, state)
    samples.append(get_params(state))

Writing the loop manually is useful if we want to do things like calculate the accuracy on a test dataset throughout the sampling.

You can find a full example here_.

.. _here: nbs/kernel.ipynb


Option 3: Build a diffusion solver:
-----------------------------------

The final level of abstraction is to build the diffusion solver for the sampler::

  init_fn, update, get_params = sgld(1e-5)

The usage of the diffusion function is very similar to JAX's optimizer_ module.

Similarly to building the kernel, we obtain 3 functions. `init_fn` and `get_params` act the same as in the kernel case above. However the `update` function takes in the iteration number, random key, gradient, and state. It returns the updated state

We can then run the sampler::

  state = init_fn(jnp.zeros(10))

  for i in tqdm(range(Nsamples)):
    key, subkey = random.split(key)
    mygrad = grad_log_post(get_params(state), *data) # use all the data
    state = update(i, subkey, mygrad, state)
    samples.append(get_params(state))

Note that we also need to build the gradient of the log-posterior (`SGMCMCJax` comes with a utility function to do this), as well as calculate the gradient at each iteration ourselves. This is useful if the data doesn't fit in memory so must be regularly read from a file. It is also useful if we want to implement our own gradient estimator.


You can find a full example of this `in this notebook`_.

.. _in this notebook: nbs/diffusion.ipynb

.. _optimizer: https://jax.readthedocs.io/en/latest/jax.experimental.optimizers.html
