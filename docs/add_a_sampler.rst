Add a sampler
=============

There are three steps to add a new sampler which correspond to the three levels of abstraction outlined in the `overview of SGMCMCJax`_. We'll illustrate how to do this for SGLD, but the other samplers will be similar.

.. _overview of SGMCMCJax: overview.rst

1. Write the diffusion solver
2. Write the kernel factory
3. Write the sampler factory


The diffusion
-------------

The way to build a diffusion follows closely JAX's optimisers_ module::

  from sgmcmcjax.diffusions import diffusion

  @diffusion
  def sgld(dt):
      dt = make_schedule(dt)

      def init_fn(x):
          return x

      def update(i, k, g, x):
          return x + dt(i)*g + jnp.sqrt(2*dt(i))*random.normal(k, shape=jnp.shape(x))

      def get_params(x):
          return x

      return init_fn, update, get_params

.. _optimisers: https://jax.readthedocs.io/en/latest/jax.experimental.optimizers.html?highlight=optimizers

After importing the `diffusion` decorator you define a function that returns three new functions. Note that we only need to define these functions to work on arrays. The diffusion decorator then extends them to work on any Pytree (ie: any parameter shape).

- `init_fn`: this function takes in the initial parameter and returns the state of the sampler. This might include momentum for underdamped Langevin samplers for example.
- `update`: takes in the iteration number, random key, gradient, and state. It returns the updated state
- `get_params`: takes in a `state` object and returns the parameter

Also note that we create the step size schedule in the first line, so the output `dt` is a function that takes in the iteration number as argument. If the initial `dt` argument is a constant then the schedule function simply returns that constant for all iterations.

The kernel factory
------------------

We then build the kernel factory. This function must return three functions that are similar to the ones in the previous section:

- `init_fn`: creates the state of the sampler from some initial parameters.
- `kernel`: takes in `i`, `key`,  and `state` and returns a new state.
- `get_params`: takes in the state and returns the parameters

You can find examples of these in `sgmcmcjax/kernels.py`::

  def build_sgld_kernel(dt, loglikelihood, logprior, data, batch_size):
      grad_log_post = build_grad_log_post(loglikelihood, logprior, data)
      init_fn, update, get_params = sgld(dt)
      estimate_gradient = build_gradient_estimation_fn(grad_log_post, data, batch_size)
      sgld_kernel = _build_langevin_kernel(update, get_params, estimate_gradient)
      return init_fn, sgld_kernel, get_params

Note that we build the gradient of the log-posterior, the diffusion, and the gradient estimation functions. We then pass these in to a helper function that builds the kernel.


The sampler factory
-------------------

To build a sampling function we use the `sgmcmc_sampler` decorator defined in `sgmcmcjax/samplers.py`. This decorator turns a kernel factory into a sampler factory::

  build_sgld_sampler = sgmcmc_sampler(build_sgld_kernel)

That's all you need to do to build the sampler! This last step is especially easy as we made sure that the kernel factorys always return the same thing.
