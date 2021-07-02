Special cases
=============

To build a new sampler there are a few special cases that don't exactly follow the previous section. Here we look at how to write splitting schemes and samplers that resample momentum.

Splitting schemes
-----------------

Splitting schemes require a gradient evaluation halfway through the diffusion solver. This changes how we define the diffusion as well as the kernel factory.

Our diffusion solver needs a different decorator and needs to include two update functions. We show how this is done with the BAOAB sampler::

  @diffusion_palindrome
  def baoab(dt, gamma, tau=1):
      dt = make_schedule(dt)

      def init_fn(x):
          v = jnp.zeros_like(x)
          return x, v

      def update1(i, k, g, state):
          x, v = state

          v = v + dt(i)*0.5*g
          x = x + v*dt(i)*0.5

          c1 = jnp.exp(-gamma*dt(i))
          c2 = jnp.sqrt(1 - c1**2)
          v = c1*v + tau*c2*random.normal(k, shape=jnp.shape(v))

          x = x + v*dt(i)*0.5

          return x, v

      def update2(i, k, g, state):
          x, v = state
          v = v + dt(i)*0.5*g
          return x, v

      def get_params(state):
          x, _ = state
          return x

      return init_fn, update1, update2, get_params

Furthermore, the gradient will need to be carried along throughout sampling so that we can re-use it at the next iteration. The kernel factory will therefore need to include the gradient as part of state. We use a different helper function to build the kernel factory and we redefine `init_fn` and `get_params`::


  def build_badodab_kernel(dt, loglikelihood, logprior, data, batch_size, a=0.01):
      grad_log_post = build_grad_log_post(loglikelihood, logprior, data)
      init_fn, update1, update2, get_params = badodab(dt, a)
      estimate_gradient = build_gradient_estimation_fn(grad_log_post, data, batch_size)
      badodab_kernel = _build_palindrome_kernel(update1, update2, get_params, estimate_gradient)

      def new_init_fn(params):
          param_grads = grad_log_post(params, *data)
          return (init_fn(params), param_grads)

      def new_get_params(state):
          state_params, _ = state
          return get_params(state_params)

      return new_init_fn, badodab_kernel, new_get_params

As the kernel factory still returns the same things, the way to build the sampler factory is unchanged.


Momentum resampling
-------------------

Samplers such as SGHMC resample momentum every `L` iterations (with `L` the number of leapfrog steps). Similarly to splitting schemes, the diffusion and kernel factory must therefore be written slightly differently.

The diffusion function must now include a `resample_momentum` function::


  @diffusion_sghmc
  def sghmc(dt, alpha=0.01, beta=0):
      "https://arxiv.org/abs/1402.4102"
      dt = make_schedule(dt)

      def init_fn(x):
          v = jnp.zeros_like(x)
          return x, v

      def update(i, k, g, state):
          x, v = state
          x = x + v
          v = v + dt(i)*g - alpha*v + jnp.sqrt(2*(alpha - beta)*dt(i))*random.normal(k, shape=jnp.shape(x))
          return x, v

      def get_params(state):
          x, _ = state
          return x

      def resample_momentum(i, k, x):
          v = jnp.sqrt(dt(i))*random.normal(k, shape=jnp.shape(x))
          return x, v

      return init_fn, update, get_params, resample_momentum

The kernel factory now simply uses a different helper function (`_build_sghmc_kernel`) to build the kernel::

  def build_sghmc_kernel(dt, L, loglikelihood, logprior, data, batch_size, alpha=0.01):
      grad_log_post = build_grad_log_post(loglikelihood, logprior, data)
      init_fn, update, get_params, resample_momentum = sghmc(dt, alpha)
      estimate_gradient = build_gradient_estimation_fn(grad_log_post, data, batch_size)
      sghmc_kernel = _build_sghmc_kernel(L, update, get_params, resample_momentum, estimate_gradient)
      return init_fn, sghmc_kernel, get_params

The sampler factory is then built in the usual way.
