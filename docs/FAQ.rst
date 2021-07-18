Frequently asked questions
==========================



I need to calculate stuff at each iteration (ex: test accuracy)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Sometimes you need to calculate something - such as test accuracy - at regular intervals. In that case you can build the transition kernel and run a Python loop yourself along with any calculations you want. How to do this is explained `here`_.

.. _here: nbs/kernel.ipynb


Another worked out example can be found in the `CNN example`_: the test accuracy is calculated every 100 iterations and saved in a list.


.. _CNN example: nbs/Flax_MNIST.ipynb


My sampler is taking ages to compile
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The sampler function can sometimes take too long to compile is the model is really big (example: a deep CNN). In that case you can set `compiled=False` which runs a native Python loop rather than a JAX scan.

Example::

  batch_size = int(0.1*N)
  dt = 1e-5
  my_sampler = build_sgld_sampler(dt, loglikelihood, logprior, (X_data,), batch_size, compiled=False)



The SGHMC kernel is taking ages to compile
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When the model in the log-posterior is big the SGHMC kernel can take a long time to compile. This is because the leapfrog steps is run by default in a JAX scan. You can fix this by setting `compiled_leapfrog=False` which runs a Python loop instead. Namely::

  init_fn, sghmc_kernel, get_params = build_sghmc_kernel(dt, L, loglikelihood,
                                 logprior, data, batch_size, compiled_leapfrog=False)

You can see this in action in the `Flax CNN notebook example`_.


.. _Flax CNN notebook example: nbs/Flax_MNIST.ipynb#SGHMC


I want to sample minibatches of data myself
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you need to sample minibatches of data yourself (perhaps your dataset doesn't fit in memory) you can build the diffusion for the sampler using `sgmcmcjax.diffusion` and write the Python loop with the gradient estimate yourself.

You can see how to do this in `this notebook`_.


.. _this notebook: nbs/diffusion.ipynb
