SGMCMC samplers
===============


Samplers:
---------

There are several SGMCMC samplers available. Each comes with its own set of tradeoffs. Here we list them and very briefly describe the pros and cons of each.

SGLD:
^^^^^

`Stochastic gradient Langevin dynamics`_ (SGLD) is the most basic SGMCMC algorithm.

.. _Stochastic gradient Langevin dynamics: https://www.ics.uci.edu/~welling/publications/papers/stoclangevin_v6.pdf

**Pros:** simple to code up and understand and fast.

**Cons:** The variance estimated from samples quickly becomes too large as the batch size decreases

SGLD-CV:
^^^^^^^^

`SGLD with control variates`_: the same update as SGLD but with a better estimate for the gradient

.. _SGLD with control variates: https://arxiv.org/abs/1706.05439

**Pros:** The gradient estimate is much more accurate for log-concave posteriors for only a small added computational cost

**Cons:** The gradient estimate will lose accuracy for posteriors that are not log-concave.  You also need to obtain the centering value (by optimising the posterior) before running the sampler


SVRG-langevin:
^^^^^^^^^^^^^^


SGHMC:
^^^^^^

SGHMC-CV:
^^^^^^^^^


SVRG-SGHMC:
^^^^^^^^^^^


pSGLD:
^^^^^^
