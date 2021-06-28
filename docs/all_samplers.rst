SGMCMC samplers
===============


Samplers:
---------

There are several SGMCMC samplers available. Each comes with its own set of tradeoffs. Here we list them and very briefly describe the pros and cons of each. You can see them in action in `this notebook`_.

.. _this notebook: nbs/sampler.ipynb

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

`SGLD with SVRG`_: Same algorithm of SGLD-CV but where the centering value is updated regularly.

**Pros:** Similarly to SGLD-CV, the gradient estimate is much more accurate. Furthermore, there is no need to find a good centering value as this is updated regularly. Finally this gradient estimator `works better`_ for posterior that are not log-concave.

**Cons:** The algorithm requires another tuning parameter: the rate at which the centering value is updated. This update can also be expensive at it requires calculating the fullbatch gradient.


.. _SGLD with SVRG: https://papers.nips.cc/paper/2016/file/9b698eb3105bd82528f23d0c92dedfc0-Paper.pdf

.. _works better: http://proceedings.mlr.press/v80/chatterji18a/chatterji18a.pdf

SGHMC:
^^^^^^

`Stochastic gradient HMC`_: overlamped Langevin with momentum resampling and stochastic gradients. The solver uses an Euler discretisation (as in the reference).

**Pros:** The momentum term can provide faster sampling compared to SGLD

**Cons:** there are 2 additional hyperparameters to tune: number of leapfrog steps `L` and the friction coefficient `alpha`

.. _Stochastic gradient HMC: https://arxiv.org/abs/1402.4102

SGHMC-CV:
^^^^^^^^^

SGHMC with control variates: the gradient estimate uses control variates as in SGLD-CV

**Pros:** Same as for SGLD-CV

**Cons:** Same as for SGLD-CV


SVRG-SGHMC:
^^^^^^^^^^^

SGHMC with SVRG: like SVRG-langevin, the centering value is regularly updated

**Pros:** Same as for SVRG-Langevin

**Cons:** Same as for SVRG-Langevin

pSGLD:
^^^^^^

`Preconditioned SGLD`_: SGLD with an adaptive (diagonal) preconditioner; this is essentially RMSProp merged with SGLD. Note that we set :math:`\Gamma(\theta)=0` as recommended in the paper.

.. _Preconditioned SGLD: https://arxiv.org/abs/1512.07666

**Pros:** The preconditioner can help with poorly scaled posteriors

**Cons:** the fact that there is no clear cutoff to the adaptation means that sometimes the samples are not of great quality.

BAOAB:
^^^^^^

`Underdamped Langevin scheme`_: The BAOAB scheme is a numerical method to solve underdamped Langevin dynamics

.. _Underdamped Langevin scheme: https://aip.scitation.org/doi/abs/10.1063/1.4802990


SGNHT:
^^^^^^

`Stochastic Gradient Nose-Hoover thermostats`_: Extends SGHMC with a third variable.

.. _Stochastic Gradient Nose-Hoover thermostats: http://people.ee.duke.edu/~lcarin/sgnht-4.pdf
