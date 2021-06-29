SGMCMC samplers
===============


There are several SGMCMC samplers available. Each comes with its own set of tradeoffs. These are build from a diffusion along with an estimator for the gradient. Here we list and very briefly describe the pros and cons of each diffusion and gradient estimator. You can see them in action in `this notebook`_.


.. _this notebook: nbs/sampler.ipynb

Diffusions:
-----------

Each of these diffusions must include an estimate for the gradient denoted :math:`\hat{\nabla} \log \pi(x_n)`. These can be either the standard estimator, the control variates estimator, or the SVRG estimator.

SGLD:
^^^^^

`Stochastic gradient Langevin dynamics`_ (SGLD) is the most basic SGMCMC algorithm. The update is given by (with :math:`\xi \sim \mathcal{N}(0,1)`):

.. _Stochastic gradient Langevin dynamics: https://www.ics.uci.edu/~welling/publications/papers/stoclangevin_v6.pdf

.. math::

  x_{n+1} = x_n + dt\hat{\nabla} \log \pi(x_n) + \sqrt{2dt}\xi

**Pros:** simple to code up and understand and is fast.

**Cons:** will not sample as efficiently as other more sophisticated samplers.


SGHMC:
^^^^^^

`Stochastic gradient HMC`_: overlamped Langevin with momentum resampling and stochastic gradients. The solver uses an Euler discretisation (as in the reference):

.. math::
  \begin{cases}
  x_{n+1} &= x_n + v_n \\
  v_{n+1} &= v_n + dt\hat{\nabla} \log \pi(x_{n+1}) - \alpha v_n + \sqrt{2(\alpha - \hat{\beta_n})dt}\xi
  \end{cases}

The tuning parameters in the update equation are :math:`dt`, :math:`\alpha`, and :math:`\beta`. The original paper recommends a small value for :math:`\alpha` (such as 0.01) and :math:`\beta=0`. The number of leapfrog steps :math:`L` must also be tuned.

**Pros:** The momentum term can provide faster sampling compared to SGLD

**Cons:** there are several additional hyperparameters to tune: number of leapfrog steps `L` and the friction coefficient :math:`\alpha`

.. _Stochastic gradient HMC: https://arxiv.org/abs/1402.4102



pSGLD:
^^^^^^

`Preconditioned SGLD`_: SGLD with an adaptive (diagonal) preconditioner; this is essentially RMSProp merged with SGLD. Note that we set :math:`\Gamma(x_n)=0` as recommended in the paper.

.. _Preconditioned SGLD: https://arxiv.org/abs/1512.07666


.. math::
  \begin{cases}
  v_{n+1} &= \alpha v_n + (1-\alpha) \hat{\nabla} \log \pi(x_n)^2 \\
  G_{n+1} &= 1/ (\sqrt{v_{n+1}}+\epsilon) \\
  x_{n+1} &= x_n + \frac{dt}{2} \left(G_{n+1} \hat{\nabla} \log \pi(x_n) + \Gamma(x_{n}) \right) + \sqrt{dt G_{n+1}}\xi
  \end{cases}

**Pros:** The preconditioner can help with poorly scaled posteriors. This algorithm was designed for training deep neural networks.

**Cons:** the fact that there is no clear cutoff to the adaptation means that sometimes the samples are not of great quality.

BAOAB:
^^^^^^

`BAOAB`_ scheme for underdamped Langevin dynamics: This splitting scheme is a numerical method to solve underdamped Langevin dynamics. This was originally derived for exact gradients.

.. _BAOAB: https://aip.scitation.org/doi/abs/10.1063/1.4802990

.. math::
  \begin{cases}
  v_{n+1/2} &= v_n +  \frac{dt}{2} \hat{\nabla} \log \pi(x_n) \\
  x_{n+1/2} &= x_n + \frac{dt}{2}v_{n+1/2} \\
  \tilde{v}_{n+1/2} &= e^{-\gamma dt}v_{n+1/2} + \sqrt{\tau(1 - e^{-2\gamma dt}) }\xi \\
  x_{n+1} &= x_{n+1/2} + \frac{dt}{2}\tilde{v}_{n+1/2} \\
  v_{n+1} &= \tilde{v}_{n+1/2} +  \frac{dt}{2} \hat{\nabla} \log \pi(x_{n+1}) \\
  \end{cases}

The tuning parameters are the step size :math:`dt`, the friction coefficient :math:`\gamma`, and the temperature :math:`\tau`. By default we set :math:`\tau=1`.

SGNHT:
^^^^^^

`Stochastic Gradient Nose-Hoover thermostats`_: Extends SGHMC with a third variable.

.. math::
  \begin{cases}
  v_{n+1} &= v_n + dt\hat{\nabla} \log \pi(x_n) - \alpha v_n + \sqrt{2a dt}\xi \\
  x_{n+1} &= x_{n+1} + v_n \\
  \alpha_{n+1} &= \alpha_n + \frac{1}{p}v_{n+1}^Tv_{n+1} - dt
  \end{cases}

.. _Stochastic Gradient Nose-Hoover thermostats: http://people.ee.duke.edu/~lcarin/sgnht-4.pdf







Gradient estimators:
--------------------

Each of these estimators can be plugged into one of the diffusions defined above.


Standard estimator:
^^^^^^^^^^^^^^^^^^^

This is simply the sample mean of the gradients of a minibatch of data. This is the estimator from the `original paper`_.

.. _original paper: https://www.ics.uci.edu/~welling/publications/papers/stoclangevin_v6.pdf

**Pros:** Easy to understand and code up.

**Cons:** The variance becomes pretty high as the minibatch size decreases. This results in poor quality samples.


Control Variates:
^^^^^^^^^^^^^^^^^

`SGLD with control variates`_: This estimator uses a centering value to lower the variance of the gradient estimator.

.. _SGLD with control variates: https://arxiv.org/abs/1706.05439

**Pros:** The gradient estimate is much more accurate for log-concave posteriors for only a small added computational cost

**Cons:** The gradient estimate will lose accuracy for posteriors that are not log-concave.  You also need to obtain the centering value (by optimising the posterior) before running the sampler


SVRG:
^^^^^

`SGLD with SVRG`_: The same control variates estimator but where the centering value is updated regularly.

**Pros:** Similarly to control variates, the gradient estimate is much more accurate. Furthermore, there is no need to find a good centering value as this is updated regularly. Finally this gradient estimator `works better`_ for posteriors that are not log-concave.

**Cons:** The algorithm requires another tuning parameter: the rate at which the centering value is updated. This update can also be expensive at it requires calculating the fullbatch gradient.


.. _SGLD with SVRG: https://papers.nips.cc/paper/2016/file/9b698eb3105bd82528f23d0c92dedfc0-Paper.pdf

.. _works better: http://proceedings.mlr.press/v80/chatterji18a/chatterji18a.pdf
