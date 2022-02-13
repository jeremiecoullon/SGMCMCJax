SGMCMC samplers
===============


There are several SGMCMC samplers available. Each comes with its own set of tradeoffs. These are built from a diffusion along with an estimator for the gradient. Here we list and very briefly describe the pros and cons of each diffusion and gradient estimator. You can see them in action in `this notebook`_.


.. _this notebook: nbs/sampler.ipynb

Diffusions:
-----------

Each of these diffusions must include an estimate for the gradient denoted :math:`\hat{\nabla} \log \pi(x_n)`. These can be either the standard estimator, the control variates estimator, or the SVRG estimator. Throughout we define :math:`\xi \sim \mathcal{N}(0,1)`.

SGLD:
^^^^^

`Stochastic gradient Langevin dynamics`_ (SGLD) is the most basic SGMCMC algorithm: it's solves the overdamped Langevin diffusion using an Euler solver. The update is given by:

.. _Stochastic gradient Langevin dynamics: https://www.ics.uci.edu/~welling/publications/papers/stoclangevin_v6.pdf

.. math::

  x_{n+1} = x_n + dt\hat{\nabla} \log \pi(x_n) + \sqrt{2dt}\xi

**Pros:** simple to code up and understand and is fast.

**Cons:** will not sample as efficiently as other more sophisticated samplers.


SGHMC:
^^^^^^

`Stochastic gradient HMC`_: Euler discretisation of overamped Langevin with momentum resampling and stochastic gradients.

.. math::
  \begin{cases}
  x_{n+1} &= x_n + v_n \\
  v_{n+1} &= v_n + dt\hat{\nabla} \log \pi(x_{n+1}) - \alpha v_n + \sqrt{2(\alpha - \hat{\beta_n})dt}\xi
  \end{cases}

The tuning parameters in the update equation are :math:`dt`, :math:`\alpha`, and :math:`\beta`. The original paper recommends a small value for :math:`\alpha` (such as 0.01) and :math:`\beta=0`. The number of leapfrog steps :math:`L` must also be tuned.

**Pros:** The momentum term can provide faster sampling compared to SGLD

**Cons:** there are several additional hyperparameters to tune: number of leapfrog steps `L` and the friction coefficient :math:`\alpha`

.. _Stochastic gradient HMC: https://arxiv.org/abs/1402.4102



SGNHT:
^^^^^^

`Stochastic Gradient Nose-Hoover thermostats`_: Extends SGHMC with a third variable and uses an Euler solver.

.. math::
  \begin{cases}
  v_{n+1} &= v_n + dt\hat{\nabla} \log \pi(x_n) - \alpha_n v_n + \sqrt{2a dt}\xi \\
  x_{n+1} &= x_{n+1} + v_n \\
  \alpha_{n+1} &= \alpha_n + \frac{1}{D}v_{n+1}^Tv_{n+1} - dt
  \end{cases}

.. _Stochastic Gradient Nose-Hoover thermostats: http://people.ee.duke.edu/~lcarin/sgnht-4.pdf

Here :math:`D` is the dimension of the parameter. The tunable parameters are :math:`dt` and :math:`a` (default :math:`a=0.01`).

**Pros:** The friction term adapts to the amount of noise in the gradient estimate.

**Cons:** The performance of the sampler is quite sensitive to step size as the Euler solver is not as accurate and stable as a splitting scheme such as `BADODAB`_. Finally, the sampler takes time to adapt the friction term to match the noise of the gradients.


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

**Cons:** Unlike RMSProp the step size does not tune automatically; a good value of :math:`dt` is necessary for good performance.

BAOAB:
^^^^^^

`BAOAB`_: This splitting scheme is a numerical method to solve underdamped Langevin dynamics. This was originally derived for exact gradients.

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


BADODAB:
^^^^^^^^

`BADODAB`_ scheme for SGNHT: This splitting scheme is a numerical method to solve the SGNHT equations:

.. _BADODAB: https://arxiv.org/pdf/1505.06889.pdf

.. math::
  \begin{cases}
  v_{n+1/2} &= v_n +  \frac{dt}{2} \hat{\nabla} \log \pi(x_n) \\
  x_{n+1/2} &= x_n + \frac{dt}{2}v_{n+1/2} \\
  \alpha_{n+1/2} &=  \alpha_n + \frac{dt}{2\mu} \left( v_{n+1/2}^Tv_{n+1/2} - D \right) \\
  \text{if } \alpha_{n+1/2} \neq 0 : \tilde{v}_{n+1/2} &= e^{-\alpha_{n+1/2} dt}v_{n+1/2} + \sigma\sqrt{(1 - e^{-2\alpha_{n+1/2} dt})/ 2\alpha_{n+1/2} } \xi \\
  \text{else }: \tilde{v}_{n+1/2} &=  v_{n+1/2} + \sigma \sqrt{dt} \xi\\
  \alpha_{n+1} &=  \alpha_{n+1/2} + \frac{dt}{2\mu} \left( \tilde{v}_{n+1/2}^T \tilde{v}_{n+1/2} - D \right) \\
  x_{n+1} &= x_{n+1/2} + \frac{dt}{2}\tilde{v}_{n+1/2} \\
  v_{n+1} &= \tilde{v}_{n+1/2} +  \frac{dt}{2} \hat{\nabla} \log \pi(x_{n+1}) \\
  \end{cases}

The tuning parameters are :math:`dt` and :math:`a` (the initial value of :math:`\alpha` with default: :math:`a=0.01`). The two other parameters are fixed: :math:`\mu=1` and :math:`\sigma=1`.

**Pros:** The friction term adapts to the amount of noise in the gradient estimate, and the splitting scheme is more accurate and stable than the Euler method in `SGNHT`_. This allows a larger range of step sizes and smaller minibatches.

**Cons:** The sampler takes time to adapt the friction term to match the noise of the gradients.

.. _SGNHT: http://people.ee.duke.edu/~lcarin/sgnht-4.pdf

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
