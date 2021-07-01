import jax.numpy as jnp
from jax import random

import pytest
from sgmcmcjax.gradient_estimation import build_gradient_estimation_fn, build_gradient_estimation_fn_CV, build_gradient_estimation_fn_SVRG
from sgmcmcjax.util import build_grad_log_post
from models import X_data, loglikelihood_array, logprior_array

Ndata, D = X_data.shape
data = (X_data,)
params = jnp.zeros(D)

def test_fullbatch_standard_estimator():
    grad_log_post = build_grad_log_post(loglikelihood_array, logprior_array, data)
    batch_size = X_data.shape[0]
    estimate_gradient = build_gradient_estimation_fn(grad_log_post, data, batch_size)
    key = random.PRNGKey(0)
    mygrad = estimate_gradient(key, jnp.zeros(D))
    assert jnp.array_equal(mygrad, grad_log_post(params, *data))

batch_size = int(0.1*X_data.shape[0])
grad_log_post = build_grad_log_post(loglikelihood_array, logprior_array, data)
estimate_gradient_standard = build_gradient_estimation_fn(grad_log_post, data, batch_size)
estimate_gradient_CV = build_gradient_estimation_fn_CV(grad_log_post, data, batch_size, params)

@pytest.mark.parametrize("estimate_gradient", [estimate_gradient_standard, estimate_gradient_CV])
def test_standard_estimator_shape(estimate_gradient):
    mygrad = estimate_gradient(random.PRNGKey(0), jnp.zeros(D))
    assert type(mygrad) == type(params)
    assert jnp.shape(mygrad) == jnp.shape(params)

def test_SVRG_estimator_shape():
    batch_size = int(0.1*X_data.shape[0])
    grad_log_post = build_grad_log_post(loglikelihood_array, logprior_array, data)
    centering_value = params
    update_rate = 100
    estimate_gradient, state_SVRG = build_gradient_estimation_fn_SVRG(grad_log_post, data,
                                                          batch_size, centering_value, update_rate)
    key = random.PRNGKey(0)
    mygrad, state_SVRG = estimate_gradient(key, params, 0, state_SVRG)
    assert type(mygrad) == type(params)
    assert jnp.shape(mygrad) == jnp.shape(params)
