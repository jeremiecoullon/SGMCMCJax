import jax.numpy as jnp
import pytest
from jax import random
# import model and dataset with 2 different parameter shapes
from models import (X_data, loglikelihood_array, loglikelihood_list_array,
                    logprior_array, logprior_list_array)

from sgmcmcjax.samplers import (build_badodab_sampler, build_badodabCV_sampler,
                                build_baoab_sampler, build_psgld_sampler,
                                build_sghmc_sampler, build_sghmc_SVRG_sampler,
                                build_sghmcCV_sampler, build_sgld_sampler,
                                build_sgld_SVRG_sampler, build_sgldCV_sampler,
                                build_sgnht_sampler, build_sgnhtCV_sampler)

Ndata, D = X_data.shape
data = (X_data,)
batch_size = int(0.01 * X_data.shape[0])


def build_sampler_list(loglikelihood, logprior, param_IC):
    "Build a list of samplers with parameter param_IC"
    sgld_sampler = build_sgld_sampler(1e-5, loglikelihood, logprior, data, batch_size)
    sgldCV_sampler = build_sgldCV_sampler(
        1e-4, loglikelihood, logprior, data, batch_size, param_IC
    )
    sgldSVRG_sampler = build_sgld_SVRG_sampler(
        1e-4, loglikelihood, logprior, data, batch_size, 5
    )
    sghmc_sampler = build_sghmc_sampler(
        1e-6, 2, loglikelihood, logprior, data, batch_size
    )
    sghmcCV_sampler = build_sghmcCV_sampler(
        1e-6, 2, loglikelihood, logprior, data, batch_size, param_IC
    )
    sghmcSVRG_sampler = build_sghmc_SVRG_sampler(
        1e-6, 2, loglikelihood, logprior, data, batch_size, 5
    )
    pSGLD_sampler = build_psgld_sampler(1e-2, loglikelihood, logprior, data, batch_size)
    baoab_sampler = build_baoab_sampler(
        1e-3, loglikelihood, logprior, data, batch_size, 5
    )
    sgnht_sampler = build_sgnht_sampler(1e-5, loglikelihood, logprior, data, batch_size)
    sgnhtCV_sampler = build_sgnhtCV_sampler(
        1e-5, loglikelihood, logprior, data, batch_size, param_IC
    )
    badodab_sampler = build_badodab_sampler(
        1e-3, loglikelihood, logprior, data, batch_size
    )
    badodabCV_sampler = build_badodabCV_sampler(
        1e-3, loglikelihood, logprior, data, batch_size, param_IC
    )

    list_samplers = [
        (sgld_sampler, "sgld"),
        (sgldCV_sampler, "sgldCV"),
        (sgldSVRG_sampler, "SGLD-SVRG"),
        (sghmc_sampler, "SGHMC"),
        (sghmcCV_sampler, "SGHMC-CV"),
        (sghmcSVRG_sampler, "SGHMC-SVRG"),
        (pSGLD_sampler, "pSGLD"),
        (baoab_sampler, "BAOAB"),
        (sgnht_sampler, "SGNHT"),
        (sgnhtCV_sampler, "SGNHT"),
        (badodab_sampler, "BADODAB"),
        (badodabCV_sampler, "BADODAB-CV"),
    ]
    return [(sam, param_IC, name) for sam, name in list_samplers]


list_samplers_param_array = build_sampler_list(
    loglikelihood_array, logprior_array, jnp.zeros(D)
)
list_samplers_param_list_array = build_sampler_list(
    loglikelihood_list_array, logprior_list_array, [jnp.zeros(D), jnp.zeros(D)]
)
list_samplers_and_names = list_samplers_param_list_array + list_samplers_param_array
list_samplers = [(sam, param_IC) for sam, param_IC, _ in list_samplers_and_names]
list_names = [name for _, _, name in list_samplers_and_names]


@pytest.mark.parametrize("sam_param", list_samplers, ids=list_names)
def test_all_samplers(sam_param):
    """
    Test all samplers for 2 parameter types:
    1. JAX array: shape (D,)
    2. list of JAX arrays: shape [(D,), (D,)]

    Check that the samples have the correct shapes and that they include no Nans.
    """
    my_sampler, param_IC = sam_param
    key = random.PRNGKey(0)
    Nsamples = 10
    samples = my_sampler(key, Nsamples, param_IC)
    assert isinstance(samples, (type(param_IC)))
    if isinstance(samples, jnp.ndarray):
        assert jnp.alltrue(~jnp.isnan(samples))
    elif isinstance(samples, list):
        for elem in samples:
            assert jnp.alltrue(~jnp.isnan(elem))
    else:
        raise ValueError(
            "Parameter shapes should be either an array of a list of arrays"
        )
