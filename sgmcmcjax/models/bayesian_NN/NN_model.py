import numpy as np
import jax.numpy as jnp
from jax import jit, vmap, nn, random, scipy


"""
# Bayesian NN

Same setup as in https://arxiv.org/pdf/1907.06986.pdf
"""

from .NN_data import X_train
N_data = X_train.shape[0]

# ==========
# Functions to initialise parameters
# initialise params: list of tuples (W, b) for each layer
def random_layer(key, m, n, scale=1e-2):
    key, subkey = random.split(key)
    return (scale*random.normal(key, (n,m))), scale*random.normal(subkey, (n,))


def init_network(key, sizes):
    keys = random.split(key, len(sizes))
    return [random_layer(k,m,n) for k,m,n in zip(keys, sizes[:-1], sizes[1:])]

# ===========
# predict and accuracy functions
@jit
def predict(params, x):
    # per-example predictions
    activations = x
    for w, b in params[:-1]:
        outputs = jnp.dot(w, activations) + b
        activations = nn.softmax(outputs)

    final_w, final_b = params[-1]
    logits = jnp.dot(final_w, activations) + final_b
    return nn.log_softmax(logits)

# =================
# Log-posterior

@jit
def loglikelihood(params, X, y):
    return jnp.sum(y*predict(params, X))

def logprior(params):
    logP = 0.0
    for w, b in params:
        logP += jnp.sum(scipy.stats.norm.logpdf(w))
        logP += jnp.sum(scipy.stats.norm.logpdf(b))
    return logP


# Accuracy for a single sample
batch_predict = vmap(predict, in_axes=(None, 0))

@jit
def accuracy(params, X, y):
    target_class = jnp.argmax(y, axis=1)
    predicted_class = jnp.argmax(batch_predict(params, X), axis=1)
    return jnp.mean(predicted_class == target_class)
