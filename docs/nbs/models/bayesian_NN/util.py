import numpy as np
import jax.numpy as jnp
import struct


def one_hot(x, k, dtype=jnp.float32):
    "Create a one-hot encoding of x of size k."
    return jnp.array(x[:, None] == jnp.arange(k), dtype)

def read_idx(filename):
    "to open idx file (for the notMNIST dataset)"
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)

def flatten_NN_params(params):
    """
    params: list of NN parameters
        Each param in a list of (mat, vect) for each layer
    """
    flattened_params = []
    for lesam in params:
        flattened_params.append(np.concatenate([np.concatenate([mat.flatten(), vect]) for mat, vect in lesam]))
    return jnp.array(flattened_params)

def _flatten_jax(layer):
    "Utility function for flatten_NN_params_jaxscan"
    a10 = np.array([e.flatten() for e in layer[0]])
    a11 = layer[1]
    return np.concatenate([a10, a11], axis=1)

def flatten_NN_params_jaxscan(params):
    """
    Flatten NN params that came out of `jax.lax.scan`
    """
    return jnp.concatenate([_flatten_jax(layer) for layer in params], axis=1)



# def load_NN_params():
#     params = np.load(f"{BASE_DIR}/data/NN_params.npy", allow_pickle=True)
#     return [tuple([jnp.array(e2) for e2 in e1]) for e1 in params]

#
# def load_NN_MAP():
#     params = np.load(f"{BASE_DIR}/data/NN_MAP.npy", allow_pickle=True)
#     return [tuple([jnp.array(e2) for e2 in e1]) for e1 in params]
