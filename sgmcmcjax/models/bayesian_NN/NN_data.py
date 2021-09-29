import jax.numpy as jnp
import tensorflow_datasets as tfds

"""
# Bayesian NN

Load MNIST data from tensorflow_datasets
"""

# 1. MNIST
# ======
# load data
data_dir = '/tmp/tfds'

# Fetch full datasets for evaluation
# tfds.load returns tf.Tensors (or tf.data.Datasets if batch_size != -1)
# You can convert them to NumPy arrays (or iterables of NumPy arrays) with tfds.dataset_as_numpy
# mnist_data, info = tfds.load(name="mnist", batch_size=-1, data_dir=data_dir, with_info=True, as_supervised=True)
mnist_data, info = tfds.load(name="mnist", batch_size=-1, with_info=True, as_supervised=True)
mnist_data = tfds.as_numpy(mnist_data)
data_train, data_test = mnist_data['train'], mnist_data['test']

def one_hot(x, k, dtype=jnp.float32):
    "Create a one-hot encoding of x of size k."
    return jnp.array(x[:, None] == jnp.arange(k), dtype)

y_train = one_hot(data_train[1], 10)
y_test = one_hot(data_test[1], 10)

X_train = data_train[0]
X_train = X_train.reshape(X_train.shape[0], 28*28)

X_test = data_test[0]
X_test = X_test.reshape(X_test.shape[0], 28*28)

# Normalizing the RGB codes by dividing it to the max RGB value.
X_train = X_train/255
X_test = X_test/255
X_train = jnp.array(X_train)
y_train = jnp.array(y_train)
N_data = X_train.shape[0]
