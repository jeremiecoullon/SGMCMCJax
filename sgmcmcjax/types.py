from typing import Dict, List, Tuple, Union
from collections import namedtuple
from typing import NamedTuple, Union, Any, Callable, Optional

import jax.numpy as jnp
from jax.tree_util import register_pytree_node
import numpy as np

Array = Union[np.ndarray, jnp.ndarray]
PyTree = Union[Dict, List, Tuple, Array]

PRNGKey = jnp.ndarray


DiffusionState = namedtuple("DiffusionState", ['packed_state', 'tree_def', 'subtree_defs'])

register_pytree_node(
    DiffusionState,
    lambda xs: ((xs.packed_state,), (xs.tree_def, xs.subtree_defs)),
    lambda data, xs : DiffusionState(xs[0], data[0], data[1]))


SVRGState = namedtuple("SVRGState", ['centering_value', 'update_rate', 'fb_grad_center'],
                       defaults=(None, None, None))

class SamplerState(NamedTuple):
    diffusion_state: DiffusionState
    param_grad: PyTree
    svrg_state: SVRGState = SVRGState() # this is empty by default
    grad_info: Optional[Any] = None
