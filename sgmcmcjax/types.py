from collections import namedtuple
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple, Union

import jax.numpy as jnp
import numpy as np
from jax.tree_util import register_pytree_node

Array = Union[np.ndarray, jnp.ndarray]
PyTree = Union[Dict, List, Tuple, Array]

PRNGKey = jnp.ndarray


DiffusionState = namedtuple(
    "DiffusionState", ["packed_state", "tree_def", "subtree_defs"]
)

register_pytree_node(
    DiffusionState,
    lambda xs: ((xs.packed_state,), (xs.tree_def, xs.subtree_defs)),
    lambda data, xs: DiffusionState(xs[0], data[0], data[1]),  # type: ignore[index]
)  # type: ignore[index]


class SVRGState(NamedTuple):
    centering_value: Any = None
    fb_grad_center: Any = None


class SamplerState(NamedTuple):
    diffusion_state: DiffusionState
    param_grad: PyTree
    svrg_state: SVRGState = SVRGState()
    grad_info: Optional[Any] = None
