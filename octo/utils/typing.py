from typing import Any, Mapping, Sequence, Union

import jax

# jax.random.KeyArray was removed in JAX 0.7.0; PRNG keys are now jax.Array
# with a special prng_key dtype. Use jax.Array for type annotations.
PRNGKey = jax.Array
PyTree = Union[jax.typing.ArrayLike, Mapping[str, "PyTree"]]
Config = Union[Any, Mapping[str, "Config"]]
Params = Mapping[str, PyTree]
Data = Mapping[str, PyTree]
Shape = Sequence[int]
Dtype = jax.typing.DTypeLike
