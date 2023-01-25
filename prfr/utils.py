from sklearn.model_selection import train_test_split
import numpy as np

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    _has_jax = False
else:
    _has_jax = True


def split_arrays(*arrays, test_size, valid_size=0):
    """
    Split arrays into train, test and (optionally) validation sets.

    Parameters
    ----------
    *arrays : list of arrays
        Arrays to split.
    test_size : float
        Fraction of the data to be used for testing.
    valid_size : float, optional
        Fraction of the remaining data to be used for validation.

    Returns
    -------
    train, test, valid : tuple of list of arrays, where each list
        is the same length as the input arrays.
    """
    split = train_test_split(*arrays, test_size=test_size)
    train_arrays = split[::2]
    test_arrays = split[1::2]

    if valid_size > 0:
        split_valid = train_test_split(*train_arrays, test_size=valid_size)
        train_arrays = split_valid[::2]
        valid_arrays = split_valid[1::2]

        return train_arrays, test_arrays, valid_arrays

    return train_arrays, test_arrays


def ecdf(x):
    if _has_jax:
        x = jnp.sort(x)
        y = jnp.arange(1, x.size + 1) / x.size
    else:
        x = np.sort(x)
        y = np.arange(1, x.size + 1) / x.size
    return x, y


def check_calibration(preds, truth):
    """
    Check calibration of the model. Applies the inverse empirical CDF to the
    predictions and compares them to the true values. The resulting
    distribution of values should be close to uniform.
    """
    if _has_jax:
        ecx, ecy = jax.vmap(jax.vmap(ecdf, in_axes=(0,)), in_axes=(0,))(preds)
        qtls = jax.vmap(jax.vmap(jnp.interp, in_axes=(0, 0, 0)), in_axes=(0, 0, 0))(
            truth, ecx, ecy
        )
        return qtls
    else:
        raise NotImplementedError("This method currently requires JAX to be installed.")
