from sklearn.model_selection import train_test_split


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
