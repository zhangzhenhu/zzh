import numpy as np
import warnings

__all__ = [
    # 'label_binarize',
    # 'LabelBinarizer',
    'LabelEncoder',
    # 'MultiLabelBinarizer',
]


def _num_samples(x):
    """Return number of samples in array-like x."""
    if hasattr(x, 'fit') and callable(x.fit):
        # Don't get num_samples from an ensembles length!
        raise TypeError('Expected sequence or array-like, got '
                        'estimator %s' % x)
    if not hasattr(x, '__len__') and not hasattr(x, 'shape'):
        if hasattr(x, '__array__'):
            x = np.asarray(x)
        else:
            raise TypeError("Expected sequence or array-like, got %s" %
                            type(x))
    if hasattr(x, 'shape'):
        if len(x.shape) == 0:
            raise TypeError("Singleton array %r cannot be considered"
                            " a valid collection." % x)
        return x.shape[0]
    else:
        return len(x)


def column_or_1d(y, warn=False):
    """ Ravel column or 1d numpy array, else raises an error

    Parameters
    ----------
    y : array-like

    warn : boolean, default False
       To control display of warnings.

    Returns
    -------
    y : array

    """
    shape = np.shape(y)
    if len(shape) == 1:
        return np.ravel(y)
    if len(shape) == 2 and shape[1] == 1:
        if warn:
            warnings.warn("A column-vector y was passed when a 1d array was"
                          " expected. Please change the shape of y to "
                          "(n_samples, ), for example using ravel().",
                          stacklevel=2)
        return np.ravel(y)

    raise ValueError("bad input shape {0}".format(shape))


def _encode_check_unknown(values, uniques, return_mask=False):
    """
    Helper function to check for unknowns in values to be encoded.

    Uses pure python method for object dtype, and numpy method for
    all other dtypes.

    Parameters
    ----------
    values : array
        Values to check for unknowns.
    uniques : array
        Allowed uniques values.
    return_mask : bool, default False
        If True, return a mask of the same shape as `values` indicating
        the valid values.

    Returns
    -------
    diff : list
        The unique values present in `values` and not in `uniques` (the
        unknown values).
    valid_mask : boolean array
        Additionally returned if ``return_mask=True``.

    """
    if values.dtype == object:
        uniques_set = set(uniques)
        diff = list(set(values) - uniques_set)
        if return_mask:
            if diff:
                valid_mask = np.array([val in uniques_set for val in values])
            else:
                valid_mask = np.ones(len(values), dtype=bool)
            return diff, valid_mask
        else:
            return diff
    else:
        unique_values = np.unique(values)
        diff = list(np.setdiff1d(unique_values, uniques, assume_unique=True))
        if return_mask:
            if diff:
                valid_mask = np.in1d(values, uniques)
            else:
                valid_mask = np.ones(len(values), dtype=bool)
            return diff, valid_mask
        else:
            return diff


def _encode_numpy(values, uniques=None, encode=False):
    # only used in _encode below, see docstring there for details
    if uniques is None:
        if encode:
            uniques, encoded = np.unique(values, return_inverse=True)
            return uniques, encoded
        else:
            # unique sorts
            return np.unique(values)
    if encode:
        diff = _encode_check_unknown(values, uniques)
        if diff:
            raise ValueError("y contains previously unseen labels: %s"
                             % str(diff))
        encoded = np.searchsorted(uniques, values)
        return uniques, encoded
    else:
        return uniques


def _encode_python(values, uniques=None, encode=False):
    # only used in _encode below, see docstring there for details
    if uniques is None:
        uniques = sorted(set(values))
        uniques = np.array(uniques, dtype=values.dtype)
    if encode:
        table = {val: i for i, val in enumerate(uniques)}
        try:
            encoded = np.array([table[v] for v in values])
        except KeyError as e:
            raise ValueError("y contains previously unseen labels: %s"
                             % str(e))
        return uniques, encoded
    else:
        return uniques


def _encode(values, uniques=None, encode=False):
    """Helper function to factorize (find uniques) and encode values.

    Uses pure python method for object dtype, and numpy method for
    all other dtypes.
    The numpy method has the limitation that the `uniques` need to
    be sorted. Importantly, this is not checked but assumed to already be
    the case. The calling method needs to ensure this for all non-object
    values.

    Parameters
    ----------
    values : array
        Values to factorize or encode.
    uniques : array, optional
        If passed, uniques are not determined from passed values (this
        can be because the user specified categories, or because they
        already have been determined in fit).
    encode : bool, default False
        If True, also encode the values into integer codes based on `uniques`.

    Returns
    -------
    uniques
        If ``encode=False``. The unique values are sorted if the `uniques`
        parameter was None (and thus inferred from the data).
    (uniques, encoded)
        If ``encode=True``.

    """
    if values.dtype == object:
        return _encode_python(values, uniques, encode)
    else:
        return _encode_numpy(values, uniques, encode)


class LabelEncoder(object):
    """Encode labels with value between 0 and n_classes-1.
    """

    def __init__(self):
        self._uniques = set()
        self._table = {}
        self._sequence = []

    def fit(self, *args):
        """Fit label encoder

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : returns an instance of self.
        """
        # uniques = set()
        for y in args:
            # y = column_or_1d(y, warn=True)
            for e in y:
                if e not in self._uniques:
                    self._uniques.add(e)
                    self._sequence.append(e)

        # self.classes_ = _encode(y)
        _sequence = np.asarray(self._sequence)
        self._table = {val: i for i, val in enumerate(_sequence)}
        return self

    def fit_transform(self, y):
        """Fit label encoder and return encoded labels

        Parameters
        ----------
        y : array-like of shape [n_samples]
            Target values.

        Returns
        -------
        y : array-like of shape [n_samples]
        """
        self.fit(y)
        return self.transform(y)
        # y = column_or_1d(y, warn=True)
        # self.classes_, y = _encode(y, encode=True)
        # return y

    def transform(self, y):
        """Transform labels to normalized encoding.

        Parameters
        ----------
        y : array-like of shape [n_samples]
            Target values.

        Returns
        -------
        y : array-like of shape [n_samples]
        """
        # check_is_fitted(self, 'classes_')
        # if not hasattr(self, 'classes_'):
        #     raise TypeError("还没训练")

        # y = column_or_1d(y, warn=True)
        # transform of empty array is empty array
        # if _num_samples(y) == 0:
        #     return np.array([])

        # _, y = _encode(y, uniques=self.classes_, encode=True)
        # return y

        try:
            encoded = np.array([self._table[v] for v in y], dtype=np.int32)
        except KeyError as e:
            raise ValueError("y contains previously unseen labels: %s"
                             % str(e))
        return encoded

    def inverse_transform(self, y):
        """Transform labels back to original encoding.

        Parameters
        ----------
        y : numpy array of shape [n_samples]
            Target values.

        Returns
        -------
        y : numpy array of shape [n_samples]
        """
        # if not hasattr(self, 'classes_'):
        if self._sequence is None:
            raise TypeError("还没训练")
        y = column_or_1d(y, warn=True)
        # inverse transform of empty array is empty array
        if _num_samples(y) == 0:
            return np.array([])
        _sequence = np.asarray(self._sequence)
        diff = np.setdiff1d(y, np.arange(len(self._sequence)))
        if len(diff):
            raise ValueError(
                "y contains previously unseen labels: %s" % str(diff))
        y = np.asarray(y)
        return _sequence[y]
