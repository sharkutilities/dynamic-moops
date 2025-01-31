# -*- encoding: utf-8 -*-

"""
A Set of Mathematical Factor/Ratio for N-Dimensional Optimization
"""

import numpy as np

def alpha(xs : np.ndarray, senses : np.ndarray) -> np.ndarray:
    """
    The Factor Squeezes an N-Dimensional Array by Feature Multiplication

    An n-dimensional array of features :attr:`(n, N)` where :attr:`n`
    is the number of observations while :attr:`N` is the number of
    features is squeezed by multiplication of all the features
    together based on the sense of the optimization.

    :type  xs: np.ndarray
    :param xs: An n-dimensional array of features :attr:`(n, N)` where
        :attr:`n` is the number of observations while :attr:`N` is the
        number of features.

    :type  senses: np.ndarray
    :param senses: Set of objective per-feature which is either
        maximization (:attr:`+1`) or minimization (:attr:`-1`). If
        scaler, then all the objective is same or individually defined
        using a vector. The senses should be a (n, -1) dimensional.

    Return Data
    -----------

    :rtype:  np.ndarray
    :return: An N-D array of shape :attr:`(n,)`, where the
        sequence is not ragged and ensures validity.
    """


    senses = np.array(senses).reshape(-1, 1)
    return np.prod(np.where(senses == -1, 1/xs, xs), axis = 0)
