# -*- encoding: utf-8 -*-

"""
Defination of Nonlinear Mathematical Ratio
"""

from typing import Iterable

import numpy as np

def delta(
        xs : np.ndarray,
        senses : np.ndarray,
        method : callable | Iterable[callable],
        **kwargs
    ) -> np.ndarray:
    """
    Non-Linear Factor that Sequeeze Dimension based on a Method

    The first non-linear factor for dynamic optimization, which squeezes
    or explodes an observation based on a descriptive method like the
    mean or median. For example, considering the function as mean, then,
    the delta is a mathematical value that tries to calculate the
    deviation of :math:`|x_i - f(x)|` where :math:`f` is the average
    function of choice.

    :type  xs: np.ndarray
    :param xs: An n-dimensional array of features :attr:`(n, N)` where
        :attr:`n` is the number of observations while :attr:`N` is the
        number of features.

    :type  senses: np.ndarray
    :param senses: Set of objective per-feature which is either
        maximization (:attr:`+1`) or minimization (:attr:`-1`). If
        scaler, then all the objective is same or individually defined
        using a vector. The senses should be a (n, -1) dimensional.

    :type  method: callable | Iterable[callable]
    :param method: An aggregation function to describe each feature,
        for example :func:`np.mean`, :func:`np.median` methods. If a
        single function is passed then the same function is applied
        across all the features, using :attr:`axis = 1`, else for an
        iterable each function is applied accross the features.

    Keyword Arguments
    -----------------

    The following keyword arguments are defined for control and
    modification of the delta factor:

    :type  absolute: bool
    :param absolute: Return absolute value of delta, which is
        derived using :math:`|x_i - f(x)|` method.

    :type  reciprocal: bool
    :param reciprocal: Inverse the value if sense is minimization,
        else keep the original, defaults to True.

    Return Data
    -----------

    :rtype:  np.ndarray
    :return: An N-D array of shape :attr:`(n,)`, where the
        sequence is not ragged and ensures validity.
    """

    absolute = kwargs.get("absolute", True)
    addonbase = kwargs.get("addonbase", True)
    reciprocal = kwargs.get("reciprocal", True)

    factors = np.array([]) # set blank ndarray, populate method
    if hasattr(method, "__call__"):
        factors = xs - method(xs, axis = 1).reshape(-1, 1)
    elif hasattr(method, "__iter__"):
        factors = np.array([
            x - method[idx](x) for idx, x in enumerate(xs)
        ])
    else:
        raise ValueError("Method must be callable or iterable")

    factors = np.abs(factors) if absolute else factors
    factors = xs + factors if addonbase else factors

    if reciprocal:
        senses = np.array(senses).reshape(-1, 1)
        factors = np.where(senses == -1, 1/factors, factors)

    return np.prod(factors, axis = 0)
