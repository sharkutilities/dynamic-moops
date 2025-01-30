# -*- encoding: utf-8 -*-

"""
A Set of Allocation Techniques for N-Dimensional Optimization

A multi objective optimization function based on allocation of
share based on n-dimension features which are either optimized basis
either minimization of maximization of all variables individually.
"""

# import statology
import numpy as np

from typing import Iterable
from dmoop.base import BaseConstruct

class SimpleLinearDeltaOptimizer(BaseConstruct):
    """
    A Linear Allocation Algorithm for N-Dimensional Optimization

    A mathematical approach to create an allocation factor basis on
    objective functions (either minimization/maximization) based on
    the number of dimensions available. The term "multi-objective" is
    defined because each individual "cost-functions" can be seperatly
    defined to be either a minimization or a maximization problem
    based on the sense set on each dimension.

    :type  xs: list
    :param xs: A typical :attr:`(N, q)`-dimensional array of cost
        function(s) where $N$ represents the number of objective
        function while $q$ represent the number of independent variable
        across each dimension.

    :type  senses: int or list
    :param senses: Set of objective per-feature which is either
        maximization (:attr:`+1`) or minimization (:attr:`-1`). If
        scaler, then all the objective is same or individually defined
        using a vector. The senses should be a (n, -1) dimensional.

    .. caution::

        If you intend to use a single object optimization, then it is
        better to use libraries like :mod:`pulp` or :mod:`scipy` where
        controls and algorithms are more efficient.

    Class Attributes
    ----------------

    The following class attributes are available once the objective is
    defined and class is initialized:

        * **:attr:`N`** - No. of dimensions for optimization. For
        example, for a two dimensional problem, it can be "cost" and
        "capacity" where typically the cost should be minimized while
        the capacity (i.e., the output) should be maximized.

        * **:attr:`q`** - No. of independent variables across each
        independent objective function. Typically, the :attr:`(N, q)`
        will create a :attr:`N x q` matrix to fully understand its
        potential functions.

    A typical use-case can be dynamically assigning "share of business"
    in a typical supply chain vendor optimization problem, that
    involves the cost associated to a vendor and the production capacity
    of the vendor.

    Keyword Arguments
    -----------------

    Addiional control on a feature/decorations can be done using the
    following keyword arguments:

        * **modelname** (*str*): The name of the model, defaults to
            the class name, else user control.
    """

    def __init__(self, xs : Iterable, senses : Iterable, **kwargs) -> None:
        super().__init__(
            xs, senses,

            # model name attribute defaults to class name
            modelname = kwargs.get("modelname", None)
        )


    def __delta__(
        self,
        method : callable | Iterable[callable],
        axis : int,
        absolute : bool,
        reciprocal : bool,
        **kwargs
    ) -> np.ndarray:
        """
        Delta Factor for Linear N-Dimensional Allocation

        The delta is a mathematical value that tries to calculate the
        deviation of :math:`|x_i - f(x)|` where :math:`f` is an
        aggregation function of choice, like mean or median.

        The factor considers the senses of each dimension and
        returns a value like :math:`i, sense = 1` (maximization) or
        reciprocal value in case of minimization.

        :type  method: callable
        :param method: An aggregation function available in the
            :mod:`numpy` library.

        :type  axis: int
        :param axis: Axis along which the method is applied, defaults
            to `1` which means the method is applied along the
            second axis. Defaults to 1.

        :type  absolute: bool
        :param absolute: Return absolute value of delta, which is
            derived using :math:`|x_i - f(x)|` method.

        :type  reciprocal: bool
        :param reciprocal: Inverse the value if sense is minimization,
            else keep the original, defaults to True.

        Keyword Arguments
        -----------------

        The following keyword arguments are defined for control and
        modification of the delta factor:

            * **described** (*dict*): The method alows keyword argument
                controls for the underlying function :func:`describe()`
                which takes in any arguments which is accepted by the
                passed method.
        """

        described = self.describe(
            self.xs,
            method = method,
            axis = axis,
            **kwargs.get("described", dict())
        ).reshape(-1, 1) # reshape to broadcast with xs

        # the base delta is the difference between self, described
        # the value is absolute or not is again parametric controlled
        delta = (self.xs - described)
        delta = np.abs(delta) if absolute else delta

        # ? finally, we inverse the value if sense if minimization, and
        # if reciprocal argument is true, else keep the original
        delta = np.array([
            list(1/xs) if sense == -1 else xs
            for xs, sense in zip(delta, self.senses)
        ]) if reciprocal else delta

        return delta

    def fit(
        self,
        method : callable | Iterable[callable] = np.mean,
        axis : int = 1,
        absolute : bool = True,
        reciprocal : bool = True,
        **kwargs
    ) -> None:
        self.delta = self.__delta__(method, axis, absolute, reciprocal, **kwargs)
        return None

    def predict(self) -> np.ndarray:
        return None
