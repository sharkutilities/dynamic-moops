# -*- encoding: utf-8 -*-

"""
Simple Non-Linear Optimizer Model Defination
"""

import numpy as np

from typing import Iterable
from dmoop.factors import nonlinear
from dmoop.base import BaseConstruct


class DeltaNonLinearOptimizer(BaseConstruct):
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


    def fit(
        self,
        method : callable | Iterable[callable] = np.mean,
        **kwargs
    ) -> None:

        deltakwargs = {
            key : kwargs[key] for key in kwargs.keys()
            if key in ["absolute", "addonbase", "reciprocal"]
        }

        self.delta = nonlinear.delta(
            self.xs, self.senses, method = method,

            # all the required kwargs are passed into the model
            **deltakwargs
        )
        return None

    def predict(self) -> np.ndarray:
        return self.delta / self.delta.sum()
