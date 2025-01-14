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

class LinearNDAllocation(BaseConstruct):
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

        # todo: outlier treatment at axis level is currently dropped
        # we need a control based on either senses, or user defined,
        # better not to keep, and make provision such that the outlier
        # are assigned a very low value thus maybe skipped externally
        # self.xs = self.treatoutlier(self.xs, *args, **kwargs)


    def delta(self, **kwargs) -> np.ndarray:
        """
        Calculate a Delta :attr:`Δ` Factor basis the Aggregation Function

        For a given objective funtion, calculates a :attr:`delta`
        factor which is based on the aggregation function like mean
        or median values. The value is mathematically represented
        as :math:`|x_i - f(x)|` where :math:`f` is the aggregation
        function of choice.

        Keyword Arguments
        -----------------
        The function can be internally controlled using only the
        defined keyword arguments:

            * **methods** (*list*): List of methods for aggregation
            across each individual objective function(s). Defaults to
            "mean" across all th objective funtions.

            * **factors** (*list*): The factor works as either a
            *penalty* in case of premiumization (when the objective
            is minimization) or *appreciation* (when the objective is
            maximization) depending upon the sense of the objective
            function. Defaults to :attr:`1` i.e., no factoring.
            Given a *factor* the final resolved mathematical expression
            is :math:`\delta = \tau_i * |x_i - f(x)|` where
            :math:`\tau_i` is the factor for the objective function.
        """

        methods = kwargs.get("methods", [np.mean] * self.N)
        factors = np.array(kwargs.get("factors", np.ones(self.N)))

        delta = np.abs([
            x - self.describe(x, method = method)
            for method, x in zip(methods, self.xs)
        ]) * factors.reshape(-1, 1)

        return delta


    def beta(self, ndigits : int = None, **kwargs) -> np.ndarray:
        """
        An Intermediate Derived Factor that Combines Senses and Delta

        The :attr:`ϐ`-factor combines the senses and the delta factor
        and provides an intermediate values along the function
        objective where if the objective is maximize then the value is
        directly proportional else inversely proportional to the
        final output allocation value.

        Typically, for a maximization function the *appreciation* is
        incorporated using the :attr:`factor` and is subtracted from
        the original value to reduce exponential growth of the
        function while reverse is true for *premiumization* i.e., the
        penalty factor where the value is added and reciprocal is
        calculated to find the allocation factor.

        :type  ndigits: int
        :param ndigits: Round the :attr:`ϐ`-factor to the number of
            digits, defaults to :attr:`None` (i.e., no rounding).

        Keyword Arguments
        -----------------
        The function accepts all keyword arguments as accepted by
        the :meth:`delta` function.
        """

        delta = self.delta(**kwargs)

        # ? calculate beta which also considers penalty/appreciation
        beta = np.array([
            1 / (x + d) if sense == -1 else x - d
            for sense, x, d in zip(self.senses, self.xs, delta)
        ])

        beta = np.where(beta == np.inf, 0, beta) # zero division
        beta = np.prod(beta, axis = 0) # final beta value
        return np.round(beta, ndigits) if ndigits else beta


    def epsilon(self, index : int = 0, method : str = np.mean, **kwargs) -> np.ndarray:
        """
        An Appreciation Factor for Objective Funtion to Maximize
        """

        x = self.xs[index]
        driver = self.describe(x, method) / x
        return self.beta(**kwargs) / driver


    def factor(self, mround : float = 0.05, appreciate : bool = True, **kwargs) -> np.ndarray:
        """
        Final Allocation Factor (in percentage) for Linear 2D Optimization

        The function calculates the final factor to allocate across a
        2D array of shape :attr:`(N, q)` and returns a percentage
        value which can be directly used for problems like share of
        business allocation in a typical supply chain problem.
        """

        appreciate_index = kwargs.get("appreciate_index", 0)
        appreciate_method = kwargs.get("appreciate_method", np.mean)

        _x = self.xs[appreciate_index]
        _beta = self.beta(**kwargs)

        _epsilon = self.describe(_x, appreciate_method)
        _epsilon = _epsilon / _x if appreciate else _beta
        appreciated_value = _beta / _epsilon
        appreciated_value = np.where(appreciated_value < 1e-9, 0, appreciated_value)

        percentage_allocation = appreciated_value / appreciated_value.sum()
        percentage_allocation_rounded = mround * np.round(percentage_allocation / mround)
        return appreciated_value, percentage_allocation, percentage_allocation_rounded
