# -*- encoding: utf-8 -*-

"""
A Set of Allocation Techniques for N-Dimensional Optimization

A multi objective optimization function based on allocation of
share based on n-dimension features which are either optimized basis
either minimization of maximization of all variables individually.
"""

import numpy as np

class Linear2DAllocation:
    """
    A Linear Allocation Algorithm for 2-Dimensional Optimization

    A mathematical approach to create an allocation factor basis on
    objective functions (either minimization/maximization) based on
    the number of dimensions available. The term "multi-objective" is
    defined because each individual "cost-functions" can be seperatly
    defined to be either a minimization or a maximization problem
    based on the sense set on each dimension.

    Important Abbreviations
    -----------------------
        * **:attr:`N`** - No. of dimensions for optimization, here the
        value is :attr:`N = 2` for 2-Dimensional Optimization. For
        example, the two dimension can be "cost" and "capacity" where
        typically the cost should be minimized while the capacity
        (i.e., the output) should be maximized.

        * **:attr:`q`** - No. of independent variables across each
        independent objective function. Typically, the :attr:`(N, q)`
        will create a :attr:`N x q` matrix to fully understand its
        potential functions.

    A typical use-case can be dynamically assigning "share of business"
    in a typical supply chain vendor optimization problem, that
    involves the cost associated to a vendor and the production capacity
    of the vendor.

    :type  xs: list
    :param xs: A typical :attr:`(2, q)`-dimensional array of cost
        function(s) where $n = 2$ represents the number of objective
        function while $q$ represent the number of independent variable
        across each dimension.

    :type  senses: int or list
    :param senses: Set of objective per-feature which is either
        maximization (:attr:`+1`) or minimization (:attr:`-1`). If
        scaler, then all the objective is same or individually defined
        using a vector. The senses should be a (n, -1) dimensional.
    """

    def __init__(self, xs : list, senses : list | int) -> None:
        self.xs = self._xs(xs) # ? conversion, assertion of `xs`
        self.senses = self._senses(senses) # ? problem objective/sense

        self.N = 2 # ! only two dimensional supported
        self.q = self.xs.shape[1] # ? q-variable in each dimension


    def _xs(self, array : list) -> np.ndarray:
        """
        Function to Assert Input Feature and Check for Ragged Nested Sequence

        The input feature (`xs`) must not be a ragged nested sequence, i.e., all
        the dimension must match the first dimension. The `VisibleDeprecationWarning`
        raised by :mod:`numpy` can however be ignore by passing `dtype = object`
        but this is rejected by the algorithm.

        :type  array: iterable
        :param array: An iterable non-ragged sequence of the shape :attr:`(2, q)`
            where :attr:`n = 2` is the number of rows, while `q` is the number of features
            and/or variables to allocate share.
        """

        array = np.stack(np.array(array, dtype = float))
        assert array.shape[0] == 2, "Only 2-Dimensional/Objective Function Supported"

        return array


    def _senses(self, senses : list | int) -> np.ndarray:
        """
        Set the Objective/Sense of all Input Dimensions (:attr:`n`) for Problem

        The objective/sense of individual element can be either a
        minimization or maximization based on the requirement. For
        minimization set :attr`sense = -1` while for maximization of set
        :attr`sense = +1`. If a scalar value is passed, then all the features
        are set to single objective based on value.

        WARNING: Typically, this algorithm should not be used in-case of
        single objective and sophisticated libraries like :mod:`PuLP` or
        :mod:`scipy` should be used.

        :type  senses: list
        :param senses: Set the sense or objective of the problem. Set
            :attr`sense = -1` for minimization while :attr`sense = +1`
            for maximization.
        """

        senses = np.array(senses) if isinstance(senses, (list, tuple, np.ndarray)) \
            else np.ones(self.xs.shape[0]) * senses

        return senses.reshape(-1, 1)


    def __describe_function_dispatch__(self, method : str) -> callable:
        """Dispatch any of the Aggregation Function"""

        methods = dict(
            min = np.min,
            max = np.max,
            mean = np.mean,
            median = np.median
        )

        return methods[method]


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

        methods = kwargs.get("methods", ["mean"] * self.N)
        factors = np.array(kwargs.get("factors", np.ones(self.N)))

        delta = np.abs([
            x - self.__describe_function_dispatch__(method = method)(x)
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

        beta = np.prod(beta, axis = 0) # final beta value
        return np.round(beta, ndigits) if ndigits else beta
