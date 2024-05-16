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

    Keyword Arguments
    -----------------
        * **drop_outliers** (*bool*): Default False, remove outlier
            from the input :attr:`xs` using quartile range.

        * **outlier_direction** (*str* or *tuple*): Default to
            :attr:`str == "both"` i.e., removes from both the direction
            if :math:`x_i >= abs(IQR(x))` else, based on the boundary
            value.

        * **outlier_bounds** (*tuple*): Left and right bounds, defaults
            to :attr:`(0.25, 0.75)` i.e., any value above IQR3 and
            below IQR1 are removed.
    """

    def __init__(self, xs : list, senses : list | int, **kwargs) -> None:
        self.xs = self._xs(xs) # ? conversion, assertion of `xs`
        self.senses = self._senses(senses) # ? problem objective/sense

        self.N = 2 # ! only two dimensional supported
        self.q = self.xs.shape[1] # ? q-variable in each dimension

        if kwargs.get("drop_outliers", False):
            outlier_bounds = kwargs.get("outlier_bounds", (0.25, 0.75))
            outlier_direction = kwargs.get("outlier_direction", "both")

            self.xs = self.__treat_outliers__(outlier_bounds, outlier_direction)


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


    def __describe_array__(self, array : np.ndarray, method : callable) -> np.ndarray:
        """Dispatch any of the Aggregation Function"""

        return method(array) # ..versionchanged:: 16-05-2024 #1 - add method as callable


    def __treat_outliers__(self, bounds : tuple, direction : str | tuple) -> np.ndarray:
        """Method to Remove Outlier from a Feature"""

        _l_quartile = np.quantile(self.xs, bounds[0], axis = 1)
        _u_quartile = np.quantile(self.xs, bounds[1], axis = 1)

        # ? expand the directions for across the axis, and treat each axis
        direction = np.array([direction] * self.N if isinstance(direction, str) else direction)

        xs_ = [] # ? remove outlier based on condition, if string then apply same method across axis
        for idx in range(self.N):
            array = self.xs[idx]
            direction_ = direction[idx]

            if direction_ in ("left", "both"):
                xs_.append([ 0 if elem > _l_quartile[idx] else elem for elem in array ])

            elif direction_ in ("right", "both"):
                xs_.append([ elem if elem < _u_quartile[idx] else 0 for elem in array ])

            else:
                xs_.append(array) # ? when any condition is None/any other value

        return np.array(xs_)


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
            x - self.__describe_array__(x, method = method)
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
        driver = self.__describe_array__(x, method) / x
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

        _epsilon = self.__describe_array__(_x, appreciate_method)
        _epsilon = _epsilon / _x if appreciate else _beta
        appreciated_value = _beta / _epsilon

        percentage_allocation = appreciated_value / appreciated_value.sum()
        percentage_allocation_rounded = mround * np.round(percentage_allocation / mround)
        return appreciated_value, percentage_allocation, percentage_allocation_rounded
