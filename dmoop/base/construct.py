# -*- encoding: utf-8 -*-

import numpy as np

from abc import ABC
from typing import Iterable

class BaseConstruct(ABC):
    """
    The Abstract Meta Class Construct Defination for DMOOP

    The abstract class configures and checks for the input features
    and senses for the optimization problem. In addition, the class
    exposes the abstract methods which are inherited and defined in
    the child classes to maintain consistency.
    """

    def __init__(self, xs : Iterable, senses : Iterable, **kwargs) -> None:
        self.xs = self._xs(xs) # ? conversion, assertion of `xs`
        self.senses = self._senses(senses) # ? problem objective(s)

        # set the class attribute `N` and `q` for the problem
        self.N, self.q = self.xs.shape

        # ? optional arguments with defaults for base construct
        self._modelname = kwargs.get("modelname", None)


    def describe(self, axis : int = None, method : str = callable) -> np.ndarray:
        """
        Describe the Input Feature based on a Callable Method

        The method is any callable function available in the
        :mod:`numpy` library. Other custom methods if passed must
        support the :attr`axis` argument.

        ..versionchanged:: 2024-12-11 GH#4

            The axis argument is now available, defaults to `None` as
            in the base :mod:`numpy` method.

        ..versionchanged:: 2024-05-16 GH#1

            The method is now a callable argument which accepts any
            user defined function or callable.
        """

        return method(self.xs, axis = axis)


    @property
    def modelname(self) -> str:
        """
        Optionally Set the Model Name or Return Default Class Name
        """

        return self._modelname or self.__class__.__name__


    def _xs(self, array : Iterable) -> np.ndarray:
        """
        Construct to Check for the Input Feature and Set Data Type

        The input feature (`xs`) must not be a ragged nested sequence,
        i.e., all the dimension must match the first dimension. The
        ``VisibleDeprecationWarning`` raised by :mod:`numpy` can
        however be ignore by passing `dtype = object` but this is
        rejected by the algorithm within the algorithm.

        :type  array: iterable
        :param array: An iterable non-ragged sequence of the shape
            :attr:`(n, q)` where :attr:`n` is the number of data points,
            while `q` is the number of features.
        """

        return np.stack(np.array(array, dtype = float))


    def _senses(self, senses : Iterable | int) -> np.ndarray:
        """
        Set the Optimization Sense for all Input Dimensions

        The optimization sense of individual feature can either be a
        minimization or maximization based on the requirement. For
        minimization set :attr`sense = -1` while for maximization of
        set :attr`sense = +1`. If a scalar value is passed, then all
        the features are set to single objective based on value.

        :type  senses: list
        :param senses: Set the sense or objective of the problem. Set
            :attr`sense = -1` for minimization while :attr`sense = +1`
            for maximization.
        """

        iterables = (list, tuple, np.ndarray) # allowed iterables
        senses = np.array(senses) if isinstance(senses, iterables) \
            else np.ones(self.xs.shape[0]) * senses

        ndim_, cur = self.xs.ndim[0], self.xs.shape[0]
        assert senses.shape[0] == self.xs.shape[0], \
            f"Number of Senses and Features must match, but got {ndim_} and {cur}"

        return senses.reshape(-1, 1)


    def __repr__(self) -> str:
        return f"{self.modelname}(nobs = {self.N}, ndims = {self.q})"
