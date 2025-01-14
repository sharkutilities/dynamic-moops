# -*- encoding: utf-8 -*-

import numpy as np

from typing import Iterable
from abc import ABC, abstractmethod

class BaseConstruct(ABC):
    """
    The Abstract Meta Class Construct Defination for DMOOP

    The abstract class configures and checks for the input features
    and senses for the optimization problem. In addition, the class
    exposes the abstract methods which are inherited and defined in
    the child classes to maintain consistency.
    """

    def __init__(
        self,
        xs : Iterable,
        senses : Iterable,
        modelname : str = None
    ) -> None:
        self.xs = self._xs(xs) # ? conversion, assertion of `xs`
        self.senses = self._senses(senses) # ? problem objective(s)

        # set the class attribute `N` and `q` for the problem
        # N: no. of observations, typically the number of data points
        # q: no. of features, which must be equal to the senses shape
        self.N, self.q = self.xs.shape[::-1]

        # ! base construct should not have a kwargs control, so
        # setting the modelname attribute here, defaults to None
        self._modelname = modelname


    @staticmethod
    def describe(
        xs : np.ndarray,
        method : callable = np.mean,
        axis : int = None,
        **kwargs
    ) -> np.ndarray:
        """
        Describe the Input Feature based on a Callable Method

        The method is any callable function available in the
        :mod:`numpy` library. Other custom methods if passed must
        support the :attr`axis` argument.

        :type  xs: np.ndarray
        :param xs: Input array of shape :attr:`(N, q)` where :attr:`N`
            is the number of data points and :attr:`q` is the number
            of features.

        :type  axis: int
        :param axis: Axis along which the method is applied, defaults
            to `None` which means the method is applied along the
            first axis. Defaults to None.

        ..versionchanged:: 2024-12-11 GH#4

            The axis argument is now available, defaults to `None` as
            in the base :mod:`numpy` method.

        :type  method: callable
        :param method: Any reduction function that supports :attr:`axis`
            argument, prefered to be a :mod:`numpy` method. Defaults to
            `np.mean`, i.e., the average.

        ..versionchanged:: 2024-05-16 GH#1

            The method is now a callable argument which accepts any
            user defined function or callable.

        Keyword Arguments
        -----------------

        The keyword argument gives the control to the underlying method
        and is directly passed to the method. This should not be used
        for other use cases for controlling.

        Return Data
        -----------

        Inclusion of :attr:`axis` makes the output array dimension
        dynamic. However, returns a reduced array based on method.

        :rtype:  np.ndarray
        :return: A reduced array based on the input function.
        """

        return method(xs, axis = axis, **kwargs)


    @abstractmethod
    def fit(self, *args, **kwargs) -> None:
        pass


    @abstractmethod
    def predict(self, *args, **kwargs) -> np.ndarray:
        pass


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

        Return Data
        -----------

        The main objective of the method is to ensure validity of the
        input feature and return it as a N-D array.

        :rtype:  np.ndarray
        :return: An N-D array of shape :attr:`(n, q)`, where the
            sequence is not ragged and ensures validity.
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

        ndim_, cur_ = self.xs.shape[0], senses.shape[0]
        assert ndim_ == cur_, \
            f"Number of Senses ({cur_}) must match" +\
            f"the Number of Features ({ndim_})."

        return senses.reshape(-1, 1)


    def __repr__(self) -> str:
        return f"{self.modelname}(nobs = {self.N}, ndims = {self.q})"
