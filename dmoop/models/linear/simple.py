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
from dmoop.factors import linear
from dmoop.base import BaseConstruct

class SimpleLinearOptimizer(BaseConstruct):
    """
    Simple Multiple Objective Optimization

    The most simple way to find a solution to a multi-objective is to
    multiply all the features together, given maximization from the
    features that needs minimizations.
    """

    def __init__(self, xs : Iterable, senses : Iterable, **kwargs) -> None:
        super().__init__(
            xs, senses,

            # model name attribute defaults to class name
            modelname = kwargs.get("modelname", None)
        )


    def fit(self) -> None:
        """
        The Simplistic Model for Linear N-Dimensional Allocation
        """

        self.alpha = linear.alpha(self.xs, self.senses)
        return None


    def predict(self) -> np.ndarray:
        return self.alpha / self.alpha.sum()
