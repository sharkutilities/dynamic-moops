# -*- encoding: utf-8 -*-

"""
A Set of Mathematical Factor/Ratio for N-Dimensional Optimization

The defined ratio and their names are available to be used by any of
the optimization algorithms. Typically, when the model is trained with
the :func:`.fit()` method the same ratio is available as a class
attribute from the called function - this gives more flexibility.
"""

from dmoop.factors import linear
from dmoop.factors import nonlinear
