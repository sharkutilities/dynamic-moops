# -*- encoding: utf-8 -*-

"""
A Set of Optimization Approach involving Linear Objective Functions

An function :attr:`f(x, y, ...)` is said to be linear if the degree of
the polynomials are either zero or one. In case of two independent
variables :attr:`f(x, y)` the graph is a straight line or in case of
three variables :attr:`f(x, y, z)` it is a plane.
"""

from dmoop.linear.allocation import Linear2DAllocation # pyright: ignore[reportMissingImports]
