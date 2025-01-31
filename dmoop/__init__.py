# -*- encoding: utf-8 -*-

"""
Dynamic Multi-Objective Optimization Problems (dynMOOPs) Source Code

A mathematical implementation of the classical :attr:`dynMOOPs` for
practical problems that typically involves a set of conflicting
objectives with premiumizationpenalty and value appreciation controls.
"""

# ? package follows https://peps.python.org/pep-0440/
# ? https://python-semver.readthedocs.io/en/latest/advanced/convert-pypi-to-semver.html
__version__ = "0.0.1.dev0"

# register for init-time options registrations
from dmoop import linear # noqa: F401, F403 # pyright: ignore[reportMissingImports]
from dmoop import factors # noqa: F401, F403 # pyright: ignore[reportMissingImports]
