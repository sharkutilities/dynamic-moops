#!/usr/bin/env python
#
# Copywright (C) 2024 Debmalya Pramanik <neuralNOD@gmail.com>
# LICENSE: MIT License

from setuptools import setup
from setuptools import find_packages

import dmoop

setup(
    name = "dynamic-moops",
    version = dmoop.__version__,
    author = "shark-utilities developers",
    author_email = "neuralNOD@outlook.com",
    description = "A mathematical implementation of dynamic MOO problems for a practical problem typically involving conflicting objectives.",
    long_description = open("README.md", "r").read(),
    long_description_content_type = "text/markdown",
    # url = "https://github.com/sharkutilities/pandas-wizard",
    packages = find_packages(),
    classifiers = [
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: End Users/Desktop",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "Operating System :: Unix",
        "Operating System :: POSIX",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License"
    ],
    project_urls = {
        # "Issue Tracker" : "https://github.com/sharkutilities/pandas-wizard/issues",
        # "Code Documentations" : "https://pandas-wizard.readthedocs.io/en/latest/index.html",
        "Org. Homepage" : "https://github.com/sharkutilities"
    },
    keywords = [
        # keywords for finding the package::
        "dynamic", "multi-objective", "optimization",
        # keywords for finding the package relevant to usecases::
        "data science", "data analysis", "data scientist", "data analyst",
        # keywords as per available functionalities::
        "statistics"
    ],
    python_requires = ">=3.8"
)
