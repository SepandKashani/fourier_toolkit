Fourier Toolkit
===============

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT


Fourier Toolkit (FTK) is a collection of utilities to compute Fourier Transforms.


Installation
------------

.. code-block:: bash

   # user install
   pip install fourier_toolkit@git+https://github.com/SepandKashani/fourier_toolkit.git

   # developer install
   git clone https://github.com/SepandKashani/fourier_toolkit.git
   cd fourier_toolkit/
   uv sync --group dev
   uv run pre-commit install


HTML Documentation
------------------

.. code-block:: bash

   cd fourier_toolkit/doc/
   uv run make html
