API Reference
=============

The FTK API is built around the scientific Python ecosystem and makes extensive use of ND-arrays.
Unless otherwise specified, all :ref:`routines <FT-routines>` support both CPU and GPU execution.
The backend is selected implicitly based on the array types provided as inputs:

- `NumPy arrays <https://numpy.org/>`_ → CPU execution
- `CuPy arrays <https://cupy.dev/>`_ → GPU execution

The backend-agnostic type hints
(:py:obj:`~fourier_toolkit.typing.ArrayR`,
:py:obj:`~fourier_toolkit.typing.ArrayC`,
:py:obj:`~fourier_toolkit.typing.ArrayRC`)
used throughout the API denote placeholders for (NumPy, CuPy) arrays of specific numeric types: their goal is to guide the reader when reading the docstrings, and not to perform runtime type-checking.


Array Types
-----------

.. automodule:: fourier_toolkit.typing
   :members:
   :no-value:


.. _FT-routines:

Computing Fourier Sums
----------------------

.. autofunction:: fourier_toolkit.u2u

.. autoclass:: fourier_toolkit.U2U
   :members:

.. autofunction:: fourier_toolkit.nu2nu

.. autofunction:: fourier_toolkit.nu2u

.. autofunction:: fourier_toolkit.u2nu


Helpers
-------

.. autoclass:: fourier_toolkit.UniformSpec
   :members:
