API Reference
=============

The FTK API is built around the scientific Python ecosystem and makes extensive use of ND-arrays.
Unless otherwise specified, all :ref:`routines <FT-routines>` support both CPU and GPU execution.

The type hints
(:py:obj:`~fourier_toolkit.typing.ArrayR`,
:py:obj:`~fourier_toolkit.typing.ArrayC`,
:py:obj:`~fourier_toolkit.typing.ArrayRC`)
used throughout the API denote placeholders for any array object compliant with the `Python array API standard <https://data-apis.org/array-api/latest/>`_: their goal is to guide the reader when reading the docstrings, and not to perform runtime type-checking.


Array Types
-----------

.. automodule:: fourier_toolkit.typing
   :members: ArrayR, ArrayC, ArrayRC
   :no-value:


.. _FT-routines:

Computing Fourier Sums
----------------------

.. autofunction:: fourier_toolkit.u2u

.. autofunction:: fourier_toolkit.nu2nu

.. autofunction:: fourier_toolkit.nu2u

.. autofunction:: fourier_toolkit.u2nu


Helpers
-------

.. autoclass:: fourier_toolkit.UniformSpec
   :members:
   :special-members: __getitem__
