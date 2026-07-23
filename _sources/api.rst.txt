API Reference
=============

The FTK API is built around the scientific Python ecosystem and makes extensive use of ND-arrays.
The type hints
(:py:obj:`~fourier_toolkit.typing.ArrayR`,
:py:obj:`~fourier_toolkit.typing.ArrayC`,
:py:obj:`~fourier_toolkit.typing.ArrayRC`)
used throughout the API denote placeholders for any array object compliant with the `Python array API standard <https://data-apis.org/array-api/latest/>`_: their goal is to guide the reader when reading the docstrings, and not to perform runtime type-checking.

The API is tested against the following array backends, unless noted otherwise:

- `NumPy <https://numpy.org/doc/stable/>`_
- `CuPy <https://docs.cupy.dev/en/latest/>`_
- `JAX <https://docs.jax.dev/en/latest/>`_ (CPU + GPU)
- `PyTorch <https://pytorch.org/docs/stable>`_ (CPU + GPU)


Array Types
-----------

.. automodule:: fourier_toolkit.typing
   :members: ArrayR, ArrayC, ArrayRC
   :no-value:


.. _FT-routines:

Computing Fourier Sums
----------------------

.. note::

   (:py:func:`~fourier_toolkit.nu2nu`, :py:func:`~fourier_toolkit.nu2u`, :py:func:`~fourier_toolkit.u2nu`) rely on:

   * CPU: `FINUFFT <https://finufft.readthedocs.io/en/latest/index.html>`_
   * GPU: `cuFINUFFT <https://finufft.readthedocs.io/en/latest/python_gpu.html>`_

   The latter is not installed by default. You can install it if needed via

   .. code::

      pip install cufinufft                        # cuda12 pre-packaged [20260711]
      pip install --no-binary cufinufft cufinufft  # cudaXX

   It may be necessary to modify ``LD_LIBRARY_PATH`` afterwards for ``libcufinufft.so`` to be importable.


.. autofunction:: fourier_toolkit.u2u

.. autofunction:: fourier_toolkit.nu2nu

.. autofunction:: fourier_toolkit.nu2u

.. autofunction:: fourier_toolkit.u2nu


Helpers
-------

.. autoclass:: fourier_toolkit.UniformSpec
   :members:
   :undoc-members: start, step, num, center, span, ndim
   :special-members: __getitem__
