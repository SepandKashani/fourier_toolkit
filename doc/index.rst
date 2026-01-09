Fourier Toolkit
===============

.. toctree::
   :maxdepth: 4
   :hidden:

   api
   references


.. |gh-badge| image:: https://img.shields.io/badge/GitHub-Repo-black?logo=github
   :alt: GitHub repository
   :target: https://github.com/SepandKashani/fourier_toolkit

.. |docs-badge| image:: https://img.shields.io/badge/Docs-GitHub%20Pages-blue?logo=github
   :alt: Documentation
   :target: https://SepandKashani.github.io/fourier_toolkit/

|gh-badge| |docs-badge|


Fourier Toolkit (FTK) is a collection of utilities to compute Fourier sums.


Rationale
---------

Fourier operators are pervasive in science and engineering as waves are the main vector of information transfer in physical systems.
Casting wave problems in discrete form leads to the evaluation of sums of the form

.. math::
   :label: dirac-sum-FT

   \bbz_{n} = \sum_{m=0}^{M-1} w_{m} \ee^{-\cj 2 \pi \innerProduct{\bbx_{m}}{\bbv_{n}}},
   \qquad n \in \{0,\ldots,N-1\},

where :math:`\{\bbx_{m} \in \bR^{D}\}` and :math:`\{\bbv_{n} \in \bR^{D}\}` are known constants refered to as *knots*.
This equation computes Fourier Transform samples of the Dirac stream :math:`f(\bbx) = \sum_{m} w_{m} \delta(\bbx - \bbx_{m})` at frequencies :math:`\bbv_{n}`.

Naive evaluation of :eq:`dirac-sum-FT` costs :math:`\cO(M N)` operations which is intractable at scale.
There is a rich literature on low-complexity methods to compute the latter more efficiently depending on the distribution of spatial/spectral knots :cite:`rabiner1969chirp,plonka2018numerical,steidl1998note,fourmont1999thesis,fourmont2003non,jacob2009optimized,barnett2019parallel,dutt1993fast,dutt1995fast`.
At their core, they all leverage the *Fast Fourier Transform* algorithm (FFT) in some form to perform the bulk of the work:

- When the :math:`(\bbx_{m}, \bbv_{n})` knots lie on uniform grids, we talk of *uniform-to-uniform* methods: one or two FFTs suffice to evaluate :eq:`dirac-sum-FT`.
- When either :math:`(\bbx_{m}, \bbv_{n})` are non-uniform, we talk of *Non-uniform FFTs*: the latter combine the FFT with convolution/interpolation steps to compute :eq:`dirac-sum-FT` within a chosen relative error :math:`\epsilon \ll 1`.

FTK is thus a toolbox to evaluate Fourier sums efficiently by building on mature libraries.


Setup
-----

.. code-block:: bash

   # user install
   pip install fourier_toolkit@git+https://github.com/SepandKashani/fourier_toolkit.git

   # developer install
   git clone https://github.com/SepandKashani/fourier_toolkit.git
   cd fourier_toolkit/
   uv sync --group dev
   uv run pre-commit install

   # run test suite
   uv run pytest

   # build HTML docs
   uv run make -C doc/ html

   # creating a release
   uv run dev/create_release.py <version>
