Fourier Toolkit
===============

.. |gh-badge| image:: https://img.shields.io/badge/GitHub-Repo-black?logo=github
   :alt: GitHub repository
   :target: https://github.com/SepandKashani/fourier_toolkit

.. |docs-badge| image:: https://img.shields.io/badge/Docs-GitHub%20Pages-blue?logo=github
   :alt: Documentation
   :target: https://SepandKashani.github.io/fourier_toolkit/

|gh-badge| |docs-badge|


Fourier Toolkit (FTK) is a collection of utilities to compute Fourier sums.


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
   uv run pytest -v

   # build HTML docs
   uv run make -C doc/ html

   # creating a release
   uv run dev/create_release.py <version>
