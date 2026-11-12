=========================
Examples & Colab Tutorial
=========================

The ``examples/`` directory in the repository provides both an interactive Jupyter Notebook
(ready for **Google Colab** or local JupyterLab) and standalone Python scripts.

Interactive Jupyter Notebook
============================

The notebook ``examples/pymu_eval_tutorial.ipynb`` walks through:

1. **Environment Setup (Google Colab & Local)**
2. **Core muParser Syntax & Scalar Expressions**
3. **Reusable Bytecode Compilation, Custom Operators & Numerical Differentiation**
4. **Vectorized Computation with NumPy ndarrays (1D Signals & 2D Wavelets)**
5. **Earth Observation & Climate Workflows with xarray.DataArray and xarray.Dataset**
6. **Integer & Bitwise Expression Parsing (ParserInt)**
7. **Localization & Structured Error Handling**

Example 1: Core Syntax, Custom Operators & Integer Mode
=======================================================

.. literalinclude:: ../examples/01_basic_syntax.py
   :language: python
   :linenos:

Example 2: Vectorized NumPy Array Expressions
=============================================

.. literalinclude:: ../examples/02_numpy_arrays.py
   :language: python
   :linenos:

Example 3: Geospatial & Climate Analysis with xarray Datasets
=============================================================

.. literalinclude:: ../examples/03_xarray_datasets.py
   :language: python
   :linenos:
