=======================================
pymu-eval Documentation
=======================================

``pymu-eval`` (imported as ``pymu_eval``) is a pure-Python implementation of the
`muParser <https://github.com/beltoforion/muparser>`_ mathematical expression syntax parser
and Reverse Polish Notation (RPN) bytecode evaluator. It brings the complete ``muParser``
syntax—including operator precedence, ternary conditionals, variadic functions, custom operators,
localization, and bytecode optimization—directly to Python scalars, **NumPy** ``ndarray`` objects,
**xarray** ``DataArray`` objects, and **xarray** ``Dataset`` objects.

Installation
============

.. code-block:: bash

   pip install pymu-eval
   # With optional xarray support:
   pip install "pymu-eval[xarray]"

   # Or from a local checkout:
   pip install -e .
   pip install -e ".[xarray]"

Key Capabilities
================

- **Compile Once, Evaluate Many Times**: Expressions are tokenized and compiled into Reverse Polish
  Notation (RPN) bytecode (:class:`~pymu_eval.ParserByteCode`) on the first evaluation. Subsequent
  evaluations reuse the compiled bytecode without re-parsing the string.
- **Constant Folding & Algebraic Optimization**: Constant sub-expressions are pre-evaluated at compile
  time, and linear/polynomial patterns (``a*x + b``, ``x^2``, ``x^3``, ``x^4``) are compiled into
  specialized bytecode instructions (``cmVARMUL``, ``cmVARPOW2``–``cmVARPOW4``).
- **Dual Vector / Scalar Execution**:

  - When all inputs are scalars, ``pymu-eval`` uses fast scalar stack evaluation with short-circuiting
    jumps for ``cond ? then : else``.
  - When any input is a ``numpy.ndarray`` or ``xarray.DataArray``, ``pymu-eval`` automatically dispatches
    to vectorized ``numpy`` / ``xarray`` operations, preserving ``xarray`` dimensions and coordinates and
    evaluating ternary ``cond ? then : else`` element-wise via ``xarray.where`` / ``numpy.where``.

- **Dual API Naming**: Every method on :class:`~pymu_eval.ParserBase` supports both Pythonic
  ``snake_case`` (e.g., ``set_expr``, ``define_var``, ``eval``) and C++ ``muParser`` ``CamelCase``
  (e.g., ``SetExpr``, ``DefineVar``, ``Eval``).

Quick Start
===========

.. code-block:: python

   import numpy as np
   import xarray as xr
   from pymu_eval import Parser, eval_expr

   # 1. One-shot evaluation with NumPy arrays
   x = np.linspace(0.0, np.pi, 5)
   y = eval_expr("sin(x)^2 + cos(x)^2", x=x)

   # 2. Direct evaluation with an xarray.Dataset
   ds = xr.Dataset(
       {
           "red": (("y", "x"), np.array([[0.1, 0.2], [0.15, 0.3]])),
           "nir": (("y", "x"), np.array([[0.5, 0.6], [0.45, 0.25]])),
           "cloud": (("y", "x"), np.array([[0, 0], [1, 0]])),
       },
       coords={"y": [10.0, 20.0], "x": [100.0, 110.0]},
   )
   ndvi = eval_expr("cloud == 0 ? (nir - red) / (nir + red) : nan", ds, nan=np.nan)

.. toctree::
   :maxdepth: 2
   :caption: User Guide & Reference

   syntax_guide
   numpy_xarray_guide
   examples
   api_reference

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
