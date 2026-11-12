====================
NumPy & xarray Guide
====================

``pymu-eval`` is designed to evaluate ``muParser`` mathematical expressions over multidimensional
**NumPy** ``ndarray`` objects, **xarray** ``DataArray`` objects, and **xarray** ``Dataset`` objects
without writing manual loops.

NumPy ``ndarray`` Evaluation
============================

One-Shot Evaluation with ``eval_expr``
--------------------------------------

Pass NumPy arrays via a dictionary or keyword arguments to :func:`~pymu_eval.eval_expr`:

.. code-block:: python

   import numpy as np
   from pymu_eval import eval_expr

   x = np.linspace(0.0, 2.0 * np.pi, 100)
   y = np.linspace(-1.0, 1.0, 100)

   # Trigonometric, exponential, and power operators
   z = eval_expr("exp(-x / 5) * sin(x)^2 + abs(y)", x=x, y=y)

Reusable Bytecode Compilation with ``Parser``
---------------------------------------------

When evaluating the same expression across many time steps, tiles, or batches, compile once with
:class:`~pymu_eval.Parser` and update the :class:`~pymu_eval.VariableRef` values in place:

.. code-block:: python

   import numpy as np
   from pymu_eval import Parser

   p = Parser("0.5 * rho * v^2 + p_static")
   rho_var = p.define_var("rho", 1.225)
   v_var = p.define_var("v", np.zeros((256, 256)))
   p_var = p.define_var("p_static", 101325.0)

   for frame in range(10):
       v_var.value = np.random.uniform(0, 30, size=(256, 256))
       p_total = p.eval()  # Reuses compiled RPN bytecode

Multidimensional Broadcasting
-----------------------------

Standard NumPy broadcasting rules apply automatically across 1D, 2D, and 3D arrays:

.. code-block:: python

   lat = np.linspace(-90, 90, 180)[:, np.newaxis]   # shape (180, 1)
   lon = np.linspace(-180, 180, 360)[np.newaxis, :] # shape (1, 360)

   insolation = eval_expr(
       "max(0, cos(lat * _pi / 180) * cos(lon * _pi / 180))",
       lat=lat,
       lon=lon,
   )
   # insolation has shape (180, 360)

``xarray.DataArray`` Evaluation
===============================

When variables are ``xarray.DataArray`` objects, ``pymu-eval`` preserves dimension names,
coordinate labels, and aligns grids automatically:

.. code-block:: python

   import numpy as np
   import xarray as xr
   from pymu_eval import eval_expr

   temp_c = xr.DataArray(
       np.array([[15.0, 22.5], [18.0, 30.0]]),
       dims=("lat", "lon"),
       coords={"lat": [51.5, 52.0], "lon": [-0.1, 0.2]},
       name="temperature",
   )
   rh = xr.DataArray(
       np.array([[80.0, 60.0], [75.0, 45.0]]),
       dims=("lat", "lon"),
       coords={"lat": [51.5, 52.0], "lon": [-0.1, 0.2]},
       name="relative_humidity",
   )

   # Magnus formula for saturation vapor pressure (hPa) and actual vapor pressure
   es = eval_expr("6.1094 * exp((17.625 * T) / (T + 243.04))", T=temp_c)
   ea = eval_expr("(RH / 100.0) * es", RH=rh, es=es)

Direct ``xarray.Dataset`` Binding
=================================

You can pass an ``xarray.Dataset`` directly to :func:`~pymu_eval.eval_expr` or
:meth:`~pymu_eval.ParserBase.define_dataset`. Both **data variables** (``ds.data_vars``) and
**coordinates** (``ds.coords``) become available by name inside the expression string:

.. code-block:: python

   import numpy as np
   import xarray as xr
   from pymu_eval import Parser, eval_expr

   ds = xr.Dataset(
       data_vars={
           "blue": (("y", "x"), np.array([[0.05, 0.08], [0.06, 0.12]])),
           "red":  (("y", "x"), np.array([[0.08, 0.25], [0.10, 0.30]])),
           "nir":  (("y", "x"), np.array([[0.45, 0.30], [0.52, 0.18]])),
           "scl":  (("y", "x"), np.array([[4, 4], [4, 9]])),  # 4=veg, 9=cloud
       },
       coords={
           "y": np.array([100.0, 200.0]),
           "x": np.array([500.0, 600.0]),
       },
   )

   # 1. One-shot evaluation directly on the Dataset
   evi = eval_expr("2.5 * (nir - red) / (nir + 6.0 * red - 7.5 * blue + 1.0)", ds)

   # 2. Cloud masking with the ternary operator (? :) and NaN fallback
   ndvi_masked = eval_expr(
       "scl != 9 && (nir + red) > 0 ? (nir - red) / (nir + red) : nan",
       ds,
       nan=np.nan,
   )

   # 3. Using spatial coordinates ('x', 'y') directly inside an expression
   lat_corrected = eval_expr("nir * cos(y / 1000.0)", ds)

Element-Wise Ternary Conditionals (``? :``)
===========================================

In C++ ``muParser``, the ternary operator ``cond ? then_expr : else_expr`` jumps over instructions
based on a scalar condition. In ``pymu-eval``:

- If ``cond`` is a **scalar** (``float``/``int``/``bool``), only the taken branch is evaluated
  (lazy evaluation).
- If ``cond`` is a **NumPy** ``ndarray`` or **xarray** ``DataArray``, ``pymu-eval`` automatically
  evaluates both branches and merges them element-wise using
  ``xarray.where(cond != 0, then_val, else_val)`` (when any operand is an ``xarray`` object) or
  ``numpy.where(cond != 0, then_val, else_val)``.

Nested ternary expressions work seamlessly for multi-class classification or piecewise functions:

.. code-block:: python

   # Classify NDVI into Water (0), Bare Soil (1), Sparse Veg (2), Dense Veg (3)
   ndvi_class = eval_expr(
       "ndvi < 0.0 ? 0 : ndvi < 0.2 ? 1 : ndvi < 0.5 ? 2 : 3",
       ndvi=ndvi_masked,
   )

Numerical Differentiation (``diff``) over Arrays and ``DataArray``
==================================================================

:meth:`~pymu_eval.Parser.diff` computes the 5-point 4th-order finite-difference derivative:

.. math::

   f'(x) \approx \frac{-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)}{12h}

When the target variable holds a ``numpy.ndarray`` or ``xarray.DataArray``, ``p.diff("x")`` evaluates
the derivative element-wise across the entire array or ``DataArray`` while preserving coordinates.
