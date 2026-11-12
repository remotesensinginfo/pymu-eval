============
Syntax Guide
============

``pymu-eval`` implements the exact grammar, operator precedence, and function semantics
of C++ ``muParser`` v2.3.6.

Built-in Constants
==================

.. list-table::
   :header-rows: 1
   :widths: 15 25 25 35

   * - Constant
     - Value
     - Availability
     - Description
   * - ``_pi``
     - :math:`3.14159265358979323846`
     - :class:`~pymu_eval.Parser`, :func:`~pymu_eval.eval_expr`
     - :math:`\pi` (ratio of circumference to diameter)
   * - ``_e``
     - :math:`2.71828182845904523536`
     - :class:`~pymu_eval.Parser`, :func:`~pymu_eval.eval_expr`
     - Euler's number :math:`e`
   * - ``pi``
     - :math:`3.14159265358979323846`
     - :func:`~pymu_eval.eval_expr` / :func:`~pymu_eval.evaluate`
     - Convenience alias for ``_pi`` (unless overridden by a variable named ``pi``)
   * - ``e``
     - :math:`2.71828182845904523536`
     - :func:`~pymu_eval.eval_expr` / :func:`~pymu_eval.evaluate`
     - Convenience alias for ``_e`` (unless overridden by a variable named ``e``)

You can register additional constants with ``parser.define_const(name, value)`` or
``parser.DefineConst(name, value)``.

Built-in Binary Operators & Precedence
======================================

Binary operators are listed below from **lowest** to **highest** precedence (matching
:class:`~pymu_eval.EOprtPrecedence`):

.. list-table::
   :header-rows: 1
   :widths: 18 15 17 50

   * - Operator
     - Precedence
     - Associativity
     - Description
   * - ``=``
     - ``-1``
     - Right
     - Variable assignment (``a = b = 5``)
   * - ``? :``
     - ``0``
     - Right
     - Ternary conditional (``cond ? then_expr : else_expr``)
   * - ``||``
     - ``1``
     - Left
     - Logical OR (non-zero is true; returns ``1.0`` or ``0.0``)
   * - ``&&``
     - ``2``
     - Left
     - Logical AND (non-zero is true; returns ``1.0`` or ``0.0``)
   * - ``|``
     - ``3``
     - Left
     - Bitwise OR (enabled in :class:`~pymu_eval.ParserInt` or when defined)
   * - ``&``
     - ``4``
     - Left
     - Bitwise AND (enabled in :class:`~pymu_eval.ParserInt` or when defined)
   * - ``!=``, ``==``
     - ``5``
     - Left
     - Inequality and equality comparison (``1.0`` or ``0.0``)
   * - ``<``, ``<=``, ``>``, ``>=``
     - ``6``
     - Left
     - Relational comparisons (``1.0`` or ``0.0``)
   * - ``+``, ``-``
     - ``7``
     - Left
     - Addition and subtraction
   * - ``*``, ``/``
     - ``8``
     - Left
     - Multiplication and division
   * - ``^``
     - ``9``
     - Right
     - Exponentiation (``2^3^2 == 2^(3^2) == 512``)

Prefix (Infix) Unary Operators
------------------------------

.. list-table::
   :header-rows: 1
   :widths: 15 20 65

   * - Operator
     - Precedence
     - Description
   * - ``-``
     - ``7`` (``prINFIX``)
     - Unary negation (``-x``). Matching C++ ``muParser``, unary ``-`` has addition/subtraction
       precedence, so ``-2^2 == -(2^2) == -4`` and ``2^-2 == 2^(-2) == 0.25``.
   * - ``+``
     - ``7`` (``prINFIX``)
     - Unary identity (``+x``).

Built-in Mathematical Functions
===============================

Single-Argument Functions
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 45 35

   * - Function
     - Formula / Description
     - Domain Notes
   * - ``sin(x)``
     - :math:`\sin(x)` (radians)
     - All real :math:`x`
   * - ``cos(x)``
     - :math:`\cos(x)` (radians)
     - All real :math:`x`
   * - ``tan(x)``
     - :math:`\tan(x)` (radians)
     - :math:`x \neq \frac{\pi}{2} + k\pi`
   * - ``cot(x)``
     - :math:`\cot(x) = 1 / \tan(x)`
     - :math:`\tan(x) \neq 0`
   * - ``asin(x)``
     - :math:`\arcsin(x)` in :math:`[-\pi/2, \pi/2]`
     - :math:`x \in [-1, 1]`
   * - ``acos(x)``
     - :math:`\arccos(x)` in :math:`[0, \pi]`
     - :math:`x \in [-1, 1]`
   * - ``atan(x)``
     - :math:`\arctan(x)` in :math:`[-\pi/2, \pi/2]`
     - All real :math:`x`
   * - ``sinh(x)``
     - :math:`\sinh(x)`
     - All real :math:`x`
   * - ``cosh(x)``
     - :math:`\cosh(x)`
     - All real :math:`x`
   * - ``tanh(x)``
     - :math:`\tanh(x)`
     - All real :math:`x`
   * - ``coth(x)``
     - :math:`\coth(x) = \cosh(x)/\sinh(x)`
     - :math:`x \neq 0`
   * - ``asinh(x)``
     - :math:`\operatorname{arsinh}(x)`
     - All real :math:`x`
   * - ``acosh(x)``
     - :math:`\operatorname{arcosh}(x)`
     - :math:`x \ge 1`
   * - ``atanh(x)``
     - :math:`\operatorname{artanh}(x)`
     - :math:`x \in (-1, 1)`
   * - ``log2(x)``
     - :math:`\log_2(x)`
     - :math:`x > 0`
   * - ``log10(x)``
     - :math:`\log_{10}(x)`
     - :math:`x > 0`
   * - ``log(x)``
     - :math:`\ln(x)` (natural logarithm, matching ``muParser`` v2.3+)
     - :math:`x > 0`
   * - ``ln(x)``
     - :math:`\ln(x)` (natural logarithm)
     - :math:`x > 0`
   * - ``exp(x)``
     - :math:`e^x`
     - All real :math:`x`
   * - ``sqrt(x)``
     - :math:`\sqrt{x}`
     - :math:`x \ge 0`
   * - ``sign(x)``
     - :math:`-1` if :math:`x < 0`, :math:`1` if :math:`x > 0`, :math:`0` if :math:`x == 0`
     - All real :math:`x`
   * - ``rint(x)``
     - Round to nearest integer: :math:`\lfloor x + 0.5 \rfloor`
     - All real :math:`x`
   * - ``abs(x)``
     - Absolute value :math:`|x|`
     - All real :math:`x`
   * - ``rnd()``
     - Random number in :math:`[0, 1)`
     - 0 arguments

Two-Argument Functions
----------------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Function
     - Description
   * - ``atan2(y, x)``
     - Four-quadrant inverse tangent :math:`\operatorname{atan2}(y, x)` in :math:`[-\pi, \pi]`

Variadic Functions (1 or More Arguments)
----------------------------------------

Variadic functions accept any number of comma-separated arguments (``sum(a, b, c, ...)``).
When arguments are NumPy arrays or ``xarray.DataArray`` objects, they operate **element-wise across
the provided arguments** (broadcasting automatically):

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Function
     - Description
   * - ``sum(x1, x2, ...)``
     - Element-wise sum: :math:`x_1 + x_2 + \dots + x_n`
   * - ``avg(x1, x2, ...)``
     - Element-wise arithmetic mean: :math:`\frac{1}{n}\sum_{i=1}^n x_i`
   * - ``min(x1, x2, ...)``
     - Element-wise minimum across all arguments
   * - ``max(x1, x2, ...)``
     - Element-wise maximum across all arguments

Ternary Conditional (``? :``)
=============================

The ``if-then-else`` operator has the syntax:

.. code-block:: text

   condition ? true_branch : false_branch

- **Nesting**: Ternary operators are right-associative and can be nested arbitrarily or enclosed in parentheses:

  .. code-block:: text

     x < 0 ? -1 : x == 0 ? 0 : 1

- **Scalar Evaluation**: Short-circuits and evaluates only the active branch.
- **Array / xarray Evaluation**: Evaluates element-wise using ``numpy.where(condition != 0, true_branch, false_branch)``
  or ``xarray.where(condition != 0, true_branch, false_branch)``, preserving ``xarray`` dimensions and coordinates.

Multi-Expression Evaluation
===========================

Multiple expressions can be separated by commas at the top level of an expression string:

.. code-block:: python

   from pymu_eval import Parser

   p = Parser()
   a = p.define_var("a", 1.0)
   b = p.define_var("b", 2.0)
   p.set_expr("a = b + 3, b = a * 2, a + b")

   # eval() returns the last expression's value (15.0)
   last_val = p.eval()

   # eval_multi() returns a list of all expression results: [5.0, 10.0, 15.0]
   all_vals = p.eval_multi()

Integer Parser (``ParserInt``)
==============================

:class:`~pymu_eval.ParserInt` is a specialized subclass for integer arithmetic, hexadecimal/binary
literals, and bitwise logic:

- **Hexadecimal literals**: ``0xff``, ``0x10`` (prefixed with ``0x``)
- **Binary literals**: ``#1010``, ``#1111`` (prefixed with ``#``)
- **Bitwise & Integer Operators**:

  - ``|`` (bitwise OR), ``&`` (bitwise AND)
  - ``<<`` (left bit-shift), ``>>`` (right bit-shift)
  - ``%`` (modulo), ``/`` (integer division truncating toward zero)
  - ``!`` (logical NOT prefix operator)
  - ``&&``, ``||`` (logical AND / OR)
