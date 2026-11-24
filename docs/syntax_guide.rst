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
     - :func:`~pymu_eval.eval_expr` / :func:`~pymu_eval.evaluate` only
     - Convenience alias for ``_pi``
   * - ``e``
     - :math:`2.71828182845904523536`
     - :func:`~pymu_eval.eval_expr` / :func:`~pymu_eval.evaluate` only
     - Convenience alias for ``_e``

Because ``pi`` and ``e`` are registered as constants by :func:`~pymu_eval.eval_expr`, passing a
variable with either name to it raises a :class:`~pymu_eval.ParserError` (``ecNAME_CONFLICT``,
message ``"Name conflict."``). A plain :class:`~pymu_eval.Parser` does not define ``pi`` or ``e``,
so you can use those names for your own variables there.

You can register additional constants with ``parser.define_const(name, value)`` or
``parser.DefineConst(name, value)``.

Built-in Binary Operators & Precedence
======================================

Binary operators are listed below from **lowest** to **highest** precedence. Precedence values
``1``–``8`` are the members of :class:`~pymu_eval.EOprtPrecedence` (``prLOR`` … ``prPOW``);
``=`` and ``? :`` are handled internally by the parser.

.. list-table::
   :header-rows: 1
   :widths: 18 15 17 50

   * - Operator
     - Precedence
     - Associativity
     - Description
   * - ``=``
     - ``-1``
     - Left
     - Variable assignment (``a = 5``). The left operand must be a variable. Chained assignment
       ``a = b = 5`` is not supported; write ``a = (b = 5)`` instead.
   * - ``? :``
     - ``0``
     - Right
     - Ternary conditional (``cond ? then_expr : else_expr``)
   * - ``||``
     - ``1`` (``prLOR``)
     - Left
     - Logical OR (non-zero is true; returns ``1.0`` or ``0.0``)
   * - ``&&``
     - ``2`` (``prLAND``)
     - Left
     - Logical AND (non-zero is true; returns ``1.0`` or ``0.0``)
   * - ``|``
     - ``3`` (``prBOR``)
     - Left
     - Bitwise OR (built into :class:`~pymu_eval.ParserInt`; available in
       :class:`~pymu_eval.Parser` only if you define it)
   * - ``&``
     - ``4`` (``prBAND``)
     - Left
     - Bitwise AND (built into :class:`~pymu_eval.ParserInt`; available in
       :class:`~pymu_eval.Parser` only if you define it)
   * - ``==``, ``!=``, ``<``, ``<=``, ``>``, ``>=``
     - ``5`` (``prCMP``)
     - Left
     - Equality, inequality and relational comparisons (``1.0`` or ``0.0``). All comparison
       operators share one precedence level, so ``1 < 2 == 1`` is ``(1 < 2) == 1``.
   * - ``+``, ``-``
     - ``6`` (``prADD_SUB``)
     - Left
     - Addition and subtraction
   * - ``*``, ``/``
     - ``7`` (``prMUL_DIV``)
     - Left
     - Multiplication and division
   * - ``^``
     - ``8`` (``prPOW``)
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
     - Unary negation (``-x``). Matching C++ ``muParser``, unary ``-`` binds more tightly than
       binary addition/subtraction (``6``) but less tightly than ``^`` (``8``), so
       ``-2^2 == -(2^2) == -4`` and ``2^-2 == 2^(-2) == 0.25``.
   * - ``+``
     - ``7`` (``prINFIX``)
     - Unary identity (``+x``).

``prINFIX`` and ``prPOSTFIX`` (both ``7``, the precedence used for custom prefix and postfix
operators) are plain ``int`` constants exported from ``pymu_eval``, not
:class:`~pymu_eval.EOprtPrecedence` members, because a Python ``IntEnum`` would turn them into
aliases of ``prMUL_DIV``. ``EOprtPrecedence.prINFIX`` and ``EOprtPrecedence.prPOSTFIX`` still work
as attribute lookups for compatibility.

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

``min`` and ``max`` propagate NaN for scalars and arrays alike: if any argument is NaN, the
result is NaN (as with ``numpy.minimum`` / ``numpy.maximum``).

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

:func:`~pymu_eval.eval_expr` behaves like ``eval()`` and returns only the last expression's value.

Integer Parser (``ParserInt``)
==============================

:class:`~pymu_eval.ParserInt` is a specialized subclass for integer arithmetic, hexadecimal/binary
literals, and bitwise logic. Operands are rounded to the nearest integer (halves away from zero)
and must fit in a signed 32-bit integer.

- **Literals**:

  - Decimal integers: ``42`` (decimal points are not accepted)
  - Hexadecimal: ``0xff``, ``0x10`` (prefixed with ``0x``)
  - Binary: ``#1010``, ``#1111`` (prefixed with ``#``; at most 31 digits)

- **Operators** (from lowest to highest precedence):

  - ``? :`` (ternary conditional)
  - ``||``, ``&&`` (logical OR / AND)
  - ``|`` (bitwise OR), ``&`` (bitwise AND)
  - ``==``, ``!=``, ``<``, ``<=``, ``>``, ``>=`` (comparisons)
  - ``+``, ``-``
  - ``*``, ``/`` (integer division truncating toward zero), ``%`` (remainder, C++ semantics)
  - ``-`` (unary negation) and ``!`` (logical NOT) prefix operators (``prINFIX``)
  - ``^`` (integer power), ``<<`` (left shift), ``>>`` (right shift), all at precedence ``8``

- **Functions**: ``sign``, ``abs``, ``if(cond, a, b)``, and the variadic ``sum``, ``min``, ``max``.
  There is no unary ``+`` and no floating-point functions such as ``sin``.

Shift semantics match a signed 32-bit ``int`` in C++, for scalars and arrays alike:

- ``>>`` is an arithmetic (sign-preserving) shift: ``-8 >> 1 == -4``.
- ``<<`` wraps to signed 32-bit: ``1 << 31 == -2147483648``.
- A shift count outside ``0``–``31`` raises a domain error.

With NumPy or ``xarray`` operands, :class:`~pymu_eval.ParserInt` evaluates element-wise and keeps
``xarray.DataArray`` dimensions and coordinates. Array values that are NaN, infinite, or outside
the signed 32-bit range raise :class:`~pymu_eval.ParserError` with code ``ecDOMAIN_ERROR``
(message ``"Domain error."``); division or remainder by zero raises ``ecDIV_BY_ZERO``
(``"Divide by zero."``).

.. code-block:: python

   import numpy as np
   from pymu_eval import ParserInt

   p = ParserInt("(0xff & mask) >> 1")
   p.define_var("mask", np.array([0x0f, -8, 0xf0]))
   p.eval()  # array([  7, 124, 120])
