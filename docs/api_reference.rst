=============
API Reference
=============

All classes and functions are importable directly from the top-level ``pymu_eval`` package
(``pip install pymu-eval``):

.. code-block:: python

   from pymu_eval import (
       Parser,
       ParserBase,
       ParserInt,
       ParserError,
       ParserCallback,
       ParserByteCode,
       VariableRef,
       Variable,
       eval_expr,
       evaluate,
       ECmdCode,
       ETypeCode,
       EErrorCodes,
       EOprtPrecedence,
       EOprtAssociativity,
       prINFIX,
       prPOSTFIX,
   )

Every ``snake_case`` method documented below also has a C++ ``muParser`` style ``CamelCase``
alias (for example ``set_expr`` / ``SetExpr``, ``define_var`` / ``DefineVar``, ``eval`` / ``Eval``,
``define_fun_user_data`` / ``DefineFunUserData``).

Top-Level Evaluation Helpers
============================

.. autofunction:: pymu_eval.eval_expr

:func:`~pymu_eval.eval_expr` creates a :class:`~pymu_eval.Parser` and registers ``pi`` and ``e`` as
constants, so a variable with either name raises :class:`~pymu_eval.ParserError`
(``"Name conflict."``). If the expression contains several comma-separated expressions, only the
last value is returned; use :meth:`~pymu_eval.ParserBase.eval_multi` to get all of them.

.. autofunction:: pymu_eval.evaluate

Parser Classes
==============

``Parser``
----------

.. autoclass:: pymu_eval.Parser
   :members:
   :inherited-members:
   :show-inheritance:

``ParserInt``
-------------

.. autoclass:: pymu_eval.ParserInt
   :members:
   :show-inheritance:

``ParserBase``
--------------

.. autoclass:: pymu_eval.ParserBase
   :members:

Variables, Callbacks & Bytecode
===============================

.. autoclass:: pymu_eval.VariableRef
   :members:

.. autoclass:: pymu_eval.ParserCallback
   :members:

.. autoclass:: pymu_eval.ParserByteCode
   :members:

Errors & Definitions
====================

.. autoclass:: pymu_eval.ParserError
   :members:
   :show-inheritance:

Error messages come from ``pymu_eval.ERROR_MESSAGES`` with the token and position filled in, for
example ``Unexpected token "a" found at position 0.`` (an undefined variable) or
``Divide by zero.``. Each property also has ``get_*`` and C++ style ``Get*`` method aliases
(``get_code()`` / ``GetCode()``, ``get_msg()`` / ``GetMsg()``, ``get_expr()`` / ``GetExpr()``,
``get_token()`` / ``GetToken()``, ``get_pos()`` / ``GetPos()``); ``pos`` is ``-1`` when unknown.

The members of the enumerations below are also exported at package level (for example
``pymu_eval.ecDIV_BY_ZERO``).

.. autoclass:: pymu_eval.EErrorCodes
   :members:
   :undoc-members:

.. autoclass:: pymu_eval.EOprtPrecedence
   :members:
   :undoc-members:

``prINFIX`` and ``prPOSTFIX`` (both ``7``) are plain ``int`` constants exported from
``pymu_eval`` rather than :class:`~pymu_eval.EOprtPrecedence` members;
``EOprtPrecedence.prINFIX`` and ``EOprtPrecedence.prPOSTFIX`` still work for compatibility.

.. autoclass:: pymu_eval.EOprtAssociativity
   :members:
   :undoc-members:

.. autoclass:: pymu_eval.ECmdCode
   :members:
   :undoc-members:

.. autoclass:: pymu_eval.ETypeCode
   :members:
   :undoc-members:
