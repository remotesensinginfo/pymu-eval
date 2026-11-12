=============
API Reference
=============

Top-Level Evaluation Helpers
============================

.. autofunction:: pymu_eval.eval_expr

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

.. autoclass:: pymu_eval.EErrorCodes
   :members:
   :undoc-members:

.. autoclass:: pymu_eval.EOprtPrecedence
   :members:
   :undoc-members:

.. autoclass:: pymu_eval.EOprtAssociativity
   :members:
   :undoc-members:

.. autoclass:: pymu_eval.ECmdCode
   :members:
   :undoc-members:

.. autoclass:: pymu_eval.ETypeCode
   :members:
   :undoc-members:
