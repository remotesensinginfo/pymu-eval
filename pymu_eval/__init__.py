"""
pymu-eval (pymu_eval): Python implementation of the muParser mathematical expression parser
with native support for scalars, NumPy ndarrays, xarray DataArrays, and xarray Datasets.
"""

from .bytecode import ParserByteCode, SToken
from .definitions import (
    CmdCode,
    DEFAULT_OPRT,
    ECmdCode,
    EErrorCodes,
    EOprtAssociativity,
    EOprtPrecedence,
    EParserVersionInfo,
    ESynCodes,
    ETypeCode,
    ErrorCode,
    MaxLenExpression,
    MaxLenIdentifier,
    MaxNestingDepth,
    OprtAssociativity,
    OprtPrecedence,
    PUBLIC_ENUMS,
    ParserVersion,
    ParserVersionDate,
    TypeCode,
)
from .errors import ERROR_MESSAGES, ParserError
from .math_impl import MathImpl, ParserIntMathImpl
from .parser import Parser, ParserBase, ParserInt, eval_expr, evaluate
from .token_reader import ParserCallback, ParserToken, ParserTokenReader, Variable, VariableRef

# Export enum members at package level for C++-style compatibility
for _enum_cls in PUBLIC_ENUMS:
    globals().update(_enum_cls.__members__)

__version__ = "2.3.6"

__all__ = [
    "Parser",
    "ParserBase",
    "ParserInt",
    "ParserError",
    "ParserByteCode",
    "SToken",
    "ParserCallback",
    "ParserToken",
    "ParserTokenReader",
    "Variable",
    "VariableRef",
    "MathImpl",
    "ParserIntMathImpl",
    "eval_expr",
    "evaluate",
    "ECmdCode",
    "CmdCode",
    "ETypeCode",
    "TypeCode",
    "EParserVersionInfo",
    "EOprtAssociativity",
    "OprtAssociativity",
    "EOprtPrecedence",
    "OprtPrecedence",
    "ESynCodes",
    "EErrorCodes",
    "ErrorCode",
    "ERROR_MESSAGES",
    "DEFAULT_OPRT",
    "MaxLenExpression",
    "MaxLenIdentifier",
    "MaxNestingDepth",
    "ParserVersion",
    "ParserVersionDate",
    "__version__",
]
