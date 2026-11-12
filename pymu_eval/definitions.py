"""
Standard definitions, enums, and constants for pymu-eval.
Mirrors muParserDef.h and muParserTokenReader.h from the C++ muParser library.
"""

# pylint: disable=invalid-name

from enum import IntEnum, IntFlag

MaxLenExpression: int = 20000
MaxLenIdentifier: int = 100
MaxNestingDepth: int = 1000

ParserVersion: str = "2.3.6 (Development)"
ParserVersionDate: str = "20261110"

DEFAULT_OPRT: tuple[str, ...] = (
    "<=",
    ">=",
    "!=",
    "==",
    "<",
    ">",
    "+",
    "-",
    "*",
    "/",
    "^",
    "&&",
    "||",
    "=",
    "(",
    ")",
    "?",
    ":",
)

BINARY_OPRT_CODES: frozenset[int] = frozenset(range(14))  # cmLE (0) .. cmASSIGN (13)
CMP_OPRT_CODES: frozenset[int] = frozenset(range(6))      # cmLE (0) .. cmGT (5)
ARITH_OPRT_CODES: frozenset[int] = frozenset(range(6, 11))  # cmADD (6) .. cmPOW (10)


class ECmdCode(IntEnum):
    """Bytecode and token command codes (matches mu::ECmdCode)."""

    cmLE = 0            # Operator item: less or equal
    cmGE = 1            # Operator item: greater or equal
    cmNEQ = 2           # Operator item: not equal
    cmEQ = 3            # Operator item: equals
    cmLT = 4            # Operator item: less than
    cmGT = 5            # Operator item: greater than
    cmADD = 6           # Operator item: add
    cmSUB = 7           # Operator item: subtract
    cmMUL = 8           # Operator item: multiply
    cmDIV = 9           # Operator item: division
    cmPOW = 10          # Operator item: y to the power of ...
    cmLAND = 11         # Operator item: logical and
    cmLOR = 12          # Operator item: logical or
    cmASSIGN = 13       # Operator item: assignment operator
    cmBO = 14           # Operator item: opening bracket
    cmBC = 15           # Operator item: closing bracket
    cmIF = 16           # For use in the ternary if-then-else operator
    cmELSE = 17         # For use in the ternary if-then-else operator
    cmENDIF = 18        # For use in the ternary if-then-else operator
    cmARG_SEP = 19      # Function argument separator
    cmVAR = 20          # Variable item
    cmVAL = 21          # Value item

    # For optimization purposes
    cmVARPOW2 = 22
    cmVARPOW3 = 23
    cmVARPOW4 = 24
    cmVARMUL = 25

    # Operators and functions
    cmFUNC = 26         # Code for a generic function item
    cmFUNC_STR = 27     # Code for a function with a string parameter
    cmFUNC_BULK = 28    # Special callbacks for Bulk mode
    cmSTRING = 29       # Code for a string token
    cmOPRT_BIN = 30     # User defined binary operator
    cmOPRT_POSTFIX = 31 # Code for postfix operators
    cmOPRT_INFIX = 32   # Code for infix operators
    cmEND = 33          # End of formula
    cmUNKNOWN = 34      # Uninitialized item


class ETypeCode(IntEnum):
    """Types internally used by the parser (matches mu::ETypeCode)."""

    tpSTR = 0   # String type
    tpDBL = 1   # Floating point / numeric / array variables
    tpVOID = 2  # Undefined type


class EParserVersionInfo(IntEnum):
    """Version info formatting flags."""

    pviBRIEF = 0
    pviFULL = 1


class EOprtAssociativity(IntEnum):
    """Parser operator associativity values (matches mu::EOprtAssociativity)."""

    oaLEFT = 0
    oaRIGHT = 1
    oaNONE = 2


class EOprtPrecedence(IntEnum):
    """Parser operator precedence values (matches mu::EOprtPrecedence)."""

    prLOR = 1       # logic or
    prLAND = 2      # logic and
    prBOR = 3       # bitwise or
    prBAND = 4      # bitwise and
    prCMP = 5       # comparison operators
    prADD_SUB = 6   # addition / subtraction
    prMUL_DIV = 7   # multiplication / division
    prPOW = 8       # power operator priority (highest)

    # infix / postfix operators
    prINFIX = 7     # Signs have a higher priority than ADD_SUB, but lower than power operator
    prPOSTFIX = 7   # Postfix operator priority


class ESynCodes(IntFlag):
    """Syntax flags controlling allowed next tokens during lexing."""

    noNONE = 0
    noBO = 1 << 0        # to avoid i.e. "cos(7)("
    noBC = 1 << 1        # to avoid i.e. "sin)" or "()"
    noVAL = 1 << 2       # to avoid i.e. "tan 2" or "sin(8)3.14"
    noVAR = 1 << 3       # to avoid i.e. "sin a" or "sin(8)a"
    noARG_SEP = 1 << 4   # to avoid i.e. ",," or "+," ...
    noFUN = 1 << 5       # to avoid i.e. "sqrt cos" or "(1)sin"
    noOPT = 1 << 6       # to avoid i.e. "(+)"
    noPOSTOP = 1 << 7    # to avoid i.e. "(5!!)" "sin!"
    noINFIXOP = 1 << 8   # to avoid i.e. "++4" "!!4"
    noEND = 1 << 9       # to avoid unexpected end of formula
    noSTR = 1 << 10      # to block numeric arguments on string functions
    noASSIGN = 1 << 11   # to block assignment to constant i.e. "4=7"
    noIF = 1 << 12
    noELSE = 1 << 13
    sfSTART_OF_LINE = noOPT | noBC | noPOSTOP | noASSIGN | noIF | noELSE | noARG_SEP
    noANY = (1 << 14) - 1


class EErrorCodes(IntEnum):
    """Error codes (matches mu::EErrorCodes)."""

    # Formula syntax errors
    ecUNEXPECTED_OPERATOR = 0
    ecUNASSIGNABLE_TOKEN = 1
    ecUNEXPECTED_EOF = 2
    ecUNEXPECTED_ARG_SEP = 3
    ecUNEXPECTED_ARG = 4
    ecUNEXPECTED_VAL = 5
    ecUNEXPECTED_VAR = 6
    ecUNEXPECTED_PARENS = 7
    ecUNEXPECTED_STR = 8
    ecSTRING_EXPECTED = 9
    ecVAL_EXPECTED = 10
    ecMISSING_PARENS = 11
    ecUNEXPECTED_FUN = 12
    ecUNTERMINATED_STRING = 13
    ecTOO_MANY_PARAMS = 14
    ecTOO_FEW_PARAMS = 15
    ecOPRT_TYPE_CONFLICT = 16
    ecSTR_RESULT = 17

    # Invalid Parser input Parameters
    ecINVALID_NAME = 18
    ecINVALID_BINOP_IDENT = 19
    ecINVALID_INFIX_IDENT = 20
    ecINVALID_POSTFIX_IDENT = 21

    ecBUILTIN_OVERLOAD = 22
    ecINVALID_FUN_PTR = 23
    ecINVALID_VAR_PTR = 24
    ecEMPTY_EXPRESSION = 25
    ecNAME_CONFLICT = 26
    ecOPT_PRI = 27

    ecDOMAIN_ERROR = 28
    ecDIV_BY_ZERO = 29
    ecGENERIC = 30
    ecLOCALE = 31

    ecUNEXPECTED_CONDITIONAL = 32
    ecMISSING_ELSE_CLAUSE = 33
    ecMISPLACED_COLON = 34

    ecUNREASONABLE_NUMBER_OF_COMPUTATIONS = 35
    ecIDENTIFIER_TOO_LONG = 36
    ecEXPRESSION_TOO_LONG = 37
    ecINVALID_CHARACTERS_FOUND = 38

    # Internal errors
    ecINTERNAL_ERROR = 39
    ecBYTECODE_IMPORT_EXPORT_DISABLED = 40
    ecUNARY_PLUS_IN_FRONT_OF_FUNCTION = 41
    ecNESTING_LIMIT = 42

    ecCOUNT = 43
    ecUNDEFINED = -1


# Convenient aliases
CmdCode = ECmdCode
TypeCode = ETypeCode
OprtAssociativity = EOprtAssociativity
OprtPrecedence = EOprtPrecedence
ErrorCode = EErrorCodes

PUBLIC_ENUMS = (
    ECmdCode,
    ETypeCode,
    EParserVersionInfo,
    EOprtAssociativity,
    EOprtPrecedence,
    EErrorCodes,
)

# Export individual enum members at module level for C++ style access
for _enum_cls in PUBLIC_ENUMS:
    globals().update(_enum_cls.__members__)
