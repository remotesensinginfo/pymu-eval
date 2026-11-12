"""
Error classes and message definitions for pymu-eval.
Mirrors muParserError.h and muParserError.cpp from the C++ muParser library.
"""

from typing import Optional, Union
from .definitions import EErrorCodes

ERROR_MESSAGES: dict[EErrorCodes, str] = {
    EErrorCodes.ecUNASSIGNABLE_TOKEN: 'Unexpected token "$TOK$" found at position $POS$.',
    EErrorCodes.ecINTERNAL_ERROR: "Internal error",
    EErrorCodes.ecINVALID_NAME: 'Invalid function-, variable- or constant name: "$TOK$".',
    EErrorCodes.ecINVALID_BINOP_IDENT: 'Invalid binary operator identifier: "$TOK$".',
    EErrorCodes.ecINVALID_INFIX_IDENT: 'Invalid infix operator identifier: "$TOK$".',
    EErrorCodes.ecINVALID_POSTFIX_IDENT: 'Invalid postfix operator identifier: "$TOK$".',
    EErrorCodes.ecINVALID_FUN_PTR: "Invalid pointer to callback function.",
    EErrorCodes.ecEMPTY_EXPRESSION: "Expression is empty.",
    EErrorCodes.ecINVALID_VAR_PTR: "Invalid pointer to variable.",
    EErrorCodes.ecUNEXPECTED_OPERATOR: 'Unexpected operator "$TOK$" found at position $POS$',
    EErrorCodes.ecUNEXPECTED_EOF: "Unexpected end of expression at position $POS$",
    EErrorCodes.ecUNEXPECTED_ARG_SEP: "Unexpected argument separator at position $POS$",
    EErrorCodes.ecUNEXPECTED_PARENS: 'Unexpected parenthesis "$TOK$" at position $POS$',
    EErrorCodes.ecUNEXPECTED_FUN: 'Unexpected function "$TOK$" at position $POS$',
    EErrorCodes.ecUNEXPECTED_VAL: 'Unexpected value "$TOK$" found at position $POS$',
    EErrorCodes.ecUNEXPECTED_VAR: 'Unexpected variable "$TOK$" found at position $POS$',
    EErrorCodes.ecUNEXPECTED_ARG: "Function arguments used without a function (position: $POS$)",
    EErrorCodes.ecMISSING_PARENS: "Missing parenthesis",
    EErrorCodes.ecTOO_MANY_PARAMS: 'Too many parameters for function "$TOK$" at expression position $POS$',
    EErrorCodes.ecTOO_FEW_PARAMS: 'Too few parameters for function "$TOK$" at expression position $POS$',
    EErrorCodes.ecDIV_BY_ZERO: "Divide by zero",
    EErrorCodes.ecDOMAIN_ERROR: "Domain error",
    EErrorCodes.ecNAME_CONFLICT: "Name conflict",
    EErrorCodes.ecOPT_PRI: "Invalid value for operator priority (must be greater or equal to zero).",
    EErrorCodes.ecBUILTIN_OVERLOAD: 'user defined binary operator "$TOK$" conflicts with a built in operator.',
    EErrorCodes.ecUNEXPECTED_STR: "Unexpected string token found at position $POS$.",
    EErrorCodes.ecUNTERMINATED_STRING: "Unterminated string starting at position $POS$.",
    EErrorCodes.ecSTRING_EXPECTED: "String function called with a non string type of argument.",
    EErrorCodes.ecVAL_EXPECTED: "String value used where a numerical argument is expected.",
    EErrorCodes.ecOPRT_TYPE_CONFLICT: 'No suitable overload for operator "$TOK$" at position $POS$.',
    EErrorCodes.ecSTR_RESULT: "Strings must only be used as function arguments!",
    EErrorCodes.ecGENERIC: "Parser error.",
    EErrorCodes.ecLOCALE: "Decimal separator is identic to function argument separator.",
    EErrorCodes.ecUNEXPECTED_CONDITIONAL: 'The "$TOK$" operator must be preceded by a closing bracket.',
    EErrorCodes.ecMISSING_ELSE_CLAUSE: "If-then-else operator is missing an else clause",
    EErrorCodes.ecMISPLACED_COLON: "Misplaced colon at position $POS$",
    EErrorCodes.ecUNREASONABLE_NUMBER_OF_COMPUTATIONS: (
        "Number of computations to small for bulk mode. (Vectorisation overhead too costly)"
    ),
    EErrorCodes.ecIDENTIFIER_TOO_LONG: "Identifier too long.",
    EErrorCodes.ecEXPRESSION_TOO_LONG: "Expression too long.",
    EErrorCodes.ecINVALID_CHARACTERS_FOUND: "Invalid non printable characters found in expression/identifer!",
    EErrorCodes.ecBYTECODE_IMPORT_EXPORT_DISABLED: (
        "Bytecode cannot be imported or exported when parser is using a variable factory!"
    ),
    EErrorCodes.ecUNARY_PLUS_IN_FRONT_OF_FUNCTION: "Unary plus operator is not allowed in front of functions!",
    EErrorCodes.ecNESTING_LIMIT: "Expression too complex. Nesting limit reached!",
}


class ParserError(Exception):
    """
    Exception raised by the muParser engine on syntax, configuration, or domain errors.
    Mirrors mu::ParserError.
    """

    def __init__(
        self,
        code_or_msg: Union[EErrorCodes, int, str] = EErrorCodes.ecUNDEFINED,
        tok_or_pos: Union[str, int] = "",
        expr_or_tok: str = "",
        pos: int = -1,
    ) -> None:
        self._expr: str = ""
        self._tok: str = ""
        self._pos: int = -1
        self._code: EErrorCodes = EErrorCodes.ecUNDEFINED
        self._msg: str = ""

        if isinstance(code_or_msg, str):
            self._code = EErrorCodes.ecGENERIC
            self._msg = code_or_msg
            if isinstance(tok_or_pos, int):
                self._pos = tok_or_pos
                self._tok = expr_or_tok
            else:
                self._tok = tok_or_pos
                self._pos = pos
        else:
            try:
                self._code = EErrorCodes(code_or_msg)
            except ValueError:
                self._code = EErrorCodes.ecUNDEFINED
            if isinstance(tok_or_pos, int):
                # ParserError(EErrorCodes, int iPos, str sTok)
                self._pos = tok_or_pos
                self._tok = expr_or_tok
            else:
                # ParserError(EErrorCodes, str sTok, str sFormula, int iPos)
                self._tok = tok_or_pos
                self._expr = expr_or_tok
                self._pos = pos
            self._msg = ERROR_MESSAGES.get(self._code, "Parser error.")

        self._msg = self._msg.replace("$POS$", str(self._pos)).replace("$TOK$", self._tok)
        super().__init__(self._msg)

    @property
    def code(self) -> EErrorCodes:
        return self._code

    @property
    def msg(self) -> str:
        return self._msg

    @property
    def expr(self) -> str:
        return self._expr

    @property
    def pos(self) -> int:
        return self._pos

    @property
    def token(self) -> str:
        return self._tok

    # C++ style accessors
    def GetCode(self) -> EErrorCodes:
        return self._code

    def GetMsg(self) -> str:
        return self._msg

    def GetExpr(self) -> str:
        return self._expr

    def GetPos(self) -> int:
        return self._pos

    def GetToken(self) -> str:
        return self._tok

    def SetFormula(self, formula: str) -> None:
        self._expr = formula

    # Pythonic snake_case accessors
    get_code = GetCode
    get_msg = GetMsg
    get_expr = GetExpr
    get_pos = GetPos
    get_token = GetToken
    set_formula = SetFormula
