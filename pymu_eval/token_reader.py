"""
Token, Callback, VariableRef, and ParserTokenReader implementations for pymu-eval.
Mirrors muParserToken.h, muParserCallback.h, and muParserTokenReader.cpp.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Optional, TYPE_CHECKING

from .definitions import (
    BINARY_OPRT_CODES,
    DEFAULT_OPRT,
    ECmdCode,
    EErrorCodes,
    EOprtAssociativity,
    EOprtPrecedence,
    ESynCodes,
    ETypeCode,
    MaxNestingDepth,
)
from .errors import ParserError
from .math_impl import MathImpl

if TYPE_CHECKING:  # pragma: no cover
    from .parser import ParserBase


class VariableRef:
    """
    Mutable container holding a variable's current value (scalar, NumPy ndarray,
    xarray DataArray, or xarray Dataset).
    Allows C++-style pointer semantics in Python as well as direct value assignment.
    """

    __slots__ = ("name", "value")

    def __init__(self, value: Any = 0.0, name: str = "") -> None:
        self.name = name
        self.value = value

    def get(self) -> Any:
        """Return the current value held by this variable reference."""
        return self.value

    def set(self, val: Any) -> Any:
        """Update the value held by this variable reference and return it."""
        self.value = val
        return val

    def __repr__(self) -> str:
        return f"VariableRef(name={self.name!r}, value={self.value!r})"


# Alias for user convenience
Variable = VariableRef


class ParserCallback:  # pylint: disable=too-many-instance-attributes
    """
    Encapsulates a function or operator callback along with its metadata
    (arity, precedence, associativity, command code, return/argument type, and optimizability).
    Mirrors mu::ParserCallback.
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        func: Optional[Callable[..., Any]],
        allow_opt: bool = True,
        prec: int = EOprtPrecedence.prINFIX,
        code: ECmdCode = ECmdCode.cmFUNC,
        associativity: EOprtAssociativity = EOprtAssociativity.oaNONE,
        argc: Optional[int] = None,
        tp: ETypeCode = ETypeCode.tpDBL,
        user_data: Any = None,
        has_user_data: bool = False,
    ) -> None:
        self.func = func
        self.allow_opt = allow_opt
        self.prec = int(prec)
        self.code = code
        self.associativity = associativity
        self.user_data = user_data
        self.has_user_data = has_user_data
        self.tp = tp
        self.argc: int = 0

        if func is not None:
            if argc is not None:
                self.argc = argc
            else:
                self._infer_signature(func, has_user_data)

    def _infer_signature(self, func: Callable[..., Any], has_user_data: bool) -> None:
        # Handle numpy ufuncs
        if hasattr(func, "nin"):
            self.argc = int(getattr(func, "nin")) - (1 if has_user_data else 0)
            return

        try:
            sig = inspect.signature(func)
        except (ValueError, TypeError):
            self.argc = -1
            return

        params = list(sig.parameters.values())
        if has_user_data and params:
            params = params[1:]

        has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)
        pos_kinds = (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
        pos_params = [p for p in params if p.kind in pos_kinds]

        # Check if first positional parameter is annotated as str
        first_is_str = bool(
            pos_params and pos_params[0].annotation in (str, "str")
        )

        if first_is_str or self.code == ECmdCode.cmFUNC_STR:
            self.code = ECmdCode.cmFUNC_STR
            self.tp = ETypeCode.tpSTR
            self.argc = -1 if has_varargs else max(0, len(pos_params) - 1)
            self.allow_opt = False
        elif has_varargs and len(pos_params) == 0:
            self.argc = -1
        else:
            # Count required positional parameters (or total positional if all required)
            required = [p for p in pos_params if p.default is inspect.Parameter.empty]
            self.argc = len(required) if required else len(pos_params)

    def is_valid(self) -> bool:
        """Return True if the underlying callback is callable."""
        return callable(self.func)

    def is_optimizable(self) -> bool:
        """Return True if constant folding is permitted for this callback."""
        return self.allow_opt

    def get_code(self) -> ECmdCode:
        """Return the command code associated with this callback."""
        return self.code

    def get_type(self) -> ETypeCode:
        """Return the value/parameter type code."""
        return self.tp

    def get_pri(self) -> int:
        """Return operator precedence."""
        return self.prec

    def get_associativity(self) -> EOprtAssociativity:
        """Return operator associativity."""
        return self.associativity

    def get_arg_count(self) -> int:
        """Return expected argument count (-1 for variadic)."""
        return self.argc

    def call_fun(self, *args: Any) -> Any:
        """Invoke a fixed-arity numeric function callback."""
        assert self.func is not None
        if self.has_user_data:
            return self.func(self.user_data, *args)
        return self.func(*args)

    def call_multfun(self, args: list[Any]) -> Any:
        """Invoke a variadic numeric function callback."""
        assert self.func is not None
        if self.has_user_data:
            return self.func(self.user_data, *args)
        return self.func(*args)

    def call_strfun(self, str_arg: str, *num_args: Any) -> Any:
        """Invoke a fixed-arity string function callback."""
        assert self.func is not None
        if self.has_user_data:
            return self.func(self.user_data, str_arg, *num_args)
        return self.func(str_arg, *num_args)

    def call_multstrfun(self, str_arg: str, num_args: list[Any]) -> Any:
        """Invoke a variadic string function callback."""
        assert self.func is not None
        if self.has_user_data:
            return self.func(self.user_data, str_arg, *num_args)
        return self.func(str_arg, *num_args)

    def call_bulkfun(self, offset: int, thread_id: int, *args: Any) -> Any:
        """Invoke a bulk-mode function callback."""
        assert self.func is not None
        if self.has_user_data:
            return self.func(self.user_data, offset, thread_id, *args)
        return self.func(offset, thread_id, *args)


class ParserToken:
    """
    Token produced by ParserTokenReader and manipulated during RPN compilation.
    Mirrors mu::ParserToken.
    """

    def __init__(self) -> None:
        self.code: ECmdCode = ECmdCode.cmUNKNOWN
        self.tp: ETypeCode = ETypeCode.tpVOID
        self.str_tok: str = ""
        self.val: Any = 0.0
        self.var_ref: Optional[VariableRef] = None
        self.callback: Optional[ParserCallback] = None
        self.str_idx: int = -1

    def set(self, code_or_cb: ECmdCode | ParserCallback, str_tok: str = "") -> ParserToken:
        """Configure this token as a command code or callback token."""
        if isinstance(code_or_cb, ParserCallback):
            cb = code_or_cb
            self.code = cb.get_code()
            self.tp = cb.get_type()
            self.str_tok = str_tok
            self.callback = cb
            self.var_ref = None
        else:
            self.code = code_or_cb
            self.tp = ETypeCode.tpVOID
            self.str_tok = str_tok
            self.callback = None
            self.var_ref = None
        return self

    def set_val(self, val: Any, str_tok: str = "") -> ParserToken:
        """Configure this token as a numeric/array value token."""
        self.code = ECmdCode.cmVAL
        self.tp = ETypeCode.tpDBL
        self.val = val
        self.str_tok = str_tok
        self.callback = None
        self.var_ref = None
        return self

    def set_var(self, var_ref: VariableRef, str_tok: str = "") -> ParserToken:
        """Configure this token as a variable reference token."""
        self.code = ECmdCode.cmVAR
        self.tp = ETypeCode.tpDBL
        self.var_ref = var_ref
        self.str_tok = str_tok
        self.callback = None
        return self

    def set_string(self, str_tok: str, idx: int) -> ParserToken:
        """Configure this token as a string literal token."""
        self.code = ECmdCode.cmSTRING
        self.tp = ETypeCode.tpSTR
        self.str_tok = str_tok
        self.str_idx = idx
        self.val = None
        self.var_ref = None
        self.callback = None
        return self

    def get_code(self) -> ECmdCode:
        """Return the token command code."""
        return self.code

    def get_type(self) -> ETypeCode:
        """Return the token value type."""
        if self.callback is not None:
            return self.callback.get_type()
        return self.tp

    def get_pri(self) -> int:
        """Return the operator priority of a callback token."""
        if self.callback is None:
            raise ParserError(EErrorCodes.ecINTERNAL_ERROR)
        return self.callback.get_pri()

    def get_associativity(self) -> EOprtAssociativity:
        """Return the operator associativity of a callback token."""
        if self.callback is None:
            raise ParserError(EErrorCodes.ecINTERNAL_ERROR)
        return self.callback.get_associativity()

    def get_val(self) -> Any:
        """Return the constant or variable value associated with this token."""
        if self.code == ECmdCode.cmVAL:
            return self.val
        if self.code == ECmdCode.cmVAR and self.var_ref is not None:
            return self.var_ref.get()
        raise ParserError(EErrorCodes.ecVAL_EXPECTED)

    def get_var(self) -> VariableRef:
        """Return the VariableRef bound to a variable token."""
        if self.code != ECmdCode.cmVAR or self.var_ref is None:
            raise ParserError(EErrorCodes.ecINTERNAL_ERROR)
        return self.var_ref

    def get_arg_count(self) -> int:
        """Return the argument count of a callback token."""
        if self.callback is None:
            raise ParserError(EErrorCodes.ecINTERNAL_ERROR)
        return self.callback.get_arg_count()

    def get_as_string(self) -> str:
        """Return the raw token string."""
        return self.str_tok

    def get_idx(self) -> int:
        """Return the string table index for a string token."""
        if self.str_idx < 0 or self.code != ECmdCode.cmSTRING:
            raise ParserError(EErrorCodes.ecINTERNAL_ERROR)
        return self.str_idx

    def get_func_addr(self) -> Optional[ParserCallback]:
        """Return the callback object bound to this token, if any."""
        return self.callback

    def is_optimizable(self) -> bool:
        """Return True if this token represents an optimizable callback."""
        return self.callback.is_optimizable() if self.callback is not None else False


class ParserTokenReader:  # pylint: disable=too-many-instance-attributes
    """
    Token reader / lexer for ParserBase.
    Mirrors mu::ParserTokenReader in muParserTokenReader.cpp.
    """

    def __init__(self, parent: ParserBase) -> None:
        self._parser: ParserBase = parent
        self._formula: str = ""
        self._pos: int = 0
        self._syn_flags: ESynCodes = ESynCodes.noNONE
        self._ignore_undef_var: bool = False
        self._ident_funs: list[Callable[[str, int], tuple[bool, int, Any]]] = []
        self._used_var: dict[str, Optional[VariableRef]] = {}
        self._zero_ref: VariableRef = VariableRef(0.0, "__zero__")
        self._bracket_stack: list[ECmdCode] = []
        self._last_tok: ParserToken = ParserToken()
        self._arg_sep: str = ","

    def clone(self, parent: ParserBase) -> ParserTokenReader:
        """Clone this token reader and bind it to parent."""
        reader = ParserTokenReader(parent)
        # pylint: disable=protected-access
        reader._formula = self._formula
        reader._pos = self._pos
        reader._syn_flags = self._syn_flags
        reader._ignore_undef_var = self._ignore_undef_var
        reader._ident_funs = list(self._ident_funs)
        reader._used_var = dict(self._used_var)
        reader._bracket_stack = list(self._bracket_stack)
        reader._last_tok = self._last_tok
        reader._arg_sep = self._arg_sep
        return reader

    def add_val_ident(self, callback: Callable[..., Any]) -> None:
        """
        Register a custom value identification callback with highest priority (push_front).
        Supports both:
        - Pythonic signature: callback(expr_substr: str) -> Optional[tuple[int, float]]
          or callback(expr: str, pos: int) -> tuple[bool, int, float]
        - C++-like signature: callback(expr_substr, pos_list, val_list) -> int/bool
        """
        self._ident_funs.insert(0, self._wrap_val_ident(callback))

    @staticmethod
    def _wrap_val_ident(
        callback: Callable[..., Any]
    ) -> Callable[[str, int], tuple[bool, int, Any]]:
        try:
            sig = inspect.signature(callback)
            n_params = len(sig.parameters)
        except (ValueError, TypeError):
            n_params = 3

        def wrapped(formula: str, pos: int) -> tuple[bool, int, Any]:
            substr = formula[pos:]
            if n_params == 1:
                res = callback(substr)
                if res and res[0] > 0:
                    return True, pos + int(res[0]), res[1]
            elif n_params == 2:
                res = callback(formula, pos)
                if isinstance(res, tuple) and len(res) == 3:
                    return bool(res[0]), int(res[1]), res[2]
                if res:
                    return True, pos + int(res[0]), res[1]
            else:
                pos_box = [pos]
                val_box = [0.0]
                if callback(substr, pos_box, val_box):
                    return True, int(pos_box[0]), val_box[0]
            return False, pos, 0.0

        return wrapped

    def get_pos(self) -> int:
        """Return the current lexer position in the formula."""
        return self._pos

    def get_expr(self) -> str:
        """Return the current formula string."""
        return self._formula

    def get_used_var(self) -> dict[str, Optional[VariableRef]]:
        """Return the dictionary of variables referenced in the formula."""
        return self._used_var

    def set_formula(self, formula: str) -> None:
        """Set a new formula string and reset lexer state."""
        self._formula = formula
        self.re_init()

    def ignore_undef_var(self, ignore: bool) -> None:
        """Enable or disable ignoring undefined variables during lexing."""
        self._ignore_undef_var = ignore

    def set_arg_sep(self, sep: str) -> None:
        """Set the function argument separator character."""
        self._arg_sep = sep

    def get_arg_sep(self) -> str:
        """Return the function argument separator character."""
        return self._arg_sep

    def re_init(self) -> None:
        """Reset lexer position, syntax flags, bracket stack, and used variables."""
        self._pos = 0
        self._syn_flags = ESynCodes.sfSTART_OF_LINE
        self._bracket_stack.clear()
        self._used_var.clear()
        self._last_tok = ParserToken()

    def _save_before_return(self, tok: ParserToken) -> ParserToken:
        self._last_tok = tok
        return tok

    def _error(self, errc: EErrorCodes, pos: int = -1, tok: str = "") -> None:
        self._parser._error(errc, pos, tok)  # pylint: disable=protected-access

    def _extract_token(self, charset: str, pos: int) -> tuple[int, str]:
        n = len(self._formula)
        end = pos
        charset_set = set(charset)
        while end < n and self._formula[end] in charset_set:
            end += 1
        return end, self._formula[pos:end]

    def _extract_operator_token(self, pos: int) -> tuple[int, str]:
        end, tok_str = self._extract_token(self._parser.valid_oprt_chars(), pos)
        if end != pos:
            return end, tok_str
        return self._extract_token("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ", pos)

    def read_next_token(self) -> ParserToken:
        """Read and return the next token from the formula."""
        n = len(self._formula)
        # Skip whitespace (0 < ord <= 0x20), rejecting control chars 14..31
        while self._pos < n:
            ch_ord = ord(self._formula[self._pos])
            if 0 < ch_ord <= 0x20:
                if 14 <= ch_ord <= 31:
                    self._error(EErrorCodes.ecINVALID_CHARACTERS_FOUND, self._pos)
                self._pos += 1
            else:
                break

        tok = ParserToken()
        token_matchers = (
            self._is_eof,
            self._is_oprt,
            self._is_fun_tok,
            self._is_built_in,
            self._is_arg_sep,
            self._is_val_tok,
            self._is_var_tok,
            self._is_str_var_tok,
            self._is_string,
            self._is_infix_op_tok,
            self._is_post_op_tok,
        )
        for matcher in token_matchers:
            if matcher(tok):
                return self._save_before_return(tok)

        # pylint: disable=protected-access
        if (
            self._ignore_undef_var or self._parser._var_factory is not None
        ) and self._is_undef_var_tok(tok):
            return self._save_before_return(tok)

        # Unknown token
        end, str_tok = self._extract_token(self._parser.valid_name_chars(), self._pos)
        if end != self._pos:
            self._error(EErrorCodes.ecUNASSIGNABLE_TOKEN, self._pos, str_tok)

        self._error(EErrorCodes.ecUNASSIGNABLE_TOKEN, self._pos, self._formula[self._pos :])
        return tok  # unreachable

    def _is_eof(self, tok: ParserToken) -> bool:
        if self._pos >= len(self._formula):
            if self._syn_flags & ESynCodes.noEND:
                self._error(EErrorCodes.ecUNEXPECTED_EOF, self._pos)
            if self._bracket_stack:
                self._error(EErrorCodes.ecMISSING_PARENS, self._pos, ")")
            self._syn_flags = ESynCodes.noNONE
            tok.set(ECmdCode.cmEND)
            return True
        return False

    def _is_built_in(self, tok: ParserToken) -> bool:  # pylint: disable=too-many-branches
        for i, op_str in enumerate(DEFAULT_OPRT):
            op_len = len(op_str)
            if self._formula.startswith(op_str, self._pos):
                code = ECmdCode(i)
                if code in BINARY_OPRT_CODES:
                    if code == ECmdCode.cmASSIGN and (self._syn_flags & ESynCodes.noASSIGN):
                        self._error(EErrorCodes.ecUNEXPECTED_OPERATOR, self._pos, op_str)

                    if not self._parser.has_built_in_oprt():
                        continue

                    if self._syn_flags & ESynCodes.noOPT:
                        if self._is_infix_op_tok(tok):
                            return True
                        self._error(EErrorCodes.ecUNEXPECTED_OPERATOR, self._pos, op_str)

                    self._syn_flags = (
                        ESynCodes.noBC
                        | ESynCodes.noOPT
                        | ESynCodes.noARG_SEP
                        | ESynCodes.noPOSTOP
                        | ESynCodes.noASSIGN
                        | ESynCodes.noIF
                        | ESynCodes.noELSE
                        | ESynCodes.noEND
                    )
                elif code == ECmdCode.cmBO:
                    if self._syn_flags & ESynCodes.noBO:
                        self._error(EErrorCodes.ecUNEXPECTED_PARENS, self._pos, op_str)

                    if self._last_tok.get_code() == ECmdCode.cmFUNC:
                        self._syn_flags = (
                            ESynCodes.noOPT
                            | ESynCodes.noEND
                            | ESynCodes.noARG_SEP
                            | ESynCodes.noPOSTOP
                            | ESynCodes.noASSIGN
                            | ESynCodes.noIF
                            | ESynCodes.noELSE
                        )
                    else:
                        self._syn_flags = (
                            ESynCodes.noBC
                            | ESynCodes.noOPT
                            | ESynCodes.noEND
                            | ESynCodes.noARG_SEP
                            | ESynCodes.noPOSTOP
                            | ESynCodes.noASSIGN
                            | ESynCodes.noIF
                            | ESynCodes.noELSE
                        )

                    if len(self._bracket_stack) >= MaxNestingDepth:
                        self._error(EErrorCodes.ecNESTING_LIMIT, self._pos, op_str)

                    self._bracket_stack.append(ECmdCode.cmBO)
                elif code == ECmdCode.cmBC:
                    if self._syn_flags & ESynCodes.noBC:
                        self._error(EErrorCodes.ecUNEXPECTED_PARENS, self._pos, op_str)

                    self._syn_flags = (
                        ESynCodes.noBO
                        | ESynCodes.noVAR
                        | ESynCodes.noVAL
                        | ESynCodes.noFUN
                        | ESynCodes.noINFIXOP
                        | ESynCodes.noSTR
                        | ESynCodes.noASSIGN
                    )

                    if self._bracket_stack:
                        self._bracket_stack.pop()
                    else:
                        self._error(EErrorCodes.ecUNEXPECTED_PARENS, self._pos, op_str)
                elif code == ECmdCode.cmELSE:
                    if self._syn_flags & ESynCodes.noELSE:
                        self._error(EErrorCodes.ecUNEXPECTED_CONDITIONAL, self._pos, op_str)

                    self._syn_flags = (
                        ESynCodes.noBC
                        | ESynCodes.noPOSTOP
                        | ESynCodes.noEND
                        | ESynCodes.noOPT
                        | ESynCodes.noIF
                        | ESynCodes.noELSE
                        | ESynCodes.noSTR
                    )
                elif code == ECmdCode.cmIF:
                    if self._syn_flags & ESynCodes.noIF:
                        self._error(EErrorCodes.ecUNEXPECTED_CONDITIONAL, self._pos, op_str)

                    self._syn_flags = (
                        ESynCodes.noBC
                        | ESynCodes.noPOSTOP
                        | ESynCodes.noEND
                        | ESynCodes.noOPT
                        | ESynCodes.noIF
                        | ESynCodes.noELSE
                        | ESynCodes.noSTR
                    )
                else:  # pragma: no cover
                    self._error(EErrorCodes.ecINTERNAL_ERROR)

                self._pos += op_len
                tok.set(code, op_str)
                return True
        return False

    def _is_arg_sep(self, tok: ParserToken) -> bool:
        if self._pos < len(self._formula) and self._formula[self._pos] == self._arg_sep:
            if self._syn_flags & ESynCodes.noARG_SEP:
                self._error(EErrorCodes.ecUNEXPECTED_ARG_SEP, self._pos, self._arg_sep)

            self._syn_flags = (
                ESynCodes.noBC
                | ESynCodes.noOPT
                | ESynCodes.noEND
                | ESynCodes.noARG_SEP
                | ESynCodes.noPOSTOP
                | ESynCodes.noASSIGN
            )
            self._pos += 1
            tok.set(ECmdCode.cmARG_SEP, self._arg_sep)
            return True
        return False

    # pylint: disable=protected-access
    def _is_infix_op_tok(self, tok: ParserToken) -> bool:
        end, s_tok = self._extract_token(self._parser.valid_infix_oprt_chars(), self._pos)
        if end == self._pos:
            return False

        # Iterate in reverse lexicographical order (longest prefix first for shared prefixes)
        for op_name in sorted(self._parser._infix_oprt_def.keys(), reverse=True):
            if not s_tok.startswith(op_name):
                continue

            cb = self._parser._infix_oprt_def[op_name]
            tok.set(cb, op_name)
            self._pos += len(op_name)

            if self._syn_flags & ESynCodes.noINFIXOP:
                self._error(EErrorCodes.ecUNEXPECTED_OPERATOR, self._pos, tok.get_as_string())

            self._syn_flags |= (
                ESynCodes.noPOSTOP
                | ESynCodes.noINFIXOP
                | ESynCodes.noOPT
                | ESynCodes.noBC
                | ESynCodes.noSTR
                | ESynCodes.noASSIGN
                | ESynCodes.noARG_SEP
            )
            return True

        return False

    def _is_fun_tok(self, tok: ParserToken) -> bool:
        end, str_tok = self._extract_token(self._parser.valid_name_chars(), self._pos)
        if end == self._pos:
            return False

        cb = self._parser._fun_def.get(str_tok)
        if cb is None:
            return False

        if end >= len(self._formula) or self._formula[end] != "(":
            return False

        # Issue #164: Unary plus in front of a function is forbidden
        last_cb = self._last_tok.get_func_addr()
        if last_cb is not None and last_cb.func is MathImpl.UnaryPlus:
            self._error(
                EErrorCodes.ecUNARY_PLUS_IN_FRONT_OF_FUNCTION,
                self._pos,
                str_tok,
            )

        tok.set(cb, str_tok)
        self._pos = end
        if self._syn_flags & ESynCodes.noFUN:
            self._error(EErrorCodes.ecUNEXPECTED_FUN, self._pos - len(str_tok), str_tok)

        self._syn_flags = ESynCodes.noANY ^ ESynCodes.noBO
        return True

    def _is_oprt(self, tok: ParserToken) -> bool:
        end, str_tok = self._extract_operator_token(self._pos)
        if end == self._pos:
            return False

        if self._parser.has_built_in_oprt() and str_tok in DEFAULT_OPRT:
            return False

        for s_id in sorted(self._parser._oprt_def.keys(), reverse=True):
            if self._formula.startswith(s_id, self._pos):
                cb = self._parser._oprt_def[s_id]
                tok.set(cb, str_tok)

                if self._syn_flags & ESynCodes.noOPT:
                    if self._is_infix_op_tok(tok):
                        return True
                    return False

                self._pos += len(s_id)
                self._syn_flags = (
                    ESynCodes.noBC
                    | ESynCodes.noOPT
                    | ESynCodes.noARG_SEP
                    | ESynCodes.noPOSTOP
                    | ESynCodes.noEND
                    | ESynCodes.noASSIGN
                )
                return True

        return False

    def _is_post_op_tok(self, tok: ParserToken) -> bool:
        if self._syn_flags & ESynCodes.noPOSTOP:
            return False

        end, s_tok = self._extract_token(self._parser.valid_oprt_chars(), self._pos)
        if end == self._pos:
            return False

        for op_name in sorted(self._parser._post_oprt_def.keys(), reverse=True):
            if not s_tok.startswith(op_name):
                continue

            cb = self._parser._post_oprt_def[op_name]
            tok.set(cb, s_tok)
            self._pos += len(op_name)

            self._syn_flags = (
                ESynCodes.noVAL
                | ESynCodes.noVAR
                | ESynCodes.noFUN
                | ESynCodes.noBO
                | ESynCodes.noPOSTOP
                | ESynCodes.noSTR
                | ESynCodes.noASSIGN
            )
            return True

        return False

    def _is_val_tok(self, tok: ParserToken) -> bool:
        # 1. Check user-defined constants
        end, str_tok = self._extract_token(self._parser.valid_name_chars(), self._pos)
        if end != self._pos and str_tok in self._parser._const_def:
            self._pos = end
            tok.set_val(self._parser._const_def[str_tok], str_tok)
            if self._syn_flags & ESynCodes.noVAL:
                self._error(EErrorCodes.ecUNEXPECTED_VAL, self._pos - len(str_tok), str_tok)

            self._syn_flags = (
                ESynCodes.noVAL
                | ESynCodes.noVAR
                | ESynCodes.noFUN
                | ESynCodes.noBO
                | ESynCodes.noINFIXOP
                | ESynCodes.noSTR
                | ESynCodes.noASSIGN
            )
            return True

        # 2. Try registered value identification callbacks
        for ident_fun in self._ident_funs:
            start = self._pos
            matched, new_pos, val = ident_fun(self._formula, self._pos)
            if matched:
                self._pos = new_pos
                str_tok = self._formula[start:new_pos]
                if self._syn_flags & ESynCodes.noVAL:
                    self._error(EErrorCodes.ecUNEXPECTED_VAL, self._pos - len(str_tok), str_tok)

                tok.set_val(val, str_tok)
                self._syn_flags = (
                    ESynCodes.noVAL
                    | ESynCodes.noVAR
                    | ESynCodes.noFUN
                    | ESynCodes.noBO
                    | ESynCodes.noINFIXOP
                    | ESynCodes.noSTR
                    | ESynCodes.noASSIGN
                )
                return True

        return False

    def _is_var_tok(self, tok: ParserToken) -> bool:
        if not self._parser._var_def:
            return False

        end, str_tok = self._extract_token(self._parser.valid_name_chars(), self._pos)
        if end == self._pos:
            return False

        var_ref = self._parser._var_def.get(str_tok)
        if var_ref is None:
            return False

        if self._syn_flags & ESynCodes.noVAR:
            self._error(EErrorCodes.ecUNEXPECTED_VAR, self._pos, str_tok)

        self._pos = end
        tok.set_var(var_ref, str_tok)
        self._used_var[str_tok] = var_ref

        self._syn_flags = (
            ESynCodes.noVAL
            | ESynCodes.noVAR
            | ESynCodes.noFUN
            | ESynCodes.noBO
            | ESynCodes.noINFIXOP
            | ESynCodes.noSTR
        )
        return True

    def _is_str_var_tok(self, tok: ParserToken) -> bool:
        if not self._parser._str_var_def:
            return False

        end, str_tok = self._extract_token(self._parser.valid_name_chars(), self._pos)
        if end == self._pos:
            return False

        idx = self._parser._str_var_def.get(str_tok)
        if idx is None:
            return False

        if self._syn_flags & ESynCodes.noSTR:
            self._error(EErrorCodes.ecUNEXPECTED_VAR, self._pos, str_tok)

        self._pos = end
        if not self._parser._string_var_buf:
            self._error(EErrorCodes.ecINTERNAL_ERROR)

        str_val = self._parser._string_var_buf[idx]
        self._parser._string_buf.append(str_val)
        tok.set_string(str_val, len(self._parser._string_buf) - 1)

        self._syn_flags = ESynCodes.noANY ^ (
            ESynCodes.noBC | ESynCodes.noOPT | ESynCodes.noEND | ESynCodes.noARG_SEP
        )
        return True

    def _is_undef_var_tok(self, tok: ParserToken) -> bool:
        end, str_tok = self._extract_token(self._parser.valid_name_chars(), self._pos)
        if end == self._pos:
            return False

        if self._syn_flags & ESynCodes.noVAR:
            self._error(
                EErrorCodes.ecUNEXPECTED_VAR,
                self._pos - len(tok.get_as_string()),
                str_tok,
            )

        if self._parser._var_factory is not None:
            f_var = self._parser._call_var_factory(str_tok)
            tok.set_var(f_var, str_tok)
            self._parser._var_def[str_tok] = f_var
            self._used_var[str_tok] = f_var
        else:
            tok.set_var(self._zero_ref, str_tok)
            self._used_var[str_tok] = None

        self._pos = end
        self._syn_flags = (
            ESynCodes.noVAL
            | ESynCodes.noVAR
            | ESynCodes.noFUN
            | ESynCodes.noBO
            | ESynCodes.noPOSTOP
            | ESynCodes.noINFIXOP
            | ESynCodes.noSTR
        )
        return True

    def _is_string(self, tok: ParserToken) -> bool:
        if self._pos >= len(self._formula) or self._formula[self._pos] != '"':
            return False

        str_buf = list(self._formula[self._pos + 1 :])
        i = 0
        skip = 0
        end_idx = -1
        while i < len(str_buf):
            if str_buf[i] == '"':
                if i == 0 or str_buf[i - 1] != "\\":
                    end_idx = i
                    break
                # Escaped quote \" -> replace with "
                str_buf[i - 1 : i + 1] = ['"']
                skip += 1
                continue
            i += 1

        if end_idx == -1:
            self._error(EErrorCodes.ecUNTERMINATED_STRING, self._pos, '"')

        str_tok = "".join(str_buf[:end_idx])
        if self._syn_flags & ESynCodes.noSTR:
            self._error(EErrorCodes.ecUNEXPECTED_STR, self._pos, str_tok)

        self._parser._string_buf.append(str_tok)
        tok.set_string(str_tok, len(self._parser._string_buf) - 1)

        self._pos += len(str_tok) + 2 + skip
        self._syn_flags = ESynCodes.noANY ^ (
            ESynCodes.noARG_SEP | ESynCodes.noBC | ESynCodes.noOPT | ESynCodes.noEND
        )
        return True
