"""
Core ParserBase, Parser, and ParserInt classes plus convenience functions
for evaluating mathematical expressions on scalars, NumPy arrays, and xarray Datasets.
Mirrors muParserBase.cpp, muParser.cpp, and muParserInt.cpp.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping
from typing import Any, Callable, Optional, Union

import numpy as np

try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:  # pragma: no cover
    xr = None  # type: ignore
    HAS_XARRAY = False

from .bytecode import ParserByteCode, SToken
from .definitions import (
    DEFAULT_OPRT,
    ECmdCode,
    EErrorCodes,
    EOprtAssociativity,
    EOprtPrecedence,
    EParserVersionInfo,
    ETypeCode,
    MaxLenExpression,
    MaxLenIdentifier,
    MaxNestingDepth,
    ParserVersion,
    ParserVersionDate,
)
from .errors import ParserError
from .math_impl import INT32_MAX, INT32_MIN, MathImpl, ParserIntMathImpl, is_array_like, is_xarray
from .token_reader import ParserCallback, ParserToken, ParserTokenReader, VariableRef


class ParserBase:
    """
    Base mathematical expressions parser engine.
    Compiles expression strings to RPN bytecode and evaluates them over scalars,
    NumPy ndarrays, xarray DataArrays, and xarray Datasets.
    Mirrors mu::ParserBase.
    """

    exception_type = ParserError
    g_DbgDumpCmdCode: bool = False
    g_DbgDumpStack: bool = False

    def __init__(self, expr: Optional[Union[str, ParserBase]] = None) -> None:
        self._rpn: ParserByteCode = ParserByteCode()
        self._string_buf: list[str] = []
        self._string_var_buf: list[str] = []
        self._fun_def: dict[str, ParserCallback] = {}
        self._post_oprt_def: dict[str, ParserCallback] = {}
        self._infix_oprt_def: dict[str, ParserCallback] = {}
        self._oprt_def: dict[str, ParserCallback] = {}
        self._const_def: dict[str, Any] = {}
        self._str_var_def: dict[str, int] = {}
        self._var_def: dict[str, VariableRef] = {}
        self._built_in_op: bool = True
        self._name_chars: str = ""
        self._oprt_chars: str = ""
        self._infix_oprt_chars: str = ""
        self._dec_sep: str = "."
        self._thousands_sep: str = ""
        self._var_factory: Optional[Callable[..., Any]] = None
        self._var_factory_data: Any = None
        self._stack_buffer: list[Any] = []
        self._final_result_idx: int = 0
        self._compiled: bool = False

        self._token_reader: ParserTokenReader = ParserTokenReader(self)

        if isinstance(expr, ParserBase):
            self.assign(expr)
        elif isinstance(expr, str):
            self.set_expr(expr)

    def assign(self, other: ParserBase) -> ParserBase:
        if other is self:
            return self

        self._re_init()
        self._const_def = dict(other._const_def)
        self._var_def = dict(other._var_def)
        self._built_in_op = other._built_in_op
        self._string_buf = list(other._string_buf)
        self._stack_buffer = list(other._stack_buffer)
        self._final_result_idx = other._final_result_idx
        self._str_var_def = dict(other._str_var_def)
        self._string_var_buf = list(other._string_var_buf)
        self._token_reader = other._token_reader.clone(self)

        self._fun_def = dict(other._fun_def)
        self._post_oprt_def = dict(other._post_oprt_def)
        self._infix_oprt_def = dict(other._infix_oprt_def)
        self._oprt_def = dict(other._oprt_def)

        self._name_chars = other._name_chars
        self._oprt_chars = other._oprt_chars
        self._infix_oprt_chars = other._infix_oprt_chars
        self._dec_sep = other._dec_sep
        self._thousands_sep = other._thousands_sep
        self._var_factory = other._var_factory
        self._var_factory_data = other._var_factory_data
        self._rpn.enable_optimizer(other._rpn._enable_optimizer)
        return self

    def copy(self) -> ParserBase:
        cls = self.__class__
        new_parser = cls.__new__(cls)
        ParserBase.__init__(new_parser)
        new_parser.assign(self)
        return new_parser

    def __copy__(self) -> ParserBase:
        return self.copy()

    def __deepcopy__(self, memo: Any) -> ParserBase:
        return self.copy()

    def _re_init(self) -> None:
        self._compiled = False
        self._string_buf.clear()
        self._rpn.clear()
        self._token_reader.re_init()

    def _error(self, errc: EErrorCodes, pos: int = -1, tok: str = "") -> None:
        raise ParserError(errc, tok, self._token_reader.get_expr(), pos)

    # Locale & separator configuration
    def set_dec_sep(self, dec_sep: str) -> None:
        self._dec_sep = dec_sep
        self._re_init()

    def set_thousands_sep(self, thousands_sep: str = "") -> None:
        self._thousands_sep = thousands_sep
        self._re_init()

    def reset_locale(self) -> None:
        self._dec_sep = "."
        self._thousands_sep = ""
        self.set_arg_sep(",")
        self._re_init()

    def set_arg_sep(self, arg_sep: str) -> None:
        self._token_reader.set_arg_sep(arg_sep)
        self._re_init()

    def get_arg_sep(self) -> str:
        return self._token_reader.get_arg_sep()

    # Character set definitions
    def define_name_chars(self, charset: str) -> None:
        self._name_chars = charset

    def define_oprt_chars(self, charset: str) -> None:
        self._oprt_chars = charset

    def define_infix_oprt_chars(self, charset: str) -> None:
        self._infix_oprt_chars = charset

    def valid_name_chars(self) -> str:
        if not self._name_chars:
            raise ParserError(EErrorCodes.ecINTERNAL_ERROR)
        return self._name_chars

    def valid_oprt_chars(self) -> str:
        if not self._oprt_chars:
            raise ParserError(EErrorCodes.ecINTERNAL_ERROR)
        return self._oprt_chars

    def valid_infix_oprt_chars(self) -> str:
        if not self._infix_oprt_chars:
            raise ParserError(EErrorCodes.ecINTERNAL_ERROR)
        return self._infix_oprt_chars

    def _check_oprt(self, name: str, cb: ParserCallback, charset: str) -> None:
        if (
            not name
            or any(c not in charset for c in name)
            or ("0" <= name[0] <= "9")
        ):
            code = cb.get_code()
            if code == ECmdCode.cmOPRT_POSTFIX:
                self._error(EErrorCodes.ecINVALID_POSTFIX_IDENT, -1, name)
            elif code == ECmdCode.cmOPRT_INFIX:
                self._error(EErrorCodes.ecINVALID_INFIX_IDENT, -1, name)
            else:
                self._error(EErrorCodes.ecINVALID_NAME, -1, name)

    def _check_name(self, name: str, charset: str) -> None:
        if (
            not name
            or any(c not in charset for c in name)
            or ("0" <= name[0] <= "9")
        ):
            self._error(EErrorCodes.ecINVALID_NAME, -1, name)

    def _add_callback(
        self,
        name: str,
        cb: ParserCallback,
        storage: dict[str, ParserCallback],
        charset: str,
    ) -> None:
        if not cb.is_valid():
            self._error(EErrorCodes.ecINVALID_FUN_PTR)

        if storage is not self._fun_def and name in self._fun_def:
            self._error(EErrorCodes.ecNAME_CONFLICT, -1, name)

        if storage is not self._post_oprt_def and name in self._post_oprt_def:
            self._error(EErrorCodes.ecNAME_CONFLICT, -1, name)

        if (
            storage is not self._infix_oprt_def
            and storage is not self._oprt_def
            and name in self._infix_oprt_def
        ):
            self._error(EErrorCodes.ecNAME_CONFLICT, -1, name)

        if (
            storage is not self._infix_oprt_def
            and storage is not self._oprt_def
            and name in self._oprt_def
        ):
            self._error(EErrorCodes.ecNAME_CONFLICT, -1, name)

        self._check_oprt(name, cb, charset)
        storage[name] = cb
        self._re_init()

    # Definitions: functions, operators, constants, variables
    def define_fun(
        self,
        name: str,
        func: Optional[Callable[..., Any]],
        allow_opt: bool = True,
        *,
        argc: Optional[int] = None,
        is_str_fun: bool = False,
        is_mult_str_fun: bool = False,
        is_bulk: bool = False,
    ) -> None:
        if is_mult_str_fun:
            cb = ParserCallback(
                func,
                allow_opt=False,
                code=ECmdCode.cmFUNC_STR,
                argc=-1,
                tp=ETypeCode.tpSTR,
            )
        elif is_str_fun:
            cb = ParserCallback(
                func,
                allow_opt=False,
                code=ECmdCode.cmFUNC_STR,
                argc=argc,
                tp=ETypeCode.tpSTR,
            )
        elif is_bulk:
            cb = ParserCallback(
                func,
                allow_opt=False,
                code=ECmdCode.cmFUNC_BULK,
                argc=argc,
                tp=ETypeCode.tpDBL,
            )
        else:
            cb = ParserCallback(func, allow_opt=allow_opt, code=ECmdCode.cmFUNC, argc=argc)
        self._add_callback(name, cb, self._fun_def, self.valid_name_chars())

    def define_str_fun(
        self,
        name: str,
        func: Optional[Callable[..., Any]],
        argc: Optional[int] = None,
    ) -> None:
        self.define_fun(name, func, False, argc=argc, is_str_fun=True)

    def define_mult_str_fun(
        self,
        name: str,
        func: Optional[Callable[..., Any]],
    ) -> None:
        self.define_fun(name, func, False, is_mult_str_fun=True)

    def define_bulk_fun(
        self,
        name: str,
        func: Optional[Callable[..., Any]],
        argc: Optional[int] = None,
    ) -> None:
        self.define_fun(name, func, False, argc=argc, is_bulk=True)

    def define_fun_user_data(
        self,
        name: str,
        func: Optional[Callable[..., Any]],
        user_data: Any,
        allow_opt: bool = True,
        *,
        argc: Optional[int] = None,
        is_str_fun: bool = False,
    ) -> None:
        if user_data is None:
            self._error(EErrorCodes.ecINVALID_FUN_PTR)
        code = ECmdCode.cmFUNC_STR if is_str_fun else ECmdCode.cmFUNC
        tp = ETypeCode.tpSTR if is_str_fun else ETypeCode.tpDBL
        cb = ParserCallback(
            func,
            allow_opt=allow_opt and not is_str_fun,
            code=code,
            argc=argc,
            tp=tp,
            user_data=user_data,
            has_user_data=True,
        )
        self._add_callback(name, cb, self._fun_def, self.valid_name_chars())

    def has_fun(self, name: str) -> bool:
        return name in self._fun_def

    def define_oprt(
        self,
        name: str,
        func: Optional[Callable[..., Any]],
        prec: int = 0,
        associativity: EOprtAssociativity = EOprtAssociativity.oaLEFT,
        allow_opt: bool = False,
    ) -> None:
        if len(name) > MaxLenIdentifier:
            self._error(EErrorCodes.ecIDENTIFIER_TOO_LONG)

        if self._built_in_op and name in DEFAULT_OPRT[: ECmdCode.cmENDIF]:
            self._error(EErrorCodes.ecBUILTIN_OVERLOAD, -1, name)

        cb = ParserCallback(
            func,
            allow_opt=allow_opt,
            prec=prec,
            code=ECmdCode.cmOPRT_BIN,
            associativity=associativity,
            argc=2,
        )
        self._add_callback(name, cb, self._oprt_def, self.valid_oprt_chars())

    def define_postfix_oprt(
        self,
        name: str,
        func: Optional[Callable[..., Any]],
        allow_opt: bool = True,
    ) -> None:
        if len(name) > MaxLenIdentifier:
            self._error(EErrorCodes.ecIDENTIFIER_TOO_LONG)

        cb = ParserCallback(
            func,
            allow_opt=allow_opt,
            prec=EOprtPrecedence.prPOSTFIX,
            code=ECmdCode.cmOPRT_POSTFIX,
            argc=1,
        )
        self._add_callback(name, cb, self._post_oprt_def, self.valid_oprt_chars())

    def define_infix_oprt(
        self,
        name: str,
        func: Optional[Callable[..., Any]],
        prec: int = EOprtPrecedence.prINFIX,
        allow_opt: bool = True,
    ) -> None:
        if len(name) > MaxLenIdentifier:
            self._error(EErrorCodes.ecIDENTIFIER_TOO_LONG)

        cb = ParserCallback(
            func,
            allow_opt=allow_opt,
            prec=prec,
            code=ECmdCode.cmOPRT_INFIX,
            argc=1,
        )
        self._add_callback(name, cb, self._infix_oprt_def, self.valid_infix_oprt_chars())

    def define_const(self, name: str, val: Any) -> None:
        if len(name) > MaxLenIdentifier:
            self._error(EErrorCodes.ecIDENTIFIER_TOO_LONG)

        self._check_name(name, self.valid_name_chars())
        self._const_def[name] = val
        self._re_init()

    def define_str_const(self, name: str, val: str) -> None:
        if name in self._str_var_def:
            self._error(EErrorCodes.ecNAME_CONFLICT)

        self._check_name(name, self.valid_name_chars())
        self._string_var_buf.append(val)
        self._str_var_def[name] = len(self._string_var_buf) - 1
        self._re_init()

    def define_var(self, name: str, var: Any = 0.0) -> VariableRef:
        if var is None:
            self._error(EErrorCodes.ecINVALID_VAR_PTR)

        if len(name) > MaxLenIdentifier:
            self._error(EErrorCodes.ecIDENTIFIER_TOO_LONG)

        if name in self._const_def:
            self._error(EErrorCodes.ecNAME_CONFLICT)

        self._check_name(name, self.valid_name_chars())

        if isinstance(var, VariableRef):
            var_ref = var
            if not var_ref.name:
                var_ref.name = name
            self._var_def[name] = var_ref
            self._re_init()
            return var_ref

        if name in self._var_def:
            # Update existing VariableRef value
            self._var_def[name].set(var)
            self._re_init()
            return self._var_def[name]

        var_ref = VariableRef(var, name=name)
        self._var_def[name] = var_ref
        self._re_init()
        return var_ref

    def define_vars(
        self,
        variables: Union[Mapping[str, Any], Any],
        *,
        include_coords: bool = True,
    ) -> None:
        """
        Define multiple variables at once from a dict/mapping or an xarray.Dataset.
        """
        if HAS_XARRAY and isinstance(variables, xr.Dataset):
            for k in variables.data_vars:
                self.define_var(str(k), variables[k])
            if include_coords:
                for k in variables.coords:
                    if str(k) not in self._var_def and str(k) not in self._const_def:
                        try:
                            self.define_var(str(k), variables.coords[k])
                        except ParserError:
                            pass
        elif isinstance(variables, Mapping):
            for k, v in variables.items():
                self.define_var(str(k), v)
        else:
            raise TypeError(f"Expected Mapping or xarray.Dataset, got {type(variables).__name__}")

    def define_dataset(self, dataset: Any, include_coords: bool = True) -> None:
        """Bind all data_vars (and optionally coords) of an xarray.Dataset as parser variables."""
        self.define_vars(dataset, include_coords=include_coords)

    def set_var(self, name: str, val: Any) -> None:
        """Update a variable's value (defining it if not yet defined)."""
        if name in self._var_def:
            was_array = is_array_like(self._var_def[name].get())
            now_array = is_array_like(val)
            self._var_def[name].set(val)
            if was_array != now_array:
                self._re_init()
        else:
            self.define_var(name, val)

    def __setitem__(self, name: str, val: Any) -> None:
        self.set_var(name, val)

    def __getitem__(self, name: str) -> Any:
        if name in self._var_def:
            return self._var_def[name].get()
        if name in self._const_def:
            return self._const_def[name]
        raise KeyError(name)

    def set_var_factory(self, factory: Optional[Callable[..., Any]], user_data: Any = None) -> None:
        self._var_factory = factory
        self._var_factory_data = user_data
        self._re_init()

    def _call_var_factory(self, name: str) -> VariableRef:
        assert self._var_factory is not None
        try:
            res = self._var_factory(name, self._var_factory_data)
        except TypeError:
            res = self._var_factory(name)
        if isinstance(res, VariableRef):
            return res
        return VariableRef(0.0 if res is None else res, name=name)

    def add_val_ident(self, callback: Callable[..., Any]) -> None:
        self._token_reader.add_val_ident(callback)

    # Clear & Remove methods
    def remove_var(self, name: str) -> None:
        if name in self._var_def:
            del self._var_def[name]
            self._re_init()

    def clear_var(self) -> None:
        self._var_def.clear()
        self._re_init()

    def clear_fun(self) -> None:
        self._fun_def.clear()
        self._re_init()

    def clear_const(self) -> None:
        self._const_def.clear()
        self._str_var_def.clear()
        self._string_var_buf.clear()
        self._re_init()

    def clear_infix_oprt(self) -> None:
        self._infix_oprt_def.clear()
        self._re_init()

    def clear_postfix_oprt(self) -> None:
        self._post_oprt_def.clear()
        self._re_init()

    def clear_oprt(self) -> None:
        self._oprt_def.clear()
        self._re_init()

    # Query methods
    def set_expr(self, expr: str) -> None:
        if self._token_reader.get_arg_sep() == self._dec_sep:
            self._error(EErrorCodes.ecLOCALE)

        if len(expr) >= MaxLenExpression:
            self._error(EErrorCodes.ecEXPRESSION_TOO_LONG, 0, expr)

        self._token_reader.set_formula(expr + " ")
        self._re_init()

    def get_expr(self) -> str:
        return self._token_reader.get_expr()

    def get_var(self) -> dict[str, VariableRef]:
        return dict(self._var_def)

    def get_var_values(self) -> dict[str, Any]:
        return {k: v.get() for k, v in self._var_def.items()}

    def get_used_var(self) -> dict[str, Optional[VariableRef]]:
        try:
            self._token_reader.ignore_undef_var(True)
            self._create_rpn()
            self._compiled = False
            self._token_reader.ignore_undef_var(False)
        except ParserError:
            self._compiled = False
            self._token_reader.ignore_undef_var(False)
            raise
        return dict(self._token_reader.get_used_var())

    def get_const(self) -> dict[str, Any]:
        return dict(self._const_def)

    def get_fun_def(self) -> dict[str, ParserCallback]:
        return dict(self._fun_def)

    def get_version(self, info: EParserVersionInfo = EParserVersionInfo.pviFULL) -> str:
        if info == EParserVersionInfo.pviFULL:
            return f"{ParserVersion} ({ParserVersionDate}; 64BIT; RELEASE; UNICODE)"
        return ParserVersion

    def get_byte_code(self) -> ParserByteCode:
        return self._rpn

    def set_byte_code(self, byte_code: ParserByteCode) -> None:
        self._rpn = byte_code.copy()
        expr, self._string_buf = self._rpn.restore_environment()
        self._token_reader.set_formula(expr)
        self._compiled = True

    def enable_optimizer(self, is_on: bool = True) -> None:
        self._rpn.enable_optimizer(is_on)
        self._re_init()

    def enable_built_in_oprt(self, is_on: bool = True) -> None:
        self._built_in_op = is_on
        self._re_init()

    def has_built_in_oprt(self) -> bool:
        return self._built_in_op

    def get_num_results(self) -> int:
        return self._final_result_idx

    # Precedence & Associativity
    def _get_oprt_precedence(self, tok: ParserToken) -> int:
        code = tok.get_code()
        if code == ECmdCode.cmEND:
            return -5
        if code == ECmdCode.cmARG_SEP:
            return -4
        if code == ECmdCode.cmASSIGN:
            return -1
        if code in (ECmdCode.cmELSE, ECmdCode.cmIF):
            return 0
        if code == ECmdCode.cmLAND:
            return int(EOprtPrecedence.prLAND)
        if code == ECmdCode.cmLOR:
            return int(EOprtPrecedence.prLOR)
        if code in (
            ECmdCode.cmLT,
            ECmdCode.cmGT,
            ECmdCode.cmLE,
            ECmdCode.cmGE,
            ECmdCode.cmNEQ,
            ECmdCode.cmEQ,
        ):
            return int(EOprtPrecedence.prCMP)
        if code in (ECmdCode.cmADD, ECmdCode.cmSUB):
            return int(EOprtPrecedence.prADD_SUB)
        if code in (ECmdCode.cmMUL, ECmdCode.cmDIV):
            return int(EOprtPrecedence.prMUL_DIV)
        if code == ECmdCode.cmPOW:
            return int(EOprtPrecedence.prPOW)
        if code in (ECmdCode.cmOPRT_INFIX, ECmdCode.cmOPRT_BIN):
            return tok.get_pri()
        raise ParserError(EErrorCodes.ecINTERNAL_ERROR, 5, "")

    def _get_oprt_associativity(self, tok: ParserToken) -> EOprtAssociativity:
        code = tok.get_code()
        if code in (
            ECmdCode.cmASSIGN,
            ECmdCode.cmLAND,
            ECmdCode.cmLOR,
            ECmdCode.cmLT,
            ECmdCode.cmGT,
            ECmdCode.cmLE,
            ECmdCode.cmGE,
            ECmdCode.cmNEQ,
            ECmdCode.cmEQ,
            ECmdCode.cmADD,
            ECmdCode.cmSUB,
            ECmdCode.cmMUL,
            ECmdCode.cmDIV,
        ):
            return EOprtAssociativity.oaLEFT
        if code == ECmdCode.cmPOW:
            return EOprtAssociativity.oaRIGHT
        if code == ECmdCode.cmOPRT_BIN:
            return tok.get_associativity()
        return EOprtAssociativity.oaNONE

    # RPN compilation helpers
    def _apply_str_func(self, fun_tok: ParserToken, args: list[ParserToken]) -> ParserToken:
        if args[-1].get_code() != ECmdCode.cmSTRING:
            self._error(
                EErrorCodes.ecSTRING_EXPECTED,
                self._token_reader.get_pos(),
                fun_tok.get_as_string(),
            )

        val_tok = ParserToken()
        cb = fun_tok.get_func_addr()
        if cb is None:
            self._error(EErrorCodes.ecINTERNAL_ERROR)

        try:
            argc = fun_tok.get_arg_count()
            if argc < 0 or argc > 5:
                self._error(EErrorCodes.ecINTERNAL_ERROR)
            val_tok.set_val(1.0)
            args[-1].get_as_string()
            for i in range(argc):
                args[i].get_val()
        except ParserError:
            self._error(
                EErrorCodes.ecVAL_EXPECTED,
                self._token_reader.get_pos(),
                fun_tok.get_as_string(),
            )

        self._rpn.add_str_fun(cb, fun_tok.get_arg_count(), args[-1].get_idx())
        return val_tok

    def _apply_mult_str_func(self, fun_tok: ParserToken, args: list[ParserToken]) -> ParserToken:
        if args[-1].get_code() != ECmdCode.cmSTRING:
            self._error(
                EErrorCodes.ecSTRING_EXPECTED,
                self._token_reader.get_pos(),
                fun_tok.get_as_string(),
            )

        val_tok = ParserToken()
        cb = fun_tok.get_func_addr()
        if cb is None:
            self._error(EErrorCodes.ecINTERNAL_ERROR)

        try:
            for i in range(len(args) - 1):
                args[i].get_val()
            args[-1].get_as_string()
            val_tok.set_val(1.0)
        except ParserError:
            self._error(
                EErrorCodes.ecVAL_EXPECTED,
                self._token_reader.get_pos(),
                fun_tok.get_as_string(),
            )

        self._rpn.add_mult_str_fun(cb, len(args) - 1, args[-1].get_idx())
        return val_tok

    def _apply_func(
        self,
        st_opt: list[ParserToken],
        st_val: list[ParserToken],
        arg_count: int,
    ) -> None:
        if not st_opt or st_opt[-1].get_func_addr() is None:
            return

        fun_tok = st_opt.pop()
        cb = fun_tok.get_func_addr()
        assert cb is not None

        i_arg_count = fun_tok.get_arg_count() if fun_tok.get_code() == ECmdCode.cmOPRT_BIN else arg_count
        i_arg_required = fun_tok.get_arg_count() + (1 if fun_tok.get_type() == ETypeCode.tpSTR else 0)
        i_arg_numerical = i_arg_count - (1 if fun_tok.get_type() == ETypeCode.tpSTR else 0)

        if fun_tok.get_code() == ECmdCode.cmFUNC_STR and i_arg_count - i_arg_numerical > 1:
            self._error(EErrorCodes.ecINTERNAL_ERROR)

        if fun_tok.get_arg_count() >= 0 and i_arg_count > i_arg_required:
            self._error(
                EErrorCodes.ecTOO_MANY_PARAMS,
                self._token_reader.get_pos() - 1,
                fun_tok.get_as_string(),
            )

        if fun_tok.get_code() != ECmdCode.cmOPRT_BIN and i_arg_count < i_arg_required:
            self._error(
                EErrorCodes.ecTOO_FEW_PARAMS,
                self._token_reader.get_pos() - 1,
                fun_tok.get_as_string(),
            )

        if (
            fun_tok.get_code() == ECmdCode.cmFUNC_STR
            and fun_tok.get_arg_count() >= 0
            and i_arg_count > i_arg_required
        ):
            self._error(
                EErrorCodes.ecTOO_MANY_PARAMS,
                self._token_reader.get_pos() - 1,
                fun_tok.get_as_string(),
            )

        st_arg: list[ParserToken] = []
        for _ in range(i_arg_numerical):
            if not st_val:
                self._error(
                    EErrorCodes.ecINTERNAL_ERROR,
                    self._token_reader.get_pos(),
                    fun_tok.get_as_string(),
                )
            st_arg.append(st_val.pop())
            if st_arg[-1].get_type() == ETypeCode.tpSTR and fun_tok.get_type() != ETypeCode.tpSTR:
                self._error(
                    EErrorCodes.ecVAL_EXPECTED,
                    self._token_reader.get_pos(),
                    fun_tok.get_as_string(),
                )

        code = fun_tok.get_code()
        if code == ECmdCode.cmFUNC_STR:
            if not st_val:
                self._error(
                    EErrorCodes.ecINTERNAL_ERROR,
                    self._token_reader.get_pos(),
                    fun_tok.get_as_string(),
                )
            st_arg.append(st_val.pop())
            if st_arg[-1].get_type() == ETypeCode.tpSTR and fun_tok.get_type() != ETypeCode.tpSTR:
                self._error(
                    EErrorCodes.ecVAL_EXPECTED,
                    self._token_reader.get_pos(),
                    fun_tok.get_as_string(),
                )

            if fun_tok.get_arg_count() == -1:
                if i_arg_numerical == 0:
                    self._error(
                        EErrorCodes.ecTOO_FEW_PARAMS,
                        self._token_reader.get_pos(),
                        fun_tok.get_as_string(),
                    )
                self._apply_mult_str_func(fun_tok, st_arg)
            else:
                self._apply_str_func(fun_tok, st_arg)

        elif code == ECmdCode.cmFUNC_BULK:
            self._rpn.add_bulk_fun(cb, len(st_arg))

        elif code in (
            ECmdCode.cmOPRT_BIN,
            ECmdCode.cmOPRT_POSTFIX,
            ECmdCode.cmOPRT_INFIX,
            ECmdCode.cmFUNC,
        ):
            if fun_tok.get_arg_count() == -1 and i_arg_count == 0:
                self._error(
                    EErrorCodes.ecTOO_FEW_PARAMS,
                    self._token_reader.get_pos(),
                    fun_tok.get_as_string(),
                )
            self._rpn.add_fun(
                cb,
                -i_arg_numerical if fun_tok.get_arg_count() == -1 else i_arg_numerical,
                fun_tok.is_optimizable(),
            )

        dummy = ParserToken()
        dummy.set_val(1.0)
        st_val.append(dummy)

    def _apply_if_else(self, st_opt: list[ParserToken], st_val: list[ParserToken]) -> None:
        while st_opt and st_opt[-1].get_code() == ECmdCode.cmELSE:
            st_opt.pop()
            if len(st_val) < 3 or not st_opt:
                self._error(EErrorCodes.ecINTERNAL_ERROR, self._token_reader.get_pos())

            v_val2 = st_val.pop()
            if v_val2.get_type() != ETypeCode.tpDBL:
                self._error(EErrorCodes.ecUNEXPECTED_STR, self._token_reader.get_pos())

            v_val1 = st_val.pop()
            if v_val1.get_type() != ETypeCode.tpDBL:
                self._error(EErrorCodes.ecUNEXPECTED_STR, self._token_reader.get_pos())

            st_val.pop()  # vExpr

            res_tok = ParserToken()
            res_tok.set_val(1.0)
            st_val.append(res_tok)

            op_if = st_opt.pop()
            if op_if.get_code() != ECmdCode.cmIF:
                self._error(EErrorCodes.ecMISPLACED_COLON, self._token_reader.get_pos())

            self._rpn.add_if_else(ECmdCode.cmENDIF)

    def _apply_bin_oprt(self, st_opt: list[ParserToken], st_val: list[ParserToken]) -> None:
        if st_opt[-1].get_code() == ECmdCode.cmOPRT_BIN:
            self._apply_func(st_opt, st_val, 2)
        else:
            if len(st_val) < 2:
                self._error(
                    EErrorCodes.ecINTERNAL_ERROR,
                    self._token_reader.get_pos(),
                    "ApplyBinOprt: not enough values in value stack!",
                )

            val_tok1 = st_val.pop()
            val_tok2 = st_val.pop()
            opt_tok = st_opt.pop()

            if val_tok1.get_type() != val_tok2.get_type() or (
                val_tok1.get_type() == ETypeCode.tpSTR and val_tok2.get_type() == ETypeCode.tpSTR
            ):
                self._error(
                    EErrorCodes.ecOPRT_TYPE_CONFLICT,
                    self._token_reader.get_pos(),
                    opt_tok.get_as_string(),
                )

            if opt_tok.get_code() == ECmdCode.cmASSIGN:
                if val_tok2.get_code() != ECmdCode.cmVAR:
                    self._error(EErrorCodes.ecUNEXPECTED_OPERATOR, -1, "=")
                self._rpn.add_assign_op(val_tok2.get_var())
            else:
                self._rpn.add_op(opt_tok.get_code())

            res_tok = ParserToken()
            res_tok.set_val(1.0)
            st_val.append(res_tok)

    def _apply_remaining_oprt(self, st_opt: list[ParserToken], st_val: list[ParserToken]) -> None:
        while st_opt and st_opt[-1].get_code() not in (ECmdCode.cmBO, ECmdCode.cmIF):
            code = st_opt[-1].get_code()
            if code in (
                ECmdCode.cmOPRT_INFIX,
                ECmdCode.cmOPRT_BIN,
                ECmdCode.cmLE,
                ECmdCode.cmGE,
                ECmdCode.cmNEQ,
                ECmdCode.cmEQ,
                ECmdCode.cmLT,
                ECmdCode.cmGT,
                ECmdCode.cmADD,
                ECmdCode.cmSUB,
                ECmdCode.cmMUL,
                ECmdCode.cmDIV,
                ECmdCode.cmPOW,
                ECmdCode.cmLAND,
                ECmdCode.cmLOR,
                ECmdCode.cmASSIGN,
            ):
                if code == ECmdCode.cmOPRT_INFIX:
                    self._apply_func(st_opt, st_val, 1)
                else:
                    self._apply_bin_oprt(st_opt, st_val)
            elif code == ECmdCode.cmELSE:
                self._apply_if_else(st_opt, st_val)
            else:
                self._error(EErrorCodes.ecINTERNAL_ERROR)

    def _create_rpn(self) -> None:
        if not self._token_reader.get_expr():
            self._error(EErrorCodes.ecUNEXPECTED_EOF, 0)

        st_opt: list[ParserToken] = []
        st_val: list[ParserToken] = []
        st_arg_count: list[int] = []
        opta = ParserToken()
        if_else_counter = 0

        self._re_init()
        st_arg_count.append(1)

        while True:
            opt = self._token_reader.read_next_token()
            code = opt.get_code()

            if code == ECmdCode.cmSTRING:
                if not st_opt:
                    self._error(
                        EErrorCodes.ecSTR_RESULT,
                        self._token_reader.get_pos(),
                        opt.get_as_string(),
                    )
                st_val.append(opt)

            elif code == ECmdCode.cmVAR:
                st_val.append(opt)
                self._rpn.add_var(opt.get_var())

            elif code == ECmdCode.cmVAL:
                st_val.append(opt)
                self._rpn.add_val(opt.get_val())

            elif code == ECmdCode.cmELSE:
                if not st_arg_count:
                    self._error(EErrorCodes.ecMISPLACED_COLON, self._token_reader.get_pos())

                if st_arg_count[-1] > 1:
                    self._error(EErrorCodes.ecUNEXPECTED_ARG_SEP, self._token_reader.get_pos())

                st_arg_count.pop()
                if_else_counter -= 1
                if if_else_counter < 0:
                    self._error(EErrorCodes.ecMISPLACED_COLON, self._token_reader.get_pos())

                self._apply_remaining_oprt(st_opt, st_val)
                self._rpn.add_if_else(ECmdCode.cmELSE)
                st_opt.append(opt)

            elif code == ECmdCode.cmARG_SEP:
                if st_opt and st_opt[-1].get_code() == ECmdCode.cmIF:
                    self._error(EErrorCodes.ecUNEXPECTED_ARG_SEP, self._token_reader.get_pos())

                if not st_arg_count:
                    self._error(EErrorCodes.ecUNEXPECTED_ARG_SEP, self._token_reader.get_pos())

                st_arg_count[-1] += 1
                self._apply_remaining_oprt(st_opt, st_val)

            elif code == ECmdCode.cmEND:
                self._apply_remaining_oprt(st_opt, st_val)

            elif code == ECmdCode.cmBC:
                if opta.get_code() == ECmdCode.cmBO:
                    st_arg_count[-1] -= 1

                self._apply_remaining_oprt(st_opt, st_val)

                if st_opt and st_opt[-1].get_code() == ECmdCode.cmBO:
                    if not st_arg_count:
                        self._error(EErrorCodes.ecINTERNAL_ERROR)
                    arg_count = st_arg_count.pop()
                    st_opt.pop()  # pop cmBO

                    if arg_count > 1 and (
                        not st_opt
                        or st_opt[-1].get_code()
                        not in (
                            ECmdCode.cmFUNC,
                            ECmdCode.cmFUNC_BULK,
                            ECmdCode.cmFUNC_STR,
                        )
                    ):
                        self._error(EErrorCodes.ecUNEXPECTED_ARG, self._token_reader.get_pos())

                    if (
                        st_opt
                        and st_opt[-1].get_code() not in (ECmdCode.cmOPRT_INFIX, ECmdCode.cmOPRT_BIN)
                        and st_opt[-1].get_func_addr() is not None
                    ):
                        self._apply_func(st_opt, st_val, arg_count)

            elif code in (
                ECmdCode.cmIF,
                ECmdCode.cmLAND,
                ECmdCode.cmLOR,
                ECmdCode.cmLT,
                ECmdCode.cmGT,
                ECmdCode.cmLE,
                ECmdCode.cmGE,
                ECmdCode.cmNEQ,
                ECmdCode.cmEQ,
                ECmdCode.cmADD,
                ECmdCode.cmSUB,
                ECmdCode.cmMUL,
                ECmdCode.cmDIV,
                ECmdCode.cmPOW,
                ECmdCode.cmASSIGN,
                ECmdCode.cmOPRT_BIN,
            ):
                if code == ECmdCode.cmIF:
                    if_else_counter += 1
                    if if_else_counter > MaxNestingDepth:
                        self._error(EErrorCodes.ecNESTING_LIMIT, self._token_reader.get_pos())
                    st_arg_count.append(1)

                while st_opt and st_opt[-1].get_code() not in (
                    ECmdCode.cmBO,
                    ECmdCode.cmELSE,
                    ECmdCode.cmIF,
                ):
                    prec1 = self._get_oprt_precedence(st_opt[-1])
                    prec2 = self._get_oprt_precedence(opt)

                    if st_opt[-1].get_code() == opt.get_code():
                        asct = self._get_oprt_associativity(opt)
                        if (asct == EOprtAssociativity.oaRIGHT and prec1 <= prec2) or (
                            asct == EOprtAssociativity.oaLEFT and prec1 < prec2
                        ):
                            break
                    elif prec1 < prec2:
                        break

                    if st_opt[-1].get_code() == ECmdCode.cmOPRT_INFIX:
                        self._apply_func(st_opt, st_val, 1)
                    else:
                        self._apply_bin_oprt(st_opt, st_val)

                if code == ECmdCode.cmIF:
                    self._rpn.add_if_else(code)

                st_opt.append(opt)

            elif code == ECmdCode.cmBO:
                st_arg_count.append(1)
                st_opt.append(opt)

            elif code in (
                ECmdCode.cmOPRT_INFIX,
                ECmdCode.cmFUNC,
                ECmdCode.cmFUNC_BULK,
                ECmdCode.cmFUNC_STR,
            ):
                st_opt.append(opt)

            elif code == ECmdCode.cmOPRT_POSTFIX:
                st_opt.append(opt)
                self._apply_func(st_opt, st_val, 1)

            else:
                self._error(EErrorCodes.ecINTERNAL_ERROR, 3)

            opta = opt

            if code == ECmdCode.cmEND:
                self._rpn.finalize()
                break

        if if_else_counter > 0:
            self._error(EErrorCodes.ecMISSING_ELSE_CLAUSE)

        if len(st_arg_count) != 1:
            self._error(EErrorCodes.ecINTERNAL_ERROR)
        self._final_result_idx = st_arg_count[0]
        if self._final_result_idx == 0:
            self._error(EErrorCodes.ecINTERNAL_ERROR, 9)

        if not st_val:
            self._error(EErrorCodes.ecEMPTY_EXPRESSION)

        while st_val:
            if st_val.pop().get_type() != ETypeCode.tpDBL:
                self._error(EErrorCodes.ecSTR_RESULT)

        self._rpn.store_environment(self._token_reader.get_expr(), self._string_buf)
        self._compiled = True

    def _parse_cmd_code_bulk(self, offset: Optional[int] = None, thread_id: int = 0) -> Any:
        rpn = self._rpn.get_base()
        stack: list[Any] = []
        cond_stack: list[list[Any]] = []
        pc = 0
        n_rpn = len(rpn)

        while pc < n_rpn:
            tok = rpn[pc]
            cmd = tok.Cmd

            if cmd == ECmdCode.cmEND:
                break

            if cmd == ECmdCode.cmVAL:
                stack.append(tok.data2)
                pc += 1
                continue

            if cmd == ECmdCode.cmVAR:
                if tok.ptr is None:
                    raise ParserError(EErrorCodes.ecINVALID_VAR_PTR)
                val = tok.ptr.get()
                stack.append(val[offset] if offset is not None else val)
                pc += 1
                continue

            if cmd == ECmdCode.cmVARPOW2:
                if tok.ptr is None:
                    raise ParserError(EErrorCodes.ecINVALID_VAR_PTR)
                buf = tok.ptr.get()
                if offset is not None:
                    buf = buf[offset]
                stack.append(buf * buf)
                pc += 1
                continue

            if cmd == ECmdCode.cmVARPOW3:
                if tok.ptr is None:
                    raise ParserError(EErrorCodes.ecINVALID_VAR_PTR)
                buf = tok.ptr.get()
                if offset is not None:
                    buf = buf[offset]
                stack.append(buf * buf * buf)
                pc += 1
                continue

            if cmd == ECmdCode.cmVARPOW4:
                if tok.ptr is None:
                    raise ParserError(EErrorCodes.ecINVALID_VAR_PTR)
                buf = tok.ptr.get()
                if offset is not None:
                    buf = buf[offset]
                stack.append(buf * buf * buf * buf)
                pc += 1
                continue

            if cmd == ECmdCode.cmVARMUL:
                if tok.ptr is None:
                    raise ParserError(EErrorCodes.ecINVALID_VAR_PTR)
                buf = tok.ptr.get()
                if offset is not None:
                    buf = buf[offset]
                stack.append(buf * tok.data + tok.data2)
                pc += 1
                continue

            if cmd == ECmdCode.cmADD:
                rhs = stack.pop()
                stack[-1] = stack[-1] + rhs
                pc += 1
                continue

            if cmd == ECmdCode.cmSUB:
                rhs = stack.pop()
                stack[-1] = stack[-1] - rhs
                pc += 1
                continue

            if cmd == ECmdCode.cmMUL:
                rhs = stack.pop()
                stack[-1] = stack[-1] * rhs
                pc += 1
                continue

            if cmd == ECmdCode.cmDIV:
                rhs = stack.pop()
                stack[-1] = MathImpl.Div(stack[-1], rhs)
                pc += 1
                continue

            if cmd == ECmdCode.cmPOW:
                rhs = stack.pop()
                stack[-1] = MathImpl.Pow(stack[-1], rhs)
                pc += 1
                continue

            if cmd == ECmdCode.cmLE:
                rhs = stack.pop()
                stack[-1] = MathImpl.LE(stack[-1], rhs)
                pc += 1
                continue

            if cmd == ECmdCode.cmGE:
                rhs = stack.pop()
                stack[-1] = MathImpl.GE(stack[-1], rhs)
                pc += 1
                continue

            if cmd == ECmdCode.cmNEQ:
                rhs = stack.pop()
                stack[-1] = MathImpl.NEQ(stack[-1], rhs)
                pc += 1
                continue

            if cmd == ECmdCode.cmEQ:
                rhs = stack.pop()
                stack[-1] = MathImpl.EQ(stack[-1], rhs)
                pc += 1
                continue

            if cmd == ECmdCode.cmLT:
                rhs = stack.pop()
                stack[-1] = MathImpl.LT(stack[-1], rhs)
                pc += 1
                continue

            if cmd == ECmdCode.cmGT:
                rhs = stack.pop()
                stack[-1] = MathImpl.GT(stack[-1], rhs)
                pc += 1
                continue

            if cmd == ECmdCode.cmLAND:
                rhs = stack.pop()
                stack[-1] = MathImpl.LAND(stack[-1], rhs)
                pc += 1
                continue

            if cmd == ECmdCode.cmLOR:
                rhs = stack.pop()
                stack[-1] = MathImpl.LOR(stack[-1], rhs)
                pc += 1
                continue

            if cmd == ECmdCode.cmASSIGN:
                rhs = stack.pop()
                stack.pop()  # lhs variable value
                if tok.ptr is None:
                    raise ParserError(EErrorCodes.ecINVALID_VAR_PTR)
                if offset is not None:
                    tok.ptr.get()[offset] = rhs
                else:
                    tok.ptr.set(rhs)
                stack.append(rhs)
                pc += 1
                continue

            if cmd == ECmdCode.cmIF:
                cond = stack.pop()
                if is_array_like(cond):
                    cond_stack.append(["array", cond, None])
                    pc += 1
                else:
                    if cond == 0:
                        cond_stack.append(["scalar_false"])
                        pc += tok.offset + 1
                    else:
                        cond_stack.append(["scalar_true"])
                        pc += 1
                continue

            if cmd == ECmdCode.cmELSE:
                if cond_stack and cond_stack[-1][0] == "array":
                    cond_stack[-1][2] = stack.pop()
                    pc += 1
                else:
                    pc += tok.offset
                continue

            if cmd == ECmdCode.cmENDIF:
                if cond_stack:
                    entry = cond_stack.pop()
                    if entry[0] == "array":
                        else_val = stack.pop()
                        then_val = entry[2]
                        cond = entry[1]
                        stack.append(MathImpl.Where(cond, then_val, else_val))
                pc += 1
                continue

            if cmd == ECmdCode.cmFUNC:
                cb = tok.cb
                assert cb is not None
                argc = tok.argc
                if argc == 0:
                    stack.append(cb.call_fun())
                elif argc > 0:
                    args = stack[-argc:]
                    del stack[-argc:]
                    stack.append(cb.call_fun(*args))
                else:
                    n_args = -argc
                    args = stack[-n_args:]
                    del stack[-n_args:]
                    stack.append(cb.call_multfun(args))
                pc += 1
                continue

            if cmd == ECmdCode.cmFUNC_STR:
                cb = tok.cb
                assert cb is not None
                idx_stack = tok.idx
                if idx_stack < 0 or idx_stack >= len(self._string_buf):
                    self._error(EErrorCodes.ecINTERNAL_ERROR, self._token_reader.get_pos())
                str_arg = self._string_buf[idx_stack]
                if tok.argc < 0:
                    n_args = -tok.argc
                    args = stack[-n_args:]
                    del stack[-n_args:]
                    stack.append(cb.call_multstrfun(str_arg, args))
                elif tok.argc == 0:
                    stack.append(cb.call_strfun(str_arg))
                else:
                    n_args = tok.argc
                    args = stack[-n_args:]
                    del stack[-n_args:]
                    stack.append(cb.call_strfun(str_arg, *args))
                pc += 1
                continue

            if cmd == ECmdCode.cmFUNC_BULK:
                cb = tok.cb
                assert cb is not None
                argc = tok.argc
                bulk_offset = offset if offset is not None else 0
                if argc == 0:
                    stack.append(cb.call_bulkfun(bulk_offset, thread_id))
                else:
                    args = stack[-argc:]
                    del stack[-argc:]
                    stack.append(cb.call_bulkfun(bulk_offset, thread_id, *args))
                pc += 1
                continue

            raise ParserError(EErrorCodes.ecINTERNAL_ERROR, 3, "")

        self._stack_buffer = stack
        return stack[self._final_result_idx - 1]

    def _parse_string(self) -> Any:
        try:
            self._create_rpn()
            return self._parse_cmd_code_bulk(None, 0)
        except ParserError as exc:
            exc.set_formula(self._token_reader.get_expr())
            raise

    def eval(
        self,
        data: Optional[Union[Mapping[str, Any], list[Any], np.ndarray, Any]] = None,
        bulk_size: Optional[int] = None,
        **variables: Any,
    ) -> Any:
        """
        Evaluate the current expression.

        Supports:
        - `parser.eval()`: evaluate using currently defined variables.
        - `parser.eval(ds)`: evaluate using variables from an `xarray.Dataset` or `dict`.
        - `parser.eval(a=arr1, b=arr2)`: evaluate with keyword variable bindings.
        - `parser.eval(results_array, bulk_size)`: C++-compatible bulk mode evaluation.
        """
        if bulk_size is not None and data is not None:
            # C++ bulk mode: Eval(results, nBulkSize)
            self._create_rpn()
            for i in range(bulk_size):
                data[i] = self._parse_cmd_code_bulk(i, 0)
            return data

        if data is not None:
            if HAS_XARRAY and isinstance(data, xr.Dataset):
                for k in data.data_vars:
                    self.set_var(str(k), data[k])
                for k in data.coords:
                    if str(k) not in self._var_def and str(k) not in self._const_def:
                        try:
                            self.set_var(str(k), data.coords[k])
                        except ParserError:
                            pass
            elif isinstance(data, Mapping):
                for k, v in data.items():
                    self.set_var(str(k), v)
            else:
                raise TypeError(
                    f"data must be a Mapping or xarray.Dataset (or pass bulk_size for bulk mode), got {type(data).__name__}"
                )

        if variables:
            for k, v in variables.items():
                self.set_var(k, v)

        if self._compiled and self._rpn.get_size() > 0:
            return self._parse_cmd_code_bulk(None, 0)
        return self._parse_string()

    def eval_multi(
        self,
        data: Optional[Union[Mapping[str, Any], Any]] = None,
        **variables: Any,
    ) -> list[Any]:
        """
        Evaluate an expression containing comma-separated subexpressions
        (e.g. "x+y, sin(x), cos(y)") and return all results as a list.
        """
        self.eval(data, **variables)
        return list(self._stack_buffer[: self._final_result_idx])

    # C++ CamelCase API aliases
    Assign = assign
    SetDecSep = set_dec_sep
    SetThousandsSep = set_thousands_sep
    ResetLocale = reset_locale
    SetArgSep = set_arg_sep
    GetArgSep = get_arg_sep
    DefineNameChars = define_name_chars
    DefineOprtChars = define_oprt_chars
    DefineInfixOprtChars = define_infix_oprt_chars
    ValidNameChars = valid_name_chars
    ValidOprtChars = valid_oprt_chars
    ValidInfixOprtChars = valid_infix_oprt_chars
    DefineFun = define_fun
    DefineStrFun = define_str_fun
    DefineMultStrFun = define_mult_str_fun
    DefineBulkFun = define_bulk_fun
    DefineFunUserData = define_fun_user_data
    HasFun = has_fun
    DefineOprt = define_oprt
    DefinePostfixOprt = define_postfix_oprt
    DefineInfixOprt = define_infix_oprt
    DefineConst = define_const
    DefineStrConst = define_str_const
    DefineVar = define_var
    DefineVars = define_vars
    DefineDataset = define_dataset
    SetVar = set_var
    SetVarFactory = set_var_factory
    AddValIdent = add_val_ident
    RemoveVar = remove_var
    ClearVar = clear_var
    ClearFun = clear_fun
    ClearConst = clear_const
    ClearInfixOprt = clear_infix_oprt
    ClearPostfixOprt = clear_postfix_oprt
    ClearOprt = clear_oprt
    SetExpr = set_expr
    GetExpr = get_expr
    GetVar = get_var
    GetUsedVar = get_used_var
    GetConst = get_const
    GetFunDef = get_fun_def
    GetVersion = get_version
    GetByteCode = get_byte_code
    SetByteCode = set_byte_code
    EnableOptimizer = enable_optimizer
    EnableBuiltInOprt = enable_built_in_oprt
    HasBuiltInOprt = has_built_in_oprt
    GetNumResults = get_num_results
    Eval = eval
    EvalMulti = eval_multi


class Parser(ParserBase):
    """
    Standard floating-point and array (NumPy / xarray) mathematical expressions parser.
    Mirrors mu::Parser in muParser.h / muParser.cpp.
    """

    def __init__(self, expr: Optional[Union[str, ParserBase]] = None) -> None:
        super().__init__()
        self.add_val_ident(self._is_val)
        self.init_char_sets()
        self.init_fun()
        self.init_const()
        self.init_oprt()

        if isinstance(expr, ParserBase):
            self.assign(expr)
        elif isinstance(expr, str):
            self.set_expr(expr)

    def _is_val(self, formula: str, pos: int) -> tuple[bool, int, Any]:
        """Default floating-point value recognizer supporting custom decimal and thousands separators."""
        substr = formula[pos:]
        if not substr:
            return False, pos, 0.0

        dsep = re.escape(self._dec_sep) if self._dec_sep else r"\."
        if self._thousands_sep:
            tsep = re.escape(self._thousands_sep)
            int_part = rf"[0-9]+(?:{tsep}[0-9]{{3}})*"
        else:
            int_part = r"[0-9]+"

        pattern = rf"^(?:{int_part}(?:{dsep}[0-9]*)?|{dsep}[0-9]+)(?:[eE][+-]?[0-9]+)?"
        m = re.match(pattern, substr)
        if not m:
            return False, pos, 0.0

        matched_str = m.group(0)
        if not matched_str:
            return False, pos, 0.0

        clean_str = matched_str
        if self._thousands_sep:
            clean_str = clean_str.replace(self._thousands_sep, "")
        if self._dec_sep and self._dec_sep != ".":
            clean_str = clean_str.replace(self._dec_sep, ".")

        try:
            val = float(clean_str)
        except ValueError:  # pragma: no cover
            return False, pos, 0.0

        return True, pos + len(matched_str), val

    def init_char_sets(self) -> None:
        self.define_name_chars("0123456789_abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
        self.define_oprt_chars(
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ+-*^/?<>=#!$%&|~'_{}"
        )
        self.define_infix_oprt_chars("/+-*^?<>=#!$%&|~'_")

    def init_fun(self) -> None:
        # Trigonometric functions
        self.define_fun("sin", MathImpl.Sin, argc=1)
        self.define_fun("cos", MathImpl.Cos, argc=1)
        self.define_fun("tan", MathImpl.Tan, argc=1)
        self.define_fun("cot", MathImpl.Cot, argc=1)
        # Inverse trigonometric (arcus) functions
        self.define_fun("asin", MathImpl.ASin, argc=1)
        self.define_fun("acos", MathImpl.ACos, argc=1)
        self.define_fun("atan", MathImpl.ATan, argc=1)
        self.define_fun("atan2", MathImpl.ATan2, argc=2)
        # Hyperbolic functions
        self.define_fun("sinh", MathImpl.Sinh, argc=1)
        self.define_fun("cosh", MathImpl.Cosh, argc=1)
        self.define_fun("tanh", MathImpl.Tanh, argc=1)
        self.define_fun("coth", MathImpl.Coth, argc=1)
        # Inverse hyperbolic functions
        self.define_fun("asinh", MathImpl.ASinh, argc=1)
        self.define_fun("acosh", MathImpl.ACosh, argc=1)
        self.define_fun("atanh", MathImpl.ATanh, argc=1)
        # Logarithm functions
        self.define_fun("log2", MathImpl.Log2, argc=1)
        self.define_fun("log10", MathImpl.Log10, argc=1)
        self.define_fun("log", MathImpl.Log, argc=1)
        self.define_fun("ln", MathImpl.Log, argc=1)
        # Misc functions
        self.define_fun("exp", MathImpl.Exp, argc=1)
        self.define_fun("sqrt", MathImpl.Sqrt, argc=1)
        self.define_fun("sign", MathImpl.Sign, argc=1)
        self.define_fun("rint", MathImpl.Rint, argc=1)
        self.define_fun("abs", MathImpl.Abs, argc=1)
        # Variadic functions
        self.define_fun("sum", MathImpl.Sum, argc=-1)
        self.define_fun("avg", MathImpl.Avg, argc=-1)
        self.define_fun("min", MathImpl.Min, argc=-1)
        self.define_fun("max", MathImpl.Max, argc=-1)
        # Random number
        self.define_fun("rnd", MathImpl.Rnd, allow_opt=False, argc=0)

    def init_const(self) -> None:
        self.define_const("_pi", MathImpl.CONST_PI)
        self.define_const("_e", MathImpl.CONST_E)

    def init_oprt(self) -> None:
        self.define_infix_oprt("-", MathImpl.UnaryMinus)
        self.define_infix_oprt("+", MathImpl.UnaryPlus)

    def diff(
        self,
        var: Union[str, VariableRef],
        pos: Any = None,
        epsilon: float = 0.0,
    ) -> Any:
        """
        Numerically differentiate with respect to a variable using a 5-point 4th-order formula.
        Works with scalars, NumPy ndarrays, and xarray DataArrays.
        Mirrors mu::Parser::Diff.
        """
        if isinstance(var, str):
            if var not in self._var_def:
                if pos is None:
                    raise ValueError(f"Variable {var!r} is not defined and pos was not provided")
                self.define_var(var, pos)
            var_ref = self._var_def[var]
        elif isinstance(var, VariableRef):
            var_ref = var
        else:
            raise TypeError("var must be a variable name (str) or VariableRef")

        f_buf = var_ref.get()
        if pos is None:
            pos = f_buf
        if epsilon == 0:
            if is_array_like(pos):
                f_eps = np.where(pos == 0, 1e-10, 1e-7 * pos)
            else:
                f_eps = 1e-10 if pos == 0 else 1e-7 * pos
        else:
            f_eps = epsilon

        try:
            var_ref.set(pos + 2 * f_eps)
            f0 = self.eval()
            var_ref.set(pos + 1 * f_eps)
            f1 = self.eval()
            var_ref.set(pos - 1 * f_eps)
            f2 = self.eval()
            var_ref.set(pos - 2 * f_eps)
            f3 = self.eval()
        finally:
            var_ref.set(f_buf)

        return (-f0 + 8 * f1 - 8 * f2 + f3) / (12 * f_eps)

    # C++ aliases
    InitCharSets = init_char_sets
    InitFun = init_fun
    InitConst = init_const
    InitOprt = init_oprt
    Diff = diff


class ParserInt(ParserBase):
    """
    Integer expression parser supporting hex (0xff), binary (#101), bitwise,
    and integer arithmetic operations.
    Mirrors mu::ParserInt in muParserInt.h / muParserInt.cpp.
    """

    def __init__(self, expr: Optional[Union[str, ParserBase]] = None) -> None:
        super().__init__()
        self.add_val_ident(self._is_val)     # lowest priority
        self.add_val_ident(self._is_bin_val)
        self.add_val_ident(self._is_hex_val) # highest priority

        self.init_char_sets()
        self.init_fun()
        self.init_const()
        self.init_oprt()

        if isinstance(expr, ParserBase):
            self.assign(expr)
        elif isinstance(expr, str):
            self.set_expr(expr)

    @staticmethod
    def _is_val(formula: str, pos: int) -> tuple[bool, int, Any]:
        substr = formula[pos:]
        m = re.match(r"^[0-9]+", substr)
        if not m:
            return False, pos, 0.0
        digits = m.group(0)
        val = int(digits, 10)
        if val > INT32_MAX or val < INT32_MIN:
            return False, pos, 0.0
        return True, pos + len(digits), float(val)

    @staticmethod
    def _is_hex_val(formula: str, pos: int) -> tuple[bool, int, Any]:
        substr = formula[pos:]
        if not substr.startswith("0x"):
            return False, pos, 0.0
        m = re.match(r"^0x([0-9a-fA-F]+)", substr)
        if not m:
            return False, pos, 0.0
        hex_digits = m.group(1)
        val = int(hex_digits, 16)
        return True, pos + 2 + len(hex_digits), float(val)

    @staticmethod
    def _is_bin_val(formula: str, pos: int) -> tuple[bool, int, Any]:
        substr = formula[pos:]
        if not substr.startswith("#"):
            return False, pos, 0.0
        m = re.match(r"^#([01]+)", substr)
        if not m:
            return False, pos, 0.0
        bin_digits = m.group(1)
        if len(bin_digits) >= 32:
            raise ParserError("Binary to integer conversion error (overflow).")
        val = int(bin_digits, 2)
        return True, pos + 1 + len(bin_digits), float(val)

    def init_char_sets(self) -> None:
        self.define_name_chars("0123456789_abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
        self.define_oprt_chars("+-*^/?<>=!%&|~'_")
        self.define_infix_oprt_chars("/+-*^?<>=!%&|~'_")

    def init_const(self) -> None:
        pass

    def init_fun(self) -> None:
        self.define_fun("sign", ParserIntMathImpl.Sign, argc=1)
        self.define_fun("abs", ParserIntMathImpl.Abs, argc=1)
        self.define_fun("if", ParserIntMathImpl.Ite, argc=3)
        self.define_fun("sum", ParserIntMathImpl.Sum, argc=-1)
        self.define_fun("min", ParserIntMathImpl.Min, argc=-1)
        self.define_fun("max", ParserIntMathImpl.Max, argc=-1)

    def init_oprt(self) -> None:
        self.enable_built_in_oprt(False)

        self.define_infix_oprt("-", ParserIntMathImpl.UnaryMinus)
        self.define_infix_oprt("!", ParserIntMathImpl.Not)

        self.define_oprt("&", ParserIntMathImpl.BitAnd, EOprtPrecedence.prBAND)
        self.define_oprt("|", ParserIntMathImpl.BitOr, EOprtPrecedence.prBOR)

        self.define_oprt("&&", ParserIntMathImpl.And, EOprtPrecedence.prLAND)
        self.define_oprt("||", ParserIntMathImpl.Or, EOprtPrecedence.prLOR)

        self.define_oprt("<", ParserIntMathImpl.Less, EOprtPrecedence.prCMP)
        self.define_oprt(">", ParserIntMathImpl.Greater, EOprtPrecedence.prCMP)
        self.define_oprt("<=", ParserIntMathImpl.LessEq, EOprtPrecedence.prCMP)
        self.define_oprt(">=", ParserIntMathImpl.GreaterEq, EOprtPrecedence.prCMP)
        self.define_oprt("==", ParserIntMathImpl.Equal, EOprtPrecedence.prCMP)
        self.define_oprt("!=", ParserIntMathImpl.NotEqual, EOprtPrecedence.prCMP)

        self.define_oprt("+", ParserIntMathImpl.Add, EOprtPrecedence.prADD_SUB)
        self.define_oprt("-", ParserIntMathImpl.Sub, EOprtPrecedence.prADD_SUB)

        self.define_oprt("*", ParserIntMathImpl.Mul, EOprtPrecedence.prMUL_DIV)
        self.define_oprt("/", ParserIntMathImpl.Div, EOprtPrecedence.prMUL_DIV)
        self.define_oprt("%", ParserIntMathImpl.Mod, EOprtPrecedence.prMUL_DIV)

        self.define_oprt("^", ParserIntMathImpl.Pow, EOprtPrecedence.prPOW, EOprtAssociativity.oaRIGHT)
        self.define_oprt(">>", ParserIntMathImpl.Shr, int(EOprtPrecedence.prMUL_DIV) + 1)
        self.define_oprt("<<", ParserIntMathImpl.Shl, int(EOprtPrecedence.prMUL_DIV) + 1)

    InitCharSets = init_char_sets
    InitFun = init_fun
    InitConst = init_const
    InitOprt = init_oprt


def eval_expr(
    expr: str,
    data: Optional[Union[Mapping[str, Any], Any]] = None,
    /,
    *,
    constants: Optional[Mapping[str, Any]] = None,
    functions: Optional[Mapping[str, Callable[..., Any]]] = None,
    **variables: Any,
) -> Any:
    """
    Convenience function to parse and evaluate a muParser math expression
    using scalars, NumPy arrays, xarray DataArrays, or an xarray Dataset.

    Parameters
    ----------
    expr : str
        Mathematical expression using muParser syntax (e.g., ``"(nir - red) / (nir + red)"``
        or ``"a > 0 ? sqrt(a) : 0"``).
    data : Mapping[str, Any] or xarray.Dataset, optional
        Dictionary of variables or an `xarray.Dataset` whose data_vars/coords
        will be bound to variables in the expression.
    constants : Mapping[str, Any], optional
        Additional named constants to register before evaluation (`pi` and `e`
        are also pre-registered alongside `_pi` and `_e`).
    functions : Mapping[str, Callable], optional
        Additional custom functions to register before evaluation.
    **variables : Any
        Additional variables passed as keyword arguments.

    Returns
    -------
    Any
        Computed scalar, `numpy.ndarray`, `xarray.DataArray`, or `xarray.Dataset`.
    """
    parser = Parser()
    parser.define_const("pi", MathImpl.CONST_PI)
    parser.define_const("e", MathImpl.CONST_E)

    if constants:
        for k, v in constants.items():
            parser.define_const(k, v)

    if functions:
        for k, fn in functions.items():
            parser.define_fun(k, fn)

    if data is not None:
        parser.define_vars(data)

    if variables:
        parser.define_vars(variables)

    parser.set_expr(expr)
    return parser.eval()


# Alias
evaluate = eval_expr
