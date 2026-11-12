"""
Bytecode representation and optimizer for pymu-eval.
Mirrors muParserBytecode.h and muParserBytecode.cpp.
"""

from __future__ import annotations

# pylint: disable=invalid-name

from dataclasses import dataclass
from typing import Any, Callable, Optional

from .definitions import ECmdCode, EErrorCodes
from .errors import ParserError
from .math_impl import MathImpl, is_array_like
from .token_reader import ParserCallback, VariableRef


@dataclass
class SToken:  # pylint: disable=too-many-instance-attributes
    """
    Single bytecode instruction in the RPN stream.
    Mirrors mu::SToken.
    """

    Cmd: ECmdCode = ECmdCode.cmUNKNOWN
    # Value / Variable fields
    ptr: Optional[VariableRef] = None
    data: Any = 0.0
    data2: Any = 0.0
    # Operator / Jump fields
    offset: int = 0
    # Function fields
    argc: int = 0
    idx: int = 0
    cb: Optional[ParserCallback] = None

    def copy(self) -> SToken:
        """Return a shallow copy of this bytecode instruction."""
        return SToken(
            Cmd=self.Cmd,
            ptr=self.ptr,
            data=self.data,
            data2=self.data2,
            offset=self.offset,
            argc=self.argc,
            idx=self.idx,
            cb=self.cb,
        )


_FOLD_OP_MAP: dict[ECmdCode, Callable[[Any, Any], Any]] = {
    ECmdCode.cmLAND: MathImpl.LAND,
    ECmdCode.cmLOR: MathImpl.LOR,
    ECmdCode.cmLT: MathImpl.LT,
    ECmdCode.cmGT: MathImpl.GT,
    ECmdCode.cmLE: MathImpl.LE,
    ECmdCode.cmGE: MathImpl.GE,
    ECmdCode.cmNEQ: MathImpl.NEQ,
    ECmdCode.cmEQ: MathImpl.EQ,
    ECmdCode.cmADD: lambda a, b: a + b,
    ECmdCode.cmSUB: lambda a, b: a - b,
    ECmdCode.cmMUL: lambda a, b: a * b,
    ECmdCode.cmPOW: MathImpl.Pow,
}

_POW_CMD_MAP: dict[int, ECmdCode] = {
    1: ECmdCode.cmVAR,
    2: ECmdCode.cmVARPOW2,
    3: ECmdCode.cmVARPOW3,
    4: ECmdCode.cmVARPOW4,
}


class ParserByteCode:
    """
    Bytecode storage and constant/polynomial optimizer.
    Mirrors mu::ParserByteCode.
    """

    def __init__(self) -> None:
        self._stack_pos: int = 0
        self._max_stack_size: int = 0
        self._rpn: list[SToken] = []
        self._enable_optimizer: bool = True
        self._string_buffer: list[str] = []
        self._expr: str = ""

    def copy(self) -> ParserByteCode:
        """Create a deep copy of this bytecode sequence."""
        bc = ParserByteCode()
        # pylint: disable=protected-access
        bc._stack_pos = self._stack_pos
        bc._max_stack_size = self._max_stack_size
        bc._rpn = [t.copy() for t in self._rpn]
        bc._enable_optimizer = self._enable_optimizer
        bc._string_buffer = list(self._string_buffer)
        bc._expr = self._expr
        return bc

    def is_optimizer_enabled(self) -> bool:
        """Return True if constant folding / algebraic optimization is enabled."""
        return self._enable_optimizer

    def enable_optimizer(self, stat: bool = True) -> None:
        """Enable or disable bytecode optimization."""
        self._enable_optimizer = stat

    def add_var(self, var_ref: VariableRef) -> None:
        """Emit a variable load instruction."""
        self._stack_pos += 1
        self._max_stack_size = max(self._max_stack_size, self._stack_pos)
        self._rpn.append(
            SToken(
                Cmd=ECmdCode.cmVAR,
                ptr=var_ref,
                data=1.0,
                data2=0.0,
            )
        )

    def add_val(self, val: Any) -> None:
        """Emit a constant value load instruction."""
        self._stack_pos += 1
        self._max_stack_size = max(self._max_stack_size, self._stack_pos)
        self._rpn.append(
            SToken(
                Cmd=ECmdCode.cmVAL,
                ptr=None,
                data=0.0,
                data2=val,
            )
        )

    def _constant_folding(self, oprt: ECmdCode) -> bool:
        sz = len(self._rpn)
        x = self._rpn[sz - 2].data2
        y = self._rpn[sz - 1].data2

        if oprt in _FOLD_OP_MAP:
            self._rpn[sz - 2].data2 = _FOLD_OP_MAP[oprt](x, y)
            self._rpn.pop()
            return True

        if oprt == ECmdCode.cmDIV:
            # Leave division by zero to runtime evaluation so it produces inf or nan
            if not is_array_like(y) and y == 0:
                return False
            self._rpn[sz - 2].data2 = MathImpl.Div(x, y)
            self._rpn.pop()
            return True

        return False

    def _opt_pow(self, tok_prev: SToken, tok_last: SToken) -> bool:
        if tok_prev.Cmd != ECmdCode.cmVAR or tok_last.Cmd != ECmdCode.cmVAL:
            return False
        exp_val = tok_last.data2
        if is_array_like(exp_val):
            return False
        if exp_val == 0 and (tok_prev.ptr is None or not is_array_like(tok_prev.ptr.get())):
            tok_prev.Cmd = ECmdCode.cmVAL
            tok_prev.ptr = None
            tok_prev.data = 0.0
            tok_prev.data2 = 1.0
            self._rpn.pop()
            return True
        if exp_val in (1, 2, 3, 4):
            tok_prev.Cmd = _POW_CMD_MAP[int(exp_val)]
            self._rpn.pop()
            return True
        return False

    def _opt_add_sub(self, oprt: ECmdCode, tok_prev: SToken, tok_last: SToken) -> bool:
        var_cmds = (ECmdCode.cmVAR, ECmdCode.cmVARMUL)
        cond = (
            (tok_last.Cmd in var_cmds and tok_prev.Cmd == ECmdCode.cmVAL)
            or (tok_last.Cmd == ECmdCode.cmVAL and tok_prev.Cmd in var_cmds)
            or (
                tok_last.Cmd in var_cmds
                and tok_prev.Cmd in var_cmds
                and tok_prev.ptr is tok_last.ptr
            )
        )
        if not cond:
            return False
        sign = -1.0 if oprt == ECmdCode.cmSUB else 1.0
        tok_prev.Cmd = ECmdCode.cmVARMUL
        tok_prev.ptr = tok_prev.ptr if tok_prev.ptr is not None else tok_last.ptr
        tok_prev.data2 = tok_prev.data2 + sign * tok_last.data2
        tok_prev.data = tok_prev.data + sign * tok_last.data
        self._rpn.pop()
        return True

    def _opt_mul(self, tok_prev: SToken, tok_last: SToken) -> bool:
        if (tok_last.Cmd == ECmdCode.cmVAR and tok_prev.Cmd == ECmdCode.cmVAL) or (
            tok_last.Cmd == ECmdCode.cmVAL and tok_prev.Cmd == ECmdCode.cmVAR
        ):
            tok_prev.Cmd = ECmdCode.cmVARMUL
            tok_prev.ptr = tok_prev.ptr if tok_prev.ptr is not None else tok_last.ptr
            tok_prev.data = tok_prev.data2 + tok_last.data2
            tok_prev.data2 = 0.0
            self._rpn.pop()
            return True

        if (tok_last.Cmd == ECmdCode.cmVAL and tok_prev.Cmd == ECmdCode.cmVARMUL) or (
            tok_last.Cmd == ECmdCode.cmVARMUL and tok_prev.Cmd == ECmdCode.cmVAL
        ):
            tok_prev.Cmd = ECmdCode.cmVARMUL
            tok_prev.ptr = tok_prev.ptr if tok_prev.ptr is not None else tok_last.ptr
            if tok_last.Cmd == ECmdCode.cmVAL:
                tok_prev.data = tok_prev.data * tok_last.data2
                tok_prev.data2 = tok_prev.data2 * tok_last.data2
            else:
                tok_prev.data = tok_last.data * tok_prev.data2
                tok_prev.data2 = tok_last.data2 * tok_prev.data2
            self._rpn.pop()
            return True

        if (
            tok_last.Cmd == ECmdCode.cmVAR
            and tok_prev.Cmd == ECmdCode.cmVAR
            and tok_last.ptr is tok_prev.ptr
        ):
            tok_prev.Cmd = ECmdCode.cmVARPOW2
            self._rpn.pop()
            return True

        return False

    def _opt_div(self, tok_prev: SToken, tok_last: SToken) -> bool:
        if (
            tok_last.Cmd == ECmdCode.cmVAL
            and tok_prev.Cmd == ECmdCode.cmVARMUL
            and not is_array_like(tok_last.data2)
            and tok_last.data2 != 0
        ):
            tok_prev.data = tok_prev.data / tok_last.data2
            tok_prev.data2 = tok_prev.data2 / tok_last.data2
            self._rpn.pop()
            return True
        return False

    def add_op(self, oprt: ECmdCode) -> None:
        """Emit a binary operator instruction, applying constant/linear folding if enabled."""
        optimized = False
        sz = len(self._rpn)

        if self._enable_optimizer and sz >= 2:
            tok_prev = self._rpn[sz - 2]
            tok_last = self._rpn[sz - 1]
            if tok_prev.Cmd == ECmdCode.cmVAL and tok_last.Cmd == ECmdCode.cmVAL:
                optimized = self._constant_folding(oprt)
            elif oprt == ECmdCode.cmPOW:
                optimized = self._opt_pow(tok_prev, tok_last)
            elif oprt in (ECmdCode.cmSUB, ECmdCode.cmADD):
                optimized = self._opt_add_sub(oprt, tok_prev, tok_last)
            elif oprt == ECmdCode.cmMUL:
                optimized = self._opt_mul(tok_prev, tok_last)
            elif oprt == ECmdCode.cmDIV:
                optimized = self._opt_div(tok_prev, tok_last)

        if not optimized:
            self._stack_pos -= 1
            self._rpn.append(SToken(Cmd=oprt))

    def add_if_else(self, oprt: ECmdCode) -> None:
        """Emit a conditional jump marker (cmIF, cmELSE, or cmENDIF)."""
        self._rpn.append(SToken(Cmd=oprt))

    def add_assign_op(self, var_ref: VariableRef) -> None:
        """Emit a variable assignment instruction."""
        self._stack_pos -= 1
        self._rpn.append(SToken(Cmd=ECmdCode.cmASSIGN, ptr=var_ref))

    def add_fun(self, cb: ParserCallback, argc: int, is_optimizable: bool) -> None:
        """Emit a function call instruction, folding constant arguments when possible."""
        sz = len(self._rpn)
        optimize = False

        if is_optimizable and self._enable_optimizer and argc > 0:
            if cb.func is MathImpl.UnaryPlus:
                return

            if sz >= argc and all(
                self._rpn[sz - i - 1].Cmd == ECmdCode.cmVAL for i in range(argc)
            ):
                optimize = True

        if optimize:
            args = [self._rpn[sz - argc + i].data2 for i in range(argc)]
            val = cb.call_fun(*args)
            del self._rpn[sz - argc :]
            self._rpn.append(SToken(Cmd=ECmdCode.cmVAL, ptr=None, data=0.0, data2=val))
        else:
            self._rpn.append(SToken(Cmd=ECmdCode.cmFUNC, argc=argc, cb=cb))

        self._stack_pos = self._stack_pos - abs(argc) + 1
        self._max_stack_size = max(self._max_stack_size, self._stack_pos)

    def add_bulk_fun(self, cb: ParserCallback, argc: int) -> None:
        """Emit a bulk-mode function call instruction."""
        self._stack_pos = self._stack_pos - argc + 1
        self._max_stack_size = max(self._max_stack_size, self._stack_pos)
        self._rpn.append(SToken(Cmd=ECmdCode.cmFUNC_BULK, argc=argc, cb=cb))

    def add_str_fun(self, cb: ParserCallback, argc: int, idx: int) -> None:
        """Emit a string-accepting function call instruction."""
        self._stack_pos = self._stack_pos - argc + 1
        self._max_stack_size = max(self._max_stack_size, self._stack_pos)
        self._rpn.append(SToken(Cmd=ECmdCode.cmFUNC_STR, argc=argc, idx=idx, cb=cb))

    def add_mult_str_fun(self, cb: ParserCallback, argc: int, idx: int) -> None:
        """Emit a variadic string-accepting function call instruction."""
        self._stack_pos = self._stack_pos - argc + 1
        self._max_stack_size = max(self._max_stack_size, self._stack_pos)
        self._rpn.append(SToken(Cmd=ECmdCode.cmFUNC_STR, argc=-argc, idx=idx, cb=cb))

    def finalize(self) -> None:
        """Append cmEND and compute jump offsets for all if-else-endif blocks."""
        self._rpn.append(SToken(Cmd=ECmdCode.cmEND))

        st_if: list[int] = []
        st_else: list[int] = []
        for i, tok in enumerate(self._rpn):
            if tok.Cmd == ECmdCode.cmIF:
                st_if.append(i)
            elif tok.Cmd == ECmdCode.cmELSE:
                st_else.append(i)
                if not st_if:
                    raise ParserError(EErrorCodes.ecINTERNAL_ERROR)
                idx = st_if.pop()
                self._rpn[idx].offset = i - idx
            elif tok.Cmd == ECmdCode.cmENDIF:
                if not st_else:
                    raise ParserError(EErrorCodes.ecINTERNAL_ERROR)
                idx = st_else.pop()
                self._rpn[idx].offset = i - idx

    def store_environment(self, expr: str, string_buf: list[str]) -> None:
        """Save the compiled expression string and string literal buffer."""
        self._expr = expr
        self._string_buffer = list(string_buf)

    def restore_environment(self) -> tuple[str, list[str]]:
        """Return the stored expression string and string literal buffer."""
        return self._expr, list(self._string_buffer)

    def get_base(self) -> list[SToken]:
        """Return the underlying list of RPN bytecode tokens."""
        return self._rpn

    def get_max_stack_size(self) -> int:
        """Return the maximum evaluation stack depth required."""
        return self._max_stack_size + 1

    def get_size(self) -> int:
        """Return the number of tokens in the RPN stream."""
        return len(self._rpn)

    def clear(self) -> None:
        """Reset the bytecode stream and stack counters."""
        self._rpn.clear()
        self._stack_pos = 0
        self._max_stack_size = 0

    # C++ style aliases
    EnableOptimizer = enable_optimizer
    AddVar = add_var
    AddVal = add_val
    AddOp = add_op
    AddIfElse = add_if_else
    AddAssignOp = add_assign_op
    AddFun = add_fun
    AddBulkFun = add_bulk_fun
    AddStrFun = add_str_fun
    AddMultStrFun = add_mult_str_fun
    Finalize = finalize
    StoreEnvironment = store_environment
    RestoreEnvironment = restore_environment
    GetBase = get_base
    GetMaxStackSize = get_max_stack_size
    GetSize = get_size
